import json
import time
from pathlib import Path
from openai import OpenAI

from metascholar.rag.schemas import LLMCallRecord
from metascholar.config import settings
from metascholar.app.db_query import pg_vector_search

INSTRUCTIONS = (
    "You are a research assistant answering questions about metagenomics literature. "
    "Use only the provided context to answer. If the answer is not in the context, "
    "say 'I don't know.' "
    "Cite every statement with the source number from the context, e.g. [1] or [2]. "
    "Example: 'Metagenomic binning uses CONCOCT and MetaBAT [1].'"
)

PROMPT_TEMPLATE = """QUESTION: {question}

CONTEXT:
{context}""".strip()


class RAG:
    """Simple keyword-search RAG over a JSONL corpus of PubMed articles."""

    _PRICING = {
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4o": (2.50, 10.00),
    }

    def __init__(self, client: OpenAI, corpus_path: Path, top_k: int = 5):
        self._client = client
        self._top_k = top_k
        self._records: list[dict] = self._load_corpus(corpus_path)

    @staticmethod
    def _load_corpus(path: Path) -> list[dict]:
        if not path.exists():
            # Corpus is fetched post-deploy via `make get_data`; until then the
            # app should still start (keyword search just returns nothing).
            print(f"corpus not found at {path}; run `make get_data` to fetch it")
            return []
        records = []
        with path.open(encoding="utf8") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    @classmethod
    def _calculate_cost(
        cls, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        for key, (in_price, out_price) in cls._PRICING.items():
            if key in model:
                return (input_tokens * in_price + output_tokens * out_price) / 1_000_000
        return 0.0

    def search_keyword(self, query: str) -> list[dict]:
        """Score records by how many query tokens appear in title + abstract."""
        _STOPWORDS = {
            "the", "a", "an", "is", "are", "was", "were", "in", "of",
            "to", "for", "with", "on", "at", "by", "from", "what", "how",
            "does", "study", "studies", "analysis", "using", "used", "based", "data",
        }
        tokens = set(query.lower().split()) - _STOPWORDS
        scored = []
        for record in self._records:
            text = f"{record.get('pmid', '')} {record.get('title', '')} {record.get('abstract', '')}".lower()
            score = sum(1 for token in tokens if token in text)
            if score > 0:
                scored.append((score, record))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[: self._top_k]]

    def search_vector(self, query: str) -> list[dict]:
        """Search via pgvector cosine similarity."""
        return pg_vector_search(query, top_k=self._top_k)

    def search_hybrid(self, query: str) -> list[dict]:
        """Combine keyword and vector results with reciprocal rank fusion."""
        k = 60
        kw_results = self.search_keyword(query)
        vec_results = self.search_vector(query)
        scores: dict[str, float] = {}
        seen: dict[str, dict] = {}
        for rank, r in enumerate(kw_results):
            scores[r["pmid"]] = scores.get(r["pmid"], 0) + 1 / (rank + k)
            seen[r["pmid"]] = r
        for rank, r in enumerate(vec_results):
            scores[r["pmid"]] = scores.get(r["pmid"], 0) + 1 / (rank + k)
            seen.setdefault(r["pmid"], r)
        merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [seen[pmid] for pmid, _ in merged[: self._top_k]]

    # Backward-compatible alias
    search = search_keyword

    def build_context(self, results: list[dict]) -> str:
        parts = []
        for i, result in enumerate(results, 1):
            parts.append(
                f"Source [{i}]: PMID {result['pmid']} - {result['title']} "
                f"({result.get('year', '?')}, {result.get('journal', '?')})\n"
                f"{result['abstract']}"
            )
        return "\n\n".join(parts)

    def build_prompt(self, question: str, context: str) -> str:
        return PROMPT_TEMPLATE.format(question=question, context=context)

    def llm(self, prompt: str) -> LLMCallRecord:
        t0 = time.perf_counter()
        response = self._client.responses.create(
            model="gpt-4o-mini",
            instructions=INSTRUCTIONS,
            input=prompt,
        )
        elapsed = time.perf_counter() - t0
        usage = response.usage
        return LLMCallRecord(
            model=response.model,
            prompt=prompt,
            instructions=INSTRUCTIONS,
            answer=response.output_text,
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            response_time=round(elapsed, 3),
            cost=self._calculate_cost(
                response.model, usage.input_tokens, usage.output_tokens
            ),
        )

    def query(self, question: str, method: str = "hybrid") -> LLMCallRecord:
        if method == "hybrid":
            results = self.search_hybrid(question)
        elif method == "vector":
            results = self.search_vector(question)
        else:
            results = self.search_keyword(question)
        if not results:
            context = "(No relevant articles found.)"
        else:
            context = self.build_context(results)
        prompt = self.build_prompt(question, context)
        record = self.llm(prompt)
        record.question = question
        record.sources = results
        return record


def create_assistant() -> RAG:
    client = OpenAI(api_key=settings.openai_api_key)
    return RAG(client=client, corpus_path=settings.corpus_path)

assistant = create_assistant()