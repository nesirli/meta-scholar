"""Evaluate retrieval and prompt approaches."""

import json
import sys
from collections.abc import Callable
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel
from tqdm.auto import tqdm

from metascholar.config import settings
from metascholar.rag.rag_init import RAG, INSTRUCTIONS
from metascholar.rag.judge import evaluate_relevance

TEST_QUERIES = [
    "What computational pipelines are used for metagenomics analysis?",
    "How does diet affect the gut microbiome?",
    "What tools are used for metagenomic binning?",
    "effect of probiotics on gut microbiota",
    "metagenomic sequencing methods for soil samples",
    "machine learning approaches for microbiome classification",
    "antibiotic resistance genes in metagenomes",
    "human gut microbiome diversity across populations",
    "viral metagenomics pipeline",
    "metatranscriptomics tools and methods",
]

PROMPT_A = INSTRUCTIONS  # current: concise with citation example

PROMPT_B = (
    "You are an academic research assistant specializing in metagenomics. "
    "Provide a thorough, well-structured answer using only the provided context. "
    "Begin with a one-sentence summary, then elaborate with key findings. "
    "If the answer is not in the context, say 'I don't know.' "
    "Cite every statement with the source number, e.g. [1] or [2]."
)


# --- cost tracking (ported from llm-zoomcamp) ---

def calc_price(usage):
    """Return the dollar cost of one OpenAI response usage object.

    Defaults to gpt-4o-mini pricing: $0.15 per million input tokens and
    $0.60 per million output tokens.
    """
    input_price_per_million = 0.15
    output_price_per_million = 0.60

    input_cost = (usage.input_tokens / 1_000_000) * input_price_per_million
    output_cost = (usage.output_tokens / 1_000_000) * output_price_per_million
    total_cost = input_cost + output_cost

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }


def calc_total_price(usages):
    """Sum the cost of many OpenAI response usage objects."""
    return sum(calc_price(u)["total_cost"] for u in usages)


# --- retrieval metrics (ported from llm-zoomcamp) ---

def compute_relevance(q, search_function: Callable):
    """Return a 0/1 list showing where the expected PMID appears."""
    expected_pmid = q["pmid"]
    results = search_function(query=q["question"])
    return [int(r["pmid"] == expected_pmid) for r in results]


def compute_relevance_total(ground_truth: list[dict], search_function: Callable):
    """Run compute_relevance for every ground-truth record."""
    relevance_total = []
    for q in tqdm(ground_truth, desc="Evaluating retrieval"):
        relevance = compute_relevance(q, search_function)
        relevance_total.append(relevance)
    return relevance_total


def hit_rate(relevance_total: list[list[int]]) -> float:
    """Fraction of queries where the expected document appears anywhere."""
    if not relevance_total:
        return 0.0
    hits = sum(1 for line in relevance_total if 1 in line)
    return hits / len(relevance_total)


def mrr(relevance_total: list[list[int]]) -> float:
    """Mean Reciprocal Rank: rewards finding the document early."""
    if not relevance_total:
        return 0.0
    total_score = 0.0
    for line in relevance_total:
        for rank, value in enumerate(line):
            if value == 1:
                total_score += 1 / (rank + 1)
                break
    return total_score / len(relevance_total)


def evaluate_search_quality(ground_truth: list[dict], search_function: Callable) -> dict:
    """Return hit_rate and mrr for a search function."""
    relevance_total = compute_relevance_total(ground_truth, search_function)
    return {
        "hit_rate": hit_rate(relevance_total),
        "mrr": mrr(relevance_total),
    }


# --- ground-truth generation (ported from llm-zoomcamp) ---

class _Questions(BaseModel):
    questions: list[str]


JUDGE_GROUND_TRUTH_INSTRUCTIONS = (
    "You emulate a researcher reading PubMed abstracts about metagenomics. "
    "Write questions this researcher might ask based on the record. "
    "The record must contain the answer. Use as few words from the record as possible. "
    "Keep questions natural and concise."
)


def generate_ground_truth_for_record(client: OpenAI, record: dict, n_questions: int = 3) -> list[dict]:
    """Ask an LLM to write questions about one corpus record."""
    response = client.responses.parse(
        model="gpt-4o-mini",
        input=[
            {"role": "developer", "content": JUDGE_GROUND_TRUTH_INSTRUCTIONS},
            {"role": "user", "content": json.dumps(record)},
        ],
        text_format=_Questions,
    )
    return [
        {"question": q, "pmid": record["pmid"]}
        for q in response.output_parsed.questions[:n_questions]
    ]


def generate_ground_truth(
    corpus_path: Path | str,
    output_path: Path | str,
    max_records: int = 20,
):
    """Generate a ground-truth file of (question, pmid) pairs from the corpus.

    This mirrors the ground-truth generation step in llm-zoomcamp. You run it
    once, then reuse the file for retrieval metrics such as Hit Rate and MRR.
    """
    corpus_path = Path(corpus_path)
    output_path = Path(output_path)

    client = OpenAI(api_key=settings.openai_api_key)
    rag = RAG(client=client, corpus_path=corpus_path, top_k=5)
    records = rag._records[:max_records]

    if not records:
        print(f"Corpus not found or empty at {corpus_path}")
        return []

    ground_truth = []
    for record in tqdm(records, desc="Generating ground truth"):
        try:
            questions = generate_ground_truth_for_record(client, record)
            ground_truth.extend(questions)
        except Exception as e:
            print(f"Skipping PMID {record.get('pmid')}: {e}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf8") as fp:
        for item in ground_truth:
            fp.write(json.dumps(item) + "\n")

    print(f"Saved {len(ground_truth)} ground-truth records to {output_path}")
    return ground_truth


# --- existing evaluation functions, enhanced ---

def evaluate_retrieval():
    """Compare keyword vs vector vs hybrid search coverage."""
    client = OpenAI(api_key=settings.openai_api_key)
    rag = RAG(client=client, corpus_path=settings.corpus_path, top_k=5)

    print(f"{'Query':<50} {'Keyword':>8} {'Vector':>8} {'Overlap':>8}")
    print("-" * 78)

    kw_total = 0
    vec_total = 0

    for query in TEST_QUERIES:
        kw = {r["pmid"] for r in rag.search_keyword(query)}
        vec = {r["pmid"] for r in rag.search_vector(query)}

        kw_total += len(kw)
        vec_total += len(vec)
        overlap = len(kw & vec)

        short = query[:48] + (".." if len(query) > 48 else "")
        print(f"{short:<50} {len(kw):>8} {len(vec):>8} {overlap:>8}")

    print("-" * 78)
    print(f"{'AVERAGE':<50} {kw_total / len(TEST_QUERIES):>8.1f} "
          f"{vec_total / len(TEST_QUERIES):>8.1f}")
    print()

    all_kw = set()
    all_vec = set()
    for query in TEST_QUERIES:
        all_kw |= {r["pmid"] for r in rag.search_keyword(query)}
        all_vec |= {r["pmid"] for r in rag.search_vector(query)}

    kw_only = all_kw - all_vec
    vec_only = all_vec - all_kw

    print("Across all queries:")
    print(f"  Found by both:     {len(all_kw & all_vec)}")
    print(f"  Keyword only:      {len(kw_only)}")
    print(f"  Vector only:       {len(vec_only)}")
    print(f"  Total unique:      {len(all_kw | all_vec)}")

    if vec_only:
        print("\nVector-only finds (not found by keyword):")
        for pmid in sorted(vec_only)[:3]:
            print(f"  PMID {pmid}")


def evaluate_prompts():
    """Compare two prompt templates by judge relevance scores."""
    client = OpenAI(api_key=settings.openai_api_key)
    rag = RAG(client=client, corpus_path=settings.corpus_path, top_k=5)

    print("Evaluating prompt variants on 5 queries...\n")
    prompts = {"Concise": PROMPT_A, "Detailed": PROMPT_B}
    scores = {name: 0 for name in prompts}
    usages = []

    for i, query in enumerate(TEST_QUERIES[:5], 1):
        results = rag.search_hybrid(query)
        context = rag.build_context(results)
        print(f"Q{i}: {query[:70]}")

        for name, prompt_instructions in prompts.items():
            full_prompt = rag.build_prompt(query, context)
            response = client.responses.create(
                model="gpt-4o-mini",
                instructions=prompt_instructions,
                input=full_prompt,
            )
            usages.append(response.usage)
            answer = response.output_text
            relevance, _ = evaluate_relevance(query, answer, client)
            # Score: RELEVANT=2, PARTLY_RELEVANT=1, NON_RELEVANT=0
            if relevance == "RELEVANT":
                scores[name] += 2
            elif relevance == "PARTLY_RELEVANT":
                scores[name] += 1
            print(f"  {name:<10} -> {relevance}")

        print()

    print("Results:")
    for name, score in scores.items():
        print(f"  {name}: {score}/10 ({score/10:.0%})")

    winner = max(scores, key=scores.get)
    print(f"\nWinner: {winner}")

    total_cost = calc_total_price(usages)
    print(f"Estimated generation cost: ${total_cost:.4f}")


def evaluate_search_with_ground_truth(ground_truth_path: Path | str | None = None):
    """Score keyword, vector, and hybrid search with Hit Rate and MRR."""
    if ground_truth_path is None:
        ground_truth_path = settings.corpus_path.parent / "ground_truth.jsonl"

    ground_truth_path = Path(ground_truth_path)
    if not ground_truth_path.exists():
        print(f"Ground truth not found at {ground_truth_path}")
        print("Generate it first with: python -m metascholar.rag.evaluate --generate-ground-truth")
        return

    ground_truth = []
    with ground_truth_path.open(encoding="utf8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                ground_truth.append(json.loads(line))

    client = OpenAI(api_key=settings.openai_api_key)
    rag = RAG(client=client, corpus_path=settings.corpus_path, top_k=5)

    def make_search(method: str):
        def search(query: str) -> list[dict]:
            if method == "hybrid":
                return rag.search_hybrid(query)
            elif method == "vector":
                return rag.search_vector(query)
            return rag.search_keyword(query)
        return search

    print(f"{'Method':<10} {'Hit Rate':>10} {'MRR':>10}")
    print("-" * 32)
    for method in ["keyword", "vector", "hybrid"]:
        scores = evaluate_search_quality(ground_truth, make_search(method))
        print(f"{method:<10} {scores['hit_rate']:>10.3f} {scores['mrr']:>10.3f}")


if __name__ == "__main__":
    if "--generate-ground-truth" in sys.argv:
        generate_ground_truth(
            settings.corpus_path,
            settings.corpus_path.parent / "ground_truth.jsonl",
        )
    else:
        evaluate_retrieval()
        print("\n" + "=" * 78 + "\n")
        evaluate_prompts()
        print("\n" + "=" * 78 + "\n")
        evaluate_search_with_ground_truth()
