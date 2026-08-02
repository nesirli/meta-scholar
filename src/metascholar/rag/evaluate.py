"""Evaluate retrieval and prompt approaches."""

from openai import OpenAI

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
            answer = response.output_text
            relevance, _ = evaluate_relevance(query, answer, client)
            # Score: RELEVANT=2, PARTLY_RELEVANT=1, NON_RELEVANT=0
            if relevance == "RELEVANT":
                scores[name] += 2
            elif relevance == "PARTLY_RELEVANT":
                scores[name] += 1
            print(f"  {name:<10} → {relevance}")

        print()

    print("Results:")
    for name, score in scores.items():
        print(f"  {name}: {score}/10 ({score/10:.0%})")

    winner = max(scores, key=scores.get)
    print(f"\nWinner: {winner}")


if __name__ == "__main__":
    evaluate_retrieval()
    print("\n" + "=" * 78 + "\n")
    evaluate_prompts()
