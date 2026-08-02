from openai import OpenAI

from metascholar.config import settings
from metascholar.rag.rag_init import RAG


def create_assistant() -> RAG:
    client = OpenAI(api_key=settings.openai_api_key)
    return RAG(client=client, corpus_path=settings.corpus_path)


if __name__ == "__main__":
    assistant = create_assistant()

    # query = "What computational pipelines are used for metagenomics analysis?"
    # query = "What does PMID 12345678 study?" # Should return "I don't know"
    query = "What does PMID 42533554 study?"  # Should return a topic about short-term travel and microbiota stability.

    print(f"Q: {query}\n")

    results = assistant.search(query)
    context = assistant.build_context(results)
    print(f"CONTEXT:\n{context}\n")
    print(f"Tokens in context: ~{len(context.split())}")

    print(f"Found {len(results)} matching articles:")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] {r['title']} ({r.get('year', '?')})")
    print()

    result = assistant.query(query)
    print(result.answer)
    print(
        f"\n— {result.model} | {result.response_time}s | {result.total_tokens} tokens"
    )
