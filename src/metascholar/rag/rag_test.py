from metascholar.rag.rag_init import assistant


if __name__ == "__main__":
    # query = "What computational pipelines are used for metagenomics analysis?"
    # query = "What does PMID 12345678 study?" # Should return "I don't know"
    query = "What does PMID 42533554 study?"  # Should return a topic about short-term travel and microbiota stability.

    result = assistant.query(query)
    print(result.answer)
    print(
        f"\n— {result.model} | {result.response_time}s | {result.total_tokens} tokens"
    )
