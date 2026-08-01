from metascholar.rag.rag_init import RAG

assistant = RAG()

query = "What computational pipelines are used for metagenomics analysis?"
answer = assistant.rag(query)
print(answer)