get_data:
	uv run python -m metascholar.ingest.fetch_data

test_rag:
	uv run python -m metascholar.rag.rag_test