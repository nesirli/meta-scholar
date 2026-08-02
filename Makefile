get_data:
	uv run python -m metascholar.ingest.fetch_data

init:
	uv run python -m metascholar.database.db_init

test_rag:
	uv run python -m metascholar.rag.rag_test

evaluate:
	uv run python -m metascholar.rag.evaluate

run_app:
	uv run streamlit run src/metascholar/app/app.py