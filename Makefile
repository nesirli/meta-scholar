# Locally, `uv run` provisions the environment automatically. Inside the
# container the venv is already on PATH, so the Dockerfile sets PYTHON=python
# and STREAMLIT=streamlit to avoid re-syncing the running app's environment.
PYTHON ?= uv run python
STREAMLIT ?= uv run streamlit

get_data:
	$(PYTHON) -m metascholar.ingest.fetch_data

init:
	$(PYTHON) -m metascholar.database.db_init

test_rag:
	$(PYTHON) -m metascholar.rag.rag_test

evaluate:
	$(PYTHON) -m metascholar.rag.evaluate

run_app:
	$(STREAMLIT) run src/metascholar/app/app.py
