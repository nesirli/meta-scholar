# MetaScholar

> RAG-based question answering over the metagenomics & microbiome literature — ask a question, get an answer grounded in PubMed abstracts, with citations.

![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![Status](https://img.shields.io/badge/status-in%20development-orange)
![License](https://img.shields.io/badge/license-MIT-green)

**Status:** 🚧 Early development — built in public, one milestone at a time. The roadmap below reflects the current state honestly; checked items work today.

---

## What it does

MetaScholar is a retrieval-augmented generation (RAG) system that answers natural-language questions about metagenomics and microbiome research. Instead of relying on what a language model happens to have memorized, it:

1. **Retrieves** relevant passages from a corpus of PubMed abstracts,
2. **Grounds** an LLM's answer in that retrieved context, and
3. **Returns** the answer with source citations — so every claim is traceable back to a paper.

The goal is answers you can *trust and verify*, not plausible-sounding hallucinations.

> This describes the target system. See the [Roadmap](#roadmap) for what's built so far.

## Why this project

Most RAG demos are built by web developers who can't evaluate whether the answers are actually correct. MetaScholar is the opposite: it pairs LLM application engineering with real domain expertise in clinical microbiology, so the system can be both *built* and *critically judged*. The metagenomics literature is a coherent, methods-rich corpus where grounded, citable answers genuinely matter.

## Tech stack

| Layer | Tool | Status |
|---|---|---|
| API & serving | FastAPI (async) | ✅ in place |
| Dependency management | uv | ✅ in place |
| Corpus | PubMed E-utilities (`httpx`) → JSONL | ✅ in place |
| LLM & embeddings | OpenAI API | ⏳ planned (M2) |
| Vector store | PostgreSQL + pgvector | ⏳ planned (M4) |
| Retrieval quality | hybrid search + reranking | ⏳ planned (M5) |
| Evaluation | LLM-as-a-judge + retrieval metrics | ⏳ planned (M7) |
| Observability | Grafana dashboards | ⏳ planned (M8) |
| Infra | Docker | ⏳ planned (M4 / M10) |

## Architecture (target)

```
            ┌─────────┐   ┌───────────┐   ┌──────────┐   ┌──────────┐
  question →│ retrieve│ → │  augment  │ → │ generate │ → │  cited   │→ answer
            │ (vector │   │  (build   │   │  (LLM)   │   │  answer  │
            │  search)│   │  prompt)  │   │          │   │          │
            └─────────┘   └───────────┘   └──────────┘   └──────────┘
                 ↑
          ┌──────────────┐
          │ PubMed corpus│  (metagenomics / microbiome abstracts)
          └──────────────┘
```

## Roadmap

Built tracer-bullet style: a crude end-to-end pipeline first (M3), then each layer upgraded in place.

- [x] **M0** — Project skeleton: uv, FastAPI app that runs
- [x] **M1** — Corpus: fetch & parse PubMed abstracts → JSONL (288 records via NCBI E-utilities, deduped, unit-tested)
- [ ] **M2** — First LLM call: provider-agnostic OpenAI client, structured output *(in progress)*
- [ ] **M3** — 🎯 Tracer bullet: crude end-to-end RAG (embeddings + cosine similarity in NumPy)
- [ ] **M4** — Real retrieval: pgvector + chunking
- [ ] **M5** — Better retrieval: hybrid search + reranking
- [ ] **M6** — Serving: FastAPI endpoint, async, streaming (SSE)
- [ ] **M7** — Evaluation: eval set, retrieval metrics, LLM-as-a-judge
- [ ] **M8** — Observability & monitoring: tracing, cost tracking, Grafana
- [ ] **M9** — Guardrails + one agentic feature (fetch fresh abstracts on demand)
- [ ] **M10** — Portfolio polish: README, Docker, deploy

## Getting started

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management
- Docker (needed from M4 onward, for Postgres + pgvector)

### Setup

```bash
git clone https://github.com/nesirli/metascholar.git metascholar
cd metascholar
uv sync                        # installs dependencies into .venv from uv.lock
cp .env.example .env           # then fill in OPENAI_API_KEY
```

### Running the API

```bash
uv run uvicorn app.main:app --reload
```

Then open http://127.0.0.1:8000/docs for the interactive Swagger UI, or check `GET /health`.

### Fetching the corpus

```bash
uv run python scripts/fetch_corpus.py
```

Queries NCBI E-utilities for metagenomics/microbiome abstracts, parses the PubMed XML (concatenating labeled abstract sections), dedupes by PMID, and writes `data/corpus.jsonl` (gitignored — regenerate locally rather than committing it).

### Running tests

```bash
uv run pytest
```

## Project structure

```
metascholar/
├── pyproject.toml       # uv-managed dependencies
├── .env / .env.example  # secrets + committed template
├── .gitignore
├── README.md
├── app/
│   ├── __init__.py
│   ├── config.py        # settings (pydantic-settings)
│   ├── main.py          # FastAPI entrypoint
│   └── corpus/
│       └── fetch.py     # NCBI E-utilities search/fetch/parse → JSONL
├── scripts/
│   └── fetch_corpus.py  # CLI entrypoint for the corpus fetch
├── tests/
│   └── test_parse.py
└── data/                # generated corpus (gitignored)
```

Folders for `retrieval/`, `rag/`, and `api/` are added at the milestones that introduce them — not before.

## License

MIT (planned — add a `LICENSE` file before making the repo public).

---

*Built in public as a learning project bridging clinical microbiology and LLM application engineering.*