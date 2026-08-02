import json
import tempfile
from pathlib import Path

import pytest

from metascholar.rag.rag_init import RAG
from metascholar.rag.schemas import LLMCallRecord


@pytest.fixture
def sample_records():
    return [
        {
            "pmid": "1",
            "title": "Metagenomics pipeline analysis",
            "abstract": "We built a pipeline for metagenomics using Snakemake.",
            "year": "2024",
            "journal": "Bioinformatics",
        },
        {
            "pmid": "2",
            "title": "Gut microbiome and diet",
            "abstract": "Diet affects gut microbiome composition significantly.",
            "year": "2023",
            "journal": "Nature",
        },
        {
            "pmid": "3",
            "title": "Computational tools for binning",
            "abstract": "CONCOCT and MetaBAT are popular binning tools.",
            "year": "2025",
            "journal": "Genome Biology",
        },
    ]


@pytest.fixture
def rag_with_data(sample_records):
    """RAG instance loaded from a temp JSONL file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for r in sample_records:
            f.write(json.dumps(r) + "\n")
        temp_path = f.name

    rag = RAG(client=None, corpus_path=Path(temp_path))
    yield rag
    Path(temp_path).unlink()


class TestRAGSearch:
    def test_exact_match_found(self, rag_with_data):
        results = rag_with_data.search("Snakemake")
        assert len(results) == 1
        assert results[0]["pmid"] == "1"

    def test_partial_match_scores(self, rag_with_data):
        results = rag_with_data.search("metagenomics")
        assert len(results) == 1
        assert results[0]["pmid"] == "1"

    def test_multiple_results(self, rag_with_data):
        rag_with_data.search("computational")  # pre-warm
        results = rag_with_data.search("computational tools")
        assert len(results) >= 1
        assert results[0]["pmid"] == "3"

    def test_no_match(self, rag_with_data):
        results = rag_with_data.search("crispr")
        assert results == []

    def test_respects_top_k(self, rag_with_data):
        results = rag_with_data.search("microbiome composition")
        assert len(results) <= 5


class TestRAGBuildContext:
    def test_formats_single_result(self, rag_with_data, sample_records):
        ctx = rag_with_data.build_context(sample_records[:1])
        assert "Source [1]:" in ctx
        assert "PMID 1" in ctx
        assert "Metagenomics pipeline analysis" in ctx
        assert "(2024, Bioinformatics)" in ctx

    def test_formats_multiple_results(self, rag_with_data, sample_records):
        ctx = rag_with_data.build_context(sample_records[:2])
        assert "Source [1]:" in ctx
        assert "Source [2]:" in ctx
        assert "PMID 1" in ctx
        assert "PMID 2" in ctx


class TestRAGBuildPrompt:
    def test_fills_template(self, rag_with_data):
        prompt = rag_with_data.build_prompt("What is metagenomics?", "Some context")
        assert "What is metagenomics?" in prompt
        assert "Some context" in prompt


class TestLLMCallRecord:
    def test_defaults(self):
        r = LLMCallRecord(
            model="gpt-4o-mini",
            prompt="test",
            instructions="test",
            answer="test",
        )
        assert r.question == ""
        assert r.cost == 0.0
        assert r.sources == []
        assert r.prompt_tokens == 0
        assert r.completion_tokens == 0

    def test_custom_fields(self):
        r = LLMCallRecord(
            model="gpt-4o-mini",
            prompt="test prompt",
            instructions="be helpful",
            answer="test answer",
            question="test question",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            response_time=0.5,
            cost=0.001,
        )
        assert r.question == "test question"
        assert r.prompt_tokens == 10
        assert r.completion_tokens == 20
        assert r.total_tokens == 30
        assert r.response_time == 0.5
        assert r.cost == 0.001
