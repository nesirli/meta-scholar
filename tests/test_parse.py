from app.corpus.fetch import parse_records

MULTI_SECTION_XML = """<PubmedArticleSet><PubmedArticle><MedlineCitation>
<PMID Version="1">12345678</PMID>
<Article><ArticleTitle>Test</ArticleTitle>
<Abstract>
<AbstractText Label="BACKGROUND">Background text.</AbstractText>
<AbstractText Label="METHODS">Methods text.</AbstractText>
<AbstractText Label="RESULTS">Results text.</AbstractText>
</Abstract>
<Journal><Title>J Test</Title></Journal>
<PubDate><Year>2026</Year></PubDate>
</Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"""


def test_abstract_concatenates_all_sections():
    records = parse_records(MULTI_SECTION_XML)
    assert len(records) == 1
    abstract = records[0]["abstract"]
    assert "Background text." in abstract
    assert "Methods text." in abstract
    assert "Results text." in abstract