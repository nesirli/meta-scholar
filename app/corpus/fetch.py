import httpx
import xml.etree.ElementTree as ET

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Focused query
SEARCH_QUERY = (
    "metagenomics AND (microbiome OR microbial community) "
    "AND (bioinformatics OR pipeline OR analysis)"
)

def search_pmids(query: str, retmax: int = 200) -> list[str]:
    url = f"{EUTILS_BASE}/esearch.fcgi"
    
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": retmax,
        "retmode": "json",
    }

    resp = httpx.get(url, params=params, timeout=30.0)
    resp.raise_for_status()
    return resp.json()["esearchresult"]["idlist"]

def fetch_records(pmids: list[str]) -> str:
    url = f"{EUTILS_BASE}/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    resp = httpx.post(url, data=params, timeout=60.0)
    resp.raise_for_status()
    return resp.text

def _parse_article(article: ET.Element) -> dict | None:
    pmid = article.findtext(".//PMID")
    title = article.findtext(".//ArticleTitle")
    abstract = article.findtext(".//AbstractText")
    year = article.findtext(".//PubDate/Year")
    journal = article.findtext(".//Journal/Title")

    if not pmid or not abstract:
        return None

    return {
        "pmid": pmid,
        "title": title or "",
        "abstract": abstract,
        "year": year or "",
        "journal": journal or "",
    }

def parse_records(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    records = []
    for article in root.findall(".//PubmedArticle"):
        parsed = _parse_article(article)
        if parsed is not None:
            records.append(parsed)
    return records