import httpx
import xml.etree.ElementTree as ET
import time
import json
from pathlib import Path

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

SEARCH_QUERY = (
    "metagenomics AND (microbiome OR microbial community) "
    "AND (bioinformatics OR pipeline OR analysis)"
)

OUT_PATH = Path("data/corpus.jsonl")


def _retry(func, *args, max_retries=3, **kwargs):
    """Call func with exponential backoff on HTTP errors (502, 429, etc.)."""
    for attempt in range(max_retries):
        try:
            resp = func(*args, **kwargs)
            resp.raise_for_status()
            return resp
        except httpx.HTTPStatusError:
            if attempt == max_retries - 1:
                raise
            time.sleep(2**attempt)  # 1s, 2s, 4s


def search_pmids(query: str, retmax: int = 200) -> list[str]:
    """Search PubMed via E-utilities and return matching PMIDs."""
    url = f"{EUTILS_BASE}/esearch.fcgi"
    params = {"db": "pubmed", "term": query, "retmax": retmax, "retmode": "json"}
    resp = _retry(httpx.get, url, params=params, timeout=30.0)
    return resp.json()["esearchresult"]["idlist"]


def fetch_records(pmids: list[str]) -> str:
    """Fetch full PubMed XML records for a batch of PMIDs (max ~100 per request)."""
    url = f"{EUTILS_BASE}/efetch.fcgi"
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
    resp = _retry(httpx.post, url, data=params, timeout=60.0)
    return resp.text


def _parse_article(article: ET.Element) -> dict | None:
    """Extract PMID, title, abstract, year, and journal from a PubmedArticle XML element."""
    pmid = article.findtext(".//PMID")
    abstract_parts = []
    for sec in article.findall(".//AbstractText"):
        label = sec.get("Label")
        text = (sec.text or "").strip()
        if text:
            abstract_parts.append(f"{label}: {text}" if label else text)
    abstract = " ".join(abstract_parts)
    if not pmid or not abstract:
        return None
    return {
        "pmid": pmid,
        "title": article.findtext(".//ArticleTitle") or "",
        "abstract": abstract,
        "year": article.findtext(".//PubDate/Year") or "",
        "journal": article.findtext(".//Journal/Title") or "",
    }


def parse_records(xml_text: str) -> list[dict]:
    """Parse an NCBI efetch XML response into a list of article dicts."""
    root = ET.fromstring(xml_text)
    records = []
    for article in root.findall(".//PubmedArticle"):
        parsed = _parse_article(article)
        if parsed is not None:
            records.append(parsed)
    return records


def fetch_all(pmids: list[str], batch_size: int = 100, delay: float = 0.4) -> list[dict]:
    """Fetch and parse all PMIDs in batches, respecting NCBI rate limits."""
    records = []
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        records.extend(parse_records(fetch_records(batch)))
        time.sleep(delay)
    return records


def write_jsonl(records: list[dict], path: Path) -> int:
    """Write records to a JSONL file, deduplicating by PMID. Returns count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            if r["pmid"] in seen:
                continue
            seen.add(r["pmid"])
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(seen)


def main() -> None:
    """Search PubMed, fetch full records, and write them to OUT_PATH."""
    pmids = search_pmids(SEARCH_QUERY, retmax=300)
    print(f"found {len(pmids)} PMIDs, fetching…")
    records = fetch_all(pmids)
    written = write_jsonl(records, OUT_PATH)
    print(f"parsed {len(records)} records, wrote {written} unique → {OUT_PATH}")


if __name__ == "__main__":
    main()