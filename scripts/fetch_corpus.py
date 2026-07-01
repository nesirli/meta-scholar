from pathlib import Path

from app.corpus.fetch import search_pmids, fetch_all, write_jsonl, SEARCH_QUERY

OUT_PATH = Path("data/corpus.jsonl")


def main() -> None:
    pmids = search_pmids(SEARCH_QUERY, retmax=300)
    print(f"found {len(pmids)} PMIDs, fetching…")
    records = fetch_all(pmids)
    written = write_jsonl(records, OUT_PATH)
    print(f"parsed {len(records)} records, wrote {written} unique → {OUT_PATH}")


if __name__ == "__main__":
    main()