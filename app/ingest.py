"""Download filings from EDGAR, chunk them, and persist chunks. Run:  python -m app.ingest"""
from __future__ import annotations

import sys

from . import edgar
from .chunking import chunk_filing
from .config import settings
from .retrieval import HybridIndex


def ingest(tickers: list[str] | None = None, per_ticker: int | None = None) -> int:
    tickers = tickers or [t.strip() for t in settings.tickers.split(",") if t.strip()]
    per_ticker = per_ticker or settings.filings_per_ticker
    all_chunks = []
    for t in tickers:
        try:
            cik, name = edgar.ticker_to_cik(t)
            for f in edgar.list_filings(cik, limit=per_ticker):
                html = edgar.fetch_filing_html(cik, f)
                url = edgar.filing_url(cik, f["accession"], f["doc"])
                cs = chunk_filing(html, ticker=t.upper(), form=f["form"], period=f["period"], url=url)
                print(f"{t} {f['form']} {f['period']}: {len(cs)} chunks", flush=True)
                all_chunks.extend(cs)
        except Exception as e:
            print(f"! {t}: {e}", file=sys.stderr)
    idx = HybridIndex()
    idx.build(all_chunks) if all_chunks else None
    idx.chunks = all_chunks
    idx.save()
    print(f"Saved {len(all_chunks)} chunks")
    return len(all_chunks)


if __name__ == "__main__":
    ingest(sys.argv[1].split(",") if len(sys.argv) > 1 else None)
