"""Minimal SEC EDGAR client with on-disk caching and polite rate limiting."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import httpx

from .config import settings

_BASE_DATA = "https://data.sec.gov"
_BASE_WWW = "https://www.sec.gov"
_last_call = 0.0


def _client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": settings.sec_user_agent, "Accept-Encoding": "gzip, deflate"},
        timeout=60,
        follow_redirects=True,
    )


def _throttle() -> None:
    """SEC fair-access policy: <= 10 req/s. We stay well below."""
    global _last_call
    wait = 0.15 - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.time()


def _cache_path(name: str) -> Path:
    p = Path(settings.data_dir) / "cache"
    p.mkdir(parents=True, exist_ok=True)
    return p / re.sub(r"[^A-Za-z0-9._-]", "_", name)


def _get(url: str, cache_name: str, as_json: bool = True):
    cp = _cache_path(cache_name)
    if cp.exists():
        txt = cp.read_text()
        return json.loads(txt) if as_json else txt
    _throttle()
    with _client() as c:
        r = c.get(url)
        r.raise_for_status()
        cp.write_text(r.text)
        return r.json() if as_json else r.text


def ticker_to_cik(ticker: str) -> tuple[str, str]:
    data = _get(f"{_BASE_WWW}/files/company_tickers.json", "company_tickers.json")
    for row in data.values():
        if row["ticker"].upper() == ticker.upper():
            return str(row["cik_str"]).zfill(10), row["title"]
    raise KeyError(f"Unknown ticker {ticker}")


def list_filings(cik: str, forms=("10-K", "10-Q"), limit=2) -> list[dict]:
    sub = _get(f"{_BASE_DATA}/submissions/CIK{cik}.json", f"sub_{cik}.json")
    recent = sub["filings"]["recent"]
    out = []
    for i, form in enumerate(recent["form"]):
        if form in forms:
            out.append(
                {
                    "form": form,
                    "accession": recent["accessionNumber"][i],
                    "date": recent["filingDate"][i],
                    "period": recent["reportDate"][i],
                    "doc": recent["primaryDocument"][i],
                }
            )
        if len(out) >= limit:
            break
    return out


def filing_url(cik: str, accession: str, doc: str) -> str:
    return f"{_BASE_WWW}/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{doc}"


def fetch_filing_html(cik: str, f: dict) -> str:
    url = filing_url(cik, f["accession"], f["doc"])
    return _get(url, f"filing_{cik}_{f['accession']}.html", as_json=False)


def company_facts(cik: str) -> dict:
    return _get(f"{_BASE_DATA}/api/xbrl/companyfacts/CIK{cik}.json", f"facts_{cik}.json")
