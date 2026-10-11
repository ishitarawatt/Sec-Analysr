"""Section-aware, table-aware chunking for 10-K / 10-Q HTML."""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict

from bs4 import BeautifulSoup

ITEM_RE = re.compile(r"^\s*item\s+(\d+[a-c]?)\.?\s*[-–—:.]?\s*(.{0,80})", re.I)

SECTION_NAMES = {
    "1": "Business",
    "1A": "Risk Factors",
    "2": "Properties",
    "3": "Legal Proceedings",
    "7": "MD&A",
    "7A": "Market Risk",
    "8": "Financial Statements",
}


@dataclass
class Chunk:
    id: str
    text: str
    ticker: str
    form: str
    period: str
    section: str
    url: str
    kind: str = "text"  # text | table

    def to_dict(self):
        return asdict(self)


def _table_to_text(tbl) -> str:
    rows = []
    for tr in tbl.find_all("tr"):
        cells = [re.sub(r"\s+", " ", td.get_text(" ", strip=True)) for td in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c and c not in {"$", ")", "%"}]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def parse_blocks(html: str) -> list[tuple[str, str]]:
    """Return ordered (kind, text) blocks from filing HTML."""
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style"]):
        t.decompose()
    blocks: list[tuple[str, str]] = []
    body = soup.body or soup
    for el in body.find_all(["p", "div", "table", "span"], recursive=True):
        if el.name == "table":
            txt = _table_to_text(el)
            if len(txt) > 40:
                blocks.append(("table", txt))
            el.decompose()
        elif el.name in ("p", "div") and not el.find(["p", "div", "table"]):
            txt = re.sub(r"\s+", " ", el.get_text(" ", strip=True))
            if len(txt) > 20:
                blocks.append(("text", txt))
    return blocks


def chunk_filing(html: str, *, ticker: str, form: str, period: str, url: str,
                 max_chars: int = 1800, overlap: int = 200) -> list[Chunk]:
    blocks = parse_blocks(html)
    section = "Preamble"
    chunks: list[Chunk] = []
    buf: list[str] = []

    def flush():
        nonlocal buf
        text = "\n".join(buf).strip()
        buf = []
        if len(text) < 80:
            return
        # split long text with overlap
        start = 0
        while start < len(text):
            piece = text[start : start + max_chars]
            cid = f"{ticker}-{form}-{period}-{len(chunks)}"
            chunks.append(Chunk(cid, piece, ticker, form, period, section, url))
            if start + max_chars >= len(text):
                break
            start += max_chars - overlap

    for kind, txt in blocks:
        m = ITEM_RE.match(txt) if kind == "text" and len(txt) < 120 else None
        if m:
            flush()
            code = m.group(1).upper()
            section = f"Item {code} {SECTION_NAMES.get(code, m.group(2).strip().title())}"
            continue
        if kind == "table":
            flush()
            cid = f"{ticker}-{form}-{period}-{len(chunks)}"
            chunks.append(Chunk(cid, txt[:max_chars * 2], ticker, form, period, section, url, "table"))
        else:
            buf.append(txt)
            if sum(len(b) for b in buf) > max_chars:
                flush()
    flush()
    return chunks
