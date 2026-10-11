import pytest

from app.chunking import chunk_filing
from app.retrieval import HybridIndex
from app.tools import safe_eval
from app.verifier import numbers_in, verify
from evals.metrics import numeric_match, summarize

HTML = """
<html><body>
<div>Item 1A. Risk Factors</div>
<p>Our business depends on supply chain partners in Asia and any disruption could harm results of operations materially.</p>
<p>Competition in the smartphone market is intense and pricing pressure may reduce our gross margin over time.</p>
<div>Item 7. Management's Discussion and Analysis</div>
<p>Total net sales increased 8 percent driven by growth in Services and strong iPhone demand across all geographic segments.</p>
<table><tr><td>Net sales</td><td>$</td><td>383,285</td><td>394,328</td></tr>
<tr><td>Research and development</td><td>$</td><td>29,915</td><td>26,251</td></tr></table>
</body></html>
"""


def _chunks():
    return chunk_filing(HTML, ticker="AAPL", form="10-K", period="2023-09-30", url="http://x")


def test_chunking_tracks_sections_and_tables():
    cs = _chunks()
    assert any("Risk Factors" in c.section for c in cs)
    assert any("MD&A" in c.section for c in cs)
    assert any(c.kind == "table" and "383,285" in c.text for c in cs)


def test_hybrid_search_filters_and_ranks():
    idx = HybridIndex()
    idx.build(_chunks())
    hits = idx.search("supply chain disruption risk", k=3, ticker="AAPL")
    assert hits and "supply chain" in hits[0]["text"].lower()
    assert idx.search("anything", ticker="MSFT") == []


def test_calculator_is_safe():
    assert safe_eval("(110-100)/100*100") == pytest.approx(10.0)
    with pytest.raises(Exception):
        safe_eval("__import__('os').system('echo hi')")


def test_numbers_units():
    assert 383.3e9 in [round(n, -8) for n in numbers_in("revenue was $383.3 billion")] or \
        any(abs(n - 383.3e9) < 1e6 for n in numbers_in("revenue was $383.3 billion"))


def test_verifier_flags_unsupported_numbers():
    evidence = [{"value": 383285000000}]
    assert verify("Revenue was $383.3 billion.", evidence)["ok"]
    bad = verify("Revenue was $999.9 billion.", evidence)
    assert not bad["ok"] and bad["unsupported_numbers"]


def test_numeric_match_and_summary():
    assert numeric_match("It was $383.3 billion", 383285000000)
    assert not numeric_match("It was $500 billion", 383285000000)
    assert numeric_match("changed by 12.3%", 12.31)
    s = summarize([
        {"type": "point", "correct": True, "verified": True, "latency_s": 1.0},
        {"type": "point", "correct": False, "verified": False, "latency_s": 2.0},
    ])
    assert s["numeric_accuracy"] == 0.5 and s["hallucination_rate"] == 0.5
