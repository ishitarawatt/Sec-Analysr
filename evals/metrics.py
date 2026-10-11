"""Scoring functions. Kept dependency-free so they are unit-testable."""
from __future__ import annotations

import re

from app.verifier import numbers_in


def numeric_match(answer: str, gold: float, rel_tol: float = 0.01, abs_tol: float = 0.06) -> bool:
    """True if any number in the answer matches gold within tolerance.
    Handles '$383.3 billion' vs 383285000000 via unit-aware extraction."""
    for n in numbers_in(answer):
        if abs(gold) >= 1000:
            if abs(n - gold) <= rel_tol * abs(gold):
                return True
        else:  # percentages / small values (EPS)
            if abs(n - gold) <= max(abs_tol, rel_tol * abs(gold)):
                return True
    return False


def summarize(results: list[dict]) -> dict:
    n = len(results) or 1
    by_type: dict[str, list[bool]] = {}
    for r in results:
        by_type.setdefault(r["type"], []).append(r["correct"])
    return {
        "n": len(results),
        "numeric_accuracy": round(sum(r["correct"] for r in results) / n, 4),
        "verified_rate": round(sum(r["verified"] for r in results) / n, 4),
        # hallucination = confident answer, flagged unsupported numbers
        "hallucination_rate": round(sum(not r["verified"] for r in results) / n, 4),
        "p50_latency_s": _pct([r["latency_s"] for r in results], 50),
        "p95_latency_s": _pct([r["latency_s"] for r in results], 95),
        "by_type": {k: round(sum(v) / len(v), 4) for k, v in by_type.items()},
    }


def _pct(xs: list[float], p: int) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(len(xs) * p / 100))], 3)
