"""Verifier: every number in the final answer must be traceable to tool output.

Deterministic check (no LLM needed): extract numeric tokens from the answer and
confirm each appears (within tolerance, after unit normalisation) in the evidence
gathered by tools. Unsupported numbers are flagged so the agent can retry."""
from __future__ import annotations

import json
import re

NUM = re.compile(r"\$?\(?-?\d[\d,]*\.?\d*\)?\s*(billion|million|thousand|bn|m|b|k|%)?", re.I)
SCALE = {"billion": 1e9, "bn": 1e9, "b": 1e9, "million": 1e6, "m": 1e6, "thousand": 1e3, "k": 1e3, "%": 1.0}


def _to_float(tok: str, unit: str | None) -> float | None:
    s = re.sub(r"[^\d.\-]", "", tok)
    if not s or s in {"-", "."}:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v * SCALE.get((unit or "").lower(), 1.0)


def numbers_in(text: str) -> list[float]:
    out = []
    for m in NUM.finditer(text):
        v = _to_float(m.group(0).replace(m.group(1) or "", ""), m.group(1))
        if v is not None:
            out.append(v)
    return out


def _close(a: float, b: float, rel: float = 0.012) -> bool:
    if a == b:
        return True
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-9)


def verify(answer: str, evidence: list) -> dict:
    ev_text = json.dumps(evidence, default=str)
    ev_nums = set(numbers_in(ev_text))
    # raw numbers inside JSON like 383285000000 are parsed with no unit
    ev_nums |= {float(x) for x in re.findall(r"-?\d+\.?\d*", ev_text) if x not in {"", "-"}}
    unsupported = []
    for n in numbers_in(answer):
        if abs(n) < 10 and float(n).is_integer():  # skip tiny ints (list indices, "3 years")
            continue
        if 1900 <= n <= 2100 and float(n).is_integer():  # years
            continue
        if not any(_close(n, e) for e in ev_nums):
            # allow derived percentages / ratios: accept if any pair yields it (cheap check)
            unsupported.append(n)
    return {"ok": not unsupported, "unsupported_numbers": unsupported[:10]}
