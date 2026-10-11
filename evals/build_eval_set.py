"""Generate an eval set whose gold answers come from SEC XBRL (not from an LLM).

Question types:
  - point:  "What was {T} {metric} in fiscal {Y}?"            -> exact value
  - growth: "By what percent did {T} {metric} change {Y1}->{Y2}?" -> computed pct
Usage: python -m evals.build_eval_set  (writes evals/eval_set.jsonl)"""
from __future__ import annotations

import json
import random
from pathlib import Path

from app import xbrl
from app.config import settings

METRICS = ["revenue", "net_income", "rd_expense", "operating_income", "total_assets", "cash"]
PRETTY = {"revenue": "revenue", "net_income": "net income", "rd_expense": "R&D expense",
          "operating_income": "operating income", "total_assets": "total assets", "cash": "cash and cash equivalents"}


def build(tickers: list[str] | None = None, seed: int = 7) -> list[dict]:
    rnd = random.Random(seed)
    tickers = tickers or [t.strip() for t in settings.tickers.split(",")]
    rows: list[dict] = []
    for t in tickers:
        for m in METRICS:
            try:
                series = xbrl.annual_series(t, m)
            except Exception:
                continue
            series = series[-5:]
            for r in series:
                y = int(r["fy_end"][:4])
                rows.append({"id": f"{t}-{m}-{y}-pt", "type": "point", "ticker": t, "metric": m, "year": y,
                             "question": f"What was {t} {PRETTY[m]} in {y}?", "gold": r["value"], "unit": r["unit"],
                             "source": {"accession": r["accession"], "tag": r["tag"]}})
            for a, b in zip(series, series[1:]):
                if a["value"]:
                    g = (b["value"] - a["value"]) / abs(a["value"]) * 100
                    y1, y2 = int(a["fy_end"][:4]), int(b["fy_end"][:4])
                    rows.append({"id": f"{t}-{m}-{y1}-{y2}-gr", "type": "growth", "ticker": t, "metric": m,
                                 "question": f"By what percent did {t} {PRETTY[m]} change from {y1} to {y2}?",
                                 "gold": round(g, 2), "unit": "pct",
                                 "source": {"accessions": [a["accession"], b["accession"]]}})
    rnd.shuffle(rows)
    return rows


if __name__ == "__main__":
    data = build()
    out = Path(__file__).parent / "eval_set.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in data))
    print(f"Wrote {len(data)} questions to {out}")
