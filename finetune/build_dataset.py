"""Build an instruction dataset for structured extraction from filing chunks.

Targets come from XBRL (verifiable), inputs are table/text chunks from the same filing.
Output JSONL: {"messages":[system,user,assistant]} — works with TRL SFTTrainer.
Hold out by TICKER (not random rows) so eval measures generalisation to unseen companies."""
from __future__ import annotations

import json
import random
from pathlib import Path

from app.config import settings
from app.retrieval import HybridIndex
from app import xbrl

SYSTEM = "Extract the requested financial metric from the filing excerpt. Reply with JSON only."
METRICS = ["revenue", "net_income", "rd_expense", "operating_income"]


def build(out_dir="finetune/data", holdout=("GOOGL",), seed=0):
    idx = HybridIndex.load()
    rnd = random.Random(seed)
    train, test = [], []
    for c in idx.chunks:
        if c.kind != "table":
            continue
        for m in METRICS:
            try:
                year = int(c.period[:4])
                row = xbrl.value_for_year(c.ticker, m, year)
            except Exception:
                continue
            if not row:
                continue
            # only keep examples where the number actually appears in the chunk
            digits = f"{row['value'] / 1e6:,.0f}"
            if digits not in c.text:
                continue
            ex = {"messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Metric: {m}\nFiscal year: {year}\n\n{c.text[:3000]}"},
                {"role": "assistant", "content": json.dumps({"metric": m, "fiscal_year": year,
                                                              "value_usd": row["value"]})},
            ]}
            (test if c.ticker in holdout else train).append(ex)
    rnd.shuffle(train)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for name, data in (("train", train), ("test", test)):
        Path(out_dir, f"{name}.jsonl").write_text("\n".join(json.dumps(x) for x in data))
    print(f"train={len(train)} test={len(test)}")


if __name__ == "__main__":
    build()
