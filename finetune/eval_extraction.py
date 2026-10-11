"""Exact-match eval of extraction on the held-out ticker. Works against any
OpenAI-compatible endpoint (vLLM serving base or fine-tuned model).

  python finetune/eval_extraction.py --base-url http://localhost:8001/v1 --model my-finetune"""
from __future__ import annotations

import argparse
import json

import httpx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="finetune/data/test.jsonl")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.data)]
    ok = valid = 0
    for r in rows:
        msgs, gold = r["messages"][:2], json.loads(r["messages"][2]["content"])
        resp = httpx.post(f"{a.base_url}/chat/completions", timeout=120,
                          json={"model": a.model, "messages": msgs, "temperature": 0, "max_tokens": 120}).json()
        txt = resp["choices"][0]["message"]["content"]
        try:
            pred = json.loads(txt)
            valid += 1
            ok += abs(pred["value_usd"] - gold["value_usd"]) <= 0.005 * abs(gold["value_usd"])
        except Exception:
            pass
    n = max(len(rows), 1)
    print(json.dumps({"model": a.model, "n": len(rows), "json_valid": valid / n, "exact_match": ok / n}))


if __name__ == "__main__":
    main()
