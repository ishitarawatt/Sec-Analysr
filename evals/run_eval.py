"""Run the agent over the XBRL-derived eval set and log metrics.

  python -m evals.run_eval --limit 50 --tag baseline
Appends a summary row to evals/history.jsonl so you can plot accuracy across versions
(this is what you quote on your resume). If MLflow is installed, also logs a run."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from app.agent import Agent
from app.retrieval import HybridIndex

from .metrics import numeric_match, summarize

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--min-accuracy", type=float, default=0.0, help="exit 1 if below (for CI gates)")
    a = ap.parse_args()

    rows = [json.loads(l) for l in (HERE / "eval_set.jsonl").read_text().splitlines()][: a.limit]
    agent = Agent(HybridIndex.load())
    results = []
    for r in rows:
        try:
            out = agent.answer(r["question"])
            ok = numeric_match(out["answer"], r["gold"])
            results.append({"id": r["id"], "type": r["type"], "correct": ok, "verified": out["verified"],
                            "latency_s": out["latency_s"], "answer": out["answer"][:300]})
        except Exception as e:
            results.append({"id": r["id"], "type": r["type"], "correct": False, "verified": False,
                            "latency_s": 0.0, "answer": f"ERROR {e}"})
    summary = {"tag": a.tag, "ts": int(time.time()), **summarize(results)}
    (HERE / f"results_{a.tag}.json").write_text(json.dumps({"summary": summary, "results": results}, indent=1))
    with (HERE / "history.jsonl").open("a") as f:
        f.write(json.dumps(summary) + "\n")
    print(json.dumps(summary, indent=2))
    try:
        import mlflow
        with mlflow.start_run(run_name=a.tag):
            mlflow.log_metrics({k: v for k, v in summary.items() if isinstance(v, float)})
    except ImportError:
        pass
    if summary["numeric_accuracy"] < a.min_accuracy:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
