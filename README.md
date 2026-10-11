# SEC Filings Analyst

An AI analyst for SEC 10-K / 10-Q filings. Ask questions in plain English and get cited answers whose **numbers are verified against SEC XBRL ground truth**.

```
question ─► agent (tool-use loop) ─┬─► search_filings  (BM25 + dense, RRF fusion, rerank, metadata filters)
                                   ├─► xbrl_lookup     (authoritative numbers from SEC companyfacts)
                                   └─► calculator      (safe AST arithmetic; no mental math)
                       draft ─► verifier (every number must appear in tool evidence) ─► repair round ─► answer + trace
```

## Features
- **Ingestion:** EDGAR client (rate-limited, cached), section-aware (Item 1A / 7 / 8…) and table-aware chunking.
- **Hybrid retrieval:** BM25 + TF-IDF/LSA vectors, reciprocal-rank fusion, heuristic reranker (swap in a cross-encoder / API embeddings via `Embedder`).
- **Agent:** planner/executor with tool use, step budget, error-tolerant tool calls, full trace returned to the UI.
- **Verifier:** deterministic numeric grounding check + one automatic repair pass → "numbers verified" badge.
- **Eval harness:** gold answers generated from XBRL (not an LLM). Metrics: numeric accuracy, hallucination rate, p50/p95 latency, by question type. Results append to `evals/history.jsonl` so you can chart improvement across versions.
- **Fine-tuning:** QLoRA dataset builder (held-out *ticker* split) + trainer + exact-match eval against any OpenAI-compatible endpoint (vLLM).
- **Serving:** FastAPI, caching, rate limiting, health check, Docker, CI with an eval gate.
- **Offline mode:** without an LLM key the API still answers metric/growth questions straight from XBRL.

## Run locally
```bash
pip install -r requirements.txt
cp .env.example .env            # set SEC_USER_AGENT="Your Name you@email.com" (SEC requires it)
python -m app.ingest            # downloads + chunks filings -> data/chunks.jsonl
uvicorn app.main:app --reload   # http://localhost:8000
pytest -q
```
Set `ANTHROPIC_API_KEY` to enable the full agent. To use your fine-tuned model behind vLLM, serve it and set `VLLM_BASE_URL` / `VLLM_MODEL`.

## Evaluate (this produces your resume numbers)
```bash
python -m evals.build_eval_set                  # ~hundreds of XBRL-derived Qs
python -m evals.run_eval --limit 100 --tag v1-baseline
# change retrieval/prompt/model, then:
python -m evals.run_eval --limit 100 --tag v2-hybrid-verifier
cat evals/history.jsonl                         # compare versions
```
Record your real numbers here once you have run them:

| version | numeric acc. | hallucination | p95 latency |
|---|---|---|---|
| v1 baseline | _run it_ | _run it_ | _run it_ |

## Fine-tune (needs a GPU)
```bash
pip install -r finetune/requirements-gpu.txt
python -m finetune.build_dataset
python finetune/qlora_train.py --base Qwen/Qwen2.5-1.5B-Instruct
vllm serve finetune/out --port 8001
python finetune/eval_extraction.py --base-url http://localhost:8001/v1 --model finetune/out
```
Report fine-tune vs base vs hosted API honestly — if prompting wins, say so and explain the cost/latency trade-off.

## Deploy
Docker (any host): `docker build -t sec-analyst . && docker run -p 8000:8000 --env-file .env sec-analyst`.
Render: connect the repo, it picks up `render.yaml`; set `SEC_USER_AGENT` and optionally `ANTHROPIC_API_KEY`.
The container ingests on first boot (`AUTO_INGEST=0` to disable). Free tiers have ephemeral disks, so the index rebuilds on restart; attach a volume or use a vector DB for persistence.

## Design decisions & limitations
- Numbers come from XBRL tools, never model memory; the verifier makes violations visible instead of silent.
- Default embeddings are classical (TF-IDF+LSA) to keep the image small and cold starts fast; semantic embeddings + a cross-encoder reranker are the first upgrade.
- XBRL covers annual (10-K) figures here; quarterly and segment data are next.
- Not investment advice.
