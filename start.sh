#!/bin/sh
# Ingest on first boot (idempotent: skipped if chunks already exist), then serve.
if [ ! -s data/chunks.jsonl ] && [ "${AUTO_INGEST:-1}" = "1" ]; then
  echo "No index found — ingesting filings (set AUTO_INGEST=0 to skip)…"
  python -m app.ingest || echo "Ingest failed; starting in degraded mode."
fi
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
