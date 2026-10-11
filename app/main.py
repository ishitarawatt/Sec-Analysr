from __future__ import annotations

import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import xbrl
from .agent import Agent
from .config import settings
from .retrieval import HybridIndex

STATE: dict = {}
_cache: dict[str, tuple[float, dict]] = {}
_hits: dict[str, deque] = defaultdict(deque)


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["index"] = HybridIndex.load()
    STATE["agent"] = Agent(STATE["index"])
    yield


app = FastAPI(title="SEC Filings Analyst", version="1.0.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if request.url.path == "/api/ask":
        ip = request.client.host if request.client else "unknown"
        q, now = _hits[ip], time.time()
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= settings.rate_limit_per_min:
            return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429)
        q.append(now)
    return await call_next(request)


@app.get("/health")
def health():
    idx = STATE.get("index")
    return {"status": "ok", "chunks": len(idx.chunks) if idx else 0,
            "mode": "llm" if settings.anthropic_api_key else "offline"}


@app.post("/api/ask")
def ask(req: AskRequest):
    key = req.question.strip().lower()
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < settings.cache_ttl_seconds:
        return {**hit[1], "cached": True}
    try:
        res = STATE["agent"].answer(req.question)
    except Exception as e:
        raise HTTPException(502, f"Agent error: {e}")
    _cache[key] = (time.time(), res)
    return {**res, "cached": False}


@app.get("/api/xbrl/{ticker}/{metric}")
def get_xbrl(ticker: str, metric: str):
    try:
        return xbrl.annual_series(ticker, metric)
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.get("/api/search")
def search(q: str, ticker: str | None = None, k: int = 5):
    return STATE["index"].search(q, k=min(k, 10), ticker=ticker)


static = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static), name="static")


@app.get("/")
def home():
    return FileResponse(static / "index.html")
