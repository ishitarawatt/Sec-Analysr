"""Hybrid retrieval: BM25 (lexical) + TF-IDF/LSA (dense) fused with RRF, then a
lightweight heuristic reranker. Embedding backend is swappable (see `Embedder`)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .chunking import Chunk
from .config import settings

_tok = re.compile(r"[A-Za-z0-9$%.]+")


def tokenize(t: str) -> list[str]:
    return [w.lower() for w in _tok.findall(t)]


class Embedder:
    """TF-IDF + LSA dense vectors. Replace with sentence-transformers / API embeddings
    by implementing fit(texts) and encode(texts) -> np.ndarray (L2-normalised)."""

    def __init__(self, dim: int = 256):
        self.tfidf = TfidfVectorizer(tokenizer=tokenize, lowercase=False, sublinear_tf=True,
                                     ngram_range=(1, 2), min_df=1, token_pattern=None)
        self.dim = dim
        self.svd: TruncatedSVD | None = None

    def fit(self, texts: list[str]) -> np.ndarray:
        X = self.tfidf.fit_transform(texts)
        k = max(2, min(self.dim, X.shape[0] - 1, X.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=0)
        return self._norm(self.svd.fit_transform(X))

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._norm(self.svd.transform(self.tfidf.transform(texts)))

    @staticmethod
    def _norm(M: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(M, axis=1, keepdims=True)
        return M / np.clip(n, 1e-9, None)


class HybridIndex:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self.bm25: BM25Okapi | None = None
        self.emb = Embedder()
        self.vecs: np.ndarray | None = None

    # ---- build / persist -------------------------------------------------
    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        texts = [f"{c.ticker} {c.form} {c.section}. {c.text}" for c in chunks]
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])
        self.vecs = self.emb.fit(texts)

    def save(self, path: str | None = None) -> None:
        p = Path(path or Path(settings.data_dir) / "chunks.jsonl")
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict()) + "\n")

    @classmethod
    def load(cls, path: str | None = None) -> "HybridIndex":
        p = Path(path or Path(settings.data_dir) / "chunks.jsonl")
        idx = cls()
        if not p.exists():
            return idx
        chunks = [Chunk(**json.loads(line)) for line in p.read_text().splitlines() if line]
        if chunks:
            idx.build(chunks)
        return idx

    # ---- search -----------------------------------------------------------
    def search(self, query: str, k: int = 6, ticker: str | None = None,
               form: str | None = None, section: str | None = None) -> list[dict]:
        if not self.chunks:
            return []
        mask = np.array([
            (ticker is None or c.ticker.upper() == ticker.upper())
            and (form is None or c.form == form)
            and (section is None or section.lower() in c.section.lower())
            for c in self.chunks
        ])
        if not mask.any():
            return []
        bm = np.array(self.bm25.get_scores(tokenize(query)))
        dn = self.vecs @ self.emb.encode([query])[0]
        bm[~mask] = -1e9
        dn[~mask] = -1e9
        pool = min(30, int(mask.sum()))
        rank_bm = {int(i): r for r, i in enumerate(np.argsort(-bm)[:pool])}
        rank_dn = {int(i): r for r, i in enumerate(np.argsort(-dn)[:pool])}
        fused = {}
        for i in set(rank_bm) | set(rank_dn):  # reciprocal rank fusion
            fused[i] = 1 / (60 + rank_bm.get(i, pool)) + 1 / (60 + rank_dn.get(i, pool))
        reranked = sorted(fused, key=lambda i: -(fused[i] + self._rerank_bonus(query, self.chunks[i])))
        out = []
        for i in reranked[:k]:
            c = self.chunks[i]
            out.append({**c.to_dict(), "score": round(float(fused[i]), 5)})
        return out

    @staticmethod
    def _rerank_bonus(query: str, c: Chunk) -> float:
        """Cheap cross-signal reranker: query-term coverage + section/table priors.
        Swap for a cross-encoder (e.g. bge-reranker) in production."""
        q = set(tokenize(query))
        if not q:
            return 0.0
        cov = len(q & set(tokenize(c.text))) / len(q)
        bonus = 0.004 * cov
        if c.kind == "table" and any(w in q for w in ("revenue", "income", "assets", "expense", "margin")):
            bonus += 0.002
        if c.section.lower().split()[-1] in q:
            bonus += 0.002
        return bonus
