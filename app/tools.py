"""Tools the agent can call. Each returns JSON-serialisable data."""
from __future__ import annotations

import ast
import operator as op

from . import xbrl
from .retrieval import HybridIndex

_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
        ast.Pow: op.pow, ast.USub: op.neg, ast.Mod: op.mod}


def safe_eval(expr: str) -> float:
    """Evaluate arithmetic only (no names, calls, attributes)."""
    def _ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](_ev(n.left), _ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](_ev(n.operand))
        raise ValueError("Unsupported expression")
    return _ev(ast.parse(expr, mode="eval").body)


TOOL_SPECS = [
    {
        "name": "search_filings",
        "description": "Hybrid search over SEC 10-K/10-Q text and tables. Use for qualitative questions "
                       "(strategy, risks, drivers, explanations). Returns cited passages.",
        "input_schema": {"type": "object", "properties": {
            "query": {"type": "string"},
            "ticker": {"type": "string", "description": "e.g. AAPL"},
            "form": {"type": "string", "enum": ["10-K", "10-Q"]},
            "section": {"type": "string", "description": "e.g. 'Risk Factors', 'MD&A'"},
            "k": {"type": "integer", "default": 5}}, "required": ["query"]},
    },
    {
        "name": "xbrl_lookup",
        "description": "Authoritative annual financial values from SEC XBRL. Use for ANY number. "
                       f"Metrics: {sorted(xbrl.METRICS)}. Returns series across fiscal years.",
        "input_schema": {"type": "object", "properties": {
            "ticker": {"type": "string"}, "metric": {"type": "string"},
            "year": {"type": "integer", "description": "Optional fiscal year filter"}},
            "required": ["ticker", "metric"]},
    },
    {
        "name": "calculator",
        "description": "Exact arithmetic (growth rates, ratios, margins). Never do math in your head.",
        "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}},
                         "required": ["expression"]},
    },
]


def run_tool(name: str, args: dict, index: HybridIndex):
    if name == "search_filings":
        return index.search(args["query"], k=int(args.get("k", 5)), ticker=args.get("ticker"),
                            form=args.get("form"), section=args.get("section"))
    if name == "xbrl_lookup":
        rows = xbrl.annual_series(args["ticker"], args["metric"])
        if args.get("year"):
            rows = [r for r in rows if r["fy_end"].startswith(str(args["year"]))]
        return rows[-8:]
    if name == "calculator":
        return {"expression": args["expression"], "result": safe_eval(args["expression"])}
    raise ValueError(f"Unknown tool {name}")
