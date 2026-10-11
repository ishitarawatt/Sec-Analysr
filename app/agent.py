"""Tool-using planner/executor agent with a verification pass.

Flow: plan+act loop (LLM chooses tools) -> draft answer -> deterministic verifier
checks every number against tool evidence -> one repair round if unsupported.
Every step is recorded in a trace returned to the caller (and loggable to Langfuse)."""
from __future__ import annotations

import json
import re
import time

from . import xbrl
from .config import settings
from .retrieval import HybridIndex
from .tools import TOOL_SPECS, run_tool, safe_eval
from .verifier import verify

SYSTEM = """You are a financial-filings analyst for SEC 10-K/10-Q documents.
Rules:
1. Every number MUST come from xbrl_lookup or calculator output or a retrieved passage. Never recall numbers from memory.
2. Use search_filings for qualitative 'why/how' questions and cite the passage as [TICKER form period, section].
3. Use calculator for all arithmetic (growth, margins, ratios).
4. If evidence is missing, say so plainly rather than guessing.
5. Be concise. State units (USD billions/millions) and fiscal years explicitly."""


class Agent:
    def __init__(self, index: HybridIndex):
        self.index = index
        self._client = None
        if settings.anthropic_api_key:
            import anthropic
            self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    # ------------------------------------------------------------------
    def answer(self, question: str) -> dict:
        t0 = time.time()
        trace: list[dict] = []
        evidence: list = []
        if self._client is None:
            return self._offline(question, trace, evidence, t0)

        msgs = [{"role": "user", "content": question}]
        final = ""
        for step in range(settings.max_agent_steps):
            resp = self._client.messages.create(
                model=settings.llm_model, max_tokens=1200, system=SYSTEM,
                tools=TOOL_SPECS, messages=msgs,
            )
            msgs.append({"role": "assistant", "content": resp.content})
            calls = [b for b in resp.content if b.type == "tool_use"]
            if not calls:
                final = "".join(b.text for b in resp.content if b.type == "text")
                break
            results = []
            for c in calls:
                try:
                    out = run_tool(c.name, c.input, self.index)
                except Exception as e:  # tool errors go back to the model, not the user
                    out = {"error": str(e)}
                evidence.append(out)
                trace.append({"step": step, "tool": c.name, "input": c.input,
                              "output_preview": json.dumps(out, default=str)[:400]})
                results.append({"type": "tool_result", "tool_use_id": c.id,
                                "content": json.dumps(out, default=str)[:12000]})
            msgs.append({"role": "user", "content": results})

        check = verify(final, evidence)
        if not check["ok"]:
            msgs.append({"role": "user", "content":
                         f"Verifier flagged numbers not found in evidence: {check['unsupported_numbers']}. "
                         "Fix using tools or remove them, then answer again."})
            resp = self._client.messages.create(model=settings.llm_model, max_tokens=1000,
                                                system=SYSTEM, tools=TOOL_SPECS, messages=msgs)
            text = "".join(b.text for b in resp.content if b.type == "text")
            if text:
                final, check = text, verify(text, evidence)
            trace.append({"step": "repair", "verifier": check})
        return {"answer": final, "verified": check["ok"], "verifier": check,
                "trace": trace, "mode": "llm", "latency_s": round(time.time() - t0, 2)}

    # ------------------------------------------------------------------
    def _offline(self, q: str, trace, evidence, t0) -> dict:
        """No-LLM mode: handles 'metric for TICKER in YEAR' and growth questions
        directly from XBRL; otherwise returns top retrieved passages."""
        ql = q.lower()
        tk = re.search(r"\b([A-Z]{2,5})\b", q)
        metric = next((m for m, kw in _KW.items() if any(k in ql for k in kw)), None)
        years = [int(y) for y in re.findall(r"\b(20\d{2})\b", q)]
        if tk and metric:
            rows = xbrl.annual_series(tk.group(1), metric)
            evidence.append(rows)
            trace.append({"tool": "xbrl_lookup", "input": {"ticker": tk.group(1), "metric": metric}})
            if rows:
                if len(years) >= 2 and ("grow" in ql or "change" in ql or "cagr" in ql):
                    a = next((r for r in rows if r["fy_end"].startswith(str(min(years)))), None)
                    b = next((r for r in rows if r["fy_end"].startswith(str(max(years)))), None)
                    if a and b:
                        g = safe_eval(f"({b['value']} - {a['value']}) / {a['value']} * 100")
                        evidence.append({"growth_pct": g})
                        ans = (f"{tk.group(1)} {metric.replace('_', ' ')} went from ${a['value']:,.0f} (FY{min(years)}) "
                               f"to ${b['value']:,.0f} (FY{max(years)}), a change of {g:.1f}%.")
                        return self._pack(ans, evidence, trace, t0)
                r = next((r for r in rows if years and r["fy_end"].startswith(str(years[0]))), rows[-1])
                ans = (f"{tk.group(1)} {metric.replace('_', ' ')} for fiscal year ending {r['fy_end']}: "
                       f"{r['value']:,.0f} {r['unit']} (XBRL tag {r['tag']}, accession {r['accession']}).")
                return self._pack(ans, evidence, trace, t0)
        hits = self.index.search(q, k=3)
        evidence.append(hits)
        trace.append({"tool": "search_filings", "input": {"query": q}})
        ans = ("(Offline mode — set ANTHROPIC_API_KEY for full agent reasoning.) Top passages:\n" +
               "\n".join(f"- [{h['ticker']} {h['form']} {h['period']}, {h['section']}] {h['text'][:300]}…" for h in hits)
               ) if hits else "No indexed passages found. Run `python -m app.ingest` first."
        return self._pack(ans, evidence, trace, t0)

    def _pack(self, ans, evidence, trace, t0):
        check = verify(ans, evidence)
        return {"answer": ans, "verified": check["ok"], "verifier": check, "trace": trace,
                "mode": "offline", "latency_s": round(time.time() - t0, 2)}


_KW = {
    "revenue": ["revenue", "sales"], "net_income": ["net income", "profit", "earnings"],
    "rd_expense": ["r&d", "research and development"], "operating_income": ["operating income"],
    "gross_profit": ["gross profit"], "total_assets": ["total assets"],
    "total_liabilities": ["liabilities"], "cash": ["cash"], "eps_diluted": ["eps", "earnings per share"],
    "capex": ["capex", "capital expenditure"], "operating_cash_flow": ["operating cash flow"],
}
