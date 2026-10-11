"""Ground-truth financial facts from SEC XBRL `companyfacts`."""
from __future__ import annotations

from . import edgar

# Friendly metric -> candidate us-gaap tags (first match wins)
METRICS: dict[str, list[str]] = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"],
    "net_income": ["NetIncomeLoss"],
    "rd_expense": ["ResearchAndDevelopmentExpense"],
    "operating_income": ["OperatingIncomeLoss"],
    "gross_profit": ["GrossProfit"],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
}


def annual_series(ticker: str, metric: str) -> list[dict]:
    """Fiscal-year (10-K, FY) values for a metric, deduplicated, oldest first."""
    if metric not in METRICS:
        raise KeyError(f"Unknown metric '{metric}'. Options: {sorted(METRICS)}")
    cik, name = edgar.ticker_to_cik(ticker)
    facts = edgar.company_facts(cik)["facts"].get("us-gaap", {})
    for tag in METRICS[metric]:
        if tag not in facts:
            continue
        units = facts[tag]["units"]
        unit = "USD" if "USD" in units else ("USD/shares" if "USD/shares" in units else next(iter(units)))
        rows = [r for r in units[unit] if r.get("form") == "10-K" and r.get("fp") == "FY"]
        by_year: dict[int, dict] = {}
        for r in rows:
            # period length ~ 1yr for flow metrics; keep latest filing for each fiscal year end
            fy_end = r["end"]
            if r.get("start"):
                from datetime import date
                s = date.fromisoformat(r["start"]); e = date.fromisoformat(r["end"])
                if not 350 <= (e - s).days <= 380:
                    continue
            key = int(fy_end[:4])
            if key not in by_year or r["filed"] > by_year[key]["filed"]:
                by_year[key] = r
        if by_year:
            return [
                {"ticker": ticker.upper(), "company": name, "metric": metric, "tag": tag,
                 "fy_end": r["end"], "value": r["val"], "unit": unit,
                 "accession": r["accn"], "filed": r["filed"]}
                for _, r in sorted(by_year.items())
            ]
    return []


def value_for_year(ticker: str, metric: str, year: int) -> dict | None:
    for row in annual_series(ticker, metric):
        if row["fy_end"].startswith(str(year)):
            return row
    return None
