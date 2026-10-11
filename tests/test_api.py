from fastapi.testclient import TestClient

from app import xbrl
from app.main import app


def test_health_and_ui():
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
        assert "SEC Filings Analyst" in c.get("/").text


def test_ask_offline_uses_xbrl(monkeypatch):
    fake = [{"ticker": "AAPL", "company": "Apple", "metric": "revenue", "tag": "Revenues",
             "fy_end": "2023-09-30", "value": 383285000000, "unit": "USD", "accession": "0000320193-23-000106",
             "filed": "2023-11-03"}]
    monkeypatch.setattr(xbrl, "annual_series", lambda t, m: fake)
    with TestClient(app) as c:
        r = c.post("/api/ask", json={"question": "What was AAPL revenue in 2023?"}).json()
        assert "383,285,000,000" in r["answer"] and r["verified"] is True


def test_validation_and_unknown_metric():
    with TestClient(app) as c:
        assert c.post("/api/ask", json={"question": "hi"}).status_code == 422
        assert c.get("/api/xbrl/AAPL/not_a_metric").status_code == 404
