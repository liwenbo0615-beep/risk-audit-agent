import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_high_risk_audit_enqueues_pending(client):
    """高风险内容 → final_action=pending_review，且进入待审队列。"""
    resp = client.post("/audit", json={"comment": "教你怎么逃税不被发现"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["risk_type"] == "illegal"
    assert body["final_action"] == "pending_review"
    assert body["human_decision"] == ""

    pend = client.get("/reviews/pending").json()
    assert any(item["trace_id"] == body["trace_id"] for item in pend)


def test_resolve_decides_and_removes_from_pending(client):
    """审核员结案 → 记录 decided，并从待审队列移除。"""
    trace_id = client.post("/audit", json={"comment": "教你怎么逃税不被发现"}).json()["trace_id"]

    r = client.post(f"/reviews/{trace_id}", json={"decision": "reject", "reviewer": "bob"})
    assert r.status_code == 200
    rec = r.json()
    assert rec["status"] == "decided"
    assert rec["final_action"] == "reject"
    assert rec["reviewer"] == "bob"

    pend = client.get("/reviews/pending").json()
    assert all(item["trace_id"] != trace_id for item in pend)


def test_resolve_unknown_returns_404(client):
    r = client.post("/reviews/does-not-exist", json={"decision": "approve"})
    assert r.status_code == 404


def test_safe_audit_not_enqueued(client):
    """低风险内容直接放行，不进待审队列。"""
    resp = client.post("/audit", json={"comment": "今天天气真好"})
    assert resp.json()["final_action"] == "approve"
    assert client.get("/reviews/pending").json() == []


def test_batch_default_enqueues_pending(client):
    """批量默认（不传 auto_decision）应与单条一致：命中复核 → pending_review 且入队，而非 skip。"""
    resp = client.post("/audit/batch", json={"comments": ["教你怎么逃税不被发现", "今天天气真好"]})
    assert resp.status_code == 200
    results = resp.json()["results"]

    illegal = next(r for r in results if r["risk_type"] == "illegal")
    assert illegal["final_action"] == "pending_review"
    assert illegal["human_decision"] == ""

    pend = client.get("/reviews/pending").json()
    assert any(it["trace_id"] == illegal["trace_id"] for it in pend)
