import audit.config as cfg
from audit import review_store
from audit.models import new_state


def _setup_store(monkeypatch, tmp_path):
    monkeypatch.setenv("REVIEW_STORE_PATH", str(tmp_path / "q.json"))
    cfg.reset_config()


def _pending_state(comment="高风险内容"):
    state = new_state(comment)
    state["risk_type"] = "illegal"
    state["risk_level"] = "high"
    state["recommended_action"] = "reject"
    state["analysis_result"] = "违法违规，建议拒绝"
    state["review_status"] = "pending"
    return state


def test_enqueue_then_list_pending(monkeypatch, tmp_path):
    _setup_store(monkeypatch, tmp_path)
    state = _pending_state()
    review_store.enqueue(state)
    pend = review_store.list_pending()
    assert len(pend) == 1
    assert pend[0]["trace_id"] == state["trace_id"]
    assert pend[0]["status"] == "pending"
    assert pend[0]["final_action"] == "pending_review"


def test_resolve_marks_decided_and_removes_from_pending(monkeypatch, tmp_path):
    _setup_store(monkeypatch, tmp_path)
    state = _pending_state()
    review_store.enqueue(state)

    rec = review_store.resolve(state["trace_id"], "approve", reviewer="alice")

    assert rec is not None
    assert rec["status"] == "decided"
    assert rec["final_action"] == "approve"
    assert rec["human_decision"] == "approve"
    assert rec["reviewer"] == "alice"
    assert review_store.list_pending() == []


def test_resolve_unknown_returns_none(monkeypatch, tmp_path):
    _setup_store(monkeypatch, tmp_path)
    assert review_store.resolve("no-such-id", "approve") is None


def test_decided_item_excluded_from_pending(monkeypatch, tmp_path):
    _setup_store(monkeypatch, tmp_path)
    s1 = _pending_state("内容1")
    review_store.enqueue(s1)
    s2 = _pending_state("内容2")
    review_store.enqueue(s2)

    review_store.resolve(s1["trace_id"], "reject")

    pend = review_store.list_pending()
    assert len(pend) == 1
    assert pend[0]["trace_id"] == s2["trace_id"]
