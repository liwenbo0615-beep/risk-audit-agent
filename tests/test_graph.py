import json

import pytest

from audit import audit_one, batch_audit
from audit.models import new_state


def test_safe_comment_skips_analyze(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "log.jsonl"))
    result = audit_one("今天天气真好，出去玩吧")
    assert result["risk_type"] == "safe"
    assert result["risk_level"] == "none"
    assert result["recommended_action"] == "approve"
    assert result["human_decision"] == ""


def test_illegal_comment_gets_high_risk(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "log.jsonl"))
    result = audit_one("教你怎么逃税不被发现")
    assert result["risk_type"] == "illegal"
    assert result["risk_level"] == "high"
    assert result["human_decision"] == "skip"


def test_political_comment_triggers_review(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "log.jsonl"))
    result = audit_one("反对政府，推翻专制统治")
    assert result["risk_type"] == "political"
    assert result["human_decision"] == "skip"


def test_final_report_is_valid_json(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "log.jsonl"))
    result = audit_one("今天天气真好")
    report = json.loads(result["final_report"])
    assert "trace_id" in report


def test_log_file_written(tmp_path, monkeypatch):
    log_path = tmp_path / "log.jsonl"
    monkeypatch.setenv("AUDIT_LOG_PATH", str(log_path))
    audit_one("加我领优惠券", auto_decision="skip")
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["risk_type"] == "spam"


def test_batch_audit_returns_all_results(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "log.jsonl"))
    comments = ["今天天气真好", "教你怎么逃税", "反对政府"]
    results = batch_audit(comments, auto_decision="skip")
    assert len(results) == 3


def test_new_state_defaults():
    state = new_state("test")
    assert state["risk_type"] == "unknown"
    assert state["risk_level"] == "unknown"
    assert state["confidence"] == 0.0
    assert state["human_decision"] == ""
    assert state["policy_tags"] == []
    assert "auto_decision" not in state


def test_new_state_with_auto_decision():
    state = new_state("test", auto_decision="skip")
    assert state["auto_decision"] == "skip"


# ---- route_by_risk: low-confidence safe must not shortcut ----

def test_low_confidence_safe_does_not_shortcut():
    """safe 但低置信度(<0.75) 不应直接出安全报告，需进入深度分析。"""
    from audit.graph import route_by_risk
    state = {"risk_type": "safe", "confidence": 0.5}
    assert route_by_risk(state) == "analyze_risk"


def test_high_confidence_safe_shortcuts():
    """safe 且高置信度(>=0.75) 直接走安全报告分支。"""
    from audit.graph import route_by_risk
    state = {"risk_type": "safe", "confidence": 0.9}
    assert route_by_risk(state) == "generate_safe_report"
