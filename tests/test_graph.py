import json

import pytest

from audit.service import audit_one, batch_audit
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
    result = audit_one("教你怎么逃税不被发现", auto_decision="skip")
    assert result["risk_type"] == "illegal"
    assert result["risk_level"] == "high"
    assert result["human_decision"] == "skip"


def test_political_comment_triggers_review(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "log.jsonl"))
    result = audit_one("反对政府，推翻专制统治", auto_decision="skip")
    assert result["risk_type"] == "political"
    assert result["human_decision"] == "skip"


def test_human_review_prompts_when_no_auto_decision(tmp_path, monkeypatch):
    """没有 auto_decision 时，human_review 必须真正停下来交互；这里 mock input 模拟审核员。"""
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "log.jsonl"))
    called = {"value": False}

    def fake_input(prompt: str) -> str:
        called["value"] = True
        assert "approve/reject/skip" in prompt
        return "reject"

    monkeypatch.setattr("builtins.input", fake_input)
    result = audit_one("教你怎么逃税不被发现")  # illegal → high → 命中人工复核

    assert called["value"] is True
    assert result["human_decision"] == "reject"
    assert json.loads(result["final_report"])["final_action"] == "reject"


def test_human_review_auto_decision_skips_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "log.jsonl"))

    def fail_input(prompt: str) -> str:
        raise AssertionError("input() should not be called when auto_decision is provided")

    monkeypatch.setattr("builtins.input", fail_input)
    result = audit_one("教你怎么逃税不被发现", auto_decision="skip")

    assert result["human_decision"] == "skip"


def test_env_auto_review_decision_does_not_override_cli_manual_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "log.jsonl"))
    monkeypatch.setenv("AUTO_REVIEW_DECISION", "skip")
    monkeypatch.setattr("builtins.input", lambda *_a, **_k: "reject")

    result = audit_one("教你怎么逃税不被发现")

    assert result["human_decision"] == "reject"
    assert json.loads(result["final_report"])["final_action"] == "reject"


def test_human_review_defers_to_pending_without_stdin(tmp_path, monkeypatch):
    """非交互环境（管道/CI/无 TTY，input() 抛 EOFError）→ 标记 pending_review，不崩溃、不伪造成 skip。"""
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "log.jsonl"))

    def no_stdin(*_a, **_k):
        raise EOFError

    monkeypatch.setattr("builtins.input", no_stdin)
    result = audit_one("教你怎么逃税不被发现")  # illegal → high → 命中人工复核

    assert result["human_decision"] == ""  # 没有真人决策
    assert json.loads(result["final_report"])["final_action"] == "pending_review"


def test_explicit_skip_is_distinct_from_pending(tmp_path, monkeypatch):
    """运营显式 -d skip 是一次真实决策，final_action 应为 skip，区别于 pending_review。"""
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "log.jsonl"))
    result = audit_one("教你怎么逃税不被发现", auto_decision="skip")

    assert result["human_decision"] == "skip"
    assert json.loads(result["final_report"])["final_action"] == "skip"


def test_pending_auto_decision_marks_pending_review(tmp_path, monkeypatch):
    """API 传 auto_decision='pending' → 不调用 input()，直接标记 pending_review。"""
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "log.jsonl"))

    def fail_input(*_a, **_k):
        raise AssertionError("pending 不应触发 input()")

    monkeypatch.setattr("builtins.input", fail_input)
    result = audit_one("教你怎么逃税不被发现", auto_decision="pending")

    assert result["human_decision"] == ""
    assert json.loads(result["final_report"])["final_action"] == "pending_review"


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
