"""Verify identify_risk's offline-first → escalate-on-demand wiring.

These tests force "online" mode (OFFLINE_DEMO_MODE=0 + fake key) and stub the
LLM so no network call happens. They assert that LLMCallJudge actually gates
whether the LLM is invoked, and that judge_reason records the decision path.
"""

import pytest

import audit.config as _cfg
import audit.judge as _judge
import audit.nodes as nodes
from audit.models import new_state


class _FakeLLM:
    """Stand-in for ChatOpenAI: records calls, returns a canned JSON reply."""

    def __init__(self):
        self.calls = 0

    def invoke(self, _messages):
        self.calls += 1

        class _Resp:
            # 一次调用同时返回一级识别 + 二级分析字段（合并调用，供 analyze_risk 复用）
            content = (
                '{"risk_type": "political", "confidence": 0.9, '
                '"policy_tags": ["涉政表达"], "evidence": ["推翻"], '
                '"risk_level": "medium", "recommended_action": "manual_review", '
                '"analysis_result": "涉政表达需结合上下文人工复核"}'
            )

        return _Resp()


@pytest.fixture
def online(monkeypatch):
    """Force online mode with a fake key + stubbed LLM; reset singletons."""
    monkeypatch.setenv("OFFLINE_DEMO_MODE", "0")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake-for-test")
    _cfg.reset_config()
    _judge.reset_judge()
    fake = _FakeLLM()
    monkeypatch.setattr(nodes, "get_llm", lambda: fake)
    yield fake
    _cfg.reset_config()
    _judge.reset_judge()


def _run(comment: str):
    state = new_state(comment)
    state = nodes.normalize_input(state)
    return nodes.identify_risk(state)


def test_escalated_comment_calls_llm_once_total(online, monkeypatch, tmp_path):
    """成本优化：升级到 LLM 的内容，整张图只调用一次 LLM（识别+分析合并），不重复付费。"""
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "log.jsonl"))
    import audit.graph as _graph
    _graph.reset_app()
    from audit.service import audit_one

    result = audit_one("反对政府，推翻专制统治", auto_decision="skip")

    assert online.calls == 1  # 不是 2
    assert result["risk_type"] == "political"
    assert result["risk_level"] == "medium"
    assert result["llm_called"] is True


def test_llm_recommended_skip_is_normalized_to_manual_review(online, monkeypatch):
    """recommended_action 不允许 skip：LLM 即使返回 skip，也应归一为 manual_review。"""
    class _SkipLLM:
        def invoke(self, _messages):
            class _R:
                content = (
                    '{"risk_type":"political","confidence":0.9,'
                    '"policy_tags":["涉政表达"],"evidence":["推翻"],'
                    '"risk_level":"medium","recommended_action":"skip",'
                    '"analysis_result":"x"}'
                )
            return _R()

    monkeypatch.setattr(nodes, "get_llm", lambda: _SkipLLM())
    state = _run("反对政府，推翻专制统治")
    assert state["recommended_action"] == "manual_review"
    assert state["recommended_action"] != "skip"


def test_safe_short_text_skips_llm(online):
    """Trivial safe text is absolutely safe → LLM must NOT be called."""
    state = _run("天气好")
    assert online.calls == 0
    assert state["llm_called"] is False
    assert state["judge_reason"] == "absolutely_safe"
    assert state["risk_type"] == "safe"


def test_compound_rule_skips_llm(online):
    """High-confidence compound rule (未成年涉色) → LLM must NOT be called."""
    state = _run("妈妈说我才12岁，鸡巴就已经超过20厘米了")
    assert online.calls == 0
    assert state["llm_called"] is False
    assert state["judge_reason"] == "high_confidence_compound_rule"


def test_political_escalates_to_llm(online):
    """LLM-first: a political keyword hit is verified by the LLM."""
    state = _run("反对政府，推翻专制统治")
    assert online.calls == 1
    assert state["llm_called"] is True
    assert state["judge_reason"] == "escalated:llm_first"


def test_minor_keyword_escalates_to_llm(online):
    """LLM-first: a simple minor keyword is verified by the LLM."""
    state = _run("未成年人不应该沉迷学习")
    assert online.calls == 1
    assert state["llm_called"] is True
    assert state["judge_reason"] == "escalated:llm_first"


def test_llm_failure_falls_back_to_offline(online, monkeypatch):
    """If the escalated LLM call raises, fall back to the offline result."""

    def _boom():
        class _Broken:
            def invoke(self, _m):
                raise RuntimeError("api down")

        return _Broken()

    monkeypatch.setattr(nodes, "get_llm", _boom)
    state = _run("反对政府，推翻专制统治")
    assert state["llm_called"] is False
    assert state["risk_type"] == "political"  # offline fallback result
    assert state["judge_reason"].startswith("escalation_failed_offline_fallback:")
    assert "api down" in state["model_error"]


def test_llm_safe_with_manual_review_not_overwritten_by_safe_report(online, monkeypatch, tmp_path):
    """LLM 返回 safe + manual_review 时，route_by_risk 不能走 generate_safe_report 把它覆盖成 approve。"""
    import json

    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "log.jsonl"))
    import audit.graph as _graph
    _graph.reset_app()
    from audit.service import audit_one

    class _SafeManualLLM:
        def invoke(self, _messages):
            class _R:
                content = (
                    '{"risk_type":"safe","confidence":0.9,'
                    '"policy_tags":[],"evidence":[],'
                    '"risk_level":"medium","recommended_action":"manual_review",'
                    '"analysis_result":"表面安全但疑似引流，建议人工复核"}'
                )
            return _R()

    monkeypatch.setattr(nodes, "get_llm", lambda: _SafeManualLLM())

    # 长文本(>白名单/阈值) → judge 升级到 LLM；auto_decision=pending 避免 input()
    result = audit_one("这条内容看起来正常其实需要模型做语义判断", auto_decision="pending")
    report = json.loads(result["final_report"])

    assert result["llm_called"] is True
    assert report["recommended_action"] == "manual_review"          # 没被改成 approve
    assert report["analysis"] == "表面安全但疑似引流，建议人工复核"   # LLM 分析被保留
    assert report["final_action"] == "pending_review"               # 进复核而非直接安全放行
