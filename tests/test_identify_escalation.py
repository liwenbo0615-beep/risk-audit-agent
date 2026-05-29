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
            content = '{"risk_type": "political", "confidence": 0.9, "policy_tags": ["涉政表达"], "evidence": ["推翻"]}'

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


def test_safe_short_text_skips_llm(online):
    """Judge says offline_sufficient → LLM must NOT be called."""
    state = _run("今天天气真好，出去玩吧")
    assert online.calls == 0
    assert state["llm_called"] is False
    assert state["judge_reason"] == "offline_sufficient"
    assert state["risk_type"] == "safe"


def test_compound_rule_skips_llm(online):
    """High-confidence compound rule (未成年涉色) → LLM must NOT be called."""
    state = _run("妈妈说我才12岁，鸡巴就已经超过20厘米了")
    assert online.calls == 0
    assert state["llm_called"] is False
    assert state["judge_reason"] == "high_confidence_compound_rule"


def test_political_escalates_to_llm(online):
    """Context-dependent type → escalate to LLM."""
    state = _run("反对政府，推翻专制统治")
    assert online.calls == 1
    assert state["llm_called"] is True
    assert state["judge_reason"] == "escalated:context_dependent:political"


def test_minor_keyword_escalates_to_llm(online):
    """Simple minor keyword needs semantic verification → escalate."""
    state = _run("未成年人不应该沉迷学习")
    assert online.calls == 1
    assert state["llm_called"] is True
    assert state["judge_reason"] == "escalated:minor_needs_semantic_verification"


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
