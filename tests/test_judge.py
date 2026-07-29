import pytest

from audit.judge import JudgeConfig, LLMCallJudge


def make_judge(**kwargs) -> LLMCallJudge:
    return LLMCallJudge(JudgeConfig(**kwargs))


# ---- master switch ----

def test_disabled_judge_always_returns_false():
    judge = make_judge(enabled=False)
    result, reason = judge.should_call("任意内容", {"risk_type": "unknown", "confidence": 0.0, "policy_tags": []})
    assert result is False
    assert reason == "judge_disabled"


# ---- compound rule bypass ----

def test_compound_tag_skips_llm():
    """High-confidence compound rule → no LLM needed even for minor type."""
    judge = make_judge()
    offline = {"risk_type": "minor", "confidence": 0.95, "policy_tags": ["未成年涉色"]}
    needs_llm, reason = judge.should_call("12岁鸡巴", offline)
    assert needs_llm is False
    assert reason == "high_confidence_compound_rule"


# ---- LLM-first: everything non-trivial is verified ----

def test_unknown_risk_type_calls_llm():
    judge = make_judge()
    offline = {"risk_type": "unknown", "confidence": 0.0, "policy_tags": []}
    needs_llm, reason = judge.should_call("模糊内容", offline)
    assert needs_llm is True
    assert reason == "llm_first"


def test_low_confidence_calls_llm():
    judge = make_judge()
    offline = {"risk_type": "spam", "confidence": 0.80, "policy_tags": ["营销引流"]}
    needs_llm, reason = judge.should_call("优惠", offline)
    assert needs_llm is True
    assert reason == "llm_first"


def test_high_confidence_clear_type_still_calls_llm():
    """LLM-first: even a confident, clear-cut offline result is verified by the LLM."""
    judge = make_judge()
    offline = {"risk_type": "illegal", "confidence": 0.90, "policy_tags": ["违法违规"]}
    needs_llm, reason = judge.should_call("逃税内容", offline)
    assert needs_llm is True
    assert reason == "llm_first"


def test_political_always_calls_llm():
    judge = make_judge()
    offline = {"risk_type": "political", "confidence": 0.90, "policy_tags": ["涉政表达"]}
    needs_llm, reason = judge.should_call("政府内容", offline)
    assert needs_llm is True
    assert reason == "llm_first"


def test_minor_without_compound_calls_llm():
    """Simple minor keyword alone is verified by the LLM — "未成年人不应该沉迷学习" is actually safe."""
    judge = make_judge()
    offline = {"risk_type": "minor", "confidence": 0.90, "policy_tags": ["未成年人保护"]}
    needs_llm, reason = judge.should_call("未成年人不应该沉迷学习", offline)
    assert needs_llm is True
    assert reason == "llm_first"


@pytest.mark.parametrize("risk_type,tag", [
    ("illegal", "违法违规"),
    ("spam",    "营销引流"),
    ("violence","暴力血腥"),
])
def test_clear_risk_types_still_call_llm(risk_type, tag):
    """Under LLM-first, clear keyword hits are no longer trusted blindly."""
    judge = make_judge()
    offline = {"risk_type": risk_type, "confidence": 0.90, "policy_tags": [tag]}
    needs_llm, reason = judge.should_call("测试内容", offline)
    assert needs_llm is True
    assert reason == "llm_first"


# ---- absolutely-safe 仅限固定白名单短语 ----

def test_whitelisted_safe_text_is_absolutely_safe():
    judge = make_judge()
    offline = {"risk_type": "safe", "confidence": 0.92, "policy_tags": []}
    needs_llm, reason = judge.should_call("今天天气真好", offline)
    assert needs_llm is False
    assert reason == "absolutely_safe"


def test_short_coded_text_is_not_absolutely_safe():
    """'约茶吗' 虽短，但不在白名单 → 仍应升级 LLM 做语义判断（防黑话漏检）。"""
    judge = make_judge()
    offline = {"risk_type": "safe", "confidence": 0.92, "policy_tags": []}
    needs_llm, reason = judge.should_call("约茶吗", offline)
    assert needs_llm is True
    assert reason == "llm_first"


def test_longer_safe_text_calls_llm():
    """Safe but non-trivial text is sent to the LLM in case keywords missed coded signals."""
    judge = make_judge()
    offline = {"risk_type": "safe", "confidence": 0.92, "policy_tags": []}
    long_text = "最近压力好大，想找个茶艺师好好品品新茶，预算不是问题。"
    needs_llm, reason = judge.should_call(long_text, offline)
    assert needs_llm is True
    assert reason == "llm_first"


# ---- env-based config ----

def test_judge_config_from_env(monkeypatch):
    monkeypatch.setenv("JUDGE_ENABLED", "1")
    monkeypatch.setenv("JUDGE_MIN_CONFIDENCE", "0.75")
    cfg = JudgeConfig.from_env()
    assert cfg.enabled is True
    assert cfg.min_confidence == 0.75


def test_judge_config_disabled_via_env(monkeypatch):
    monkeypatch.setenv("JUDGE_ENABLED", "0")
    cfg = JudgeConfig.from_env()
    assert cfg.enabled is False
