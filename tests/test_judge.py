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


# ---- unknown type ----

def test_unknown_risk_type_calls_llm():
    judge = make_judge()
    offline = {"risk_type": "unknown", "confidence": 0.0, "policy_tags": []}
    needs_llm, reason = judge.should_call("模糊内容", offline)
    assert needs_llm is True
    assert reason == "offline_unknown"


# ---- confidence threshold ----

def test_low_confidence_calls_llm():
    judge = make_judge(min_confidence=0.90)
    offline = {"risk_type": "spam", "confidence": 0.80, "policy_tags": ["营销引流"]}
    needs_llm, reason = judge.should_call("优惠", offline)
    assert needs_llm is True
    assert reason == "low_confidence"


def test_at_threshold_calls_llm():
    judge = make_judge(min_confidence=0.90)
    offline = {"risk_type": "spam", "confidence": 0.89, "policy_tags": []}
    needs_llm, _ = judge.should_call("内容", offline)
    assert needs_llm is True


def test_above_threshold_skips_llm_for_clear_types():
    judge = make_judge(min_confidence=0.90)
    offline = {"risk_type": "illegal", "confidence": 0.90, "policy_tags": ["违法违规"]}
    needs_llm, reason = judge.should_call("逃税内容", offline)
    assert needs_llm is False
    assert reason == "offline_sufficient"


# ---- context-dependent types ----

def test_political_always_calls_llm():
    judge = make_judge()
    offline = {"risk_type": "political", "confidence": 0.90, "policy_tags": ["涉政表达"]}
    needs_llm, reason = judge.should_call("政府内容", offline)
    assert needs_llm is True
    assert "political" in reason


# ---- minor verification ----

def test_minor_without_compound_calls_llm():
    """Simple minor keyword alone should trigger LLM — "未成年人不应该沉迷学习" is actually safe."""
    judge = make_judge()
    offline = {"risk_type": "minor", "confidence": 0.90, "policy_tags": ["未成年人保护"]}
    needs_llm, reason = judge.should_call("未成年人不应该沉迷学习", offline)
    assert needs_llm is True
    assert reason == "minor_needs_semantic_verification"


# ---- safe long text ----

def test_safe_short_text_skips_llm():
    judge = make_judge(safe_max_len=150)
    offline = {"risk_type": "safe", "confidence": 0.92, "policy_tags": []}
    needs_llm, reason = judge.should_call("今天天气真好", offline)
    assert needs_llm is False
    assert reason == "offline_sufficient"


def test_safe_long_text_calls_llm():
    judge = make_judge(safe_max_len=20)
    offline = {"risk_type": "safe", "confidence": 0.92, "policy_tags": []}
    long_text = "这是一段超过阈值的文本，里面可能有离线分类器没捕捉到的细微风险信号。"
    needs_llm, reason = judge.should_call(long_text, offline)
    assert needs_llm is True
    assert reason == "safe_long_text"


# ---- clear-case types that skip LLM ----

@pytest.mark.parametrize("risk_type,tag", [
    ("illegal", "违法违规"),
    ("spam",    "营销引流"),
    ("violence","暴力血腥"),
])
def test_clear_risk_types_skip_llm(risk_type, tag):
    judge = make_judge()
    offline = {"risk_type": risk_type, "confidence": 0.90, "policy_tags": [tag]}
    needs_llm, reason = judge.should_call("测试内容", offline)
    assert needs_llm is False
    assert reason == "offline_sufficient"


# ---- env-based config ----

def test_judge_config_from_env(monkeypatch):
    monkeypatch.setenv("JUDGE_ENABLED", "1")
    monkeypatch.setenv("JUDGE_MIN_CONFIDENCE", "0.75")
    monkeypatch.setenv("JUDGE_SAFE_MAX_LEN", "50")
    cfg = JudgeConfig.from_env()
    assert cfg.enabled is True
    assert cfg.min_confidence == 0.75
    assert cfg.safe_max_len == 50


def test_judge_config_disabled_via_env(monkeypatch):
    monkeypatch.setenv("JUDGE_ENABLED", "0")
    cfg = JudgeConfig.from_env()
    assert cfg.enabled is False
