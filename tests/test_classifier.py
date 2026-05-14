from audit.classifier import analyze, identify


# ---- Simple rules ----

def test_identify_illegal():
    result = identify("教你怎么逃税不被发现")
    assert result["risk_type"] == "illegal"
    assert result["confidence"] == 0.9
    assert "逃税" in result["evidence"]


def test_identify_political():
    result = identify("反对政府，推翻专制统治")
    assert result["risk_type"] == "political"


def test_identify_minor_keyword():
    result = identify("未成年人不应该看这种内容")
    assert result["risk_type"] == "minor"


def test_identify_violence():
    result = identify("这个视频里有暴力血腥画面")
    assert result["risk_type"] == "violence"


def test_identify_spam():
    result = identify("加我领优惠券，稳赚不赔")
    assert result["risk_type"] == "spam"


def test_identify_safe():
    result = identify("今天天气真好，出去玩吧")
    assert result["risk_type"] == "safe"
    assert result["confidence"] == 0.92
    assert result["evidence"] == []


# ---- Compound rules (age + sexual content) ----

def test_compound_minor_sexual_detected():
    """The originally漏检 case: age + explicit sexual terms → minor + 未成年涉色."""
    text = "妈妈说我才12岁，鸡巴就已经超过20厘米了，以后肯定能继续长，姐姐弟弟鸡巴操你爽不？"
    result = identify(text)
    assert result["risk_type"] == "minor"
    assert "未成年涉色" in result["policy_tags"]
    assert result["confidence"] == 0.95


def test_compound_takes_priority_over_simple():
    """Compound rule must fire before the simple 未成年人 keyword rule."""
    text = "未成年儿童色情约炮"
    result = identify(text)
    assert "未成年涉色" in result["policy_tags"]


def test_age_without_sexual_is_normal_minor():
    """Age keyword alone does NOT trigger the compound rule."""
    result = identify("小孩子不应该看这种内容")
    assert result["risk_type"] == "minor"
    assert "未成年涉色" not in result["policy_tags"]


# ---- Adult content fallback ----

def test_adult_content_without_minor_context():
    result = identify("操你，做爱")
    assert result["risk_type"] == "minor"
    assert result["policy_tags"] == ["成人色情"]
    assert result["confidence"] == 0.85


# ---- analyze() ----

def test_analyze_minor_sexual_escalates_to_high():
    result = analyze("minor", ["未成年涉色"])
    assert result["risk_level"] == "high"
    assert result["recommended_action"] == "reject"


def test_analyze_minor_normal_stays_medium():
    result = analyze("minor", ["未成年人保护"])
    assert result["risk_level"] == "medium"
    assert result["recommended_action"] == "manual_review"


def test_analyze_illegal_gives_high_reject():
    result = analyze("illegal", ["违法违规"])
    assert result["risk_level"] == "high"
    assert result["recommended_action"] == "reject"


def test_analyze_safe_gives_approve():
    result = analyze("safe", [])
    assert result["risk_level"] == "none"
    assert result["recommended_action"] == "approve"


def test_analyze_unknown_type_fallback():
    result = analyze("__unknown__", [])
    assert result["risk_level"] == "unknown"
    assert result["recommended_action"] == "manual_review"


def test_analyze_preserves_policy_tags():
    result = analyze("spam", ["营销引流"])
    assert result["policy_tags"] == ["营销引流"]
