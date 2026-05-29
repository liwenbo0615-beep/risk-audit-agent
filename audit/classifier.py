"""Offline deterministic classifier — no LLM, no network required.

Matching priority:
  1. Compound rules  (group_a AND group_b) — catches context-dependent patterns
  2. Simple rules    (any keyword)
  3. Adult content   (explicit terms without minor context)
"""

from typing import Any

# fmt: off
# Simple rules: any single keyword triggers the rule
_SIMPLE_RULES: list[tuple[str, list[str], str]] = [
    ("illegal",   ["逃税", "诈骗", "毒品", "武器", "赌博", "黑产", "洗钱", "走私"],         "违法违规"),
    ("minor",     ["小孩子", "未成年人", "儿童", "小学生", "初中生", "幼儿园", "幼童"],      "未成年人保护"),
    ("political", ["政府", "推翻", "专制", "政权"],                                         "涉政表达"),
    ("violence",  ["暴力", "血腥", "威胁", "杀人"],                                         "暴力血腥"),
    ("spam",      ["优惠券", "加我", "稳赚", "返利", "领券"],                               "营销引流"),
]

# Compound rules: at least one hit from EACH group triggers the rule
# Use tag "未成年涉色" so analyze() can escalate to high severity
_SEXUAL_TERMS: list[str] = [
    "鸡巴", "操你", "操我", "做爱", "性爱", "裸体", "淫", "约炮", "性交",
    "小穴", "阴道", "阴茎", "阴部", "阴户", "阴蒂", "射精", "强奸",
    "乱伦", "肛交", "口交", "自慰", "插入", "勃起",
]

_COMPOUND_RULES: list[tuple[str, list[str], list[str], str]] = [
    (
        "minor",
        ["岁", "小学", "初中", "未成年", "儿童", "小孩", "孩子", "少女", "少年"],
        _SEXUAL_TERMS,
        "未成年涉色",
    ),
    (
        "illegal",
        ["妹妹", "姐姐", "妈妈", "女儿", "哥哥", "爸爸", "弟弟", "兄妹", "母子", "父女"],
        _SEXUAL_TERMS,
        "乱伦涉色",
    ),
]

# Explicit adult content without a minor/family context indicator
_ADULT_KEYWORDS: list[str] = [
    "鸡巴", "操你", "操我", "做爱", "性交", "色情", "淫秽", "约炮",
    "小穴", "阴道", "阴茎", "阴部", "阴户", "阴蒂", "射精", "强奸",
    "乱伦", "肛交", "口交", "自慰", "插入", "勃起",
]
# fmt: on

_ANALYSIS_MAP: dict[str, tuple[str, str, str]] = {
    "illegal":   ("high",   "内容涉及违法违规方法或规避监管，需要拒绝或人工确认。",         "reject"),
    "political": ("medium", "内容涉及政治表达，语义较敏感，建议人工复核判断上下文。",       "manual_review"),
    "minor":     ("medium", "内容涉及未成年人语境，需结合上下文判断是否存在保护风险。",     "manual_review"),
    "violence":  ("medium", "内容提到暴力或血腥画面，建议人工复核素材上下文。",             "manual_review"),
    "spam":      ("low",    "内容疑似营销引流，风险较低，可按平台策略处理。",               "approve"),
    "safe":      ("none",   "未识别到明显内容安全风险。",                                   "approve"),
}


def identify(text: str) -> dict[str, Any]:
    # 1. Compound rules have higher priority
    for risk_type, group_a, group_b, tag in _COMPOUND_RULES:
        hit_a = [kw for kw in group_a if kw in text]
        hit_b = [kw for kw in group_b if kw in text]
        if hit_a and hit_b:
            return {
                "risk_type": risk_type,
                "confidence": 0.95,
                "policy_tags": [tag],
                "evidence": (hit_a + hit_b)[:3],
            }

    # 2. Simple single-keyword rules
    for risk_type, keywords, tag in _SIMPLE_RULES:
        evidence = [kw for kw in keywords if kw in text]
        if evidence:
            return {
                "risk_type": risk_type,
                "confidence": 0.9,
                "policy_tags": [tag],
                "evidence": evidence[:3],
            }

    # 3. Explicit adult content (no minor indicator found above)
    adult_hits = [kw for kw in _ADULT_KEYWORDS if kw in text]
    if adult_hits:
        return {
            "risk_type": "minor",
            "confidence": 0.85,
            "policy_tags": ["成人色情"],
            "evidence": adult_hits[:3],
        }

    return {"risk_type": "safe", "confidence": 0.92, "policy_tags": [], "evidence": []}


def analyze(risk_type: str, policy_tags: list[str]) -> dict[str, Any]:
    risk_level, analysis_result, action = _ANALYSIS_MAP.get(
        risk_type,
        ("unknown", "信息不足，建议进入人工复核。", "manual_review"),
    )
    # Escalate: CSAM-type content must be rejected immediately
    if "未成年涉色" in policy_tags:
        risk_level = "high"
        action = "reject"
        analysis_result = "内容涉及未成年人性相关表达，属于严重违规，建议立即拒绝并上报。"
    # Escalate: incest sexual content is illegal and must be rejected
    if "乱伦涉色" in policy_tags:
        risk_level = "high"
        action = "reject"
        analysis_result = "内容涉及家庭成员间性行为描写，属于违法违规内容，建议立即拒绝。"
    return {
        "risk_level": risk_level,
        "analysis_result": analysis_result,
        "recommended_action": action,
        "policy_tags": policy_tags,
    }
