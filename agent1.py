import json
import os
import re
from collections import Counter
from datetime import datetime
from typing import Any, Literal, NotRequired, TypedDict
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


RiskType = Literal["political", "minor", "illegal", "violence", "spam", "safe", "unknown"]
RiskLevel = Literal["high", "medium", "low", "none", "unknown"]
AuditAction = Literal["approve", "reject", "manual_review", "skip"]

ALLOWED_RISK_TYPES: set[str] = {"political", "minor", "illegal", "violence", "spam", "safe", "unknown"}
ALLOWED_RISK_LEVELS: set[str] = {"high", "medium", "low", "none", "unknown"}
REVIEW_LEVELS = {"high", "medium", "unknown"}
REVIEW_TYPES = {"political", "minor", "illegal", "unknown"}
LOG_PATH = os.getenv("AUDIT_LOG_PATH", "audit_log.jsonl")
AUTO_REVIEW_DECISION = os.getenv("AUTO_REVIEW_DECISION", "").strip().lower()
OFFLINE_DEMO_MODE = os.getenv("OFFLINE_DEMO_MODE", "1").strip().lower() not in {"0", "false", "no"}
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "8"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "0"))


class RiskState(TypedDict):
    trace_id: str
    user_input: str
    normalized_input: str
    risk_type: RiskType
    risk_level: RiskLevel
    confidence: float
    policy_tags: list[str]
    evidence: list[str]
    analysis_result: str
    recommended_action: AuditAction
    human_decision: str
    final_report: str
    model_error: NotRequired[str]


_llm: ChatOpenAI | None = None


def create_llm() -> ChatOpenAI:
    """Initialize model from env so the demo is safe for interview and production."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DEEPSEEK_API_KEY. Please set it before running agent1.py.")

    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        api_key=api_key,
        temperature=0,
        timeout=LLM_TIMEOUT,
        max_retries=LLM_MAX_RETRIES,
    )


def get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = create_llm()
    return _llm


def new_state(user_input: str) -> RiskState:
    return {
        "trace_id": str(uuid4()),
        "user_input": user_input,
        "normalized_input": "",
        "risk_type": "unknown",
        "risk_level": "unknown",
        "confidence": 0.0,
        "policy_tags": [],
        "evidence": [],
        "analysis_result": "",
        "recommended_action": "manual_review",
        "human_decision": "",
        "final_report": "",
    }


def extract_json(text: str) -> dict[str, Any]:
    """Parse model JSON with a conservative fallback for markdown-wrapped output."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def clamp_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))


def normalize_label(value: Any, allowed: set[str], default: str) -> str:
    label = str(value or "").strip().lower()
    return label if label in allowed else default


def normalize_input(state: RiskState) -> RiskState:
    text = re.sub(r"\s+", " ", state["user_input"]).strip()
    state["normalized_input"] = text[:2000]
    return state


def offline_identify_risk(text: str) -> dict[str, Any]:
    """Small deterministic classifier for demos when the LLM/network is unavailable."""
    rules: list[tuple[str, list[str], str]] = [
        ("illegal", ["逃税", "诈骗", "毒品", "武器", "赌博", "黑产"], "违法违规"),
        ("minor", ["小孩子", "未成年人", "儿童", "学生"], "未成年人保护"),
        ("political", ["政府", "推翻", "专制", "政权"], "涉政表达"),
        ("violence", ["暴力", "血腥", "威胁", "杀"], "暴力血腥"),
        ("spam", ["优惠券", "加我", "稳赚", "返利", "领券"], "营销引流"),
    ]
    for risk_type, keywords, tag in rules:
        evidence = [keyword for keyword in keywords if keyword in text]
        if evidence:
            return {
                "risk_type": risk_type,
                "confidence": 0.9,
                "policy_tags": [tag],
                "evidence": evidence[:3],
            }
    return {
        "risk_type": "safe",
        "confidence": 0.92,
        "policy_tags": [],
        "evidence": [],
    }


def offline_analyze_risk(state: RiskState) -> dict[str, Any]:
    risk_map: dict[str, tuple[str, str, str]] = {
        "illegal": ("high", "内容涉及违法违规方法或规避监管，需要拒绝或人工确认。", "reject"),
        "political": ("medium", "内容涉及政治表达，语义较敏感，建议人工复核判断上下文。", "manual_review"),
        "minor": ("medium", "内容涉及未成年人语境，需结合上下文判断是否存在保护风险。", "manual_review"),
        "violence": ("medium", "内容提到暴力或血腥画面，建议人工复核素材上下文。", "manual_review"),
        "spam": ("low", "内容疑似营销引流，风险较低，可按平台策略处理。", "approve"),
        "safe": ("none", "未识别到明显内容安全风险。", "approve"),
    }
    risk_level, analysis_result, action = risk_map.get(
        state["risk_type"],
        ("unknown", "信息不足，建议进入人工复核。", "manual_review"),
    )
    return {
        "risk_level": risk_level,
        "analysis_result": analysis_result,
        "recommended_action": action,
        "policy_tags": state["policy_tags"],
    }


def identify_risk(state: RiskState) -> RiskState:
    """Classify user content into a moderation risk category with confidence."""
    if OFFLINE_DEMO_MODE:
        data = offline_identify_risk(state["normalized_input"])
        state["risk_type"] = data["risk_type"]  # type: ignore[typeddict-item]
        state["confidence"] = data["confidence"]
        state["policy_tags"] = data["policy_tags"]
        state["evidence"] = data["evidence"]
        return state

    prompt = """
你是内容安全 Agent 的风险识别节点，负责给 UGC 评论做一级分流。
请只返回 JSON，不要输出 markdown。

字段要求：
- risk_type: political/minor/illegal/violence/spam/safe/unknown
- confidence: 0 到 1 的小数
- policy_tags: 命中的策略标签数组，例如 ["涉政表达", "违法教程"]
- evidence: 从原文中摘取不超过 3 个关键证据，不能编造

判定口径：
- political: 涉及政治人物、制度、政权、敏感公共事件或煽动性政治表达
- minor: 涉及未成年人保护、诱导、伤害、成人内容等
- illegal: 涉及违法犯罪、规避监管、黑灰产、毒品、武器、诈骗等
- violence: 暴力、血腥、威胁、仇恨攻击
- spam: 广告引流、刷量、博彩、疑似营销灌水
- safe: 普通闲聊、生活分享、无明显风险
- unknown: 信息不足或模型无法确定
"""
    try:
        response = get_llm().invoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(content=state["normalized_input"]),
            ]
        )
        data = extract_json(response.content)
        state["risk_type"] = normalize_label(data.get("risk_type"), ALLOWED_RISK_TYPES, "unknown")  # type: ignore[typeddict-item]
        state["confidence"] = clamp_float(data.get("confidence"))
        state["policy_tags"] = [str(tag)[:40] for tag in data.get("policy_tags", [])][:5]
        state["evidence"] = [str(item)[:120] for item in data.get("evidence", [])][:3]
    except Exception as exc:
        state["risk_type"] = "unknown"
        state["confidence"] = 0.0
        state["model_error"] = f"identify_risk failed: {exc}"
    return state


def analyze_risk(state: RiskState) -> RiskState:
    """Run a second-pass risk analysis and decide the first-line action."""
    if OFFLINE_DEMO_MODE:
        data = offline_analyze_risk(state)
        state["risk_level"] = data["risk_level"]  # type: ignore[typeddict-item]
        state["analysis_result"] = data["analysis_result"]
        state["recommended_action"] = data["recommended_action"]  # type: ignore[typeddict-item]
        state["policy_tags"] = data["policy_tags"]
        return state

    prompt = f"""
你是内容安全 Agent 的深度分析节点。当前一级分类为 {state["risk_type"]}。
请结合平台审核策略，输出可落地的二级判断。只返回 JSON。

字段要求：
- risk_level: high/medium/low/none/unknown
- analysis_result: 用中文说明主要风险、上下文判断、误杀可能性，120 字以内
- recommended_action: approve/reject/manual_review/skip
- policy_tags: 可以补充或修正策略标签数组

处置口径：
- high: 明确违法违规、严重伤害、明确煽动或高危未成年人风险，建议 reject 或 manual_review
- medium: 有明显风险但需要上下文，建议 manual_review
- low: 轻微风险或边界表达，建议 approve 或 manual_review
- none: 无风险，建议 approve
- unknown: 信息不足、模型异常或证据不充分，建议 manual_review
"""
    try:
        response = get_llm().invoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(content=state["normalized_input"]),
            ]
        )
        data = extract_json(response.content)
        state["risk_level"] = normalize_label(data.get("risk_level"), ALLOWED_RISK_LEVELS, "unknown")  # type: ignore[typeddict-item]
        state["analysis_result"] = str(data.get("analysis_result", "")).strip()[:500]
        state["recommended_action"] = normalize_label(
            data.get("recommended_action"),
            {"approve", "reject", "manual_review", "skip"},
            "manual_review",
        )  # type: ignore[typeddict-item]
        merged_tags = [*state["policy_tags"], *data.get("policy_tags", [])]
        state["policy_tags"] = list(dict.fromkeys(str(tag)[:40] for tag in merged_tags if tag))[:8]
    except Exception as exc:
        state["risk_level"] = "unknown"
        state["recommended_action"] = "manual_review"
        state["analysis_result"] = "模型分析失败，已进入人工复核兜底。"
        state["model_error"] = f"analyze_risk failed: {exc}"
    return state


def generate_safe_report(state: RiskState) -> RiskState:
    state["risk_level"] = "none"
    state["recommended_action"] = "approve"
    state["analysis_result"] = "未识别到明显内容安全风险。"
    state["final_report"] = json.dumps(
        {
            "trace_id": state["trace_id"],
            "risk_type": "safe",
            "risk_level": "none",
            "confidence": state["confidence"],
            "action": "approve",
            "reason": state["analysis_result"],
        },
        ensure_ascii=False,
        indent=2,
    )
    return state


def human_review(state: RiskState) -> RiskState:
    print(f"\n{'!' * 60}")
    print("命中人工复核队列")
    print(f"Trace ID: {state['trace_id']}")
    print(f"原始内容: {state['user_input']}")
    print(f"风险类型: {state['risk_type']} | 风险等级: {state['risk_level']} | 置信度: {state['confidence']:.2f}")
    print(f"策略标签: {', '.join(state['policy_tags']) or '无'}")
    print(f"模型分析: {state['analysis_result']}")
    print(f"{'!' * 60}")

    if AUTO_REVIEW_DECISION in {"approve", "reject", "skip"}:
        state["human_decision"] = AUTO_REVIEW_DECISION
        print(f"已使用 AUTO_REVIEW_DECISION={AUTO_REVIEW_DECISION} 作为人工决策")
        return state

    while True:
        decision = input("请输入人工决策 [approve/reject/skip]: ").strip().lower()
        if decision in {"approve", "reject", "skip"}:
            state["human_decision"] = decision
            break
        print("输入无效，请重新输入 approve / reject / skip")
    return state


def generate_report(state: RiskState) -> RiskState:
    final_action = state["human_decision"] or state["recommended_action"]
    report = {
        "trace_id": state["trace_id"],
        "risk_type": state["risk_type"],
        "risk_level": state["risk_level"],
        "confidence": state["confidence"],
        "policy_tags": state["policy_tags"],
        "evidence": state["evidence"],
        "analysis": state["analysis_result"],
        "recommended_action": state["recommended_action"],
        "human_decision": state["human_decision"],
        "final_action": final_action,
        "model_error": state.get("model_error", ""),
    }
    state["final_report"] = json.dumps(report, ensure_ascii=False, indent=2)
    return state


def route_by_risk(state: RiskState) -> Literal["generate_safe_report", "analyze_risk"]:
    if state["risk_type"] == "safe" and state["confidence"] >= 0.75:
        return "generate_safe_report"
    return "analyze_risk"


def route_by_level(state: RiskState) -> Literal["human_review", "generate_report"]:
    if state["risk_type"] in REVIEW_TYPES and state["risk_level"] not in {"none", "low"}:
        return "human_review"
    if state["risk_level"] in REVIEW_LEVELS:
        return "human_review"
    if state["recommended_action"] == "manual_review":
        return "human_review"
    return "generate_report"


def build_graph():
    workflow = StateGraph(RiskState)
    workflow.add_node("normalize_input", normalize_input)
    workflow.add_node("identify_risk", identify_risk)
    workflow.add_node("analyze_risk", analyze_risk)
    workflow.add_node("generate_safe_report", generate_safe_report)
    workflow.add_node("human_review", human_review)
    workflow.add_node("generate_report", generate_report)

    workflow.set_entry_point("normalize_input")
    workflow.add_edge("normalize_input", "identify_risk")
    workflow.add_conditional_edges(
        "identify_risk",
        route_by_risk,
        {
            "generate_safe_report": "generate_safe_report",
            "analyze_risk": "analyze_risk",
        },
    )
    workflow.add_conditional_edges(
        "analyze_risk",
        route_by_level,
        {
            "human_review": "human_review",
            "generate_report": "generate_report",
        },
    )
    workflow.add_edge("human_review", "generate_report")
    workflow.add_edge("generate_safe_report", END)
    workflow.add_edge("generate_report", END)
    return workflow.compile()


app = build_graph()


def save_log(result: RiskState) -> None:
    log = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "trace_id": result["trace_id"],
        "input": result["user_input"],
        "risk_type": result["risk_type"],
        "risk_level": result["risk_level"],
        "confidence": result["confidence"],
        "recommended_action": result["recommended_action"],
        "human_decision": result["human_decision"],
        "final_report": result["final_report"],
    }
    with open(LOG_PATH, "a", encoding="utf-8") as file:
        file.write(json.dumps(log, ensure_ascii=False) + "\n")


def audit_one(comment: str) -> RiskState:
    result = app.invoke(new_state(comment))
    save_log(result)
    return result


def batch_audit(comments: list[str]) -> list[RiskState]:
    results: list[RiskState] = []
    for index, comment in enumerate(comments, start=1):
        print(f"\n{'=' * 60}")
        print(f"处理第 {index}/{len(comments)} 条: {comment}")
        result = audit_one(comment)
        results.append(result)
        print(result["final_report"])

    risk_counter = Counter(result["risk_type"] for result in results)
    level_counter = Counter(result["risk_level"] for result in results)
    review_count = sum(1 for result in results if result["human_decision"])

    print(f"\n{'=' * 60}")
    print(f"批量审核完成，共处理 {len(results)} 条")
    print(f"风险类型分布: {dict(risk_counter)}")
    print(f"风险等级分布: {dict(level_counter)}")
    print(f"人工复核数量: {review_count}")
    print(f"日志已保存至 {LOG_PATH}")
    return results


if __name__ == "__main__":
    if not AUTO_REVIEW_DECISION:
        AUTO_REVIEW_DECISION = "skip"
        print("演示批处理默认使用 AUTO_REVIEW_DECISION=skip，避免人工复核阻塞。")

    test_comments = [
        "这个政策真的太好了，支持！",
        "小孩子不应该看这种内容",
        "教你怎么逃税不被发现",
        "今天天气真好，出去玩吧",
        "未成年人不应该沉迷学习",
        "反对政府，推翻专制统治",
        "这个视频里有暴力血腥画面",
        "加我领优惠券，稳赚不赔",
    ]
    batch_audit(test_comments)
