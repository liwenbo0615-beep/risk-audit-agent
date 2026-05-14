import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from . import classifier
from .config import get_config
from .judge import get_judge
from .models import ALLOWED_RISK_LEVELS, ALLOWED_RISK_TYPES, RiskState

_llm: "ChatOpenAI | None" = None


def get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        cfg = get_config()
        if not cfg.deepseek_api_key:
            raise RuntimeError("Missing DEEPSEEK_API_KEY. Set it in .env before running.")
        _llm = ChatOpenAI(
            model=cfg.deepseek_model,
            base_url=cfg.deepseek_base_url,
            api_key=cfg.deepseek_api_key,
            temperature=0,
            timeout=cfg.llm_timeout,
            max_retries=cfg.llm_max_retries,
        )
    return _llm


def reset_llm() -> None:
    global _llm
    _llm = None


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def _clamp_float(value: Any, default: float = 0.0) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, n))


def _normalize_label(value: Any, allowed: set[str], default: str) -> str:
    label = str(value or "").strip().lower()
    return label if label in allowed else default


# ---- LangGraph node functions ----

def normalize_input(state: RiskState) -> RiskState:
    text = re.sub(r"\s+", " ", state["user_input"]).strip()
    state["normalized_input"] = text[:2000]
    return state


def identify_risk(state: RiskState) -> RiskState:
    """Classify content risk type.

    Always runs the offline classifier first, then the judge decides whether
    to escalate to the LLM API. The LLM is only called when offline is insufficient.
    """
    text = state["normalized_input"]
    cfg = get_config()

    # Step 1: offline classifier (always runs, zero latency)
    offline = classifier.identify(text)

    # Step 2: judge — should we call LLM?
    needs_llm, reason = get_judge().should_call(text, offline)
    state["judge_reason"] = reason
    state["llm_called"] = False

    # Step 3: try LLM if judge says yes and we're not in offline-only mode
    if needs_llm and not cfg.offline_demo_mode and cfg.deepseek_api_key:
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
            response = get_llm().invoke([
                SystemMessage(content=prompt),
                HumanMessage(content=text),
            ])
            data = _extract_json(response.content)
            state["risk_type"] = _normalize_label(data.get("risk_type"), ALLOWED_RISK_TYPES, "unknown")  # type: ignore[typeddict-item]
            state["confidence"] = _clamp_float(data.get("confidence"))
            state["policy_tags"] = [str(t)[:40] for t in data.get("policy_tags", [])][:5]
            state["evidence"] = [str(e)[:120] for e in data.get("evidence", [])][:3]
            state["llm_called"] = True
            return state
        except Exception as exc:
            # LLM failed → fall through to offline result
            state["model_error"] = f"identify_risk LLM failed, using offline fallback: {exc}"

    # Step 4: use offline result (either judge said no, or LLM failed)
    state["risk_type"] = offline["risk_type"]  # type: ignore[typeddict-item]
    state["confidence"] = offline["confidence"]
    state["policy_tags"] = offline["policy_tags"]
    state["evidence"] = offline["evidence"]
    return state


def analyze_risk(state: RiskState) -> RiskState:
    """Run second-pass risk analysis.

    Uses LLM only when identify_risk already used LLM for this request.
    If identify_risk fell back to offline, so do we.
    """
    cfg = get_config()
    use_llm = state.get("llm_called", False) and not cfg.offline_demo_mode

    if not use_llm:
        data = classifier.analyze(state["risk_type"], state["policy_tags"])
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
        response = get_llm().invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=state["normalized_input"]),
        ])
        data = _extract_json(response.content)
        state["risk_level"] = _normalize_label(data.get("risk_level"), ALLOWED_RISK_LEVELS, "unknown")  # type: ignore[typeddict-item]
        state["analysis_result"] = str(data.get("analysis_result", "")).strip()[:500]
        state["recommended_action"] = _normalize_label(  # type: ignore[typeddict-item]
            data.get("recommended_action"),
            {"approve", "reject", "manual_review", "skip"},
            "manual_review",
        )
        merged = [*state["policy_tags"], *data.get("policy_tags", [])]
        state["policy_tags"] = list(dict.fromkeys(str(t)[:40] for t in merged if t))[:8]
    except Exception as exc:
        data = classifier.analyze(state["risk_type"], state["policy_tags"])
        state["risk_level"] = data["risk_level"]  # type: ignore[typeddict-item]
        state["analysis_result"] = data["analysis_result"]
        state["recommended_action"] = data["recommended_action"]  # type: ignore[typeddict-item]
        state["model_error"] = f"analyze_risk LLM failed, using offline fallback: {exc}"
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
            "policy_tags": state["policy_tags"],
            "evidence": state["evidence"],
            "analysis": state["analysis_result"],
            "recommended_action": "approve",
            "human_decision": "",
            "final_action": "approve",
            "llm_called": state.get("llm_called", False),
            "judge_reason": state.get("judge_reason", ""),
            "model_error": "",
        },
        ensure_ascii=False,
        indent=2,
    )
    return state


def human_review(state: RiskState) -> RiskState:
    print(f"\n{'!' * 60}")
    print("命中人工复核队列")
    print(f"Trace ID    : {state['trace_id']}")
    print(f"原始内容    : {state['user_input']}")
    print(f"风险类型    : {state['risk_type']} | 风险等级: {state['risk_level']} | 置信度: {state['confidence']:.2f}")
    print(f"策略标签    : {', '.join(state['policy_tags']) or '无'}")
    print(f"模型分析    : {state['analysis_result']}")
    print(f"LLM 已调用  : {state.get('llm_called', False)} ({state.get('judge_reason', '')})")
    print(f"{'!' * 60}")

    cfg = get_config()
    override = state.get("auto_decision") or cfg.auto_review_decision
    if override in {"approve", "reject", "skip"}:
        state["human_decision"] = override
        print(f"自动决策: {override}")
        return state

    while True:
        decision = input("请输入人工决策 [approve/reject/skip]: ").strip().lower()
        if decision in {"approve", "reject", "skip"}:
            state["human_decision"] = decision
            break
        print("输入无效，请重新输入")
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
        "llm_called": state.get("llm_called", False),
        "judge_reason": state.get("judge_reason", ""),
        "model_error": state.get("model_error", ""),
    }
    state["final_report"] = json.dumps(report, ensure_ascii=False, indent=2)
    return state
