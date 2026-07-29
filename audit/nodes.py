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


def _apply_offline_result(state: RiskState, offline: dict[str, Any]) -> None:
    state["risk_type"] = offline["risk_type"]  # type: ignore[typeddict-item]
    state["confidence"] = offline["confidence"]
    state["policy_tags"] = offline["policy_tags"]
    state["evidence"] = offline["evidence"]


def identify_risk(state: RiskState) -> RiskState:
    """Classify content risk type — offline-first, escalate to LLM on demand.

    1. Always run the cheap offline classifier first.
    2. Ask LLMCallJudge whether that offline result is trustworthy.
    3. Only call the LLM API when the judge says escalation is needed AND the
       API is configured (not in offline demo mode). This avoids paying for an
       LLM call on the cases the offline classifier already handles confidently.
    On any LLM error, fall back to the offline result we already computed.
    """
    text = state["normalized_input"]
    cfg = get_config()
    state["llm_called"] = False

    # Step 1: offline classifier (cheap, deterministic, always runs)
    offline = classifier.identify(text)

    # Step 2: judge decides whether the offline result needs LLM verification
    needs_llm, reason = get_judge().should_call(text, offline)

    # Step 3: skip the LLM when the judge says it's unnecessary, or when the
    # LLM is simply unavailable (offline demo mode / no API key).
    llm_available = (not cfg.offline_demo_mode) and bool(cfg.deepseek_api_key)
    if not needs_llm or not llm_available:
        _apply_offline_result(state, offline)
        if not needs_llm:
            state["judge_reason"] = reason  # e.g. offline_sufficient / high_confidence_compound_rule
        elif cfg.offline_demo_mode:
            state["judge_reason"] = "offline_demo_mode"
        else:
            state["judge_reason"] = "llm_unavailable_no_api_key"
        return state

    # Step 4: judge requested escalation → 一次 LLM 调用同时产出"识别 + 分析"，
    # analyze_risk 直接复用，避免对同一条内容付两次费（成本优化）。
    prompt = """
你是内容安全 Agent 的风险研判节点，对 UGC 评论做一次性的"识别 + 分级"。
请只返回 JSON，不要输出 markdown。

字段要求：
- risk_type: political/minor/adult/illegal/violence/spam/safe/unknown
- confidence: 0 到 1 的小数
- policy_tags: 命中的策略标签数组，例如 ["涉政表达", "违法教程", "成人色情", "乱伦涉色"]
- evidence: 从原文中摘取不超过 3 个关键证据，不能编造
- risk_level: high/medium/low/none/unknown
- recommended_action: approve/reject/manual_review（不允许 skip；skip 只属于人工决策）
- analysis_result: 用中文说明主要风险、上下文判断、误杀可能性，120 字以内

风险类型口径：
- political: 政治人物 / 制度 / 政权 / 敏感公共事件 / 煽动性政治表达
- minor: 涉及未成年人保护、诱导、伤害，或未成年人 × 性相关内容
- adult: 成人（非未成年）色情、淫秽、性交易等明确成人性内容
- illegal: 违法犯罪、规避监管、黑灰产、毒品、武器、诈骗
- violence: 暴力、血腥、威胁、仇恨攻击
- spam: 广告引流、刷量、博彩、营销灌水
- safe: 普通闲聊、生活分享、无明显风险
- unknown: 信息不足或无法确定

处置口径：
- high → 建议 reject / manual_review；medium → manual_review；
  low → approve / manual_review；none → approve；unknown → manual_review
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
        # 同一次调用顺带产出二级分析，供 analyze_risk 复用
        state["risk_level"] = _normalize_label(data.get("risk_level"), ALLOWED_RISK_LEVELS, "unknown")  # type: ignore[typeddict-item]
        state["recommended_action"] = _normalize_label(  # type: ignore[typeddict-item]
            data.get("recommended_action"),
            {"approve", "reject", "manual_review"},  # skip 不是模型建议动作，只属于人工决策
            "manual_review",
        )
        state["analysis_result"] = str(data.get("analysis_result", "")).strip()[:500]
        state["analysis_from_llm"] = True
        state["llm_called"] = True
        state["judge_reason"] = f"escalated:{reason}"
        return state
    except Exception as exc:
        # LLM failed → fall back to the offline result already computed in step 1
        _apply_offline_result(state, offline)
        state["model_error"] = f"identify_risk LLM failed, using offline fallback: {exc}"
        state["judge_reason"] = f"escalation_failed_offline_fallback:{reason}"
        return state


def analyze_risk(state: RiskState) -> RiskState:
    """二级分析（定 risk_level + recommended_action）。

    成本优化：identify_risk 升级 LLM 时已在同一次调用里产出了二级分析
    （analysis_from_llm=True），这里直接复用，不再二次调用 LLM——单条内容
    最多一次付费调用。其余情况（离线 / LLM 失败降级）走离线确定性分析映射。
    """
    if state.get("analysis_from_llm"):
        return state  # risk_level / recommended_action / analysis_result 已由 identify_risk 写入

    data = classifier.analyze(state["risk_type"], state["policy_tags"])
    state["risk_level"] = data["risk_level"]  # type: ignore[typeddict-item]
    state["analysis_result"] = data["analysis_result"]
    state["recommended_action"] = data["recommended_action"]  # type: ignore[typeddict-item]
    state["policy_tags"] = data["policy_tags"]
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

    # 只认显式传入的 state["auto_decision"]（CLI -d / API）。不再读取
    # cfg.auto_review_decision，避免环境变量在背后把人工复核静默跳过。
    override = state.get("auto_decision", "")
    # 显式"待人工"信号（API / 服务化无人值守）：不调用 input()，直接挂起为 pending。
    if override == "pending":
        print("标记为 pending_review（待人工复核）")
        state["review_status"] = "pending"
        state["human_decision"] = ""
        return state
    if override in {"approve", "reject", "skip"}:
        state["human_decision"] = override
        state["review_status"] = "decided"
        print(f"自动决策: {override}")
        return state

    while True:
        try:
            decision = input("请输入人工决策 [approve/reject/skip]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            # 非交互环境（管道/CI/无 TTY）或用户中断：标记为待人工复核(pending)，
            # 既不崩溃丢结果，也不伪造成"真人 skip"，语义清晰。
            print("\n无交互输入，标记为 pending_review（待人工复核）")
            state["review_status"] = "pending"
            state["human_decision"] = ""
            return state
        if decision in {"approve", "reject", "skip"}:
            state["human_decision"] = decision
            state["review_status"] = "decided"
            break
        print("输入无效，请重新输入")
    return state


def generate_report(state: RiskState) -> RiskState:
    # final_action 语义：真人/显式决策优先；命中复核但无人决策 → pending_review；
    # 其余（未进复核的低风险）→ 采用模型建议动作。
    if state["human_decision"]:
        final_action = state["human_decision"]
    elif state.get("review_status") == "pending":
        final_action = "pending_review"
    else:
        final_action = state["recommended_action"]
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
