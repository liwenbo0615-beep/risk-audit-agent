from typing import Literal

from langgraph.graph import END, StateGraph

from .models import REVIEW_LEVELS, REVIEW_TYPES, RiskState
from .nodes import (
    analyze_risk,
    generate_report,
    generate_safe_report,
    human_review,
    identify_risk,
    normalize_input,
)


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
    wf = StateGraph(RiskState)
    wf.add_node("normalize_input", normalize_input)
    wf.add_node("identify_risk", identify_risk)
    wf.add_node("analyze_risk", analyze_risk)
    wf.add_node("generate_safe_report", generate_safe_report)
    wf.add_node("human_review", human_review)
    wf.add_node("generate_report", generate_report)

    wf.set_entry_point("normalize_input")
    wf.add_edge("normalize_input", "identify_risk")
    wf.add_conditional_edges("identify_risk", route_by_risk, {
        "generate_safe_report": "generate_safe_report",
        "analyze_risk": "analyze_risk",
    })
    wf.add_conditional_edges("analyze_risk", route_by_level, {
        "human_review": "human_review",
        "generate_report": "generate_report",
    })
    wf.add_edge("human_review", "generate_report")
    wf.add_edge("generate_safe_report", END)
    wf.add_edge("generate_report", END)
    return wf.compile()


_app = None


def get_app():
    global _app
    if _app is None:
        _app = build_graph()
    return _app


def reset_app() -> None:
    global _app
    _app = None
