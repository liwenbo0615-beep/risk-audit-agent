from typing import Any, Literal, NotRequired, TypedDict
from uuid import uuid4

from pydantic import BaseModel, Field

RiskType = Literal["political", "minor", "illegal", "violence", "spam", "safe", "unknown"]
RiskLevel = Literal["high", "medium", "low", "none", "unknown"]
AuditAction = Literal["approve", "reject", "manual_review", "skip"]

ALLOWED_RISK_TYPES: set[str] = {"political", "minor", "illegal", "violence", "spam", "safe", "unknown"}
ALLOWED_RISK_LEVELS: set[str] = {"high", "medium", "low", "none", "unknown"}
REVIEW_LEVELS: set[str] = {"high", "medium", "unknown"}
REVIEW_TYPES: set[str] = {"political", "minor", "illegal", "unknown"}


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
    auto_decision: NotRequired[str]
    model_error: NotRequired[str]
    llm_called: NotRequired[bool]
    judge_reason: NotRequired[str]


def new_state(user_input: str, auto_decision: str = "") -> RiskState:
    state: RiskState = {
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
    if auto_decision:
        state["auto_decision"] = auto_decision
    return state


# ---- Pydantic schemas for REST API ----

class AuditRequest(BaseModel):
    comment: str = Field(..., min_length=1, max_length=5000)
    auto_decision: Literal["approve", "reject", "skip"] | None = Field(
        None, description="设置后跳过交互式人工审核"
    )


class BatchAuditRequest(BaseModel):
    comments: list[str] = Field(..., min_length=1, max_length=100)
    auto_decision: Literal["approve", "reject", "skip"] = Field("skip")


class AuditResponse(BaseModel):
    trace_id: str
    risk_type: str
    risk_level: str
    confidence: float
    policy_tags: list[str]
    evidence: list[str]
    analysis: str
    recommended_action: str
    human_decision: str
    final_action: str
    llm_called: bool
    judge_reason: str
    model_error: str


class BatchSummary(BaseModel):
    total: int
    risk_type_dist: dict[str, int]
    risk_level_dist: dict[str, int]
    manual_review_count: int


class BatchAuditResponse(BaseModel):
    results: list[AuditResponse]
    summary: BatchSummary
