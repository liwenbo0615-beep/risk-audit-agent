"""FastAPI service for the content audit agent.

Run with:
    uvicorn api.main:app --reload --port 8000
"""

import sys
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure project root is on the path when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from audit import review_store
from audit.service import audit_one, batch_audit
from audit.models import (
    AuditRequest,
    AuditResponse,
    BatchAuditRequest,
    BatchAuditResponse,
    BatchSummary,
    PendingReviewItem,
    ReviewDecisionRequest,
    RiskState,
)
from audit.storage import load_logs


def _enqueue_if_pending(result: RiskState) -> None:
    """命中复核但无真人决策的内容入异步待审队列。"""
    if result.get("review_status") == "pending":
        review_store.enqueue(result)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from audit.graph import get_app
    get_app()  # warm-up: compile graph once at startup
    yield


app = FastAPI(
    title="内容审核 Agent API",
    description="基于 LangGraph 的 UGC 内容风险审核服务",
    version="1.0.0",
    lifespan=lifespan,
)


def _state_to_response(result: RiskState) -> AuditResponse:
    import json
    try:
        report = json.loads(result["final_report"]) if result["final_report"] else {}
    except Exception:
        report = {}
    return AuditResponse(
        trace_id=result["trace_id"],
        risk_type=result["risk_type"],
        risk_level=result["risk_level"],
        confidence=result["confidence"],
        policy_tags=result["policy_tags"],
        evidence=result["evidence"],
        analysis=result["analysis_result"],
        recommended_action=result["recommended_action"],
        human_decision=result["human_decision"],
        final_action=report.get("final_action") or result["recommended_action"],
        llm_called=report.get("llm_called", result.get("llm_called", False)),
        judge_reason=report.get("judge_reason", result.get("judge_reason", "")),
        model_error=result.get("model_error", ""),
    )


@app.get("/health", summary="健康检查")
async def health():
    return {"status": "ok"}


@app.post("/audit", response_model=AuditResponse, summary="单条审核")
async def audit_single(req: AuditRequest):
    try:
        # API 无法做交互式人工复核：客户端未显式给决策时默认 "pending"，
        # 命中复核队列的内容会被诚实标注为 final_action="pending_review"（待人工），
        # 而不是伪造成"已 skip"。客户端也可显式传 approve/reject/skip 覆盖。
        result = await run_in_threadpool(audit_one, req.comment, req.auto_decision or "pending")
        _enqueue_if_pending(result)
        return _state_to_response(result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/audit/batch", response_model=BatchAuditResponse, summary="批量审核")
async def audit_batch(req: BatchAuditRequest):
    try:
        results = await run_in_threadpool(batch_audit, req.comments, req.auto_decision or "pending")
        for r in results:
            _enqueue_if_pending(r)
        responses = [_state_to_response(r) for r in results]
        summary = BatchSummary(
            total=len(responses),
            risk_type_dist=dict(Counter(r.risk_type for r in responses)),
            risk_level_dist=dict(Counter(r.risk_level for r in responses)),
            manual_review_count=sum(1 for r in responses if r.human_decision in {"approve", "reject"}),
        )
        return BatchAuditResponse(results=responses, summary=summary)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/logs", summary="查询审核日志")
async def get_logs(limit: int = Query(50, ge=1, le=500, description="返回条数")):
    return await run_in_threadpool(load_logs, limit)


@app.get("/reviews/pending", response_model=list[PendingReviewItem], summary="拉取待人工复核队列")
async def list_pending_reviews(limit: int = Query(50, ge=1, le=500, description="返回条数")):
    return await run_in_threadpool(review_store.list_pending, limit)


@app.post("/reviews/{trace_id}", response_model=PendingReviewItem, summary="审核员结案")
async def resolve_review(trace_id: str, req: ReviewDecisionRequest):
    record = await run_in_threadpool(review_store.resolve, trace_id, req.decision, req.reviewer)
    if record is None:
        raise HTTPException(status_code=404, detail=f"待审记录不存在: {trace_id}")
    return record
