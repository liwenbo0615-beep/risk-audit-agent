"""异步人工复核待审队列（Plan B）。

命中复核但无真人决策的内容在此持久化为 pending，审核员事后异步结案。
存储用单个 JSON 文件 {trace_id: record}，读-改-写。

⚠️ 并发限制：单进程 / 低并发场景适用（作品集 Demo）。高并发或多副本部署需换
   带行锁/事务的数据库（如 Postgres），否则读-改-写存在竞态。
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import get_config
from .models import RiskState


def _path() -> Path:
    return Path(get_config().review_store_path)


def _load() -> dict[str, dict[str, Any]]:
    p = _path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save(data: dict[str, dict[str, Any]]) -> None:
    _path().write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def enqueue(state: RiskState) -> dict[str, Any]:
    """把一条命中复核的内容写入待审队列（status=pending）。"""
    data = _load()
    trace_id = state["trace_id"]
    record = {
        "trace_id": trace_id,
        "comment": state["user_input"],
        "risk_type": state["risk_type"],
        "risk_level": state["risk_level"],
        "confidence": state["confidence"],
        "policy_tags": state["policy_tags"],
        "recommended_action": state["recommended_action"],
        "analysis": state["analysis_result"],
        "status": "pending",
        "human_decision": "",
        "final_action": "pending_review",
        "reviewer": "",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "resolved_at": None,
    }
    data[trace_id] = record
    _save(data)
    return record


def list_pending(limit: int = 50) -> list[dict[str, Any]]:
    """返回待审（pending）记录，按创建时间倒序。"""
    items = [r for r in _load().values() if r.get("status") == "pending"]
    items.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return items[:limit]


def get(trace_id: str) -> dict[str, Any] | None:
    return _load().get(trace_id)


def resolve(trace_id: str, decision: str, reviewer: str = "") -> dict[str, Any] | None:
    """审核员结案：把待审记录置为 decided 并写入最终决策。

    返回更新后的记录；trace_id 不存在时返回 None（由上层转 404）。
    重复结案为幂等更新（再次写入最新决策）。
    """
    data = _load()
    record = data.get(trace_id)
    if record is None:
        return None
    record["status"] = "decided"
    record["human_decision"] = decision
    record["final_action"] = decision
    record["reviewer"] = reviewer
    record["resolved_at"] = datetime.now().isoformat(timespec="seconds")
    data[trace_id] = record
    _save(data)
    return record


def reset_store() -> None:
    """清空待审队列（测试隔离用）。"""
    p = _path()
    if p.exists():
        p.unlink()
