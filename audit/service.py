"""审核服务入口（重依赖：会拉起 LangGraph 编排）。

把 audit_one / batch_audit 放这里，让 audit/__init__.py 保持轻量——
这样 `from audit.classifier import identify` 等纯规则模块可独立导入，不强依赖 langgraph。
"""

from collections import Counter

from .config import get_config
from .graph import get_app
from .models import RiskState, new_state
from .storage import save_log


def audit_one(comment: str, auto_decision: str = "") -> RiskState:
    result = get_app().invoke(new_state(comment, auto_decision))
    save_log(result)
    return result


def batch_audit(comments: list[str], auto_decision: str = "") -> list[RiskState]:
    results: list[RiskState] = []
    for i, comment in enumerate(comments, start=1):
        print(f"\n{'=' * 60}")
        print(f"处理第 {i}/{len(comments)} 条: {comment}")
        result = audit_one(comment, auto_decision)
        results.append(result)
        print(result["final_report"])

    risk_counter = Counter(r["risk_type"] for r in results)
    level_counter = Counter(r["risk_level"] for r in results)
    review_queue_count = sum(
        1 for r in results
        if r["risk_type"] in {"political", "minor", "adult", "illegal", "unknown"}
        or r["risk_level"] in {"high", "medium", "unknown"}
        or r["recommended_action"] == "manual_review"
    )
    manual_decision_count = sum(1 for r in results if r.get("human_decision") in {"approve", "reject"})
    auto_skip_count = sum(1 for r in results if r.get("human_decision") == "skip")
    pending_count = sum(1 for r in results if r.get("review_status") == "pending")

    print(f"\n{'=' * 60}")
    print(f"批量审核完成，共处理 {len(results)} 条")
    print(f"风险类型分布: {dict(risk_counter)}")
    print(f"风险等级分布: {dict(level_counter)}")
    print(f"进入复核队列数量: {review_queue_count}")
    print(f"人工决策数量: {manual_decision_count}")
    print(f"自动 skip 数量: {auto_skip_count}")
    print(f"待人工(pending)数量: {pending_count}")
    print(f"日志已保存至 {get_config().log_path}")
    return results
