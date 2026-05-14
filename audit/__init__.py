from collections import Counter

from .config import get_config
from .graph import get_app
from .models import RiskState, new_state
from .storage import save_log


def audit_one(comment: str, auto_decision: str = "") -> RiskState:
    result = get_app().invoke(new_state(comment, auto_decision))
    save_log(result)
    return result


def batch_audit(comments: list[str], auto_decision: str = "skip") -> list[RiskState]:
    results: list[RiskState] = []
    for i, comment in enumerate(comments, start=1):
        print(f"\n{'=' * 60}")
        print(f"处理第 {i}/{len(comments)} 条: {comment}")
        result = audit_one(comment, auto_decision)
        results.append(result)
        print(result["final_report"])

    risk_counter = Counter(r["risk_type"] for r in results)
    level_counter = Counter(r["risk_level"] for r in results)
    review_count = sum(1 for r in results if r["human_decision"])

    print(f"\n{'=' * 60}")
    print(f"批量审核完成，共处理 {len(results)} 条")
    print(f"风险类型分布: {dict(risk_counter)}")
    print(f"风险等级分布: {dict(level_counter)}")
    print(f"人工复核数量: {review_count}")
    print(f"日志已保存至 {get_config().log_path}")
    return results
