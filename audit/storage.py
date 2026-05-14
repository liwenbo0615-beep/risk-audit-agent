import json
from datetime import datetime
from pathlib import Path

from .config import get_config
from .models import RiskState


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
    path = Path(get_config().log_path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(log, ensure_ascii=False) + "\n")


def load_logs(limit: int = 50) -> list[dict]:
    path = Path(get_config().log_path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    tail = lines[-limit:] if len(lines) > limit else lines
    result = []
    for line in reversed(tail):
        line = line.strip()
        if line:
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return result
