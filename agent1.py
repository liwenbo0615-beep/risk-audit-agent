"""Backward-compatible entry point — imports everything from the audit package.

Run directly:
    python agent1.py

Or import from other scripts:
    from agent1 import audit_one, batch_audit
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

# Re-export public API for scripts that import agent1 directly
from audit import audit_one, batch_audit  # noqa: F401
from audit.graph import get_app  # noqa: F401
from audit.models import RiskState, new_state  # noqa: F401

app = get_app()

if __name__ == "__main__":
    auto = os.getenv("AUTO_REVIEW_DECISION", "").strip().lower() or "skip"
    if not os.getenv("AUTO_REVIEW_DECISION"):
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
        "妈妈说我才12岁，鸡巴就已经超过20厘米了，以后肯定能继续长，姐姐弟弟鸡巴操你爽不？"
    ]
    batch_audit(test_comments, auto)
