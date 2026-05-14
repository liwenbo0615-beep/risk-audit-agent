"""LLM mode real-API test — shows judge routing in practice.

    python run_llm_test.py
"""

import json
import os
import time
from pathlib import Path

os.environ["OFFLINE_DEMO_MODE"] = "0"
os.environ["AUTO_REVIEW_DECISION"] = "skip"
os.environ["AUDIT_LOG_PATH"] = "audit_log_llm_test.jsonl"

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from audit import audit_one
from audit.config import reset_config
from audit.judge import reset_judge

reset_config()
reset_judge()

TEST_CASES = [
    ("今天天气真好，出去玩吧",                                  "offline / safe"),
    ("加我领优惠券，稳赚不赔",                                  "offline / spam"),
    ("教你怎么逃税不被发现",                                    "offline / illegal"),
    ("这个视频里有暴力血腥画面",                                "offline / violence"),
    ("妈妈说我才12岁，鸡巴就已经超过20厘米了，鸡巴操你爽不？", "offline / minor+compound"),
    ("小孩子不应该看这种内容",                                  "LLM   / minor verify"),
    ("未成年人不应该沉迷学习",                                  "LLM   / minor verify"),
    ("这个政策真的太好了，支持！",                              "LLM   / political"),
    ("反对政府，推翻专制统治",                                  "LLM   / political"),
]

SEP  = "=" * 72
SEP2 = "-" * 72

print()
print(SEP)
print("  LLM mode routing test  (OFFLINE_DEMO_MODE=0)")
print(SEP)
print()

rows = []
total_llm = 0
total_offline = 0

for i, (comment, expected) in enumerate(TEST_CASES, 1):
    print(f"[{i}/{len(TEST_CASES)}] {comment}")
    t0 = time.perf_counter()
    result = audit_one(comment, auto_decision="skip")
    elapsed = time.perf_counter() - t0

    llm_called   = result.get("llm_called", False)
    judge_reason = result.get("judge_reason", "")
    risk_type    = result["risk_type"]
    risk_level   = result["risk_level"]
    confidence   = result["confidence"]
    final_action = json.loads(result["final_report"]).get("final_action", "?")
    model_error  = result.get("model_error", "")

    route_tag = "[LLM ]" if llm_called else "[FAST]"
    if llm_called:
        total_llm += 1
    else:
        total_offline += 1

    print(f"  route    : {route_tag}  reason={judge_reason}")
    print(f"  expected : {expected}")
    print(f"  result   : type={risk_type}  level={risk_level}  conf={confidence:.2f}  action={final_action}  {elapsed:.2f}s")
    if model_error:
        print(f"  ERROR    : {model_error}")
    print()

    rows.append({
        "no": i, "input": comment, "expected": expected,
        "llm_called": llm_called, "judge_reason": judge_reason,
        "risk_type": risk_type, "risk_level": risk_level,
        "confidence": confidence, "final_action": final_action,
        "elapsed_s": round(elapsed, 2),
    })

print(SEP)
print(f"  Routing summary")
print(f"  offline (FAST) : {total_offline} items  -- no API cost")
print(f"  LLM    (SLOW)  : {total_llm} items  -- semantic understanding needed")
print(f"  LLM call rate  : {total_llm / len(TEST_CASES) * 100:.0f}%  ({total_llm}/{len(TEST_CASES)})")
print(SEP)
print()
print("Detailed table:")
print(f"{'No':>3}  {'LLM':>5}  {'type':>10}  {'lvl':>7}  {'conf':>5}  {'action':>13}  {'t(s)':>5}  reason")
print(SEP2)
for r in rows:
    llm_flag = "YES" if r["llm_called"] else "no"
    print(
        f"{r['no']:>3}  {llm_flag:>5}  {r['risk_type']:>10}  {r['risk_level']:>7}"
        f"  {r['confidence']:>5.2f}  {r['final_action']:>13}  {r['elapsed_s']:>5.2f}  {r['judge_reason']}"
    )
print(SEP2)
