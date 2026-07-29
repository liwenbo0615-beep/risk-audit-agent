"""LLMCallJudge — decides whether to escalate from offline classifier to LLM API.

Single responsibility: given an offline result + raw text, return (needs_llm, reason_code).
This module never touches LangGraph state, the LLM, or any I/O.

LLM-first policy
──────────────────────────────────────────────────────────────────────
The model is the primary judge. The offline classifier is only trusted to
make the final call in two clear-cut situations; everything else is sent to
the LLM for semantic verification (this is what catches coded / 黑话 content
that keyword rules miss).

Condition                                          LLM needed?
──────────────────────────────────────────────────────────────────────
Judge disabled                                     No  — master switch off
Compound rule matched (未成年涉色 / 乱伦涉色 tag)   No  — authoritative violation hit
risk_type == "safe" AND text in ABSOLUTE_SAFE_TEXTS No — 固定无害短语白名单
everything else                                    Yes — LLM-first verification
──────────────────────────────────────────────────────────────────────

为什么用"固定白名单"而不是"长度阈值"：短不等于无害（如"约茶吗"是涉黄黑话）。
只有明确无害的固定寒暄短语才跳过 LLM，其余一律交给模型做语义判断，避免短黑话漏检。

Note: `min_confidence` is still parsed from the environment for backward
compatibility but is no longer used as a gate — under the LLM-first policy any
non-trivial result is verified regardless of the offline confidence score.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

# Tags produced by compound rules — classifier is authoritative when these are present
_HIGH_CONFIDENCE_TAGS: frozenset[str] = frozenset({"未成年涉色", "乱伦涉色"})

# 明确无害的固定寒暄短语白名单——只有命中这里的 safe 文本才跳过 LLM。
# 不用长度阈值：短文本也可能是黑话（"约茶吗"），必须交给模型判断。
ABSOLUTE_SAFE_TEXTS: frozenset[str] = frozenset({
    "你好", "谢谢", "哈哈", "早上好", "晚安", "今天天气真好", "天气好",
})


@dataclass(frozen=True)
class JudgeConfig:
    enabled: bool = True
    min_confidence: float = 0.90  # parsed for compat; unused under LLM-first policy

    @classmethod
    def from_env(cls) -> "JudgeConfig":
        return cls(
            enabled=os.getenv("JUDGE_ENABLED", "1").strip().lower() not in {"0", "false", "no"},
            min_confidence=float(os.getenv("JUDGE_MIN_CONFIDENCE", "0.90")),
        )


class LLMCallJudge:
    def __init__(self, config: JudgeConfig) -> None:
        self._cfg = config

    def should_call(self, text: str, offline_result: dict[str, Any]) -> tuple[bool, str]:
        """Return (needs_llm, reason_code) under the LLM-first policy."""
        if not self._cfg.enabled:
            return False, "judge_disabled"

        risk_type: str = offline_result.get("risk_type", "unknown")
        policy_tags: list[str] = offline_result.get("policy_tags", [])

        # Compound rule matched → authoritative violation, LLM adds nothing.
        if any(tag in _HIGH_CONFIDENCE_TAGS for tag in policy_tags):
            return False, "high_confidence_compound_rule"

        # Absolutely safe: 仅命中固定无害短语白名单的 safe 文本跳过 LLM。
        if risk_type == "safe" and text.strip() in ABSOLUTE_SAFE_TEXTS:
            return False, "absolutely_safe"

        # LLM-first: verify everything else with the model (catches coded content).
        return True, "llm_first"


_judge: "LLMCallJudge | None" = None


def get_judge() -> LLMCallJudge:
    global _judge
    if _judge is None:
        _judge = LLMCallJudge(JudgeConfig.from_env())
    return _judge


def reset_judge() -> None:
    global _judge
    _judge = None
