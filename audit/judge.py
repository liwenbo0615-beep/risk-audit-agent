"""LLMCallJudge — decides whether to escalate from offline classifier to LLM API.

Single responsibility: given an offline result + raw text, return (needs_llm, reason_code).
This module never touches LangGraph state, the LLM, or any I/O.

Decision table
──────────────────────────────────────────────────────────────────────
Condition                                          LLM needed?
──────────────────────────────────────────────────────────────────────
Compound rule matched (未成年涉色 tag)             No  — authoritative hit
risk_type == "unknown"                             Yes — offline couldn't classify
confidence < min_confidence (default 0.90)         Yes — uncertain result
risk_type in {political, unknown}                  Yes — context-dependent (支持 vs 推翻)
risk_type == "minor" without compound tag          Yes — keyword alone misleads
                                                         ("未成年人不应该沉迷学习" is safe)
risk_type == "safe" AND len(text) > safe_max_len   Yes — may have missed subtle signals
everything else                                    No  — offline result is sufficient
──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

# Risk types where a keyword match is unreliable without context
_CONTEXT_DEPENDENT: frozenset[str] = frozenset({"political", "unknown"})

# Risk types where simple keyword match often needs semantic verification
_VERIFY_TYPES: frozenset[str] = frozenset({"minor"})

# Tags produced by compound rules — classifier is authoritative when these are present
_HIGH_CONFIDENCE_TAGS: frozenset[str] = frozenset({"未成年涉色"})


@dataclass(frozen=True)
class JudgeConfig:
    enabled: bool = True
    min_confidence: float = 0.90
    safe_max_len: int = 150

    @classmethod
    def from_env(cls) -> "JudgeConfig":
        return cls(
            enabled=os.getenv("JUDGE_ENABLED", "1").strip().lower() not in {"0", "false", "no"},
            min_confidence=float(os.getenv("JUDGE_MIN_CONFIDENCE", "0.90")),
            safe_max_len=int(os.getenv("JUDGE_SAFE_MAX_LEN", "150")),
        )


class LLMCallJudge:
    def __init__(self, config: JudgeConfig) -> None:
        self._cfg = config

    def should_call(self, text: str, offline_result: dict[str, Any]) -> tuple[bool, str]:
        """Return (needs_llm, reason_code)."""
        if not self._cfg.enabled:
            return False, "judge_disabled"

        risk_type: str = offline_result.get("risk_type", "unknown")
        confidence: float = float(offline_result.get("confidence", 0.0))
        policy_tags: list[str] = offline_result.get("policy_tags", [])

        # Compound rule matched → high-confidence, skip LLM
        if any(tag in _HIGH_CONFIDENCE_TAGS for tag in policy_tags):
            return False, "high_confidence_compound_rule"

        # Offline couldn't classify
        if risk_type == "unknown":
            return True, "offline_unknown"

        # Low confidence → offline result unreliable
        if confidence < self._cfg.min_confidence:
            return True, "low_confidence"

        # Context-dependent type: identical keywords can mean opposite things
        if risk_type in _CONTEXT_DEPENDENT:
            return True, f"context_dependent:{risk_type}"

        # Simple keyword match for minor is insufficient — needs semantic understanding
        if risk_type in _VERIFY_TYPES:
            return True, "minor_needs_semantic_verification"

        # Safe but long text — offline may have missed subtle multi-token signals
        if risk_type == "safe" and len(text) > self._cfg.safe_max_len:
            return True, "safe_long_text"

        return False, "offline_sufficient"


_judge: "LLMCallJudge | None" = None


def get_judge() -> LLMCallJudge:
    global _judge
    if _judge is None:
        _judge = LLMCallJudge(JudgeConfig.from_env())
    return _judge


def reset_judge() -> None:
    global _judge
    _judge = None
