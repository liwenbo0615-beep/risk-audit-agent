import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    deepseek_api_key: str
    deepseek_model: str
    deepseek_base_url: str
    log_path: str
    auto_review_decision: str
    offline_demo_mode: bool
    llm_timeout: float
    llm_max_retries: int

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            log_path=os.getenv("AUDIT_LOG_PATH", "audit_log.jsonl"),
            auto_review_decision=os.getenv("AUTO_REVIEW_DECISION", "").strip().lower(),
            offline_demo_mode=os.getenv("OFFLINE_DEMO_MODE", "1").strip().lower() not in {"0", "false", "no"},
            llm_timeout=float(os.getenv("LLM_TIMEOUT", "8")),
            llm_max_retries=int(os.getenv("LLM_MAX_RETRIES", "0")),
        )


_config: "Config | None" = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def reset_config() -> None:
    global _config
    _config = None
