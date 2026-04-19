"""Configuration management — loads settings.yaml + environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "settings.yaml"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"


def _resolve_env_vars(obj: Any) -> Any:
    """Recursively replace ${ENV_VAR} placeholders with actual env values."""
    if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        env_key = obj[2:-1]
        return os.environ.get(env_key, "")
    if isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_vars(v) for v in obj]
    return obj


class APIKeysConfig(BaseSettings):
    """API credentials — `groq_api_key` and `fred_api_key` come from .env;
    `sec_user_agent` still comes from settings.yaml."""

    groq_api_key: str = ""
    fred_api_key: str = ""
    sec_user_agent: str = "StockAnalysis stock@analysis.com"

    model_config = SettingsConfigDict(
        env_file=str(DEFAULT_ENV_PATH),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


class ScreeningConfig(BaseModel):
    ema_fast: int = 9
    ema_slow: int = 21
    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    volume_surge_multiplier: float = 1.5
    min_market_cap: float = 10_000_000_000
    max_candidates: int = 30


class DCFConfig(BaseModel):
    projection_years: int = 5
    terminal_growth_rate: float = 0.025
    risk_free_rate_source: str = "fred"
    risk_free_rate_manual: float = 0.043
    market_risk_premium: float = 0.055
    tax_rate: float = 0.21
    wacc_sensitivity_range: list[float] = Field(
        default_factory=lambda: [0.06, 0.08, 0.10, 0.12, 0.14]
    )
    terminal_growth_sensitivity_range: list[float] = Field(
        default_factory=lambda: [0.015, 0.020, 0.025, 0.030, 0.035]
    )


class LLMConfig(BaseModel):
    provider: str = "groq"
    model_main: str = "llama-3.3-70b-versatile"
    model_qc: str = "llama-3.1-8b-instant"
    max_retries: int = 3
    temperature: float = 0.3
    max_tokens_thesis: int = 2000
    max_tokens_risk: int = 1500
    max_tokens_industry: int = 1500
    max_tokens_annual: int = 2000
    max_tokens_meetings: int = 1500
    max_tokens_qc: int = 500


class ReportConfig(BaseModel):
    page_size: str = "A4"
    output_format: str = "pdf"
    output_dir: str = "output"
    include_appendix: bool = True
    include_esg: bool = True


class DatabaseConfig(BaseModel):
    path: str = "data/stock_analysis.db"
    cache_ttl_hours: int = 24


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "json"


class Settings(BaseSettings):
    api_keys: APIKeysConfig = Field(default_factory=APIKeysConfig)
    screening: ScreeningConfig = Field(default_factory=ScreeningConfig)
    dcf: DCFConfig = Field(default_factory=DCFConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_settings(config_path: Path | None = None) -> Settings:
    """Load settings from YAML; API keys (groq, fred) come from .env via APIKeysConfig."""
    path = config_path or DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = {}
    if path.exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        raw = _resolve_env_vars(raw)

    yaml_api_keys = raw.pop("api_keys", {}) if isinstance(raw, dict) else {}
    # yaml only contributes sec_user_agent; groq/fred are always read from env
    api_keys_overrides = {
        k: v for k, v in yaml_api_keys.items()
        if k == "sec_user_agent" and v
    }
    api_keys = APIKeysConfig(**api_keys_overrides)

    return Settings(api_keys=api_keys, **raw)


# Singleton instance
settings = load_settings()
