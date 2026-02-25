"""Runtime settings for the FastAPI rollout backend."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)
DEFAULT_ANTHROPIC_KEY_PATH = Path.home() / ".anthropic-usc-key.txt"


class Settings(BaseSettings):
    model_name: str = Field(default="gpt-5-mini", alias="POLICY_AGENT_MODEL")
    cache_path: str = Field(default="data/cache/policy_cache.sqlite", alias="POLICY_AGENT_CACHE_PATH")
    use_cached: bool = Field(default=True, alias="POLICY_AGENT_USE_CACHED")
    temperature: float = Field(default=0.7, alias="POLICY_AGENT_TEMPERATURE")
    langsmith_project: Optional[str] = Field(default=None, alias="LANGSMITH_PROJECT")
    langsmith_api_key: Optional[str] = Field(default=None, alias="LANGSMITH_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    langchain_tracing_v2: bool = Field(default=False, alias="LANGCHAIN_TRACING_V2")
    langchain_endpoint: Optional[str] = Field(default=None, alias="LANGCHAIN_ENDPOINT")
    langchain_api_key: Optional[str] = Field(default=None, alias="LANGCHAIN_API_KEY")
    claude_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    claude_model_name: str = Field(default="claude-sonnet-4-6", alias="CLAUDE_CITATIONS_MODEL")
    use_claude_submit_tool: bool = Field(default=True, alias="USE_CLAUDE_SUBMIT_TOOL")

    class Config:
        env_file = os.environ.get(
            "POLICY_BACKEND_ENV_FILE", str(Path(__file__).resolve().parent / ".env")
        )
        env_file_encoding = "utf-8"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    if not settings.claude_api_key:
        try:
            if DEFAULT_ANTHROPIC_KEY_PATH.exists():
                key = DEFAULT_ANTHROPIC_KEY_PATH.read_text().strip()
                if key:
                    settings.claude_api_key = key
        except OSError:
            logger.debug("Unable to read default Anthropic key path", exc_info=True)
    return settings
