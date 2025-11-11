"""
Configuration helpers for loading per-source settings defined in YAML.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "sources.yaml"


class ConfigError(RuntimeError):
    """Raised when the configuration file cannot be loaded."""


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load the crawler configuration YAML. Falls back to the default file shipped
    with the repository unless a custom path is provided or the environment
    variable `AI_CORPUS_CONFIG` is set.
    """
    candidate: Optional[Path] = path
    if candidate is None:
        env_override = os.environ.get("AI_CORPUS_CONFIG")
        if env_override:
            candidate = Path(env_override).expanduser()
    if candidate is None:
        candidate = DEFAULT_CONFIG_PATH
    if not candidate.exists():
        raise ConfigError(f"Config file not found at {candidate}")
    with candidate.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data

