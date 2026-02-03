"""LangGraph-based policy research agent helpers."""

from __future__ import annotations

from typing import Any

__all__ = ["run_one_rollout"]


def __getattr__(name: str) -> Any:
    if name == "run_one_rollout":
        from .rollout import run_one_rollout as _run_one_rollout

        return _run_one_rollout
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
