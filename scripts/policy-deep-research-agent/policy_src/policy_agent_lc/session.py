"""Research session state shared across LangGraph nodes and tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ResearchSession:
    """Lightweight container describing the ongoing research run."""

    task_id: str
    question: str
    instructions: str
    use_cached: bool
    max_steps: int
    cache_path: str
    bibliography_enabled: bool = True

    step_count: int = 0
    bib: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    last_tool_result: Dict[str, Any] = field(default_factory=dict)

    final_memo: Optional[str] = None
    final_reward: float = 0.0
    reward_breakdown: Dict[str, float] = field(default_factory=dict)

    messages: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    memo_blocks: List[Dict[str, Any]] = field(default_factory=list)
    source_documents: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def done(self) -> bool:
        """Return True when the memo is submitted or steps exhausted."""
        return self.final_memo is not None or self.step_count >= self.max_steps
