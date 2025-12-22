# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Typed data models for the policy deep research OpenEnv environment."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from openenv_core.env_server.types import Action, Observation, State

ActionType = Literal[
    "SEARCH SEMANTIC SCHOLAR",
    "FETCH PAPER",
    "ADD TO BIB",
    "WRITE NOTE",
    "SUBMIT",
    # Legacy aliases
    "SEARCH",
    "FETCH_PAPER",
    "ADD_TO_BIB",
    "WRITE_NOTE",
]


@dataclass(kw_only=True)
class ResearchAction(Action):
    """Structured action emitted by the policy research agent."""

    type: ActionType
    query: Optional[str] = None
    paper_id: Optional[str] = None
    top_k: int = 10
    filters: Dict[str, Any] = field(default_factory=dict)
    content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class ResearchObservation(Observation):
    """Observation returned to the agent each step."""

    task_id: str = ""
    question: str = ""
    instructions: str = ""
    last_tool_result: Dict[str, Any] = field(default_factory=dict)
    bib: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    remaining_steps: int = 0


@dataclass
class ResearchState(State):
    """Server-side environment state exposed via HTTP."""

    episode_id: Optional[str] = None
    step_count: int = 0
    task_id: str = ""
    done: bool = False
    use_cached: bool = False
    cache_path: str = ""
