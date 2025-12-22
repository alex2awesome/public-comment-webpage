# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Implementation of the policy deep research OpenEnv environment."""

from __future__ import annotations

import os
import random
import uuid
from typing import Any, Dict, Iterable, List, Optional

from openenv_core.env_server.interfaces import Environment

try:  # pragma: no cover - handle both package and flat module layouts
    from ..models import ResearchAction, ResearchObservation, ResearchState
except ImportError:  # pragma: no cover
    from models import ResearchAction, ResearchObservation, ResearchState
from .cache_db import CacheDB
from .reward import compute_reward
from .semanticscholar_api import SemanticScholarClient
from .tasks import load_tasks


class PolicyDeepResearchEnvironment(Environment):
    """Environment where an agent researches policy questions via Semantic Scholar."""

    def __init__(self) -> None:
        self._tasks = load_tasks()
        self._rng = random.Random(int(os.getenv("OPENENV_TASK_SEED", "0")))
        self._task_order = list(range(len(self._tasks)))
        if self._task_order:
            self._rng.shuffle(self._task_order)
        self._task_cursor = 0

        self._use_cached = os.getenv("OPENENV_USE_CACHED", "0") == "1"
        self._cache_path = os.getenv("OPENENV_CACHE_PATH", "/app/env/data/cache.sqlite")
        self._max_steps = int(os.getenv("MAX_STEPS", "12"))

        self._db = CacheDB(self._cache_path)
        self._s2 = SemanticScholarClient()

        self._state = ResearchState(
            episode_id=None,
            step_count=0,
            task_id="",
            done=False,
            use_cached=self._use_cached,
            cache_path=self._cache_path,
        )
        self._bib: List[Dict[str, Any]] = []
        self._notes: List[str] = []
        self._last_tool_result: Dict[str, Any] = {}
        self._current_task: Optional[Dict[str, Any]] = None

    def reset(self) -> ResearchObservation:
        run_id = str(uuid.uuid4())
        task_idx = self._select_task_index()
        task = self._tasks[task_idx]

        self._state = ResearchState(
            episode_id=run_id,
            step_count=0,
            task_id=task.get("task_id", str(task_idx)),
            done=False,
            use_cached=self._use_cached,
            cache_path=self._cache_path,
        )
        self._current_task = task
        self._bib = []
        self._notes = []
        self._last_tool_result = {}
        self._db.record_run(run_id, self._state.task_id, self._use_cached)

        return ResearchObservation(
            task_id=self._state.task_id,
            question=task.get("question", ""),
            instructions=self._instructions(),
            last_tool_result={},
            bib=[],
            notes=[],
            remaining_steps=self._max_steps,
            done=False,
            reward=0.0,
        )

    def step(self, action: ResearchAction) -> ResearchObservation:  # type: ignore[override]
        if self._state.done:
            return self._obs(err="Episode already completed.", done=True, reward=0.0)

        self._state.step_count += 1
        try:
            if action.type in {"SEARCH", "SEARCH SEMANTIC SCHOLAR"}: # default for "SEARCH" is "SEARCH SEMANTIC SCHOLAR"
                if not action.query:
                    raise ValueError("SEARCH action requires 'query'.")
                self._last_tool_result = self._handle_search_semantic_scholar(action)
            elif action.type in {"FETCH SEMANTIC SCHOLAR PAPER", "FETCH PAPER"}: # default for "FETCH PAPER" is "FETCH SEMANTIC SCHOLAR PAPER"
                if not action.paper_id:
                    raise ValueError("FETCH SEMANTIC SCHOLAR PAPER requires 'paper_id'.")
                self._last_tool_result = self._handle_fetch_semantic_scholar(action)
            elif action.type in {"ADD_TO_BIB", "ADD TO BIB"}:
                if not action.paper_id:
                    raise ValueError("ADD TO BIB requires 'paper_id'.")
                self._last_tool_result = self._handle_add_to_bib(action)
            elif action.type in {"WRITE_NOTE", "WRITE NOTE"}:
                content = action.content or ""
                self._notes.append(content)
                self._last_tool_result = {"note_saved": len(self._notes) - 1}
            elif action.type == "SUBMIT":
                memo = action.content or ""
                reward = compute_reward(
                    question=self._current_question(),
                    memo=memo,
                    bib=self._bib,
                    step_count=self._state.step_count,
                )
                self._state.done = True
                self._last_tool_result = {
                    "submitted": True,
                    "reward_breakdown": reward.breakdown,
                }
                return self._obs(done=True, reward=float(reward.total), metadata={"final_memo": memo})
            else:
                raise ValueError(f"Unknown action.type={action.type}")
        except Exception as exc: 
            print(f"Error in step: {exc}")
            self._last_tool_result = {"error": str(exc)}

        if self._state.step_count >= self._max_steps:
            self._state.done = True
            return self._obs(done=True, reward=-1.0)
        return self._obs(done=False, reward=0.0)

    @property
    def state(self) -> ResearchState:
        return self._state

    # ----------------------------- Helpers --------------------------------- #
    def _select_task_index(self) -> int:
        override = os.getenv("OPENENV_TASK_INDEX")
        if override is not None:
            return int(override) % len(self._tasks)
        idx = self._task_order[self._task_cursor % len(self._task_order)]
        self._task_cursor += 1
        return idx

    def _current_question(self) -> str:
        if self._current_task:
            return self._current_task.get("question", "")
        return ""

    def _handle_search_semantic_scholar(self, action: ResearchAction) -> Dict[str, Any]:
        fields = "paperId,title,year,venue,url,abstract,citationCount"
        year_filter = None
        if action.filters:
            year_filter = action.filters.get("year")
        fetch_limit = max(20, action.top_k or 1)
        result = self._db.cached_or_fetch_semantic_scholar_search(
            use_cached=self._use_cached,
            query=action.query or "",
            params={"year": year_filter, "fields": fields, "top_k": action.top_k},
            fetch_fn=lambda: self._take_n(
                self._s2.bulk_search(action.query or "", fields=fields, year=year_filter, limit=fetch_limit),
                fetch_limit,
            ),
            top_k=max(1, action.top_k),
        )
        return {"type": "search_results", **result}

    def _handle_fetch_semantic_scholar(self, action: ResearchAction) -> Dict[str, Any]:
        fields = "paperId,title,year,venue,url,abstract,citationCount,authors,openAccessPdf"
        payload = self._db.cached_or_fetch_semantic_scholar_paper(
            use_cached=self._use_cached,
            paper_id=action.paper_id or "",
            fields=fields,
            fetch_fn=lambda: self._s2.paper_details(action.paper_id or "", fields=fields),
        )
        return {"type": "paper_details", **payload}

    def _handle_add_to_bib(self, action: ResearchAction) -> Dict[str, Any]:
        paper = self._db.get_paper(action.paper_id or "")
        if not paper:
            raise ValueError("Paper must be fetched before adding to bibliography.")
        for entry in self._bib:
            if entry.get("paperId") == action.paper_id:
                entry["reason"] = action.metadata.get("reason", entry.get("reason", ""))
                return {"already_present": True, "paperId": action.paper_id}

        entry = {
            "paperId": paper.get("paperId"),
            "title": paper.get("title"),
            "year": paper.get("year"),
            "venue": paper.get("venue"),
            "url": paper.get("url"),
            "reason": action.metadata.get("reason", ""),
        }
        self._bib.append(entry)
        return {"added": True, "paperId": action.paper_id}

    def _obs(
        self,
        *,
        err: str | None = None,
        done: bool,
        reward: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ResearchObservation:
        remaining = max(self._max_steps - self._state.step_count, 0)
        last = {"error": err} if err else self._last_tool_result
        return ResearchObservation(
            task_id=self._state.task_id,
            question=self._current_question(),
            instructions=self._instructions(),
            last_tool_result=last,
            bib=list(self._bib),
            notes=list(self._notes),
            remaining_steps=remaining,
            done=done,
            reward=reward,
            metadata=metadata or {},
        )

    def _instructions(self) -> str:
        return (
            "Valid actions:\n"
            "1) SEARCH SEMANTIC SCHOLAR - query Semantic Scholar for academic papers. Use SHORT, KEYWORD-FOCUSED queries (3–6 terms). Avoid punctuation like quotes or OR clauses. Be sure to keep your queries short!!! Requires {\"type\":\"SEARCH SEMANTIC SCHOLAR\",\"query\":\"...\"}.\n"
            "2) FETCH SEMANTIC SCHOLAR PAPER - fetch a specific semantic scholar paper from the search results. Requires {\"type\":\"FETCH SEMANTIC SCHOLAR PAPER\",\"paper_id\":\"...\"}.\n"
            "3) ADD TO BIB - add a paper to your bibliography with a justification. Requires {\"type\":\"ADD TO BIB\",\"paper_id\":\"...\",\"metadata\":{\"reason\":\"...\"}}.\n"
            "4) WRITE NOTE - write a free-form note. Requires {\"type\":\"WRITE NOTE\",\"content\":\"...\"}.\n"
            "5) SUBMIT - deliver the final memo. Requires {\"type\":\"SUBMIT\",\"content\":\"...\"}.\n"
        )

    @staticmethod
    def _take_n(iterable: Iterable[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for item in iterable:
            items.append(item)
            if len(items) >= limit:
                break
        return items
