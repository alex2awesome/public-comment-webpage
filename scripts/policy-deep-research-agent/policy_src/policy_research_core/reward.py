"""Reward shaping utilities for the deep research policy environment."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class RewardResult:
    total: float
    breakdown: Dict[str, float] = field(default_factory=dict)


def compute_reward(
    *,
    question: str,
    memo: str,
    bib: List[Dict[str, Any]],
    step_count: int,
) -> RewardResult:
    """Deterministically score the memo given the working notes."""
    memo_lower = memo.lower()
    question_tokens = _tokenize(question)
    memo_tokens = _tokenize(memo)

    # Papers cited in memo by ID or by title keyword overlap
    citation_hits = 0
    for entry in bib:
        pid = (entry.get("paperId") or "").lower()
        title = (entry.get("title") or "").lower()
        if pid and pid in memo_lower:
            citation_hits += 1
            continue
        if title and _title_in_text(title, memo_lower):
            citation_hits += 1
    citation_reward = min(citation_hits, len(bib)) * 1.0

    venues = [entry.get("venue") for entry in bib if entry.get("venue")]
    unique_venues = len({v.lower() for v in venues})
    diversity_reward = max(unique_venues * 0.4 - (len(venues) - unique_venues) * 0.2, -1.0)

    coverage_fraction = len(question_tokens & memo_tokens) / max(len(question_tokens), 1)
    length_bonus = min(len(memo) / 1200.0, 1.0)
    coverage_reward = min(coverage_fraction * 3.0 + length_bonus * 2.0, 4.0)

    budget_penalty = -0.15 * max(step_count - 12, 0)

    total = round(citation_reward + diversity_reward + coverage_reward + budget_penalty, 3)
    breakdown = {
        "citation_reward": round(citation_reward, 3),
        "diversity_reward": round(diversity_reward, 3),
        "coverage_reward": round(coverage_reward, 3),
        "budget_penalty": round(budget_penalty, 3),
    }
    return RewardResult(total=total, breakdown=breakdown)


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return set(tokens)


def _title_in_text(title: str, text: str) -> bool:
    words = [w for w in re.findall(r"[a-z0-9]+", title) if len(w) > 3]
    hits = sum(1 for w in words if w in text)
    if not words:
        return False
    return hits / len(words) >= 0.3
