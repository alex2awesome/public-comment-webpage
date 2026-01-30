"""Prompt loading helpers for the LangGraph rollout."""

from __future__ import annotations

from pathlib import Path


PROMPT_SUBDIR = "policy_prompts"


def load_system_prompt(base_dir: Path) -> str:
    """Return the concatenated system prompt + example + tool instructions."""
    prompt_dir = base_dir / PROMPT_SUBDIR
    system = (prompt_dir / "system.txt").read_text()
    example = (prompt_dir / "example_1.txt").read_text()
    footer = """
TOOL USAGE RULES:
- You MUST call exactly ONE tool per step.
- Do NOT call tools in parallel.
- Choose from: search_semantic_scholar, fetch_paper, add_to_bib, write_note, submit
- When you have enough info, call submit(memo=...).
"""
    return f"{system.strip()}\n\n{example.strip()}\n\n{footer.strip()}"
