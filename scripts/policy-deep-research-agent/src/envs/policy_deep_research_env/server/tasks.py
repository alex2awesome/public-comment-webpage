"""Task loading utilities for the policy deep research environment."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List


@lru_cache(maxsize=1)
def load_tasks() -> List[Dict]:
    """Load policy questions from the configured JSONL file."""
    paths = _candidate_paths()
    for path in paths:
        if not path.exists():
            continue
        tasks: List[Dict] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                tasks.append(json.loads(line))
        if tasks:
            return tasks.copy()
    raise FileNotFoundError(f"No tasks file found in {paths}")


def _candidate_paths() -> List[Path]:
    env_path = os.getenv("TASKS_PATH")
    if env_path:
        return [Path(env_path)]
    package_root = Path(__file__).resolve().parents[1]
    return [
        Path("/app/env/data/tasks/policy_questions.jsonl"),
        package_root / "data" / "tasks" / "policy_questions.jsonl",
    ]
