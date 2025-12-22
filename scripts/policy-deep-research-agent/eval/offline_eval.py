"""Deterministic offline evaluation for recorded policy deep research rollouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from src.envs.policy_deep_research_env.server.reward import compute_reward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score cached trajectories with the environment reward.")
    parser.add_argument("--trajectories", required=True, help="JSONL file with memo + bib data.")
    parser.add_argument(
        "--tasks",
        default="data/tasks/policy_questions.jsonl",
        help="Task file to map task_id to question text.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_map = load_tasks(Path(args.tasks))
    rows = list(load_jsonl(Path(args.trajectories)))
    if not rows:
        print("No trajectories found.")
        return

    summaries = []
    for row in rows:
        task_id = row.get("task_id")
        question = row.get("question") or task_map.get(task_id, "")
        memo = row.get("final_memo") or row.get("memo") or ""
        step_count = int(row.get("steps") or row.get("step_count") or 0)
        bib = row.get("bib") or []
        result = compute_reward(question=question, memo=memo, bib=bib, step_count=step_count)
        summaries.append(
            {
                "task_id": task_id,
                "reward": result.total,
                "citation_reward": result.breakdown["citation_reward"],
                "diversity_reward": result.breakdown["diversity_reward"],
                "coverage_reward": result.breakdown["coverage_reward"],
                "budget_penalty": result.breakdown["budget_penalty"],
            }
        )

    print_report(summaries)


def load_tasks(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    mapping: Dict[str, str] = {}
    for record in load_jsonl(path):
        mapping[record.get("task_id")] = record.get("question", "")
    return mapping


def load_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def print_report(rows: List[Dict]) -> None:
    total = sum(r["reward"] for r in rows)
    avg = total / max(len(rows), 1)
    print(f"Evaluated {len(rows)} trajectories. Average reward: {avg:.3f}")
    print("task_id,reward,citation,diversity,coverage,budget_penalty")
    for row in rows:
        print(
            f"{row['task_id']},{row['reward']:.3f},{row['citation_reward']:.3f},"
            f"{row['diversity_reward']:.3f},{row['coverage_reward']:.3f},{row['budget_penalty']:.3f}"
        )


if __name__ == "__main__":
    main()
