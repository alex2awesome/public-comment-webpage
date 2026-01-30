"""Command-line replica of the LangGraph rollout notebook for debugging."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Ensure repo root is on sys.path so `policy_src.` and `eval.` imports resolve.
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from eval.offline_eval import load_jsonl as eval_load_jsonl
from eval.offline_eval import load_tasks as eval_load_tasks
from eval.offline_eval import print_report
from policy_src.policy_research_core.reward import compute_reward
from policy_src.policy_agent_lc.io import append_jsonl
from policy_src.policy_agent_lc.rollout import run_one_rollout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a LangGraph policy rollout (notebook parity).")
    parser.add_argument("--model-name", default="gpt-5-mini", help="Chat model to use via langchain-openai.")
    parser.add_argument("--task-index", type=int, default=0, help="Index of the policy question to run.")
    parser.add_argument("--use-cached", action="store_true", help="Only read Semantic Scholar data from cache.")
    parser.add_argument("--max-steps", type=int, default=12, help="Max tool executions before forcing termination.")
    parser.add_argument("--cache-path", default=REPO_ROOT / "data/cache/policy_cache.sqlite", help="Path to CacheDB file.")
    parser.add_argument("--output-path", default=REPO_ROOT / "data/cache/rollouts_langgraph.jsonl", help="Where to append episodes.")
    parser.add_argument("--tasks-path", default=REPO_ROOT / "data/tasks/policy_questions.jsonl", help="Tasks JSONL for eval summary.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--openai-key-file",
        type=Path,
        help="Optional text file whose contents populate OPENAI_API_KEY for convenience.",
    )
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Do not recompute offline evaluation across saved episodes.",
    )
    return parser.parse_args()


def maybe_load_api_key(path: Path | None) -> None:
    if os.environ.get("OPENAI_API_KEY"):
        return
    if not path:
        return
    expanded = path.expanduser()
    if not expanded.exists():
        raise FileNotFoundError(f"OPENAI key file not found: {expanded}")
    os.environ["OPENAI_API_KEY"] = expanded.read_text().strip()


def format_bibliography(bib: List[dict]) -> str:
    if not bib:
        return "(empty)"
    lines = []
    for idx, entry in enumerate(bib, start=1):
        lines.append(f"{idx}. {entry.get('title')} ({entry.get('year')}) - {entry.get('reason', '')}")
    return "\n".join(lines)


def main() -> None:
    load_dotenv()
    args = parse_args()
    maybe_load_api_key(args.openai_key_file)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    episode = run_one_rollout(
        model_name=args.model_name,
        task_index=args.task_index,
        use_cached=args.use_cached,
        max_steps=args.max_steps,
        cache_path=args.cache_path,
        temperature=args.temperature,
    )
    append_jsonl(str(output_path), episode)

    print(f"Task: {episode['task_id']} -> {episode['question']}")
    print(f"Reward: {episode['reward']:.3f} | Steps: {episode['steps']} | Bib entries: {len(episode['bib'])}")
    print("-" * 80)
    print("Final memo:")
    print(episode["final_memo"] or "(empty memo)")
    print("-" * 80)
    print("Bibliography:")
    print(format_bibliography(episode["bib"]))
    print("-" * 80)
    print("Tool calls:")
    for call in episode["tool_calls"]:
        print(call)

    if args.skip_summary:
        return

    tasks_map = eval_load_tasks(Path(args.tasks_path))
    rows = list(eval_load_jsonl(output_path))
    summaries = []
    for row in rows:
        question = row.get("question") or tasks_map.get(row.get("task_id"), "")
        memo = row.get("final_memo") or row.get("memo") or ""
        step_count = int(row.get("steps") or row.get("step_count") or 0)
        bib = row.get("bib") or []
        reward = compute_reward(question=question, memo=memo, bib=bib, step_count=step_count)
        summaries.append(
            {
                "task_id": row.get("task_id"),
                "reward": reward.total,
                "citation_reward": reward.breakdown["citation_reward"],
                "diversity_reward": reward.breakdown["diversity_reward"],
                "coverage_reward": reward.breakdown["coverage_reward"],
                "budget_penalty": reward.breakdown["budget_penalty"],
            }
        )
    if summaries:
        print("-" * 80)
        print_report(summaries)
    else:
        print("No rollouts saved yet; skipping summary.")


if __name__ == "__main__":
    main()
