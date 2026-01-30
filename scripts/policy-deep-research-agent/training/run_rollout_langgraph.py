"""CLI helper to run single LangGraph rollouts for debugging."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policy_src.policy_agent_lc.io import append_jsonl
from policy_src.policy_agent_lc.rollout import run_one_rollout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a LangGraph rollout for a policy task.")
    parser.add_argument("--model", default="gpt-4o-mini", help="Chat model name for LangChain.")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--use-cached", action="store_true")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--cache-path", default="data/cache/policy_cache.sqlite")
    parser.add_argument("--out", default="data/cache/rollouts_langgraph.jsonl")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--disable-bibliography",
        action="store_true",
        help="Skip bibliography tools and caching-heavy actions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode = run_one_rollout(
        model_name=args.model,
        task_index=args.task_index,
        use_cached=args.use_cached,
        max_steps=args.max_steps,
        cache_path=args.cache_path,
        temperature=args.temperature,
        enable_bibliography=not args.disable_bibliography,
    )
    print(f"reward={episode['reward']:.3f} steps={episode['steps']} bib={len(episode['bib'])}")
    append_jsonl(args.out, episode)


if __name__ == "__main__":
    main()
