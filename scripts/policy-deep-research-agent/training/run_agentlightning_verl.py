"""AgentLightning runner that reuses the LangGraph rollout + OpenEnv reward."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean
from typing import Dict, List

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policy_src.policy_agent_lc.io import append_jsonl
from policy_src.policy_agent_lc.rollout import run_one_rollout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/eval policy agent with AgentLightning VERL.")
    parser.add_argument("--phase", choices=["sanity", "verl"], default="sanity")
    parser.add_argument("--tasks-path", default="data/tasks/policy_questions.jsonl")
    parser.add_argument("--num-tasks", type=int, default=4, help="Number of tasks to cycle through.")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--use-cached", action="store_true")
    parser.add_argument("--cache-path", default="data/cache/policy_cache.sqlite")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--sanity-out", default="data/cache/rollouts_langgraph.jsonl")
    parser.add_argument("--rollout-log-prefix", default="data/cache/rollouts_agentlightning")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--rollouts-per-epoch", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--max-response-tokens", type=int, default=900)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = load_tasks_from_jsonl(Path(args.tasks_path))
    limited_tasks = tasks[: max(1, args.num_tasks)]

    if args.phase == "sanity":
        run_sanity_phase(args, limited_tasks)
    else:
        run_verl_phase(args, limited_tasks)


def load_tasks_from_jsonl(path: Path) -> List[Dict]:
    """Load JSONL tasks and attach a task_index for downstream loops."""
    items: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            payload.setdefault("task_index", idx)
            items.append(payload)
    if not items:
        raise ValueError(f"No tasks found in {path}")
    return items


def run_sanity_phase(args: argparse.Namespace, tasks: List[Dict]) -> None:
    """Phase A: run LangGraph rollouts and report aggregate reward."""
    rewards: List[float] = []
    for idx, task in enumerate(tasks):
        ep = run_one_rollout(
            model_name=args.model,
            task_index=int(task["task_index"]),
            use_cached=args.use_cached or os.getenv("OPENENV_USE_CACHED", "1") == "1",
            max_steps=args.max_steps,
            cache_path=args.cache_path,
            temperature=args.temperature,
        )
        append_jsonl(args.sanity_out, ep)
        rewards.append(float(ep["reward"]))
        print(f"[sanity] task={task.get('task_id')} idx={idx} reward={ep['reward']:.3f} steps={ep['steps']}")
    avg_reward = mean(rewards) if rewards else 0.0
    print(f"[sanity] completed {len(rewards)} rollouts avg_reward={avg_reward:.3f}")


def run_verl_phase(args: argparse.Namespace, tasks: List[Dict]) -> None:
    """Phase B: wire the rollout into AgentLightning VERL training."""
    try:
        import agentlightning as agl
    except ImportError as exc:
        raise SystemExit(
            "AgentLightning is not installed. Install it per requirements.txt before running phase 'verl'."
        ) from exc

    rollout_fn = build_policy_rollout(agl, args)

    verl_config_cls = getattr(agl, "VERLConfig", None)
    if verl_config_cls:
        verl_config = verl_config_cls(
            adv_estimator="grpo",
            rollout_batch_size=args.rollouts_per_epoch,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            max_steps=args.max_steps,
        )
    else:
        verl_config = {
            "adv_estimator": "grpo",
            "rollout_batch_size": args.rollouts_per_epoch,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "max_steps": args.max_steps,
        }

    llm_cls = getattr(agl, "LLM", None)
    llm = None
    if llm_cls:
        llm = llm_cls(model=args.model, temperature=args.temperature, max_tokens=args.max_response_tokens)

    trainer_cls = getattr(agl, "Trainer", None)
    if trainer_cls is None:
        raise SystemExit("agentlightning.Trainer is missing; upgrade AgentLightning to a version with VERL support.")

    trainer_kwargs = {
        "tasks": tasks,
        "rollout": rollout_fn,
        "config": verl_config,
        "epochs": args.epochs,
    }
    if llm is not None:
        trainer_kwargs["llm"] = llm

    trainer = trainer_cls(**trainer_kwargs)
    trainer.fit()


def build_policy_rollout(agl_module, args: argparse.Namespace):
    """Wrap run_one_rollout with AgentLightning's rollout decorator."""
    rollout_path = f"{args.rollout_log_prefix}_{os.getpid()}.jsonl"
    use_cached = args.use_cached or os.getenv("OPENENV_USE_CACHED", "1") == "1"

    def map_llm_name(name: str) -> str:
        overrides = {
            "openai/gpt-4o-mini": "gpt-4o-mini",
            "gpt-4o-mini": "gpt-4o-mini",
        }
        return overrides.get(name, name or args.model)

    @agl_module.rollout
    def policy_rollout(task: Dict, llm) -> float:
        task_index = int(task.get("task_index", 0))
        model_name = map_llm_name(getattr(llm, "model", args.model))
        episode = run_one_rollout(
            model_name=model_name,
            task_index=task_index,
            use_cached=use_cached,
            max_steps=args.max_steps,
            cache_path=args.cache_path,
            temperature=args.temperature,
        )
        append_jsonl(rollout_path, episode)
        return float(episode["reward"])

    return policy_rollout


if __name__ == "__main__":
    main()
