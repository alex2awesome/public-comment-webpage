"""TRL GRPO harness for the policy deep research environment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

TRAINING_DIR = Path(__file__).resolve().parent
REPO_ROOT = TRAINING_DIR.parents[0]
for path in (TRAINING_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rollout_openenv import rollout_openenv
from utils.envs.policy_deep_research_env.client import PolicyDeepResearchEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a policy research agent with TRL GRPO.")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--env-image", type=str, default="openenv-policy-deep-research-env:latest")
    parser.add_argument("--env-url", type=str, default=None, help="Optional running OpenEnv URL.")
    parser.add_argument("--use-cached", action="store_true", help="Do not hit Semantic Scholar when cache is warm.")
    parser.add_argument("--cache-path", type=str, default="../data/cache/policy_cache.sqlite", help="Host cache DB path.")
    parser.add_argument(
        "--container-cache-path",
        type=str,
        default="/app/env/data/cache.sqlite",
        help="Cache path seen inside the container.",
    )
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--rollouts-per-batch", type=int, default=4)
    parser.add_argument("--training-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--output-dir", type=str, default="../data/checkpoints")
    parser.add_argument("--system-prompt", type=str, default="../policy_src/policy_prompts/system.txt")
    parser.add_argument("--action-schema", type=str, default="../policy_src/policy_prompts/action_schema.md")
    parser.add_argument("--record-rollouts", type=str, default="../data/cache/rollouts.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()

    training_dir = Path(__file__).resolve().parent
    system_prompt_path = Path(args.system_prompt)
    if not system_prompt_path.is_absolute():
        system_prompt_path = training_dir / system_prompt_path
    system_prompt = system_prompt_path.read_text()
    example_path = system_prompt_path.parent / "example_1.txt"
    if example_path.exists():
        system_prompt = f"{system_prompt.strip()}\n\n{example_path.read_text().strip()}"
    action_schema_path = Path(args.action_schema)
    if not action_schema_path.is_absolute():
        action_schema_path = training_dir / action_schema_path
    action_schema = action_schema_path.read_text()

    env_client = build_env_client(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    for step in range(args.training_steps):
        step_rollouts = []
        model.eval()
        for ridx in range(args.rollouts_per_batch):
            episode = rollout_openenv(
                model=model,
                tokenizer=tokenizer,
                env_client=env_client,
                system_prompt=system_prompt,
                action_schema=action_schema,
                max_steps=args.max_steps,
            )
            step_rollouts.append(episode)
            print(f"[rollout] step={step} sample={ridx} reward={episode['reward'][0]:.3f}")
        dataset = episodes_to_dataset(step_rollouts)
        trainer = build_trainer(model, tokenizer, dataset, args)
        trainer.train()
        model.eval()
        save_rollouts(step_rollouts, Path(args.record_rollouts))

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    env_client.close()


def build_env_client(args: argparse.Namespace) -> PolicyDeepResearchEnv:
    env_vars = {
        "OPENENV_USE_CACHED": "1" if args.use_cached else "0",
        "OPENENV_CACHE_PATH": args.container_cache_path,
        "OPENENV_TASK_INDEX": str(args.task_index),
        "MAX_STEPS": str(args.max_steps),
        "S2_API_KEY": os.getenv("S2_API_KEY", ""),
    }
    cache_path = Path(args.cache_path).resolve()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.env_url:
        print(
            (
                "Launching Docker image {image}.\n"
                "To persist the cache between runs, restart with:\n"
                f"  docker run -v {cache_path}:{args.container_cache_path} ...\n"
            ).format(image=args.env_image)
        )
        return PolicyDeepResearchEnv.from_docker_image(args.env_image, env_vars=env_vars)
    return PolicyDeepResearchEnv(base_url=args.env_url, request_timeout_s=30.0)


def episodes_to_dataset(episodes: List[Dict]) -> Dataset:
    prompts = []
    completions = []
    rewards = []
    for episode in episodes:
        prompts.append(episode["prompt"][0])
        completions.append(episode["completion"][0])
        rewards.append(episode["reward"][0])
    return Dataset.from_dict(
        {
            "prompt": prompts,
            "cached_reward": rewards,
            "reference_transcript": completions,
        }
    )


def build_trainer(model, tokenizer, dataset: Dataset, args: argparse.Namespace) -> GRPOTrainer:
    def cached_reward(prompts, completions, cached_reward, **_) -> List[float]:
        return cached_reward

    config = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        remove_unused_columns=False,
        per_device_train_batch_size=1,
        num_generations=1,
        logging_steps=1,
        max_steps=1,
    )
    return GRPOTrainer(
        model=model,
        reward_funcs=cached_reward,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )


def save_rollouts(episodes: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
