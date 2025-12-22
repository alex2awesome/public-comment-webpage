"""Utility helps TRL/GRPO talk to the OpenEnv policy research environment."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.envs.policy_deep_research_env.models import ResearchAction, ResearchObservation


def rollout_openenv(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    env_client,
    system_prompt: str,
    action_schema: str,
    max_steps: int,
    temperature: float = 0.7,
) -> Dict[str, List[Any]]:
    """Run a single environment episode and return GRPO-compatible tuples."""
    transcript: List[Dict[str, str]] = []
    messages = [
        {"role": "system", "content": system_prompt.strip()},
    ]

    reset_result = env_client.reset()
    observation: ResearchObservation = reset_result.observation
    messages.append(
        {
            "role": "user",
            "content": f"{observation.question}\n\n{observation.instructions}",
        }
    )

    final_reward = 0.0
    final_memo = ""
    task_id = observation.task_id
    steps_taken = 0
    for _ in range(max_steps):
        completion, prompt = _generate_action(model, tokenizer, messages, temperature)
        transcript.append({"prompt": prompt, "completion": completion})

        try:
            action_dict, _ = _parse_action_payload(completion)
        except ValueError as exc:
            messages.append({"role": "assistant", "content": completion})
            messages.append(
                {
                    "role": "user",
                    "content": f"Invalid JSON ({exc}). Respond with ONLY a JSON object that matches:\n{action_schema}",
                }
            )
            continue

        action = ResearchAction(**action_dict)
        step_result = env_client.step(action)
        observation = step_result.observation
        final_reward = float(step_result.reward or 0.0)
        task_id = observation.task_id
        steps_taken += 1
        metadata = observation.metadata or {}
        if metadata.get("final_memo"):
            final_memo = metadata.get("final_memo", "")
        messages.append({"role": "assistant", "content": completion})
        messages.append(
            {
                "role": "user",
                "content": _format_feedback(observation),
            }
        )
        if step_result.done:
            break

    return {
        "prompt": [system_prompt],
        "completion": [json.dumps(transcript, ensure_ascii=False)],
        "reward": [final_reward],
        "task_id": task_id,
        "final_memo": final_memo,
        "bib": observation.bib,
        "steps": steps_taken,
    }


def _generate_action(model, tokenizer, messages, temperature: float):
    prompt = _render_prompt(tokenizer, messages)
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    do_sample = temperature > 0
    gen_kwargs = {"max_new_tokens": 512, "do_sample": do_sample}
    if do_sample:
        gen_kwargs["temperature"] = temperature
    with torch.no_grad():
        generation = model.generate(**inputs, **gen_kwargs)
    output_ids = generation[0][inputs["input_ids"].shape[-1] :]
    completion = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return completion, prompt


def _render_prompt(tokenizer, messages):
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    rendered = []
    for message in messages:
        rendered.append(f"{message['role'].upper()}: {message['content']}")
    rendered.append("ASSISTANT:")
    return "\n".join(rendered)


ACTION_TAG = re.compile(r"<action(?:\s[^>]*)?>(.*?)</action>", re.DOTALL | re.IGNORECASE)
THINK_TAG = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)

CANONICAL_ACTION_TYPES = (
    "SEARCH SEMANTIC SCHOLAR",
    "FETCH PAPER",
    "ADD TO BIB",
    "WRITE NOTE",
    "SUBMIT",
)
ACTION_TYPE_ALIASES = {
    "SEARCH_SEMANTIC_SCHOLAR": "SEARCH SEMANTIC SCHOLAR",
    "FETCH_PAPER": "FETCH PAPER",
    "ADD_TO_BIB": "ADD TO BIB",
    "WRITE_NOTE": "WRITE NOTE",
}


def _normalize_type_key(value: str) -> str:
    """Normalize whitespace and separators so aliases can be matched."""
    return re.sub(r"[\s_-]+", " ", value.strip()).upper()


ACTION_TYPE_LOOKUP = {_normalize_type_key(name): name for name in CANONICAL_ACTION_TYPES}
ACTION_TYPE_LOOKUP.update(
    {_normalize_type_key(alias): canonical for alias, canonical in ACTION_TYPE_ALIASES.items()}
)


def _parse_action_payload(raw: str) -> Tuple[Dict[str, Any], str | None]:
    """Parse DRTulu-style output containing optional <think> and <action> tags."""
    action_blob = None
    action_match = ACTION_TAG.search(raw)
    if action_match:
        action_blob = action_match.group(1).strip()
    if not action_blob:
        action_blob = extract_json(raw)
    action = json.loads(action_blob)
    action_type = action.get("type")
    if isinstance(action_type, str):
        normalized_type = ACTION_TYPE_LOOKUP.get(_normalize_type_key(action_type))
        if normalized_type:
            action["type"] = normalized_type
    action.setdefault("metadata", {})
    action.setdefault("filters", {})
    action.setdefault("top_k", 10)
    thought = None
    think_match = THINK_TAG.search(raw)
    if think_match:
        thought = think_match.group(1).strip()
    return action, thought


def extract_json(text: str) -> str:
    """Extract the first JSON object from the string."""
    start = text.find("{")
    if start == -1:
        raise ValueError("missing '{'")
    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_string:
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
    raise ValueError("unbalanced braces")


def _format_feedback(obs: ResearchObservation) -> str:
    payload = {
        "last_tool_result": obs.last_tool_result,
        "bib_count": len(obs.bib),
        "notes": obs.notes[-2:],
        "remaining_steps": obs.remaining_steps,
    }
    return json.dumps(payload, ensure_ascii=False)
