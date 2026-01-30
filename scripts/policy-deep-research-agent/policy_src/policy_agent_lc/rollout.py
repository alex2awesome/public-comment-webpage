"""Single rollout helper that orchestrates LangGraph + legacy tools."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage, messages_to_dict
from langchain_openai import ChatOpenAI

from policy_src.policy_research_core.reward import compute_reward
from policy_src.policy_research_core.tasks import load_tasks

from .graph import build_graph
from .prompts import load_system_prompt
from .session import ResearchSession
from .tools import build_tools


logger = logging.getLogger(__name__)
FALLBACK_TASK = {
    "task_id": "default-policy-question",
    "question": (
        "Summarize notable recent developments in U.S. federal policy related to AI governance, "
        "highlighting key agencies and actions."
    ),
}


def select_task(task_index: Optional[int] = None) -> Dict[str, Any]:
    """Return a task description by index (wraps around if needed)."""
    try:
        tasks = load_tasks()
    except FileNotFoundError:
        logger.warning("Policy task file missing; using fallback question.")
        tasks = [FALLBACK_TASK]
    if task_index is None:
        task_index = 0
    task = tasks[task_index % len(tasks)]
    task.setdefault("task_id", str(task_index))
    return task


def run_one_rollout(
    model_name: str,
    task_index: int,
    use_cached: bool,
    max_steps: int,
    cache_path: str,
    temperature: float = 0.7,
    enable_bibliography: bool = True,
    question_override: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Execute one LangGraph rollout and return an episode dict."""
    task = select_task(task_index)
    if question_override:
        task = {
            **task,
            "question": question_override,
            "task_id": task.get("task_id", f"custom-{task_index}"),
        }
    run_id = str(uuid.uuid4())

    instructions = (
        "Use the tools to search Semantic Scholar, fetch papers, build a bibliography with reasons, "
        "take notes, then submit a final memo."
    )
    if max_steps:
        instructions += f" You may issue at most {max_steps} tool calls; plan to use the final step for `submit`."
    if not enable_bibliography:
        instructions += " Bibliography tools are disabled for this run; rely on search results, notes, and your final memo."

    session = ResearchSession(
        task_id=task.get("task_id", str(task_index)),
        question=task.get("question", ""),
        instructions=instructions,
        use_cached=use_cached,
        max_steps=max_steps,
        cache_path=cache_path,
        bibliography_enabled=enable_bibliography,
    )

    repo_root = Path(__file__).resolve().parents[2]
    policy_root = repo_root / "policy_src"
    system_prompt = load_system_prompt(policy_root)

    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        timeout=300,
        max_retries=2,
        parallel_tool_calls=False,
        api_key=openai_api_key,
    )

    if event_callback:
        try:
            event_callback(
                {
                    "type": "run_started",
                    "task_id": session.task_id,
                    "question": session.question,
                    "max_steps": max_steps,
                }
            )
        except Exception:
            pass

    tools = build_tools(session, enable_bibliography=enable_bibliography, event_callback=event_callback)
    graph = build_graph(llm, tools, max_steps=max_steps)

    state = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"{session.question}\n\n{session.instructions}"),
        ],
        "steps_executed": 0,
        "max_steps": max_steps,
        "submitted": False,
        "pending_tool_call": None,
        "phase": "start_phase",
        "note_pending": False,
        "force_submit_mode": bool(max_steps),
        "force_submit_active": False,
        "force_submit_prompted": False,
    }

    invoke_config = {"metadata": {"episode_run_id": run_id}}
    final_state = graph.invoke(state, config=invoke_config)
    session.messages = messages_to_dict(final_state["messages"])

    if session.final_memo is None:
        session.final_reward = -1.0
        session.reward_breakdown = {}
    else:
        reward = compute_reward(
            question=session.question,
            memo=session.final_memo,
            bib=session.bib,
            step_count=session.step_count,
        )
        session.final_reward = float(reward.total)
        session.reward_breakdown = dict(reward.breakdown)

    episode = {
        "run_id": run_id,
        "task_id": session.task_id,
        "question": session.question,
        "final_memo": session.final_memo or "",
        "reward": session.final_reward,
        "reward_breakdown": session.reward_breakdown,
        "bib": session.bib,
        "notes": session.notes,
        "steps": session.step_count,
        "tool_calls": session.tool_calls,
        "messages": session.messages,
    }
    if event_callback:
        try:
            event_callback({"type": "run_completed", "episode": episode})
        except Exception:
            pass
    return episode
