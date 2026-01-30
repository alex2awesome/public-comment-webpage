"""Tests enforcing the one-tool-per-step invariant + reward determinism."""

from __future__ import annotations

from typing import List

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolCall

from policy_src.policy_research_core.reward import compute_reward
from policy_src.policy_agent_lc.graph import build_graph
from policy_src.policy_agent_lc.session import ResearchSession
from policy_src.policy_agent_lc.tools import build_tools


class FakeLLM:
    """Deterministic stub that replays scripted AI messages."""

    def __init__(self, responses: List[AIMessage]):
        self._responses = responses

    def bind_tools(self, tools, tool_choice: str = "any"):
        return self

    def invoke(self, messages):
        if not self._responses:
            raise RuntimeError("No scripted responses remain.")
        return self._responses.pop(0)


def test_one_tool_per_step_enforced(tmp_path) -> None:
    """Invalid tool counts should not increment steps until a valid call occurs."""
    cache_path = tmp_path / "policy_cache.sqlite"
    session = ResearchSession(
        task_id="test",
        question="Test question?",
        instructions="Do things.",
        use_cached=True,
        max_steps=1,
        cache_path=str(cache_path),
    )
    tools = build_tools(session)

    responses = [
        AIMessage(content="No tool here.", tool_calls=[]),
        AIMessage(
            content="Too many tools.",
            tool_calls=[
                ToolCall(name="write_note", args={"content": "first"}, id="call_1"),
                ToolCall(name="write_note", args={"content": "second"}, id="call_2"),
            ],
        ),
        AIMessage(
            content="Valid write note.",
            tool_calls=[ToolCall(name="write_note", args={"content": "final note"}, id="call_3")],
        ),
    ]
    llm = FakeLLM(responses)
    graph = build_graph(llm, tools, max_steps=1)
    state = {
        "messages": [
            SystemMessage(content="system"),
            HumanMessage(content="question"),
        ],
        "steps_executed": 0,
        "max_steps": 1,
        "submitted": False,
        "pending_tool_call": None,
        "phase": "write_note",
    }

    final_state = graph.invoke(state)
    assert final_state["steps_executed"] == 1
    assert session.step_count == 1
    assert len(session.tool_calls) == 1
    assert session.tool_calls[0]["tool"] == "write_note"
    assert session.notes == ["final note"]


def test_compute_reward_deterministic() -> None:
    """Reward helper should be deterministic for identical inputs."""
    question = "How do states use carbon pricing for climate resilience?"
    memo = (
        "This memo cites policy_1 and describes Carbon Pricing Strategies with Nature Energy data. "
        "policy_1 also shows resilience funding for policy_2."
    )
    bib = [
        {"paperId": "policy_1", "title": "Carbon Pricing Strategies", "venue": "Nature Energy"},
        {"paperId": "policy_2", "title": "Resilience Funding Approaches", "venue": "Science"},
    ]

    first = compute_reward(question=question, memo=memo, bib=bib, step_count=10)
    second = compute_reward(question=question, memo=memo, bib=bib, step_count=10)

    assert first.total == pytest.approx(4.37, abs=1e-3)
    assert first.total == second.total
    assert first.breakdown == second.breakdown
    assert first.breakdown["citation_reward"] == pytest.approx(2.0, abs=1e-6)
