"""Tests for the summary-before-submit flow."""

from __future__ import annotations

from typing import List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolCall

from policy_src.policy_agent_lc.graph import build_graph
from policy_src.policy_agent_lc.session import ResearchSession
from policy_src.policy_agent_lc.tools import build_tools


class FakeLLM:
    """Deterministic stub mirroring the LangChain ChatModel interface."""

    def __init__(self, responses: List[AIMessage]):
        self._responses = responses

    def bind_tools(self, tools, tool_choice: str = "any"):
        return self

    def invoke(self, messages):
        if not self._responses:
            raise RuntimeError("No scripted responses remain.")
        return self._responses.pop(0)


def _build_session(tmp_path, max_steps: int = 6) -> tuple[ResearchSession, list]:
    cache_path = tmp_path / "policy_cache.sqlite"
    session = ResearchSession(
        task_id="test-session",
        question="How should FEMA prioritize resilience grants?",
        instructions="Use tools wisely.",
        use_cached=True,
        max_steps=max_steps,
        cache_path=str(cache_path),
    )
    tools = build_tools(session)
    return session, tools


def test_summarize_findings_normalizes_payload(tmp_path) -> None:
    """The summarize tool should coerce aliases + strings into normalized dicts."""
    session, tools = _build_session(tmp_path)
    summarize_tool = next(tool for tool in tools if tool.name == "summarize_findings")
    payload = {
        "top arguments": ["  Funding gaps in small towns  ", 42],
        "top articles": [
            {
                "title": "Community Resilience Finance",
                "paper_id": "S2:xyz",
                "authors": ["Dr. Jane Doe", "Dr. Alex Kim"],
                "reason": "Details grant programs.",
            },
            "ignore-me",
        ],
        "top people": ["Dr. Jane Doe", None],
        "top recommendations": ["  Fund resilience hubs  ", ""],
    }

    result = summarize_tool.invoke({"summary": payload})
    assert session.summary["top_arguments"] == ["Funding gaps in small towns", "42"]
    assert len(session.summary["top_articles"]) == 1
    article = session.summary["top_articles"][0]
    assert article["paperId"] == "S2:xyz"
    assert article["title"] == "Community Resilience Finance"
    assert article["authors"] == ["Dr. Jane Doe", "Dr. Alex Kim"]
    assert session.summary["top_people"] == ["Dr. Jane Doe"]
    assert session.summary["top_recommendations"] == ["Fund resilience hubs"]
    # Tool response should mirror normalized payload.
    assert result["summary"] == session.summary


def test_submit_requires_summary_first(tmp_path) -> None:
    """Submit tool calls should be rejected until summarize_findings runs."""
    session, tools = _build_session(tmp_path, max_steps=4)
    responses = [
        AIMessage(
            content="Submitting early.",
            tool_calls=[ToolCall(name="submit", args={"memo": "Premature memo"}, id="call_1")],
        ),
        AIMessage(
            content="Here is the summary JSON.",
            tool_calls=[
                ToolCall(
                    name="summarize_findings",
                    args={
                        "summary": {
                            "top arguments": ["Argument A"],
                            "top articles": [{"paperId": "abc123", "title": "Paper A"}],
                            "top people": ["Author A"],
                            "top recommendations": ["Add matching funds"],
                        }
                    },
                    id="call_2",
                )
            ],
        ),
        AIMessage(
            content="Submitting after summary.",
            tool_calls=[ToolCall(name="submit", args={"memo": "Final memo body."}, id="call_3")],
        ),
    ]
    llm = FakeLLM(responses)
    graph = build_graph(llm, tools, max_steps=4)
    state = {
        "messages": [
            SystemMessage(content="system"),
            HumanMessage(content="question"),
        ],
        "steps_executed": 0,
        "max_steps": 4,
        "submitted": False,
        "pending_tool_call": None,
        "phase": "start_phase",
        "note_pending": False,
        "force_submit_mode": True,
        "force_submit_active": False,
        "force_submit_prompted": False,
        "summary_ready": False,
        "force_summary_active": False,
        "force_summary_prompted": False,
    }

    final_state = graph.invoke(state)
    assert final_state["submitted"] is True
    # Only summarize + submit should be counted as tool calls; initial invalid submit is rejected.
    assert session.step_count == 2
    assert [call["tool"] for call in session.tool_calls] == ["summarize_findings", "submit"]
    assert session.final_memo == "Final memo body."
    assert session.summary["top_arguments"] == ["Argument A"]
