"""LangGraph wiring that enforces a single tool call per step."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, TypedDict

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import END, StateGraph


class GraphState(TypedDict):
    """Shared state passed between LangGraph nodes."""

    messages: List[Any]
    steps_executed: int
    max_steps: int
    submitted: bool
    pending_tool_call: Optional[Dict[str, Any]]
    phase: str
    note_pending: bool
    force_submit_mode: bool
    force_submit_active: bool
    force_submit_prompted: bool
    summary_ready: bool
    force_summary_active: bool
    force_summary_prompted: bool
    summary_prompted: bool


TOOL_NODE_LABELS = {
    "search_semantic_scholar": "semantic_search",
    "fetch_paper": "semantic_fetch",
    "write_note": "write_note",
    "review_bibliography": "review_bibliography",
    "fetch_bibliography_paper": "inspect_saved_paper",
    "wait": "wait",
    "summarize_findings": "summarize_findings",
    "submit": "submit_memo",
}

PHASE_TRANSITIONS: Dict[str, Sequence[str]] = {
    "start_phase": ("review_bibliography", "search_semantic_scholar", "wait", "write_note", "summarize_findings"),
    "review_bibliography": (
        "review_bibliography",
        "fetch_bibliography_paper",
        "search_semantic_scholar",
        "write_note",
        "wait",
        "summarize_findings",
    ),
    "fetch_bibliography_paper": ("write_note", "review_bibliography", "summarize_findings", "submit", "wait"),
    "search_semantic_scholar": ("fetch_paper", "write_note", "review_bibliography", "wait", "summarize_findings"),
    "fetch_paper": ("write_note", "review_bibliography", "wait", "summarize_findings"),
    "write_note": (
        "write_note",
        "review_bibliography",
        "search_semantic_scholar",
        "fetch_paper",
        "fetch_bibliography_paper",
        "summarize_findings",
        "submit",
        "wait",
    ),
    "wait": ("review_bibliography", "search_semantic_scholar", "write_note", "summarize_findings"),
    "force_summary_phase": ("summarize_findings",),
    "force_submit_phase": ("submit",),
    "summarize_findings": ("submit", "write_note", "review_bibliography", "search_semantic_scholar"),
    "submit": (),
}
INITIAL_PHASE = "start_phase"


def _filter_phase_transitions(available_tools: Sequence[str]) -> Dict[str, Sequence[str]]:
    """Limit the transition table to the tools that are actually available."""
    allowed = set(available_tools)
    filtered: Dict[str, Sequence[str]] = {}
    for phase, transitions in PHASE_TRANSITIONS.items():
        filtered[phase] = tuple(tool for tool in transitions if tool in allowed)
    return filtered


def build_graph(llm, tools, max_steps: int):
    """Return a compiled LangGraph with strict one-tool-per-step enforcement."""
    llm = llm.bind_tools(tools, tool_choice="any")
    tool_map = {tool.name: tool for tool in tools}
    phase_transitions = _filter_phase_transitions(tool_map.keys())

    def agent_node(state: GraphState) -> GraphState:
        response = llm.invoke(state["messages"])
        state["messages"].append(response)
        return state

    def validate_tool_sequence(state: GraphState) -> GraphState:
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []

        if len(tool_calls) != 1:
            if len(tool_calls) == 0:
                correction = "You MUST call exactly ONE tool this step. Call one tool now."
            else:
                correction = "You called multiple tools. Call EXACTLY ONE tool this step."
            call = tool_calls[0] if tool_calls else {}
            tool_call_id = call.get("id", "invalid_tool_call")
            state["messages"].append(
                ToolMessage(content="Tool call failed: invalid number of tool calls.", tool_call_id=tool_call_id)
            )
            state["messages"].append(HumanMessage(content=correction))
            state["pending_tool_call"] = None
            return state

        call = tool_calls[0]
        name = call.get("name")
        tool = tool_map.get(name)
        if not tool:
            state["messages"].append(
                ToolMessage(content=f"Tool call failed: unknown tool '{name}'.", tool_call_id=call.get("id", name))
            )
            state["messages"].append(
                HumanMessage(content=f"Unknown tool '{name}'. Call exactly one valid tool.")
            )
            state["pending_tool_call"] = None
            return state

        phase = state.get("phase") or INITIAL_PHASE
        allowed = phase_transitions.get(phase, phase_transitions.get(INITIAL_PHASE, ()))
        if state.get("note_pending") and name not in ("write_note", "wait"):
            state["messages"].append(
                ToolMessage(
                    content="Tool call rejected: you must call write_note before the next non-note action.",
                    tool_call_id=call.get("id", name),
                )
            )
            state["messages"].append(
                HumanMessage(
                    content="Record a note summarizing your thinking or findings before issuing another action."
                )
            )
            state["pending_tool_call"] = None
            return state
        if name not in allowed:
            allowed_str = ", ".join(allowed) if allowed else "no tools"
            state["messages"].append(
                ToolMessage(
                    content=f"Tool call rejected: '{name}' not allowed during phase '{phase}'.",
                    tool_call_id=call.get("id", name),
                )
            )
            state["messages"].append(
                HumanMessage(
                    content=(
                        f"Invalid tool '{name}' for this phase. "
                        f"Choose one of: {allowed_str}. After fetching, inspect or record the paper before searching again."
                    )
                )
            )
            state["pending_tool_call"] = None
            return state

        if name == "submit" and not state.get("summary_ready"):
            state["messages"].append(
                ToolMessage(
                    content="Tool call rejected: summarize_findings must run immediately before submit.",
                    tool_call_id=call.get("id", name),
                )
            )
            state["messages"].append(
                HumanMessage(
                    content=(
                        "Call summarize_findings with JSON outlining top arguments, sources, and people. "
                        "Then call submit(memo=...)."
                    )
                )
            )
            state["pending_tool_call"] = None
            return state

        state["pending_tool_call"] = {
            "tool_name": name,
            "tool_call": call,
        }
        if name in ("write_note", "summarize_findings"):
            state["note_pending"] = False
        elif name == "wait":
            state["note_pending"] = state.get("note_pending", False)
        else:
            state["note_pending"] = True
        return state

    def route_tool(state: GraphState) -> str:
        pending = state.get("pending_tool_call")
        if not pending:
            return "retry"
        return pending.get("tool_name") or "retry"

    def build_tool_node(tool_name: str, tool):
        def node(state: GraphState) -> GraphState:
            pending = state.get("pending_tool_call") or {}
            call = pending.get("tool_call") or {}
            args = call.get("args", {}) or {}
            result = tool.invoke(args)
            tool_call_id = call.get("id", tool_name or "tool")
            state["messages"].append(ToolMessage(content=str(result), tool_call_id=tool_call_id))
            state["steps_executed"] += 1
            previous_phase = state.get("phase") or INITIAL_PHASE
            if tool_name == "wait":
                state["phase"] = previous_phase
            else:
                state["phase"] = tool_name
            if tool_name == "submit":
                state["submitted"] = True
            if tool_name == "summarize_findings":
                state["summary_ready"] = True
                state["force_summary_active"] = False
            elif tool_name != "submit":
                state["summary_ready"] = False
            state["pending_tool_call"] = None
            return state

        return node

    def _inject_summary_prompt(state: GraphState) -> None:
        if state.get("summary_prompted"):
            return
        state["messages"].append(
            HumanMessage(
                content=(
                    "Before you submit you must call summarize_findings(summary={...}) with a JSON object that locks in your plan. "
                    "Include: (1) `top arguments`: 3–5 clear statements; (2) `top articles`: each with title, paperId or url, authors, "
                    "and a `reason_chosen` describing how it supports your argument; (3) `top people`: experts you intend to highlight. "
                    "After calling summarize_findings with that JSON, immediately call submit(memo=...)."
                )
            )
        )
        state["summary_prompted"] = True

    def should_continue(state: GraphState) -> str:
        if not state.get("summary_ready"):
            threshold = state["max_steps"] - 2 if state["max_steps"] else 2
            if state["steps_executed"] >= max(threshold, 2):
                _inject_summary_prompt(state)
        if state["submitted"]:
            return "end"

        summary_ready = state.get("summary_ready", False)
        if state.get("force_submit_mode") and not state["submitted"]:
            summary_threshold = max(state["max_steps"] - 2, 0)
            if (
                state["steps_executed"] >= summary_threshold
                and not summary_ready
                and not state.get("force_summary_active")
            ):
                state["phase"] = "force_summary_phase"
                state["note_pending"] = False
                state["pending_tool_call"] = None
                state["force_summary_active"] = True
                if not state.get("force_summary_prompted"):
                    _inject_summary_prompt(state)
                    state["force_summary_prompted"] = True
            if state.get("force_summary_active") and not summary_ready:
                return "continue"

            final_threshold = max(state["max_steps"] - 1, 0)
            if state["steps_executed"] >= final_threshold and not state.get("force_submit_active"):
                state["phase"] = "force_submit_phase"
                state["note_pending"] = False
                state["pending_tool_call"] = None
                state["force_submit_active"] = True
                if not state.get("force_submit_prompted"):
                    state["messages"].append(
                        HumanMessage(
                            content="You are out of tool calls. Summarize your findings and call the `submit` tool now."
                        )
                    )
                    state["force_submit_prompted"] = True
            if state.get("force_submit_active") and not state["submitted"]:
                if state["steps_executed"] >= state["max_steps"]:
                    return "continue"

        if state["steps_executed"] >= state["max_steps"]:
            return "end"
        return "continue"

    graph = StateGraph(GraphState)
    graph.add_node("agent", agent_node)

    tool_nodes: Dict[str, str] = {}
    for tool in tools:
        node_name = TOOL_NODE_LABELS.get(tool.name, f"Tool::{tool.name}")
        graph.add_node(node_name, build_tool_node(tool.name, tool))
        graph.add_conditional_edges(node_name, should_continue, {"continue": "agent", "end": END})
        tool_nodes[tool.name] = node_name

    graph.add_node("validate_tool_sequence", validate_tool_sequence)
    route_map = {tool: node for tool, node in tool_nodes.items()}
    route_map["retry"] = "agent"
    graph.add_conditional_edges("validate_tool_sequence", route_tool, route_map)

    graph.set_entry_point("agent")
    graph.add_edge("agent", "validate_tool_sequence")
    return graph.compile()
