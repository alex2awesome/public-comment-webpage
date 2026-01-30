"""Pydantic request/response objects for the FastAPI endpoints."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class RolloutRequest(BaseModel):
    question: str = Field(..., min_length=5, description="Policy research question to investigate.")
    max_steps: int = Field(12, ge=4, le=40, description="Maximum LangGraph tool calls.")
    enable_bibliography: bool = Field(False, description="Allow bibliography caching + fetch tools.")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)


class BibliographyEntry(BaseModel):
    paperId: Optional[str] = None
    title: Optional[str] = None
    year: Optional[int] = None
    url: Optional[str] = None
    reason: Optional[str] = None


class ToolCall(BaseModel):
    step: int
    tool: str
    args: Dict[str, object]


class RolloutResponse(BaseModel):
    run_id: str
    task_id: str
    question: str
    final_memo: str
    reward: float
    reward_breakdown: Dict[str, float]
    bib: List[BibliographyEntry]
    notes: List[str]
    steps: int
    tool_calls: List[ToolCall]
    langsmith_run_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    run_id: str = Field(..., min_length=4)
    langsmith_run_id: Optional[str] = Field(None, min_length=4)
    sentiment: Literal["positive", "negative"]
    note: Optional[str] = Field(None, max_length=2000)
