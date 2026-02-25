"""Pydantic request/response objects for the FastAPI endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

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
    abstract: Optional[str] = None


class SummaryArticle(BaseModel):
    paperId: Optional[str] = Field(default=None, alias="paper_id")
    title: Optional[str] = None
    url: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    reason_chosen: Optional[str] = None

    class Config:
        populate_by_name = True


class FindingsSummary(BaseModel):
    top_arguments: List[str] = Field(default_factory=list, alias="top_arguments")
    top_recommendations: List[str] = Field(default_factory=list, alias="top_recommendations")
    top_articles: List[SummaryArticle] = Field(default_factory=list, alias="top_articles")
    top_people: List[str] = Field(default_factory=list, alias="top_people")

    class Config:
        populate_by_name = True


class ToolCall(BaseModel):
    step: int
    tool: str
    args: Dict[str, object]


class MemoCitation(BaseModel):
    type: Optional[str] = None
    document_index: Optional[int] = None
    document_id: Optional[str] = None
    document_title: Optional[str] = None
    document_url: Optional[str] = None
    paper_id: Optional[str] = None
    cited_text: Optional[str] = None
    start_char_index: Optional[int] = None
    end_char_index: Optional[int] = None
    start_page_number: Optional[int] = None
    end_page_number: Optional[int] = None
    start_block_index: Optional[int] = None
    end_block_index: Optional[int] = None


class MemoBlock(BaseModel):
    text: str
    citations: List[MemoCitation] = Field(default_factory=list)


class SourceDocument(BaseModel):
    document_index: int
    document_id: Optional[str] = None
    paper_id: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None
    year: Optional[int] = None
    reason: Optional[str] = None
    authors: Optional[str] = None
    text_preview: Optional[str] = None
    kind: Optional[str] = None
    argument_index: Optional[int] = None
    person_index: Optional[int] = None
    recommendation_index: Optional[int] = None


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
    summary: Optional[FindingsSummary] = None
    final_memo_blocks: Optional[List[MemoBlock]] = Field(default=None, alias="final_memo_blocks")
    source_documents: List[SourceDocument] = Field(default_factory=list, alias="source_documents")


class FeedbackRequest(BaseModel):
    run_id: str = Field(..., min_length=4)
    langsmith_run_id: Optional[str] = Field(None, min_length=4)
    sentiment: Literal["positive", "negative"]
    note: Optional[str] = Field(None, max_length=2000)


class ResubmitRequest(BaseModel):
    run_id: str = Field(..., min_length=4)
    question: str = Field(..., min_length=5)
    summary: FindingsSummary
    notes: List[str] = Field(default_factory=list)
    directives: Optional[str] = Field(
        default=None, description="Optional editing guidance or structure instructions supplied by the user."
    )
    tool_events: List[Dict[str, Any]] = Field(default_factory=list, description="Recent tool call/result payloads.")
    prior_memo: Optional[str] = Field(default=None, description="Most recent memo text before regeneration.")


class ResubmitResponse(BaseModel):
    memo: str
    memo_blocks: Optional[List[MemoBlock]] = None
    source_documents: List[SourceDocument] = Field(default_factory=list)
