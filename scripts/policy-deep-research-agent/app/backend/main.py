"""FastAPI application exposing rollout + feedback endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.concurrency import run_in_threadpool
from langsmith import Client as LangSmithClient
from langchain_core.messages import HumanMessage, SystemMessage

from policy_src.policy_research_core.cache_db import CacheDB
from policy_src.policy_research_core.semanticscholar_api import SemanticScholarClient

from .config import Settings, get_settings
from .schemas import (
    FeedbackRequest,
    FindingsSummary,
    ResubmitRequest,
    ResubmitResponse,
    RolloutRequest,
    RolloutResponse,
)

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_LEVEL = os.getenv("POLICY_BACKEND_LOG_LEVEL", "INFO").upper()
root_logger = logging.getLogger()
if not root_logger.handlers:
    logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, LOG_LEVEL, logging.INFO))
else:
    root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

logger = logging.getLogger(__name__)
app = FastAPI(title="Policy Rollout API", version="0.1.0")
S2_CLIENT = SemanticScholarClient()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=("*",),
    allow_headers=("*",),
    expose_headers=("*",),
)


@app.get("/", tags=["meta"])
def root() -> Dict[str, str]:
    return {"message": "Policy agent is awake and ready to research 🚀"}


@app.get("/healthz", tags=["meta"])
def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


def resolve_api_key(authorization: Optional[str], settings: Settings) -> Optional[str]:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization.split(" ", 1)[1].strip() or None
    return settings.openai_api_key


def configure_langsmith_env(settings: Settings) -> None:
    if settings.langsmith_project:
        os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    if settings.langchain_api_key:
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    if settings.langchain_endpoint:
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
    if settings.langchain_tracing_v2:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"


def _find_langsmith_run_id(external_run_id: Optional[str], project_name: Optional[str]) -> Optional[str]:
    if not project_name or not external_run_id:
        return None
    try:
        client = LangSmithClient()
        runs = client.list_runs(
            project_name=project_name,
            order="desc",
            limit=20,
        )
        for run in runs:
            metadata = getattr(run, "metadata", {}) or {}
            if metadata.get("episode_run_id") == external_run_id:
                langsmith_id = getattr(run, "id", None) or getattr(run, "run_id", None)
                return str(langsmith_id) if langsmith_id else None
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Unable to map LangSmith run id", exc_info=exc)
    return None


def attach_langsmith_run_id(episode: Dict[str, Any], project_name: Optional[str]) -> None:
    langsmith_run_id = _find_langsmith_run_id(episode.get("run_id"), project_name)
    if langsmith_run_id:
        episode["langsmith_run_id"] = langsmith_run_id


def _run_rollout(*args, **kwargs):
    from policy_src.policy_agent_lc.rollout import run_one_rollout

    return run_one_rollout(*args, **kwargs)


def _semantic_scholar_paper_lookup(paper_id: str, settings: Settings) -> Dict[str, Any]:
    fields = "paperId,title,year,venue,url,abstract,citationCount,authors,openAccessPdf"
    db = CacheDB(settings.cache_path)
    payload = db.cached_or_fetch_semantic_scholar_paper(
        use_cached=settings.use_cached,
        paper_id=paper_id,
        fields=fields,
        fetch_fn=lambda: S2_CLIENT.paper_details(paper_id, fields=fields),
    )
    raw = payload.get("raw") or {}
    if "authors" not in payload and "authors" in raw:
        payload["authors"] = raw["authors"]
    if not payload.get("url"):
        payload["url"] = raw.get("url") or raw.get("openAccessPdf", {}).get("url")
    return payload


def _summary_to_json(summary: FindingsSummary) -> str:
    payload = summary.model_dump(mode="python", by_alias=True)
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _notes_block(notes: List[str]) -> str:
    cleaned = [note.strip() for note in notes if isinstance(note, str) and note.strip()]
    if not cleaned:
        return "No persistent researcher notes were recorded."
    return "\n".join(f"- {note}" for note in cleaned)


def _tool_events_block(events: List[Dict[str, Any]], limit: int = 12) -> str:
    if not events:
        return "No recent tool call context was provided."
    lines: List[str] = []
    for event in events[:limit]:
        step = event.get("step")
        tool = event.get("tool") or event.get("type")
        result = event.get("result")
        message = event.get("message")
        if isinstance(result, dict):
            snippet = json.dumps(result)
        else:
            snippet = str(result or message or event.get("args") or "")
        snippet = snippet.replace("\n", " ")
        if len(snippet) > 280:
            snippet = f"{snippet[:277]}..."
        prefix = f"Step {step}: {tool}" if step is not None else f"{tool}"
        lines.append(f"- {prefix} -> {snippet}")
    return "\n".join(lines)


def _render_memo_from_summary(
    question: str,
    summary: FindingsSummary,
    notes: List[str],
    directives: Optional[str],
    model_name: str,
    temperature: float,
    api_key: str,
    tool_events: List[Dict[str, Any]],
    prior_memo: Optional[str],
) -> str:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        timeout=180,
        max_retries=2,
        api_key=api_key,
    )

    directives_text = directives.strip() if directives else "Follow the agent's default memo style."
    summary_blob = _summary_to_json(summary)
    notes_blob = _notes_block(notes)
    events_blob = _tool_events_block(tool_events)
    prior_memo_section = ""
    if prior_memo:
        prior_memo_section = f"Previous memo submitted:\n---\n{prior_memo.strip()}\n---\n"
    human_prompt = (
        "Draft a formal response to a policy Request for Information (RFI) using the updated findings summary and prior agent context.\n\n"
        f"Research question:\n{question.strip()}\n\n"
        f"Finding summary JSON (preserve article ordering when possible):\n{summary_blob}\n\n"
        f"Researcher notes:\n{notes_blob}\n\n"
        f"Recent tool interactions:\n{events_blob}\n\n"
        f"{prior_memo_section}"
        f"Additional directives from the user:\n{directives_text}\n\n"
        "Requirements:\n"
        "- Treat this as a polished RFI response with salutation/introduction, numbered recommendations, and clear calls to action.\n"
        "- Weave the listed top arguments into the memo structure and cite articles inline using <cite id=\"PAPER_ID\">Title</cite>.\n"
        "- Align the memo’s numbered recommendations with the `top recommendations` provided in the summary whenever possible.\n"
        "- Cite each referenced article using <cite id=\"PAPER_ID\">Title</cite> with the provided paperId when available.\n"
        "- Mention recommended contributors (top people) in a final section if the list is not empty.\n"
        "- Reference previous memo learnings if helpful, but the new memo should supersede it.\n"
        "- Keep tone professional and targeted for policy stakeholders."
    )
    response = llm.invoke(
        [
            SystemMessage(
                content="You are an expert policy researcher who produces clear, defensible memos grounded in cited evidence."
            ),
            HumanMessage(content=human_prompt),
        ]
    )
    if isinstance(response.content, str):
        return response.content
    if isinstance(response.content, list):
        return "\n".join(part.get("text", "") for part in response.content if isinstance(part, dict))
    return str(response.content)


@app.post("/rollouts", response_model=RolloutResponse)
async def create_rollout(
    payload: RolloutRequest,
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
    settings: Settings = Depends(get_settings),
) -> RolloutResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Question cannot be empty.")
    api_key = resolve_api_key(authorization, settings)
    if not api_key:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing OpenAI API key.")

    configure_langsmith_env(settings)

    temperature = payload.temperature if payload.temperature is not None else settings.temperature

    logger.info(
        "Rollout request accepted | steps=%s | temperature=%s | bibliography=%s",
        payload.max_steps,
        temperature,
        payload.enable_bibliography,
    )

    episode = await run_in_threadpool(
        _run_rollout,
        settings.model_name,
        0,
        settings.use_cached,
        payload.max_steps,
        settings.cache_path,
        temperature,
        payload.enable_bibliography,
        question,
        api_key,
        settings.claude_api_key,
        settings.claude_model_name,
        settings.use_claude_submit_tool,
    )
    attach_langsmith_run_id(episode, settings.langsmith_project)
    logger.info("Rollout complete | run_id=%s | task_id=%s | reward=%s", episode["run_id"], episode["task_id"], episode.get("reward"))

    return RolloutResponse(
        run_id=episode["run_id"],
        task_id=episode["task_id"],
        question=episode["question"],
        final_memo=episode.get("final_memo", ""),
        reward=episode.get("reward", 0.0),
        reward_breakdown=episode.get("reward_breakdown", {}),
        bib=episode.get("bib", []),
        notes=episode.get("notes", []),
        steps=episode.get("steps", 0),
        tool_calls=episode.get("tool_calls", []),
        langsmith_run_id=episode.get("langsmith_run_id"),
        summary=episode.get("summary"),
        final_memo_blocks=episode.get("final_memo_blocks"),
        source_documents=episode.get("source_documents", []),
    )


@app.get("/rollouts/stream")
async def stream_rollout(
    question: str,
    max_steps: int = Query(12, ge=4, le=40),
    enable_bibliography: bool = Query(True),
    temperature: Optional[float] = Query(None, ge=0.0, le=2.0),
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
    settings: Settings = Depends(get_settings),
):
    question_text = question.strip()
    if not question_text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Question cannot be empty.")
    api_key = resolve_api_key(authorization, settings)
    if not api_key:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing OpenAI API key.")

    configure_langsmith_env(settings)
    queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def emit_event(event: Dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type in {"run_started", "complete", "run_completed", "error"}:
            logger.info("Rollout event | type=%s | payload_keys=%s", event_type, list(event.keys()))
        loop.call_soon_threadsafe(queue.put_nowait, event)

    async def rollout_task() -> None:
        try:
            rollout_temperature = temperature if temperature is not None else settings.temperature
            episode = await run_in_threadpool(
                _run_rollout,
                settings.model_name,
                0,
                settings.use_cached,
                max_steps,
                settings.cache_path,
                rollout_temperature,
                enable_bibliography,
                question_text,
                api_key,
                settings.claude_api_key,
                settings.claude_model_name,
                settings.use_claude_submit_tool,
                emit_event,
            )
            attach_langsmith_run_id(episode, settings.langsmith_project)
            emit_event({"type": "complete", "episode": episode})
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Rollout stream failed", exc_info=exc)
            emit_event({"type": "error", "message": str(exc)})
        finally:
            await queue.put(None)

    async def event_generator():
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=10.0)
            except asyncio.TimeoutError:
                heartbeat = {"type": "heartbeat", "ts": time.time()}
                logger.debug("Emitting heartbeat keepalive")
                yield f"event: heartbeat\ndata: {json.dumps(heartbeat)}\n\n"
                continue
            if event is None:
                break
            event_name = event.get("type", "message")
            payload = json.dumps(event)
            # Server-Sent Events expect newline-delimited fields.
            yield f"event: {event_name}\ndata: {payload}\n\n"

    asyncio.create_task(rollout_task())
    logger.info(
        "Streaming rollout started | question_len=%s | steps=%s | bibliography=%s",
        len(question_text),
        max_steps,
        enable_bibliography,
    )
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.post("/rollouts/resubmit", response_model=ResubmitResponse)
async def regenerate_memo_from_summary(
    payload: ResubmitRequest,
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
    settings: Settings = Depends(get_settings),
) -> ResubmitResponse:
    api_key = resolve_api_key(authorization, settings)
    if not api_key:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing OpenAI API key.")
    if (
        not payload.summary.top_arguments
        and not payload.summary.top_articles
        and not payload.summary.top_recommendations
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide at least one top argument, recommendation, or article before regenerating the memo.",
        )
    configure_langsmith_env(settings)
    try:
        memo = await run_in_threadpool(
            _render_memo_from_summary,
            payload.question,
            payload.summary,
            payload.notes,
            payload.directives,
            settings.model_name,
            settings.temperature,
            api_key,
            payload.tool_events,
            payload.prior_memo,
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Memo regeneration failed", exc_info=exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to regenerate memo from summary. Please try again.",
        ) from exc
    return ResubmitResponse(memo=memo, memo_blocks=None, source_documents=[])


@app.get("/semantic-scholar/paper/{paper_id}")
async def get_semantic_scholar_paper(
    paper_id: str,
    settings: Settings = Depends(get_settings),
):
    try:
        payload = await run_in_threadpool(_semantic_scholar_paper_lookup, paper_id, settings)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Semantic Scholar paper lookup failed | paper_id=%s", paper_id)
        raise HTTPException(status_code=502, detail="Unable to fetch Semantic Scholar paper metadata.") from exc
    return payload


@app.get("/semantic-scholar/authors")
async def search_semantic_scholar_authors(
    query: str = Query(..., min_length=2),
    limit: int = Query(5, ge=1, le=10),
):
    try:
        response = await run_in_threadpool(
            S2_CLIENT.author_search,
            query,
            "name,authorId,url,paperCount,hIndex",
            limit,
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Semantic Scholar author search failed | query=%s", query)
        raise HTTPException(status_code=502, detail="Unable to search Semantic Scholar authors.") from exc
    return response


@app.post("/feedback", status_code=status.HTTP_204_NO_CONTENT)
async def submit_feedback(
    payload: FeedbackRequest,
    settings: Settings = Depends(get_settings),
):
    if not settings.langsmith_project:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="LangSmith project not configured.")

    configure_langsmith_env(settings)
    client = LangSmithClient()

    score = 1.0 if payload.sentiment == "positive" else 0.0
    comment = payload.note.strip() if payload.note else None
    target_run_id = payload.langsmith_run_id or _find_langsmith_run_id(payload.run_id, settings.langsmith_project)
    if not target_run_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Unable to locate LangSmith run for this rollout. Refresh the page and try again.",
        )
    try:
        client.create_feedback(
            run_id=target_run_id,
            key="user_sentiment",
            score=score,
            value={"sentiment": payload.sentiment},
            comment=comment,
            source="policy-rollout-app",
        )
        logger.info("Feedback submitted | run_id=%s | sentiment=%s", target_run_id, payload.sentiment)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Failed to send LangSmith feedback", exc_info=exc)
        raise HTTPException(status_code=502, detail="Failed to send feedback to LangSmith.") from exc
