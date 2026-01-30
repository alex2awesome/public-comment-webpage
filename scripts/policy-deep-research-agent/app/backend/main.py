"""FastAPI application exposing rollout + feedback endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from langsmith import Client as LangSmithClient
from sse_starlette.sse import EventSourceResponse

from policy_src.policy_agent_lc.rollout import run_one_rollout

from .config import Settings, get_settings
from .schemas import FeedbackRequest, RolloutRequest, RolloutResponse

logger = logging.getLogger(__name__)
app = FastAPI(title="Policy Rollout API", version="0.1.0")

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


def attach_langsmith_run_id(episode: Dict[str, Any], project_name: Optional[str]) -> None:
    external_run_id = episode.get("run_id")
    if not project_name or not external_run_id:
        return
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
                episode["langsmith_run_id"] = str(getattr(run, "id", "") or getattr(run, "run_id", ""))
                break
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Unable to map LangSmith run id", exc_info=exc)


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

    episode = await run_in_threadpool(
        run_one_rollout,
        settings.model_name,
        0,
        settings.use_cached,
        payload.max_steps,
        settings.cache_path,
        temperature,
        payload.enable_bibliography,
        question,
        api_key,
    )
    attach_langsmith_run_id(episode, settings.langsmith_project)

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
        loop.call_soon_threadsafe(queue.put_nowait, event)

    async def rollout_task() -> None:
        try:
            rollout_temperature = temperature if temperature is not None else settings.temperature
            episode = await run_in_threadpool(
                run_one_rollout,
                settings.model_name,
                0,
                settings.use_cached,
                max_steps,
                settings.cache_path,
                rollout_temperature,
                enable_bibliography,
                question_text,
                api_key,
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
            event = await queue.get()
            if event is None:
                break
            yield {
                "event": event.get("type", "message"),
                "data": json.dumps(event),
            }

    asyncio.create_task(rollout_task())
    return EventSourceResponse(event_generator())


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
    try:
        client.create_feedback(
            run_id=payload.run_id,
            key="user_sentiment",
            score=score,
            value={"sentiment": payload.sentiment},
            comment=comment,
            source="policy-rollout-app",
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Failed to send LangSmith feedback", exc_info=exc)
        raise HTTPException(status_code=502, detail="Failed to send feedback to LangSmith.") from exc
