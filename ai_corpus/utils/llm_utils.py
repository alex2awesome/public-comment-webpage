"""
Utilities for interacting with local Ollama models and OpenAI chat completions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Type, TypeVar

from pydantic import BaseModel

__all__ = [
    "BACKEND_LABELS",
    "BackendClients",
    "classify_prompt",
    "classify_prompt_async",
    "coerce_message_content",
    "init_backend",
    "parse_label",
    "request_structured_response",
    "request_structured_response_async",
]

BACKEND_LABELS = {"ollama": "Ollama", "openai": "OpenAI"}

StructuredModel = TypeVar("StructuredModel", bound=BaseModel)


@dataclass(frozen=True)
class BackendClients:
    sync: Any
    async_client: Any | None
    error: Optional[type[BaseException]]
    async_error: Optional[type[BaseException]]


def parse_label(response_text: str) -> int:
    import re

    matches = re.findall(r"[01]", response_text)
    if not matches:
        raise ValueError(f"Model returned an unexpected response:\n{response_text}")
    return int(matches[-1])


def coerce_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
        return "".join(parts)
    return ""


def init_backend(backend: str, host: Optional[str] = None) -> BackendClients:
    if backend == "ollama":
        try:
            import ollama
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "The ollama Python client is required. Install with: pip install ollama"
            ) from exc
        client = ollama.Client(host=host) if host else ollama.Client()
        return BackendClients(
            sync=client,
            async_client=None,
            error=getattr(ollama, "ResponseError", Exception),
            async_error=None,
        )
    if backend == "openai":
        try:
            from openai import AsyncOpenAI, OpenAI, OpenAIError
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "The openai Python client is required. Install with: pip install openai"
            ) from exc
        sync_client = OpenAI()
        async_client = AsyncOpenAI()
        return BackendClients(
            sync=sync_client,
            async_client=async_client,
            error=OpenAIError,
            async_error=OpenAIError,
        )
    raise ValueError(f"Unsupported backend '{backend}'. Choose 'ollama' or 'openai'.")


def classify_prompt(
    backend: str,
    client: Any,
    model: str,
    prompt_text: str,
) -> int:
    if backend == "ollama":
        response = client.generate(
            model=model,
            prompt=prompt_text,
            stream=False,
        )
        if hasattr(response, "response"):
            response_text = response.response  # type: ignore[attr-defined]
        elif isinstance(response, dict):
            response_text = response.get("response", "")
        else:
            response_text = str(response)
        return parse_label(response_text)

    if backend == "openai":
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=1,
        )
        content = ""
        if response.choices:
            message = response.choices[0].message
            if message and message.content is not None:
                content = coerce_message_content(message.content)
        return parse_label(content)

    raise ValueError(f"Unsupported backend '{backend}'")


async def classify_prompt_async(
    backend: str,
    async_client: Any,
    model: str,
    prompt_text: str,
) -> int:
    if backend != "openai":
        raise ValueError("Asynchronous classification is only supported for the OpenAI backend.")
    response = await async_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt_text}],
        max_tokens=1,
    )
    content = ""
    if response.choices:
        message = response.choices[0].message
        if message and message.content is not None:
            content = coerce_message_content(message.content)
    return parse_label(content)


def request_structured_response(
    client: Any,
    model: str,
    prompt_text: str,
    schema: Type[StructuredModel],
    *,
    max_output_tokens: int = 32,
    temperature: float | None = 0.0,
) -> StructuredModel:
    responses = getattr(client, "responses", None)
    if responses is None:
        raise ValueError("OpenAI client does not expose the responses API.")

    response = responses.parse(
        model=model,
        input=[{"role": "user", "content": prompt_text}],
        text_format=schema,
        max_output_tokens=max_output_tokens,
        **({"temperature": temperature} if temperature is not None else {}),
    )
    parsed = response.output_parsed
    if parsed is None:
        raise ValueError("Structured output missing parsed content.")
    if isinstance(parsed, schema):
        return parsed
    if isinstance(parsed, dict):
        try:
            return schema.model_validate(parsed)  # type: ignore[attr-defined]
        except AttributeError:
            return schema(**parsed)
    return schema.model_validate(parsed)  # type: ignore[attr-defined]


async def request_structured_response_async(
    async_client: Any,
    model: str,
    prompt_text: str,
    schema: Type[StructuredModel],
    *,
    max_output_tokens: int = 32,
    temperature: float | None = 0.0,
) -> StructuredModel:
    responses = getattr(async_client, "responses", None)
    if responses is None:
        raise ValueError("OpenAI async client does not expose the responses API.")

    response = await responses.parse(
        model=model,
        input=[{"role": "user", "content": prompt_text}],
        text_format=schema,
        max_output_tokens=max_output_tokens,
        **({"temperature": temperature} if temperature is not None else {}),
    )
    parsed = response.output_parsed
    if parsed is None:
        raise ValueError("Structured output missing parsed content.")
    if isinstance(parsed, schema):
        return parsed
    if isinstance(parsed, dict):
        try:
            return schema.model_validate(parsed)  # type: ignore[attr-defined]
        except AttributeError:
            return schema(**parsed)
    return schema.model_validate(parsed)  # type: ignore[attr-defined]
