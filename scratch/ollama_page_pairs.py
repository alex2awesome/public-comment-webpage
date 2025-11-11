#!/usr/bin/env python3
"""
Classify whether adjacent PDF pages belong to the same comment using an LLM backend.

Usage:
  python scratch/ollama_page_pairs.py path/to/comments.pdf [--backend ollama|openai] [--model ...]

By default the script runs a scaffolded text extraction pipeline:
1. pdfminer.six
2. PyMuPDF for any blank pages
3. Tesseract (LSTM OCR) for remaining blanks

The resulting per-page text is fed into the selected LLM to label each adjacent pair.
The model must respond with a single "0" (different comment) or "1" (same comment).
"""

from __future__ import annotations

import argparse
import asyncio
import textwrap
from pathlib import Path
import sys
from typing import Any, List, Optional, Literal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from ai_corpus.utils.llm_utils import (
    BACKEND_LABELS,
    classify_prompt,
    classify_prompt_async,
    init_backend,
    request_structured_response,
    request_structured_response_async,
)
from ai_corpus.utils.extraction import AVAILABLE_PARSERS, extract_pages

from pydantic import BaseModel

PROMPT_TEMPLATE = textwrap.dedent(
    """\
    I am trying to parse a large PDF with hundreds of public comments emails or letters all in one file. Some emails or letters stretch across multiple pages and I am trying to chunk them up.
    I will show you two consecutive pages in this file and ask you to judge whether these two pages are from the same public comment email or letter.
    Please pay attention to text that might be headers, footers, and signatures to identify the same comment email or letter. 
    Also pay attention to the names, company names and personal names to identify the same comment email or letter.
    
    Respond with the following JSON:
    {{
        "label": "1" if Page A and Page B ARE from the same comment email or letter, "0" if Page A ends a comment and Page B starts a new comment email or letter.
        "a_is_error": true if Page A is not a comment page or is incorrectly parsed, false otherwise.
        "b_is_error": true if Page B is not a comment page or is incorrectly parsed, false otherwise.
    }}
    No explanation, no extra characters.

    [Page A]
    {page_a}

    [Page B]
    {page_b}

    Answer:"""
)


class PagePairJudgment(BaseModel):
    """Structured output schema for OpenAI responses."""

    label: Literal["0", "1"]
    a_is_error: bool
    b_is_error: bool


PROMPT_STRUCTURED_OUTPUT = PagePairJudgment


def classify_pages(
    backend: str,
    client: Any,
    model: str,
    page_a: str,
    page_b: str,
) -> int:
    prompt_text = PROMPT_TEMPLATE.format(page_a=page_a, page_b=page_b)
    if backend == "openai":
        judgment = judge_pages_openai_structured(client, model, prompt_text)
        return int(judgment.label)
    return classify_prompt(backend, client, model, prompt_text)


async def classify_pages_async(
    backend: str,
    async_client: Any,
    model: str,
    page_a: str,
    page_b: str,
) -> int:
    prompt_text = PROMPT_TEMPLATE.format(page_a=page_a, page_b=page_b)
    if backend == "openai":
        judgment = await judge_pages_openai_structured_async(async_client, model, prompt_text)
        return int(judgment.label)
    return await classify_prompt_async(backend, async_client, model, prompt_text)


def judge_pages_openai_structured(client: Any, model: str, prompt_text: str, *, max_output_tokens: int = 32, temperature: float | None = None) -> PagePairJudgment:
    try:
        return request_structured_response(
            client,
            model,
            prompt_text,
            PROMPT_STRUCTURED_OUTPUT,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
    except Exception:
        if model.lower().startswith("gpt-5"):
            raise
        fallback_label = classify_prompt("openai", client, model, prompt_text)
        return PagePairJudgment(label=str(fallback_label), a_is_error=False, b_is_error=False)


async def judge_pages_openai_structured_async(
    async_client: Any,
    model: str,
    prompt_text: str,
) -> PagePairJudgment:
    try:
        return await request_structured_response_async(
            async_client,
            model,
            prompt_text,
            PROMPT_STRUCTURED_OUTPUT,
            max_output_tokens=32,
            temperature=None,
        )
    except Exception:
        if model.lower().startswith("gpt-5"):
            raise
        fallback_label = await classify_prompt_async("openai", async_client, model, prompt_text)
        return PagePairJudgment(label=str(fallback_label), a_is_error=False, b_is_error=False)


def _classify_pages_openai_structured(client: Any, model: str, prompt_text: str) -> int:
    judgment = judge_pages_openai_structured(client, model, prompt_text)
    return int(judgment.label)


async def _classify_pages_openai_structured_async(async_client: Any, model: str, prompt_text: str) -> int:
    judgment = await judge_pages_openai_structured_async(async_client, model, prompt_text)
    return int(judgment.label)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf_path", help="Path to the PDF file with comments.")
    parser.add_argument(
        "--backend",
        choices=("ollama", "openai"),
        default="ollama",
        help="LLM backend to use (default: ollama).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name. Defaults: llama3 for Ollama, gpt5-mini for OpenAI.",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Ollama host, e.g., http://localhost:11434 (defaults to local daemon).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N page pairs (useful for smoke testing).",
    )
    parser.add_argument(
        "--parser",
        choices=sorted((*AVAILABLE_PARSERS, "auto")),
        default="auto",
        help="PDF text extraction strategy (default: auto).",
    )
    parser.add_argument(
        "--async-openai",
        action="store_true",
        help="Use asyncio for OpenAI requests (ignored for other backends).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Maximum concurrent OpenAI requests when using --async-openai (default: 5).",
    )
    args = parser.parse_args()

    backend_label = BACKEND_LABELS.get(args.backend, args.backend)
    model = args.model or ("llama3" if args.backend == "ollama" else "gpt5-mini")

    try:
        pages, stats = extract_pages(args.pdf_path, args.parser)
    except (ImportError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if len(pages) < 2:
        raise SystemExit("Need at least two pages to compare.")

    host = args.host if args.backend == "ollama" else None
    clients = init_backend(args.backend, host)
    client = clients.sync
    backend_error = clients.error
    async_client = clients.async_client
    async_error = clients.async_error
    if args.backend == "openai" and args.host:
        print("Note: --host is ignored for OpenAI backend.")
    if args.async_openai and args.backend != "openai":
        print("--async-openai is only available for the OpenAI backend; ignoring flag.")
    async_enabled = args.backend == "openai" and args.async_openai
    if async_enabled and async_client is None:
        raise SystemExit("Async OpenAI client is unavailable; ensure openai>=1.0 is installed.")
    max_concurrency = max(1, args.concurrency if args.concurrency is not None else 5)

    print(
        f"Parsed {stats['total_pages']} pages from {args.pdf_path} using '{args.parser}' strategy"
    )
    print(f"Using backend {backend_label} with model '{model}'")
    if args.parser == "auto":
        print(
            "Extraction summary: "
            f"initial blanks={stats['initial_blanks']}, "
            f"filled by PyMuPDF={stats['pymupdf_filled']}, "
            f"filled by OCR={stats['ocr_filled']}, "
            f"remaining blanks={stats['remaining_blanks']}"
        )
        if stats["pymupdf_attempted"] and stats["pymupdf_failed"]:
            print("PyMuPDF fallback unavailable; install pymupdf to enable this stage.")
        if stats["ocr_attempted"] and stats["ocr_failed"]:
            print(
                "OCR fallback unavailable; install pytesseract, Pillow, and ensure Tesseract is on PATH."
            )
    elif stats["remaining_blanks"]:
        print(
            f"Warning: {stats['remaining_blanks']} pages are blank after '{args.parser}' extraction."
        )

    total_pairs = len(pages) - 1
    pair_limit = args.limit if args.limit is not None else total_pairs
    pair_limit = min(pair_limit, total_pairs)

    if async_enabled:
        async def classify_all_async() -> List[Any]:
            semaphore = asyncio.Semaphore(max_concurrency)

            async def worker(index: int) -> Any:
                page_a = pages[index]
                page_b = pages[index + 1]
                async with semaphore:
                    return await classify_pages_async(
                        args.backend, async_client, model, page_a, page_b
                    )

            tasks = [worker(index) for index in range(pair_limit)]
            return await asyncio.gather(*tasks, return_exceptions=True)

        results = asyncio.run(classify_all_async())
        for index, result in enumerate(results):
            if isinstance(result, Exception):
                if async_error and isinstance(result, async_error):
                    raise SystemExit(
                        f"{backend_label} async error while evaluating pages {index + 1} and {index + 2}: {result}"
                    ) from result
                raise SystemExit(
                    f"Unexpected async error while evaluating pages {index + 1} and {index + 2}: {result}"
                ) from result
            print(f"Page {index + 1} -> {index + 2}: {result}")
    else:
        for index in range(pair_limit):
            page_a = pages[index]
            page_b = pages[index + 1]
            if backend_error is not None:
                try:
                    label = classify_pages(args.backend, client, model, page_a, page_b)
                except backend_error as exc:
                    raise SystemExit(
                        f"{backend_label} error while evaluating pages {index + 1} and {index + 2}: {exc}"
                    ) from exc
            else:
                label = classify_pages(args.backend, client, model, page_a, page_b)

            print(f"Page {index + 1} -> {index + 2}: {label}")


if __name__ == "__main__":
    main()
