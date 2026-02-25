"""Claude Citations helper used to render the final memo with inline references."""

from __future__ import annotations

import json
import logging
import re
from textwrap import shorten
from typing import Any, Dict, List, Optional, Sequence, Tuple

from anthropic import Anthropic
from anthropic.types import (
    CitationCharLocation,
    CitationContentBlockLocation,
    CitationPageLocation,
    CitationsSearchResultLocation,
    CitationsWebSearchResultLocation,
    DocumentBlockParam,
    TextBlock,
)

logger = logging.getLogger(__name__)


class CitationGenerationError(RuntimeError):
    """Raised when the Claude Citations API cannot produce a memo."""


def _slugify(text: str, default: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return cleaned or default


def _format_summary(summary: Optional[Dict[str, Any]]) -> str:
    if not summary:
        return "No findings summary was captured."
    return json.dumps(summary, indent=2, ensure_ascii=False)


def _format_notes(notes: Sequence[str], limit: int = 6) -> str:
    cleaned = [note.strip() for note in notes if isinstance(note, str) and note.strip()]
    if not cleaned:
        return "No persistent notes were recorded."
    trimmed = cleaned[-limit:]
    return "\n".join(f"- {note}" for note in trimmed)


def _stringify_authors(raw_authors: Any) -> Optional[str]:
    if not raw_authors:
        return None
    names: List[str] = []
    for author in raw_authors:
        if isinstance(author, dict):
            name = author.get("name") or author.get("fullName")
        else:
            name = str(author)
        if name:
            names.append(name)
    if names:
        return ", ".join(names)
    return None


def _build_bibliography_payload(
    entry: Dict[str, Any],
    summary_entry: Optional[Dict[str, Any]],
) -> Tuple[Optional[str], Optional[str]]:
    sections: List[str] = []
    title = entry.get("title") or (summary_entry or {}).get("title")
    if title:
        sections.append(str(title).strip())
    venue = entry.get("venue")
    year = entry.get("year")
    if venue or year:
        sections.append(f"Publication: {venue or 'Unknown venue'} ({year or 'n.d.'})".strip())
    authors = _stringify_authors(entry.get("authors") or (summary_entry or {}).get("authors"))
    if authors:
        sections.append(f"Authors: {authors}")
    abstract = entry.get("abstract")
    if abstract and isinstance(abstract, str):
        sections.append(f"Abstract: {abstract.strip()}")
    summary_reason = (summary_entry or {}).get("reason_chosen")
    reason = entry.get("reason") or summary_reason
    if reason:
        sections.append(f"Why it matters: {reason.strip()}")
    url = entry.get("url") or (summary_entry or {}).get("url")
    if url:
        sections.append(f"URL: {url}")
    if not sections:
        return None, authors
    return "\n\n".join(sections).strip(), authors


def _select_documents(
    bibliography: List[Dict[str, Any]],
    summary: Optional[Dict[str, Any]],
    question: str,
    notes: Sequence[str],
    max_docs: Optional[int] = 24,
) -> Tuple[List[DocumentBlockParam], List[Dict[str, Any]]]:
    bib_by_id: Dict[str, Dict[str, Any]] = {}
    for entry in bibliography:
        paper_id = entry.get("paperId")
        if paper_id:
            bib_by_id[paper_id] = entry

    summary_articles = (summary or {}).get("top_articles") or []
    summary_arguments = (summary or {}).get("top_arguments") or []
    summary_recommendations = (summary or {}).get("top_recommendations") or []
    summary_people = (summary or {}).get("top_people") or []

    def unique_doc_id(base: str, used: set[str]) -> str:
        slug = _slugify(base, base)
        if slug not in used:
            used.add(slug)
            return slug
        idx = 2
        while f"{slug}-{idx}" in used:
            idx += 1
        new_id = f"{slug}-{idx}"
        used.add(new_id)
        return new_id

    document_blocks: List[DocumentBlockParam] = []
    document_metadata: List[Dict[str, Any]] = []
    used_ids: set[str] = set()

    def add_document(
        *,
        doc_id: str,
        title: str,
        text: str,
        metadata: Dict[str, Any],
    ) -> None:
        text_value = (text or "").strip()
        if not text_value:
            return
        if max_docs is not None and len(document_blocks) >= max_docs:
            return
        block: DocumentBlockParam = {
            "type": "document",
            "title": title,
            "citations": {"enabled": True},
            "source": {
                "type": "text",
                "media_type": "text/plain",
                "data": text_value[:8000],
            },
        }
        document_blocks.append(block)
        document_metadata.append(
            {
                "document_index": len(document_blocks) - 1,
                "document_id": doc_id,
                **metadata,
            }
        )

    ordered_entries: List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]] = []
    seen_ids: set[str] = set()
    for article in summary_articles:
        paper_id = article.get("paperId")
        bib_entry = bib_by_id.get(paper_id or "")
        if bib_entry and paper_id not in seen_ids:
            ordered_entries.append((bib_entry, article))
            seen_ids.add(paper_id)
    for entry in bibliography:
        paper_id = entry.get("paperId")
        if paper_id and paper_id in seen_ids:
            continue
        ordered_entries.append((entry, None))
        if paper_id:
            seen_ids.add(paper_id)

    for entry, summary_entry in ordered_entries:
        paper_id = entry.get("paperId") or (summary_entry or {}).get("paperId")
        title = entry.get("title") or (summary_entry or {}).get("title") or "Untitled Source"
        doc_id = unique_doc_id(paper_id or title or f"doc-{len(document_blocks)+1}", used_ids)
        doc_text, authors_text = _build_bibliography_payload(entry, summary_entry)
        reason_text = entry.get("reason") or (summary_entry or {}).get("reason_chosen")
        text_preview = shorten(doc_text, width=200, placeholder="…") if doc_text else None
        add_document(
            doc_id=doc_id,
            title=title,
            text=doc_text or (reason_text or ""),
            metadata={
                "paper_id": paper_id,
                "title": title,
                "url": entry.get("url") or (summary_entry or {}).get("url"),
                "year": entry.get("year"),
                "reason": reason_text,
                "text_preview": text_preview,
                "authors": authors_text,
                "kind": "bibliography",
            },
        )

    # Add top articles that were not present in the bibliography cache.
    for article in summary_articles:
        paper_id = article.get("paperId")
        if paper_id and paper_id in seen_ids:
            continue
        text_parts: List[str] = []
        if article.get("title"):
            text_parts.append(article["title"])
        if article.get("reason_chosen"):
            text_parts.append(f"Why it matters: {article['reason_chosen']}")
        if article.get("url"):
            text_parts.append(f"URL: {article['url']}")
        authors_text = None
        raw_authors = article.get("authors")
        if raw_authors:
            if isinstance(raw_authors, list):
                authors_text = ", ".join(str(name) for name in raw_authors if name)
            else:
                authors_text = str(raw_authors)
        doc_text = "\n\n".join(text_parts).strip()
        if not doc_text:
            continue
        doc_id = unique_doc_id(paper_id or article.get("title") or "article", used_ids)
        add_document(
            doc_id=doc_id,
            title=article.get("title") or f"Source {len(document_blocks) + 1}",
            text=doc_text,
            metadata={
                "paper_id": paper_id,
                "title": article.get("title"),
                "url": article.get("url"),
                "reason": article.get("reason_chosen"),
                "authors": authors_text,
                "kind": "summary_article",
            },
        )

    for idx, argument in enumerate(summary_arguments, start=1):
        raw_text = str(argument or "").strip()
        if not raw_text:
            continue
        doc_id = unique_doc_id(f"argument-{idx}", used_ids)
        add_document(
            doc_id=doc_id,
            title=f"Argument #{idx}",
            text=f"Argument: {raw_text}",
            metadata={
                "kind": "argument",
                "argument_index": idx,
                "title": f"Argument #{idx}",
                "reason": raw_text,
            },
        )

    for idx, recommendation in enumerate(summary_recommendations, start=1):
        raw_text = str(recommendation or "").strip()
        if not raw_text:
            continue
        doc_id = unique_doc_id(f"recommendation-{idx}", used_ids)
        add_document(
            doc_id=doc_id,
            title=f"Recommendation #{idx}",
            text=f"Recommendation: {raw_text}",
            metadata={
                "kind": "recommendation",
                "recommendation_index": idx,
                "title": f"Recommendation #{idx}",
                "reason": raw_text,
            },
        )

    for idx, person in enumerate(summary_people, start=1):
        text = str(person or "").strip()
        if not text:
            continue
        doc_id = unique_doc_id(f"person-{idx}", used_ids)
        add_document(
            doc_id=doc_id,
            title=f"Top Person #{idx}",
            text=text,
            metadata={
                "kind": "person",
                "person_index": idx,
                "title": text,
            },
        )

    if not document_blocks:
        fallback_notes = _format_notes(notes)
        add_document(
            doc_id=unique_doc_id("session-context", used_ids),
            title="Session Context",
            text=f"Question: {question.strip()}\n\nNotes:\n{fallback_notes}",
            metadata={"kind": "context"},
        )

    return document_blocks, document_metadata


def _normalize_citation(
    citation: Any,
    document_catalog: List[Dict[str, Any]],
) -> Dict[str, Any]:
    doc_meta: Optional[Dict[str, Any]] = None
    doc_index = getattr(citation, "document_index", None)
    if doc_index is not None and 0 <= doc_index < len(document_catalog):
        doc_meta = document_catalog[doc_index]
    payload = {
        "type": getattr(citation, "type", type(citation).__name__),
        "document_index": doc_index,
        "document_id": doc_meta.get("document_id") if doc_meta else None,
        "document_title": getattr(citation, "document_title", None) or (doc_meta or {}).get("title"),
        "document_url": (doc_meta or {}).get("url"),
        "paper_id": (doc_meta or {}).get("paper_id"),
        "cited_text": getattr(citation, "cited_text", None),
    }
    if isinstance(citation, CitationCharLocation):
        payload["start_char_index"] = citation.start_char_index
        payload["end_char_index"] = citation.end_char_index
    elif isinstance(citation, CitationPageLocation):
        payload["start_page_number"] = citation.start_page_number
        payload["end_page_number"] = citation.end_page_number
    elif isinstance(citation, CitationContentBlockLocation):
        payload["start_block_index"] = citation.start_block_index
        payload["end_block_index"] = citation.end_block_index
    elif isinstance(citation, (CitationsSearchResultLocation, CitationsWebSearchResultLocation)):
        payload["url"] = getattr(citation, "url", None)
    return payload


def render_memo_with_claude(
    *,
    question: str,
    summary: Optional[Dict[str, Any]],
    notes: Sequence[str],
    bibliography: List[Dict[str, Any]],
    api_key: str,
    model_name: str,
    directives: Optional[str] = None,
    max_tokens: int = 2400,
) -> Dict[str, Any]:
    """Call Claude's Citations API to render the final memo and inline citation metadata."""
    if not api_key:
        raise CitationGenerationError("Claude API key is missing.")

    document_blocks, document_metadata = _select_documents(bibliography, summary, question, notes)
    kind_counts: Dict[str, int] = {}
    for meta in document_metadata:
        kind = meta.get("kind") or "unknown"
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
    logger.info(
        "Claude citation document catalog | total=%s | kind_breakdown=%s",
        len(document_metadata),
        kind_counts,
    )

    summary_blob = _format_summary(summary)
    notes_blob = _format_notes(notes)
    directives_text = directives.strip() if directives else "Follow the memo template described below."

    intro_text = (
        "You are a senior policy analyst. Review the attached source documents and craft a formal memo "
        "answering the question below. Cite evidence inline using the Citations API output."
    )
    requirements = (
        "Memo requirements:\n"
        "- Start with a salutation and short executive summary paragraph.\n"
        "- Provide 3 numbered recommendations or findings with supporting detail.\n"
        "- Reference experts or stakeholders (top people) when relevant.\n"
        "- Close with concrete next steps.\n"
        "- Use a professional tone suitable for an RFI response.\n"
        "- Only cite the provided documents; do not introduce external facts.\n"
        "- Aggressively cite every substantive sentence or clause—repeat citations as needed so almost no text remains uncited and readers can trace claims back to arguments, people, or sources.\n"
        "- When you cite, capture the full sentence or clause (not just a short fragment) so the cited_text spans as much relevant context as possible.\n"
        "- When you mention specific people or stakeholders, cite the matching Top Person document so the reader can inspect their biography context.\n"
        "- Use the provided recommendations as explicit anchors (e.g., Recommendation #1) and cite those recommendation documents whenever they appear."
    )
    context_blob = (
        f"Research question:\n{question.strip()}\n\n"
        f"Top recommendations / arguments / articles / people (JSON):\n{summary_blob}\n\n"
        f"Researcher notes:\n{notes_blob}\n\n"
        f"Additional directives:\n{directives_text}\n"
    )

    user_content: List[Dict[str, Any]] = [
        {"type": "text", "text": intro_text},
        {"type": "text", "text": requirements},
    ]
    user_content.extend(document_blocks)
    user_content.append({"type": "text", "text": context_blob})

    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        temperature=0.2,
        messages=[
            {
                "role": "user",
                "content": user_content,
            }
        ],
    )
    try:
        raw_response_json = response.model_dump_json(indent=2)
    except AttributeError:  # pragma: no cover - fallback for unexpected SDK changes
        try:
            raw_response_json = json.dumps(response.model_dump(), indent=2)
        except Exception:
            raw_response_json = repr(response)
    logger.info("Claude raw response:\n%s", raw_response_json)
    memo_blocks: List[Dict[str, Any]] = []
    memo_text_parts: List[str] = []
    for block in response.content:
        if isinstance(block, TextBlock):
            memo_text_parts.append(block.text)
            citations = []
            if block.citations:
                citations = [_normalize_citation(citation, document_metadata) for citation in block.citations]
            memo_blocks.append({"text": block.text, "citations": citations})
    memo_text = "".join(memo_text_parts).strip()
    if not memo_text:
        raise CitationGenerationError("Claude returned an empty memo response.")

    total_citations = sum(len(block["citations"]) for block in memo_blocks)
    logger.info(
        "Claude citations memo generated | chars=%s | blocks=%s | documents=%s | citations=%s",
        len(memo_text),
        len(memo_blocks),
        len(document_metadata),
        total_citations,
    )
    logger.info("Claude memo text:\n%s", memo_text)

    return {
        "text": memo_text,
        "blocks": memo_blocks,
        "documents": document_metadata,
    }
