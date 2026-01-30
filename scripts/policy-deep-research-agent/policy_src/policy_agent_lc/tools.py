"""LangChain tool definitions wired to the legacy OpenEnv helpers."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from langchain_core.tools import tool

from policy_src.policy_research_core.cache_db import CacheDB
from policy_src.policy_research_core.semanticscholar_api import (
    SemanticScholarClient,
    SemanticScholarRateLimitError,
)

from .session import ResearchSession


def build_tools(
    session: ResearchSession,
    enable_bibliography: bool = True,
    event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[Any]:
    """Return LangChain tool objects bound to the provided session."""
    db = CacheDB(session.cache_path)
    s2 = SemanticScholarClient()

    def _bump_step(tool_name: str, args: Dict[str, Any]) -> None:
        session.step_count += 1
        session.tool_calls.append({"step": session.step_count, "tool": tool_name, "args": args})
        _emit_event("tool_call", {"step": session.step_count, "tool": tool_name, "args": args})

    def _emit_event(event_type: str, payload: Dict[str, Any]) -> None:
        if not event_callback:
            return
        try:
            event_callback({"type": event_type, **payload})
        except Exception:
            pass

    @tool
    def search_semantic_scholar(query: str, top_k: int = 10, year: Optional[str] = None) -> Dict[str, Any]:
        """Search Semantic Scholar. Keep queries short (3–6 terms)."""
        _bump_step("search_semantic_scholar", {"query": query, "top_k": top_k, "year": year})
        fields = "paperId,title,year,venue,url,abstract,citationCount"
        desired = max(int(top_k or 0), 1)
        fetch_limit = min(100, max(20, desired))
        try:
            result = db.cached_or_fetch_semantic_scholar_search(
                use_cached=session.use_cached,
                query=query,
                params={"year": year, "fields": fields, "top_k": desired},
                fetch_fn=lambda: s2.bulk_search(
                    query, fields=fields, year=year, limit=fetch_limit, max_items=desired
                ),
                top_k=desired,
            )
        except SemanticScholarRateLimitError as exc:
            session.last_tool_result = {
                "type": "search_results",
                "error": (
                    "Semantic Scholar search is busy after waiting "
                    f"{exc.waited_seconds:.1f}s. Review existing papers, take notes, or draft your memo."
                ),
                "waited_seconds": exc.waited_seconds,
            }
            return session.last_tool_result
        except RuntimeError as exc:
            session.last_tool_result = {
                "type": "search_results",
                "error": str(exc),
            }
            return session.last_tool_result
        except Exception as exc:
            session.last_tool_result = {
                "type": "search_results",
                "error": f"Semantic Scholar search failed: {exc}",
            }
            return session.last_tool_result
        session.last_tool_result = {"type": "search_results", **result}
        _emit_event(
            "tool_result",
            {"step": session.step_count, "tool": "search_semantic_scholar", "result": session.last_tool_result},
        )
        return session.last_tool_result

    @tool
    def fetch_paper(paper_id: str, reason: str = "") -> Dict[str, Any]:
        """Fetch metadata for a paper and add it to the bibliography."""
        _bump_step("fetch_paper", {"paper_id": paper_id, "reason": reason})
        fields = "paperId,title,year,venue,url,abstract,citationCount,authors,openAccessPdf"
        try:
            payload = db.cached_or_fetch_semantic_scholar_paper(
                use_cached=session.use_cached,
                paper_id=paper_id,
                fields=fields,
                fetch_fn=lambda: s2.paper_details(paper_id, fields=fields),
            )
        except SemanticScholarRateLimitError as exc:
            session.last_tool_result = {
                "type": "paper_details",
                "error": (
                    "Semantic Scholar fetch is busy after waiting "
                    f"{exc.waited_seconds:.1f}s. Reuse bibliography entries, take notes, or submit."
                ),
                "waited_seconds": exc.waited_seconds,
            }
            return session.last_tool_result
        except RuntimeError as exc:
            session.last_tool_result = {
                "type": "paper_details",
                "error": str(exc),
            }
            return session.last_tool_result
        except Exception as exc:
            session.last_tool_result = {
                "type": "paper_details",
                "error": f"Semantic Scholar fetch failed: {exc}",
            }
            return session.last_tool_result
        entry = {
            "paperId": payload.get("paperId"),
            "title": payload.get("title"),
            "year": payload.get("year"),
            "venue": payload.get("venue"),
            "url": payload.get("url"),
            "reason": reason,
            "abstract": payload.get("abstract"),
            "citationCount": payload.get("citationCount"),
            "authors": payload.get("authors"),
            "openAccessPdf": payload.get("openAccessPdf"),
        }
        session.bib = [b for b in session.bib if b.get("paperId") != paper_id]
        session.bib.append(entry)
        session.last_tool_result = {
            "type": "paper_details",
            **payload,
            "bibliography_entry": entry,
            "bib_count": len(session.bib),
        }
        _emit_event(
            "tool_result",
            {"step": session.step_count, "tool": "fetch_paper", "result": session.last_tool_result},
        )
        return session.last_tool_result

    @tool
    def write_note(content: str) -> Dict[str, Any]:
        """Write a free-form note."""
        _bump_step("write_note", {"content": content[:2000]})
        session.notes.append(content)
        session.last_tool_result = {
            "type": "note",
            "note_saved": len(session.notes) - 1,
            "content": content,
        }
        _emit_event("tool_result", {"step": session.step_count, "tool": "write_note", "result": session.last_tool_result})
        return session.last_tool_result

    @tool
    def submit(memo: str) -> Dict[str, Any]:
        """Submit the final memo and terminate the run."""
        _bump_step("submit", {"memo": memo[:2000]})
        session.final_memo = memo
        session.last_tool_result = {"submitted": True}
        _emit_event("tool_result", {"step": session.step_count, "tool": "submit", "result": session.last_tool_result})
        return session.last_tool_result

    @tool
    def review_bibliography() -> Dict[str, Any]:
        """Inspect the current bibliography entries without hitting Semantic Scholar."""
        _bump_step("review_bibliography", {})
        session.last_tool_result = {
            "type": "bibliography",
            "entries": list(session.bib),
            "count": len(session.bib),
        }
        _emit_event(
            "tool_result",
            {"step": session.step_count, "tool": "review_bibliography", "result": session.last_tool_result},
        )
        return session.last_tool_result

    @tool
    def fetch_bibliography_paper(paper_id: str) -> Dict[str, Any]:
        """Load cached metadata for a paper that is already in the bibliography."""
        _bump_step("fetch_bibliography_paper", {"paper_id": paper_id})
        bib_entry = next((entry for entry in session.bib if entry.get("paperId") == paper_id), None)
        if not bib_entry:
            raise ValueError("Paper must be present in the bibliography to inspect it.")
        cached = db.get_paper(paper_id)
        if not cached:
            raise ValueError("Cached metadata not found for this bibliography entry.")
        payload = {
            "type": "bib_paper_details",
            "paperId": paper_id,
            "bib_entry": bib_entry,
            "cached_metadata": cached,
        }
        session.last_tool_result = payload
        _emit_event(
            "tool_result",
            {"step": session.step_count, "tool": "fetch_bibliography_paper", "result": session.last_tool_result},
        )
        return session.last_tool_result

    @tool
    def wait(seconds: float = 5.0) -> Dict[str, Any]:
        """Pause for the requested time (in seconds) before continuing."""
        clamped = max(0.0, min(float(seconds), 60.0))
        _bump_step("wait", {"seconds": clamped})
        time.sleep(clamped)
        session.last_tool_result = {"type": "wait", "seconds": clamped}
        _emit_event("tool_result", {"step": session.step_count, "tool": "wait", "result": session.last_tool_result})
        return session.last_tool_result

    tools: List[Any] = [
        search_semantic_scholar,
        write_note,
        wait,
        submit,
    ]
    if enable_bibliography:
        tools.insert(1, fetch_paper)
        tools.insert(3, review_bibliography)
        tools.insert(4, fetch_bibliography_paper)
    return tools
