"""
Lightweight heuristics for spotting comment-response sections in rule text.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional, Tuple

COMMENT_PATTERNS = [
    re.compile(r"\bresponse[s]?\s+to\s+comment", re.I),
    re.compile(r"\bcomment(?:er)?s?\s+(?:argued|suggested|requested|recommended)", re.I),
    re.compile(r"\bcomment\s+\d+\b", re.I),
    re.compile(r"\bafter\s+reviewing\s+the\s+comments\b", re.I),
    re.compile(r"\bwe\s+received\s+.*\bcomment", re.I),
]


def detect_comment_citations(*texts: Optional[str]) -> Tuple[bool, Optional[str]]:
    """
    Scan the supplied text blobs and return (has_citation, snippet).
    The snippet is trimmed to ~280 characters for storage.
    """

    for text in texts:
        if not text:
            continue
        snippet = _find_snippet(text)
        if snippet:
            return True, snippet
    return False, None


def _find_snippet(text: str) -> Optional[str]:
    for pattern in COMMENT_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        start = max(match.start() - 120, 0)
        end = min(match.end() + 160, len(text))
        snippet = " ".join(text[start:end].split())
        if len(snippet) > 280:
            snippet = snippet[:277] + "..."
        return snippet
    return None


__all__ = ["detect_comment_citations"]
