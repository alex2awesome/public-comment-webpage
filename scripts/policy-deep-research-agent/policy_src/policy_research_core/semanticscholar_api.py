"""Thin wrapper around the Semantic Scholar Graph API with retry handling."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Iterable, Optional

import requests

BASE = "https://api.semanticscholar.org/graph/v1"


class SemanticScholarRateLimitError(RuntimeError):
    """Raised when Semantic Scholar returns 429s for too long."""

    def __init__(self, waited_seconds: float):
        super().__init__(f"Semantic Scholar rate-limited too long after waiting {waited_seconds:.1f}s (429).")
        self.waited_seconds = waited_seconds


class SemanticScholarClient:
    """HTTP client for the Semantic Scholar Academic Graph API."""

    def __init__(self, api_key: Optional[str] = None, timeout_s: int = 30, max_retries: int = 12):
        self.api_key = api_key or os.getenv("S2_API_KEY")
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{BASE}{path}"
        backoff = 2.0
        total_wait = 0.0
        for attempt in range(self.max_retries):
            response = requests.get(
                url,
                params=params,
                headers=self._headers(),
                timeout=self.timeout_s,
            )
            if response.status_code == 429:
                sleep_for = min(backoff, 60)
                time.sleep(sleep_for)
                backoff *= 1.5
                total_wait += sleep_for
                continue
            response.raise_for_status()
            return response.json()
        raise SemanticScholarRateLimitError(total_wait)

    def bulk_search(
        self,
        query: str,
        fields: str,
        year: Optional[str],
        limit: int = 100,
        max_items: Optional[int] = None,
    ) -> Iterable[Dict[str, Any]]:
        """Iterate over paginated search results."""
        params: Dict[str, Any] = {"query": query, "fields": fields, "limit": limit}
        if year:
            params["year"] = year
        offset: Optional[int] = 0
        yielded = 0
        while True:
            page_params = dict(params)
            if offset:
                page_params["offset"] = offset
            payload = self._get("/paper/search", page_params)
            data = payload.get("data", [])
            for item in data:
                if max_items is not None and yielded >= max_items:
                    return
                yielded += 1
                yield item
            offset = payload.get("next")
            if not offset or (max_items is not None and yielded >= max_items):
                return

    def paper_details(self, paper_id: str, fields: str) -> Dict[str, Any]:
        """Fetch metadata for a single paper."""
        return self._get(f"/paper/{paper_id}", {"fields": fields})

    def author_search(self, query: str, fields: str, limit: int = 5) -> Dict[str, Any]:
        """Search for authors."""
        params: Dict[str, Any] = {"query": query, "fields": fields, "limit": limit}
        return self._get("/author/search", params)
