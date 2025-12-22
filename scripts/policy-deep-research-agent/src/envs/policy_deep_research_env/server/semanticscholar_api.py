"""Thin wrapper around the Semantic Scholar Graph API with retry handling."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Iterable, Optional

import requests

BASE = "https://api.semanticscholar.org/graph/v1"


class SemanticScholarClient:
    """HTTP client for the Semantic Scholar Academic Graph API."""

    def __init__(self, api_key: Optional[str] = None, timeout_s: int = 30):
        self.api_key = api_key or os.getenv("S2_API_KEY")
        self.timeout_s = timeout_s

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{BASE}{path}"
        for attempt in range(8):
            response = requests.get(
                url,
                params=params,
                headers=self._headers(),
                timeout=self.timeout_s,
            )
            if response.status_code == 429:
                time.sleep(min(2**attempt, 30))
                continue
            response.raise_for_status()
            return response.json()
        raise RuntimeError("Semantic Scholar rate-limited too long (429).")

    def bulk_search(
        self,
        query: str,
        fields: str,
        year: Optional[str],
        limit: int = 100,
    ) -> Iterable[Dict[str, Any]]:
        """Iterate over paginated search results."""
        params: Dict[str, Any] = {"query": query, "fields": fields, "limit": limit}
        if year:
            params["year"] = year
        offset: Optional[int] = 0
        while True:
            page_params = dict(params)
            if offset:
                page_params["offset"] = offset
            payload = self._get("/paper/search", page_params)
            data = payload.get("data", [])
            for item in data:
                yield item
            offset = payload.get("next")
            if not offset:
                return

    def paper_details(self, paper_id: str, fields: str) -> Dict[str, Any]:
        """Fetch metadata for a single paper."""
        return self._get(f"/paper/{paper_id}", {"fields": fields})
