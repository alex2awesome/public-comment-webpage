"""
Connector for Oregon's historical Administrative Orders dataset (Socrata).

- Dataset landing page: https://data.oregon.gov/d/gir4-yym3
- API endpoint: https://data.oregon.gov/resource/gir4-yym3.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import urljoin

from ai_corpus.connectors.base import BaseConnector, Collection, DocMeta
from ai_corpus.utils.http import RateLimiter, backoff_get, get_http_session, next_user_agent

logger = logging.getLogger(__name__)


class OregonAdminOrdersConnector(BaseConnector):
    name = "oregon_admin_orders"

    def __init__(
        self,
        config: Dict,
        global_config: Optional[Dict] = None,
        session=None,
    ) -> None:
        self.config = config or {}
        self.global_config = global_config or {}
        self.session = session or get_http_session(
            {"User-Agent": self.global_config.get("user_agent") or next_user_agent()}
        )
        self.base_url = self.config.get("base_url", "https://data.oregon.gov/resource/gir4-yym3.json")
        self.rate_limiter = RateLimiter(min_interval=float(self.config.get("rate_limit_seconds", 0.2)))
        self.max_entries = int(self.config.get("max_entries", 200))

    # --------------------------------------------------------------------- discovery
    def discover(self, **_) -> Iterable[Collection]:
        yield Collection(
            source=self.name,
            collection_id="oregon_admin_orders",
            title="Oregon Administrative Orders 2006-2017",
            url="https://data.oregon.gov/d/gir4-yym3",
            jurisdiction="US-OR",
            topic="State Rulemaking",
            extra={"dataset": "gir4-yym3"},
        )

    # ----------------------------------------------------------------- list documents
    def list_documents(self, collection_id: str, **_) -> Iterable[DocMeta]:
        rows = self._fetch_rows()
        documents: List[DocMeta] = []
        for row in rows:
            doc = self._convert_row(row)
            if doc:
                documents.append(doc)
        return documents

    # -------------------------------------------------------------------------- fetch
    def fetch(self, doc: DocMeta, out_dir: Path, **_) -> Dict[str, str]:
        link = doc.urls.get("html")
        if not link:
            raise ValueError("Document missing source URL")
        response = backoff_get(
            link,
            session=self.session,
            rate_limiter=self.rate_limiter,
            headers={"User-Agent": self.session.headers.get("User-Agent") or next_user_agent()},
            raise_for_status=True,
        )
        if response is None:
            raise RuntimeError(f"Failed to download {link}")
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{doc.doc_id}.html"
        out_path = out_dir / filename
        response.encoding = response.encoding or "utf-8"
        out_path.write_text(response.text, encoding="utf-8")
        return {"html": str(out_path)}

    # ---------------------------------------------------------------------- utilities
    def _fetch_rows(self) -> List[Dict]:
        params = {
            "$limit": self.max_entries,
            "$order": "year DESC",
        }
        if self.config.get("year_start"):
            params["$where"] = f"year >= '{self.config['year_start']}'"
        response = backoff_get(
            self.base_url,
            params=params,
            session=self.session,
            rate_limiter=self.rate_limiter,
            headers={"User-Agent": self.session.headers.get("User-Agent") or next_user_agent()},
            raise_for_status=True,
        )
        if response is None:
            return []
        try:
            return json.loads(response.text)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to decode Oregon dataset: %s", exc)
            return []

    def _convert_row(self, row: Dict) -> Optional[DocMeta]:
        order_number = row.get("administrative_order_number")
        link_info = row.get("link_to_document") or {}
        url = link_info.get("url")
        if not order_number or not url:
            return None
        title = row.get("title") or order_number
        year = row.get("year")
        return DocMeta(
            source=self.name,
            collection_id="oregon_admin_orders",
            doc_id=order_number,
            title=title,
            submitter=row.get("document_type"),
            submitter_type=None,
            org=None,
            submitted_at=f"{year}-01-01T00:00:00" if year else None,
            language="en",
            urls={"html": url},
            extra={"year": year},
            kind="order",
        )


__all__ = ["OregonAdminOrdersConnector"]
