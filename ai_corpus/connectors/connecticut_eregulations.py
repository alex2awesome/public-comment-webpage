"""
Connector for Connecticut's eRegulations portal.

Key documentation / endpoints:
- Open comment periods grid JSON: https://eregulations.ct.gov/eRegsPortal/Browse/getOpenCommentPeriods
- Regulation history entries: https://eregulations.ct.gov/eRegsPortal/Search/getRMRHistory?trackingNumber=<tracking>
- Document download: https://eregulations.ct.gov/eRegsPortal/Search/getDocument?guid=<guid>
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import urljoin

from ai_corpus.connectors.base import BaseConnector, Collection, DocMeta
from ai_corpus.utils.http import RateLimiter, backoff_get, get_http_session, next_user_agent

logger = logging.getLogger(__name__)
MS_DATE_RE = re.compile(r"/Date\((?P<value>-?\d+)\)/")


class ConnecticutEregsConnector(BaseConnector):
    name = "connecticut_eregulations"

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
        self.base_url = self.config.get("base_url", "https://eregulations.ct.gov").rstrip("/")
        self.rate_limiter = RateLimiter(min_interval=float(self.config.get("rate_limit_seconds", 0.25)))
        self.max_entries = int(self.config.get("max_entries", 100))

    # --------------------------------------------------------------------- discovery
    def discover(self, **_) -> Iterable[Collection]:
        periods = self._fetch_open_comment_periods()
        for entry in periods[: self.max_entries]:
            tracking_number = entry.get("TrackingNumber")
            if not tracking_number:
                continue
            comment_due = self._ms_date_to_iso(entry.get("CommentPeriodEndDate"))
            yield Collection(
                source=self.name,
                collection_id=tracking_number,
                title=entry.get("ShortName") or tracking_number,
                url=urljoin(
                    self.base_url + "/",
                    f"eRegsPortal/Search/RMRView/{tracking_number}",
                ),
                jurisdiction="US-CT",
                topic="State Rulemaking",
                extra={
                    "agency": entry.get("AgencyName"),
                    "comment_due": comment_due,
                    "sections": entry.get("SectionNumber"),
                    "status": entry.get("Status"),
                },
            )

    # ----------------------------------------------------------------- list documents
    def list_documents(self, collection_id: str, **_) -> Iterable[DocMeta]:
        history = self._fetch_history(collection_id)
        documents: List[DocMeta] = []
        for item in history:
            doc_meta = self._convert_history_row(collection_id, item)
            if doc_meta:
                documents.append(doc_meta)
        return documents

    # -------------------------------------------------------------------------- fetch
    def fetch(self, doc: DocMeta, out_dir: Path, **_) -> Dict[str, str]:
        guid = doc.extra.get("document_guid") if doc.extra else None
        if not guid:
            raise ValueError(f"Document {doc.doc_id} is metadata-only and has no downloadable asset.")
        url = urljoin(self.base_url + "/", f"eRegsPortal/Search/getDocument?guid={guid}")
        response = backoff_get(
            url,
            session=self.session,
            rate_limiter=self.rate_limiter,
            headers={"User-Agent": self.session.headers.get("User-Agent") or next_user_agent()},
            raise_for_status=True,
        )
        if response is None:
            raise RuntimeError(f"Failed to download document {doc.doc_id}")
        filename = doc.extra.get("filename") or f"{doc.doc_id}.bin"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename
        out_path.write_bytes(response.content)
        return {"binary": str(out_path)}

    # ---------------------------------------------------------------------- utilities
    def _fetch_open_comment_periods(self) -> List[Dict]:
        url = urljoin(self.base_url + "/", "eRegsPortal/Browse/getOpenCommentPeriods")
        return self._get_json(url)

    def _fetch_history(self, tracking_number: str) -> List[Dict]:
        url = urljoin(
            self.base_url + "/",
            f"eRegsPortal/Search/getRMRHistory?trackingNumber={tracking_number}",
        )
        return self._get_json(url)

    def _convert_history_row(self, collection_id: str, row: Dict) -> Optional[DocMeta]:
        assoc_doc = row.get("AssociatedDocument")
        if not assoc_doc:
            return None
        guid = assoc_doc.get("GUID")
        doc_id = guid or str(row.get("RMRID") or row.get("GUID") or row.get("Description"))
        title = row.get("Description") or assoc_doc.get("Title") or doc_id
        posted = self._ms_date_to_iso(row.get("PostedDate"))
        filename = assoc_doc.get("Name") or assoc_doc.get("Title") or f"{doc_id}.bin"
        mime_type = assoc_doc.get("MimeType")
        urls = {
            "download": urljoin(
                self.base_url + "/",
                f"eRegsPortal/Search/getDocument?guid={guid}",
            )
        }
        return DocMeta(
            source=self.name,
            collection_id=collection_id,
            doc_id=doc_id,
            title=title,
            submitter=None,
            submitter_type=None,
            org=row.get("CaseID"),
            submitted_at=posted,
            language="en",
            urls=urls,
            extra={
                "document_guid": guid,
                "filename": filename,
                "mime_type": mime_type,
            },
            kind="document",
        )

    def _get_json(self, url: str) -> List[Dict]:
        response = backoff_get(
            url,
            session=self.session,
            rate_limiter=self.rate_limiter,
            headers={"User-Agent": self.session.headers.get("User-Agent") or next_user_agent()},
            raise_for_status=True,
        )
        if response is None:
            return []
        response.encoding = response.encoding or "utf-8"
        try:
            return json.loads(response.text)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to decode JSON from %s: %s", url, exc)
            return []

    def _ms_date_to_iso(self, raw: Optional[str]) -> Optional[str]:
        if not raw:
            return None
        match = MS_DATE_RE.match(raw)
        if not match:
            return None
        try:
            millis = int(match.group("value"))
        except ValueError:
            return None
        dt = datetime.fromtimestamp(millis / 1000, tz=timezone.utc)
        return dt.isoformat()


__all__ = ["ConnecticutEregsConnector"]
