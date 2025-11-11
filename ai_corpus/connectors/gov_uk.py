"""
Connector implementation for GOV.UK consultation pages and attachments.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin

from bs4 import BeautifulSoup, FeatureNotFound  # type: ignore

from ai_corpus.connectors.base import Collection, DocMeta, docmeta_to_rule_row
from ai_corpus.utils.http import RateLimiter, backoff_get, get_http_session

DEFAULT_UA = "alex-ai-policy-crawler/1.0 (+contact: you@usc.edu)"
DEFAULT_QUERY = "artificial intelligence"
DEFAULT_DOCUMENT_TYPES = [
    "open_consultation",
    "closed_consultation",
    "consultation_outcome",
]
DEFAULT_ORG_SLUGS = [
    "department-for-science-innovation-and-technology",
    "department-for-business-energy-and-industrial-strategy",
    "department-for-digital-culture-media-and-sport",
]


class GovUkConnector:
    """
    Connector that discovers GOV.UK consultations via the Search API and pulls
    consultation metadata plus attachments through the Content API / HTML pages.
    """

    name = "gov_uk"

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        global_config: Optional[Dict[str, Any]] = None,
        session=None,
    ) -> None:
        self.config = config or {}
        self.global_config = global_config or {}
        self.user_agent = (
            self.global_config.get("user_agent")
            or self.config.get("user_agent")
            or DEFAULT_UA
        )
        self.session = session or get_http_session({"User-Agent": self.user_agent})
        self.search_base = self.config.get(
            "search_base", "https://www.gov.uk/api/search.json"
        )
        self.content_base = self.config.get(
            "content_base", "https://www.gov.uk/api/content"
        )
        self.site_base = self.config.get("site_base", "https://www.gov.uk").rstrip("/")
        rps = float(self.config.get("requests_per_second", 4.0))
        min_interval = 1.0 / max(rps, 0.1)
        self.rate_limiter = RateLimiter(min_interval=min_interval)
        self.timeout = int(self.config.get("timeout", 30))
        self.default_query = self.config.get("query", DEFAULT_QUERY)
        self.default_phrase = bool(self.config.get("phrase_query", False))
        self.default_document_types: List[str] = self.config.get("document_types", DEFAULT_DOCUMENT_TYPES)  # type: ignore[assignment]
        self.default_org_slugs: List[str] = self.config.get("organisations", DEFAULT_ORG_SLUGS)  # type: ignore[assignment]
        self.extra_filter_params: Dict[str, Any] = self.config.get("extra_filters", {})
        self._page_cache: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def discover(
        self,
        *,
        q: Optional[str] = None,
        org_slugs: Optional[List[str]] = None,
        document_types: Optional[List[str]] = None,
        count: int = 1000,
        max_results: int = 2000,
        order: str = "-public_timestamp",
        phrase: Optional[bool] = None,
        extra_filters: Optional[Dict[str, Any]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **_: Any,
    ) -> Iterable[Collection]:
        """
        Query the GOV.UK Search API for consultations.
        """
        start = 0
        yielded = 0
        start_dt = self._parse_date(start_date) if start_date else None
        end_dt = self._parse_date(end_date) if end_date else None
        while True:
            params: Dict[str, Any] = {
                "order": order,
                "count": min(count, 1000),
                "start": start,
            }
            query = q if q is not None else self.default_query
            if query:
                use_phrase = self.default_phrase if phrase is None else phrase
                params["q"] = query if not use_phrase else self._as_phrase(query)
            doc_types = document_types or self.default_document_types
            if doc_types:
                params["filter_content_store_document_type"] = doc_types
            org_list = org_slugs or self.default_org_slugs
            if org_list:
                params["filter_organisations"] = org_list
            merged_filters = dict(self.extra_filter_params)
            if extra_filters:
                merged_filters.update(extra_filters)
            for key, value in merged_filters.items():
                if value is None:
                    continue
                params[key] = value

            data = self._get_json(self.search_base, params=params) or {}
            results = data.get("results") or []
            if not results:
                break

            stop_due_to_date = False
            for item in results:
                link = item.get("link")
                if not link:
                    continue
                base_path = self._norm_path(link)
                title = item.get("title") or base_path.split("/")[-1].replace("-", " ").title()
                url = urljoin(self.site_base, base_path)
                collection_id = base_path
                title_lower = title.lower()
                topic = (
                    "AI"
                    if "artificial intelligence" in title_lower or " ai " in f" {title_lower} "
                    else None
                )
                pub_ts_raw = item.get("public_timestamp")
                pub_ts = self._parse_date(pub_ts_raw)
                if end_dt and pub_ts and pub_ts > end_dt:
                    continue
                if start_dt and pub_ts and pub_ts < start_dt:
                    stop_due_to_date = True
                    continue

                yield Collection(
                    source=self.name,
                    collection_id=collection_id,
                    title=title,
                    url=url,
                    jurisdiction="UK",
                    topic=topic,
                    extra={
                        "public_timestamp": item.get("public_timestamp"),
                        "content_store_document_type": item.get("content_store_document_type"),
                        "organisations": item.get("organisations"),
                    },
                )
                yielded += 1
                if yielded >= max_results:
                    return

            start += params["count"]
            if stop_due_to_date and start_dt:
                break
            total = data.get("total")
            if total is not None and start >= total:
                break

    # ------------------------------------------------------------------
    # Document listing
    # ------------------------------------------------------------------
    def list_documents(self, collection_id: str, **_: Any) -> Iterable[DocMeta]:
        """
        Enumerate the consultation page itself plus any attachments.
        """
        base_path = self._norm_path(collection_id)
        page = self._load_page(base_path)
        if not page:
            return []

        web_url = urljoin(self.site_base, base_path)
        locale = page.get("locale") or "en"
        public_updated_at = page.get("public_updated_at") or page.get("first_published_at")

        org_names: List[str] = []
        for org in (page.get("links", {}).get("organisations") or []):
            name = org.get("title") or org.get("internal_name")
            if name:
                org_names.append(name)
        owning_org = org_names[0] if org_names else None

        detail = page.get("details") or {}
        attachments: List[Dict[str, Any]] = []

        if isinstance(detail.get("attachments"), list):
            for attachment in detail["attachments"]:
                url = (
                    attachment.get("url")
                    or attachment.get("web_url")
                    or attachment.get("attachment_url")
                )
                if not url:
                    continue
                attachments.append(
                    {
                        "title": attachment.get("title"),
                        "content_type": attachment.get("content_type")
                        or attachment.get("file_type"),
                        "url": self._absolute_url(url),
                    }
                )

        if isinstance(detail.get("documents"), list):
            for document in detail["documents"]:
                if isinstance(document, dict):
                    attachment = document.get("attachment") or {}
                    url = attachment.get("url") or attachment.get("web_url")
                    if not url:
                        continue
                    attachments.append(
                        {
                            "title": document.get("title"),
                            "content_type": attachment.get("content_type")
                            or attachment.get("file_type"),
                            "url": self._absolute_url(url),
                        }
                    )

        html = self._get_html(web_url)
        if html:
            try:
                soup = BeautifulSoup(html, "lxml")
            except FeatureNotFound:
                soup = BeautifulSoup(html, "html.parser")
            for anchor in soup.select("a[href]"):
                href = anchor.get("href")
                if not href:
                    continue
                lower_href = href.lower()
                if not lower_href.endswith(
                    (".pdf", ".doc", ".docx", ".odt", ".csv", ".xlsx", ".zip")
                ):
                    continue
                url = href if href.startswith("http") else urljoin(self.site_base, href)
                attachments.append(
                    {
                        "title": anchor.get_text(strip=True) or None,
                        "content_type": None,
                        "url": url,
                    }
                )

        seen: set[str] = set()
        for attachment in attachments:
            att_url = attachment.get("url")
            if not att_url or att_url in seen:
                continue
            seen.add(att_url)
            filename = att_url.rsplit("/", 1)[-1]
            doc_id = f"attachment:{base_path}:{filename}"
            yield DocMeta(
                source=self.name,
                collection_id=base_path,
                doc_id=doc_id,
                title=attachment.get("title") or filename,
                submitter=owning_org,
                submitter_type="government",
                org=owning_org,
                submitted_at=public_updated_at,
                language=locale,
                urls={"file": att_url, "page": web_url},
                extra={"content_type": attachment.get("content_type")},
            )

    def get_call_document(self, collection_id: str, **_: Any) -> Optional[DocMeta]:
        base_path = self._norm_path(collection_id)
        page = self._load_page(base_path)
        if not page:
            return None
        web_url = urljoin(self.site_base, base_path)
        api_url = f"{self.content_base.rstrip('/')}{base_path}"
        org_names: List[str] = []
        for org in (page.get("links", {}).get("organisations") or []):
            name = org.get("title") or org.get("internal_name")
            if name:
                org_names.append(name)
        owning_org = org_names[0] if org_names else None
        return DocMeta(
            source=self.name,
            collection_id=base_path,
            doc_id=f"page:{base_path}",
            title=page.get("title"),
            submitter=owning_org,
            submitter_type="government",
            org=owning_org,
            submitted_at=page.get("public_updated_at") or page.get("first_published_at"),
            language=page.get("locale") or "en",
            urls={"html": web_url, "json": api_url},
            extra={"document_type": page.get("document_type"), "document_role": "call"},
            kind="call",
        )

    def _load_page(self, base_path: str) -> Dict[str, Any]:
        if base_path in self._page_cache:
            return self._page_cache[base_path]
        api_url = f"{self.content_base.rstrip('/')}{base_path}"
        page = self._get_json(api_url) or {}
        self._page_cache[base_path] = page
        return page

    # ------------------------------------------------------------------
    # Fetching
    # ------------------------------------------------------------------
    def fetch(self, doc: DocMeta, out_dir: Path, **kwargs: Any) -> Dict[str, Any]:
        """
        Download the HTML page and/or attachment file for a document.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        artifacts: Dict[str, Any] = {}

        html_url = doc.urls.get("html") or doc.urls.get("page")
        if html_url:
            html_text = self._get_html(html_url)
            if html_text:
                html_path = out_dir / self._safe_name(f"{doc.doc_id}.html")
                html_path.write_text(html_text, encoding="utf-8")
                artifacts["html"] = str(html_path)

        file_url = doc.urls.get("file") or doc.urls.get("pdf")
        if file_url:
            response = backoff_get(
                file_url,
                timeout=self.timeout,
                headers={"User-Agent": self.user_agent},
                session=self.session,
                rate_limiter=self.rate_limiter,
                raise_for_status=False,
            )
            if response is not None and response.status_code == 200:
                filename = self._safe_name(file_url.rsplit("/", 1)[-1])
                file_path = out_dir / filename
                file_path.write_bytes(response.content)
                artifacts["file"] = str(file_path)

        return artifacts

    def iter_rule_versions(self, collection_id: str, **_: Any) -> Iterable[Dict[str, Any]]:
        """
        Represent the consultation page plus its attachments as sequential rule versions.
        """

        rows: List[Dict[str, Any]] = []
        call_doc = self.get_call_document(collection_id)
        if call_doc:
            rows.append(docmeta_to_rule_row(self.name, call_doc, history_rank=1, relationship="seed"))
        documents = list(self.list_documents(collection_id))
        start_rank = len(rows) + 1
        for offset, doc in enumerate(documents, start=start_rank):
            rows.append(
                docmeta_to_rule_row(
                    self.name,
                    doc,
                    history_rank=offset,
                    relationship="attachment",
                )
            )
        return rows

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        response = backoff_get(
            url,
            params=params,
            timeout=self.timeout,
            headers={"User-Agent": self.user_agent},
            session=self.session,
            rate_limiter=self.rate_limiter,
            raise_for_status=True,
        )
        if response is None:
            return {}
        return response.json()

    def _get_html(self, url: str) -> str:
        response = backoff_get(
            url,
            timeout=self.timeout,
            headers={"User-Agent": self.user_agent},
            session=self.session,
            rate_limiter=self.rate_limiter,
            raise_for_status=True,
        )
        if response is None:
            return ""
        return response.text

    @staticmethod
    def _norm_path(path: str) -> str:
        if not path:
            return "/"
        if not path.startswith("/"):
            path = "/" + path
        return path.split("?", 1)[0]

    def _absolute_url(self, url: str) -> str:
        if not url:
            return url
        if url.startswith(("http://", "https://")):
            return url
        return urljoin(self.site_base, self._norm_path(url))

    @staticmethod
    def _safe_name(name: str) -> str:
        cleaned = re.sub(r"[^-_.A-Za-z0-9]+", "_", name)
        return cleaned.strip("_") or "file"

    @staticmethod
    def _as_phrase(query: str) -> str:
        stripped = query.strip()
        if stripped.startswith('"') and stripped.endswith('"'):
            return stripped
        return f'"{stripped}"'

    def _as_timestamp(self, value: str, *, start: bool) -> str:
        parsed = self._parse_date(value)
        if parsed is None:
            return value
        if start:
            return parsed.strftime("%Y-%m-%dT00:00:00Z")
        return parsed.strftime("%Y-%m-%dT23:59:59Z")

    def _parse_date(self, value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            val = value.replace("Z", "+00:00")
            dt = datetime.fromisoformat(val)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            return None


__all__ = ["GovUkConnector"]
