"""
Connector implementation for the Regulations.gov v4 API.
"""

from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

try:
    import xmltodict
except ImportError:  # pragma: no cover - optional dependency
    xmltodict = None

FR_DOC_FIELDS = [
    "title",
    "agencies",
    "agency_names",
    "type",
    "publication_date",
    "html_url",
    "document_number",
    "comments_close_on",
    "regulations_dot_gov_url",
    "regulations_dot_gov_info",
    "docket_id",
    "docket_ids",
    "abstract",
    "action",
    "dates",
    "full_text_xml_url",
]

from ai_corpus.connectors.base import BaseConnector, Collection, DocMeta
from ai_corpus.rules import detect_comment_citations
from ai_corpus.utils.http import RateLimiter, backoff_get, get_http_session, next_user_agent, regs_backoff_get

logger = logging.getLogger(__name__)


class RegulationsGovConnector(BaseConnector):
    name = "regulations_gov"

    def __init__(
        self,
        config: Dict,
        global_config: Optional[Dict] = None,
        session=None,
    ) -> None:
        self.config = config or {}
        self.global_config = global_config or {}
        self.session = session or get_http_session(
            {"User-Agent": self.global_config.get("user_agent")}
        )
        self.user_agent = self.session.headers.get("User-Agent")
        if not self.user_agent:
            self.user_agent = next_user_agent()
        self.base_url = f"{self.config.get('base_url', 'https://api.regulations.gov').rstrip('/')}/{self.config.get('version', 'v4').lstrip('/')}"
        auth = self.config.get("auth", {})
        self.api_key_env = auth.get("env_key", "REGS_GOV_API_KEY")
        self.api_key_value = auth.get("value")
        self.api_key = os.environ.get(self.api_key_env) or self.api_key_value
        self.rate_limiter = RateLimiter(min_interval=1.0)
        downloads_cfg = self.config.get("downloads", {})
        self.download_base = downloads_cfg.get("base_url", "https://downloads.regulations.gov").rstrip("/")
        self.max_workers = int(self.config.get("max_workers", 2))
        self.fr_base = self.global_config.get("federal_register_base", "https://www.federalregister.gov/api/v1").rstrip("/")
        self.fr_rate_limiter = RateLimiter(min_interval=0.5)
        min_updates = self.global_config.get("regulations_min_updates")
        try:
            self.min_update_documents = int(min_updates) if min_updates is not None else 0
        except (TypeError, ValueError):
            self.min_update_documents = 0
        min_comments = self.global_config.get("regulations_min_comments")
        try:
            self.min_comment_count = int(min_comments) if min_comments is not None else 0
        except (TypeError, ValueError):
            self.min_comment_count = 0
        self.drop_null_comments = bool(self.global_config.get("regulations_drop_null_comments"))
        self._document_detail_cache: Dict[str, Optional[Dict[str, Any]]] = {}

    # Discovery -----------------------------------------------------
    def discover(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        query: Optional[str] = None,
        docket_ids: Optional[Iterable[str]] = None,
        page_size: int = 100,
        max_pages: Optional[int] = None,
        **_,
    ) -> Iterable[Collection]:
        seeds = self.config.get("corpora_seeds", [])
        filters_provided = any([start_date, end_date, query, docket_ids])

        if filters_provided:
            yield from self._discover_remote(
                start_date=start_date,
                end_date=end_date,
                query=query,
                docket_ids=docket_ids,
                page_size=page_size,
                max_pages=max_pages,
            )
            return

        for seed in seeds:
            docket_id = seed.get("docket_id")
            if not docket_id:
                continue
            title = seed.get("label") or docket_id
            url = f"https://www.regulations.gov/docket/{docket_id}"
            yield Collection(
                source=self.name,
                collection_id=docket_id,
                title=title,
                url=url,
                jurisdiction="US-Federal",
                topic="AI",
                extra={"seed": True},
            )

        if query:
            yield from self._discover_remote(
                query=query,
                page_size=page_size,
                max_pages=max_pages,
            )

    def _discover_remote(
        self,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        query: Optional[str] = None,
        docket_ids: Optional[Iterable[str]] = None,
        page_size: int = 100,
        max_pages: Optional[int] = None,
    ) -> Iterator[Collection]:
        if docket_ids:
            for docket_id in docket_ids:
                yield from self._fetch_dockets(
                    start_date=start_date,
                    end_date=end_date,
                    query=query,
                    page_size=page_size,
                    max_pages=1,
                    docket_id=docket_id,
                )
        else:
            yield from self._fetch_dockets(
                start_date=start_date,
                end_date=end_date,
                query=query,
                page_size=page_size,
                max_pages=max_pages,
            )

    def _fetch_dockets(
        self,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        query: Optional[str] = None,
        page_size: int = 100,
        max_pages: Optional[int] = None,
        docket_id: Optional[str] = None,
    ) -> Iterator[Collection]:
        if not self.api_key:
            return
        params: Dict[str, object] = {
            "page[size]": min(page_size, 250),
            "sort": "-lastModifiedDate",
            "fields[dockets]": ",".join(
                [
                    "docketId",
                    "title",
                    "lastModifiedDate",
                    "commentDueDate",
                    "commentPeriodStartDate",
                    "commentPeriodEndDate",
                    "commentsCloseDate",
                    "commentCount",
                    "numberOfDocuments",
                    "agencyAcronym",
                    "docketType",
                ]
            ),
        }
        if start_date:
            params["filter[lastModifiedDate][ge]"] = self._normalize_date(start_date, "00:00:00")
        if end_date:
            params["filter[lastModifiedDate][le]"] = self._normalize_date(end_date, "23:59:59")
        if query:
            params["filter[searchTerm]"] = query
        if docket_id:
            params["filter[docketId]"] = docket_id

        page_number = 1
        pages_fetched = 0
        while True:
            params["page[number]"] = page_number
            resp = regs_backoff_get(
                "dockets",
                api_key=self.api_key,
                base_url=self.base_url,
                params=params,
                user_agent=next_user_agent(),
                rate_limiter=self.rate_limiter,
                raise_for_status=False,
            )
            if resp is None or resp.status_code != 200:
                break
            payload = resp.json() or {}
            data = payload.get("data") or []
            for item in data:
                coll = self._build_collection_from_docket(item)
                if coll:
                    yield coll
            pages_fetched += 1
            if max_pages and pages_fetched >= max_pages:
                break
            meta = payload.get("meta") or {}
            has_next = meta.get("hasNextPage")
            total_pages = meta.get("totalPages") or meta.get("page", {}).get("totalPages")
            if not has_next and total_pages and page_number >= total_pages:
                break
            if not has_next and not total_pages:
                break
            page_number += 1

            # When querying a specific docketId, the first page is sufficient.
            if docket_id:
                break

    def _build_collection_from_docket(self, item: Dict) -> Optional[Collection]:
        attrs = item.get("attributes") or {}
        docket_id = item.get("id") or attrs.get("docketId")
        if not docket_id:
            return None
        title = attrs.get("title") or docket_id
        agency = attrs.get("agencyAcronym") or attrs.get("agencyId")
        document_count = attrs.get("numberOfDocuments")
        comment_count = attrs.get("commentCount")
        if (
            self.min_update_documents
            and document_count is not None
            and document_count < self.min_update_documents
        ):
            return None
        if (
            self.min_comment_count
            and comment_count is not None
            and comment_count < self.min_comment_count
        ):
            return None
        if self.drop_null_comments and comment_count is None:
            return None
        extra = {
            "last_modified": attrs.get("lastModifiedDate"),
            "comment_due_date": attrs.get("commentDueDate") or attrs.get("commentsCloseDate"),
            "comment_period_start": attrs.get("commentPeriodStartDate"),
            "comment_period_end": attrs.get("commentPeriodEndDate"),
            "comment_count": comment_count,
            "document_count": document_count,
            "agency": agency,
            "docket_type": attrs.get("docketType"),
        }
        return Collection(
            source=self.name,
            collection_id=docket_id,
            title=title,
            url=f"https://www.regulations.gov/docket/{docket_id}",
            jurisdiction="US-Federal",
            topic="AI" if "AI" in title.upper() else attrs.get("docketType"),
            extra={k: v for k, v in extra.items() if v is not None},
        )

    def _normalize_date(self, value: str, fallback_time: str) -> str:
        value = value.strip()
        if len(value) == 10:
            return f"{value} {fallback_time}"
        return value

    # Harvest -------------------------------------------------------
    def list_documents(
        self,
        collection_id: str,
        document_type: Optional[str] = None,
        page_size: int = 250,
        max_pages: Optional[int] = None,
        **_,
    ) -> Iterable[DocMeta]:
        """
        Yield both public comments and agency-authored documents for the given
        docket. By default (`document_type=None`), the connector pulls every
        government-issued document (notices, supporting material, agency
        comment summaries, etc.) *and* all submitted comments so a pipeline run
        automatically mirrors the full Regulations.gov docket.
        """

        mode = (document_type or "all").lower()
        if mode not in {"comments", "documents", "all"}:
            mode = "comments"

        if mode in {"documents", "all"}:
            yield from self._iter_docket_items(
                collection_id=collection_id,
                endpoint="documents",
                is_comment=False,
                page_size=page_size,
                max_pages=max_pages,
            )
        if mode in {"comments", "all"}:
            yield from self._iter_docket_items(
                collection_id=collection_id,
                endpoint="comments",
                is_comment=True,
                page_size=page_size,
                max_pages=max_pages,
            )

    def _iter_docket_items(
        self,
        *,
        collection_id: str,
        endpoint: str,
        is_comment: bool,
        page_size: int,
        max_pages: Optional[int],
    ) -> Iterable[DocMeta]:
        fields_param = "fields[comments]" if is_comment else "fields[documents]"
        field_values = (
            [
                "commentId",
                "postedDate",
                "submitterName",
                "submitterType",
                "organization",
                "comment",
                "title",
                "fileFormats",
                "agencyId",
                "language",
            ]
            if is_comment
            else [
                "documentId",
                "postedDate",
                "title",
                "documentType",
                "documentSubtype",
                "fileFormats",
                "rin",
                "summary",
                "effectiveDate",
                "signingDate",
                "commentStartDate",
                "commentEndDate",
                "agencyId",
                "language",
            ]
        )
        params = {
            "filter[docketId]": collection_id,
            "page[size]": min(page_size, 250),
            "include": "attachments",
            "sort": "-postedDate",
            fields_param: ",".join(field_values),
        }
        pages = 0
        page_number = 1
        while True:
            params["page[number]"] = page_number
            params.pop("page[token]", None)
            resp = regs_backoff_get(
                endpoint,
                api_key=self.api_key,
                base_url=self.base_url,
                params=params,
                user_agent=next_user_agent(),
                rate_limiter=self.rate_limiter,
                raise_for_status=False,
            )
            if resp is None or resp.status_code != 200:
                break
            payload = resp.json() or {}
            attachments = self._index_attachments(payload.get("included", []))
            for item in payload.get("data", []):
                meta = self._build_docmeta(
                    collection_id,
                    item,
                    attachments,
                    is_comment=is_comment,
                )
                if not meta:
                    continue
                if is_comment:
                    meta.kind = "response"
                else:
                    meta.kind = "call"
                    attrs = meta.extra.get("raw_attributes", {})
                    agency = attrs.get("agencyId")
                    if agency:
                        meta.submitter = meta.submitter or agency
                        meta.org = meta.org or agency
                    meta.submitter_type = meta.submitter_type or "agency"
                    meta.extra.setdefault("document_role", "call")
                    meta.extra.setdefault("government_document", True)
                yield meta
            pages += 1
            if max_pages and pages >= max_pages:
                break
            meta = payload.get("meta") or {}
            pagination = meta.get("page") if isinstance(meta.get("page"), dict) else {}
            has_next = pagination.get("hasNextPage")
            total_pages = pagination.get("totalPages") or meta.get("totalPages")
            if has_next is None:
                has_next = meta.get("hasNextPage")
            if not has_next and total_pages and page_number >= total_pages:
                break
            if not has_next and not total_pages:
                break
            page_number += 1

    def _index_attachments(self, included: List[Dict]) -> Dict[str, Dict]:
        out: Dict[str, Dict] = {}
        for entry in included or []:
            if entry.get("type") != "attachments":
                continue
            att_id = entry.get("id")
            if att_id:
                out[att_id] = entry
        return out

    def _extract_pdf_url(self, formats: Optional[List[Dict[str, Any]]]) -> Optional[str]:
        if not formats:
            return None
        for fmt in formats:
            fmt_name = (fmt.get("format") or "").lower()
            file_url = fmt.get("fileUrl")
            if fmt_name == "pdf" and file_url:
                return file_url
        return None

    def _extract_pdf_url_from_attachments(self, attachments: List[Dict]) -> Optional[str]:
        for attachment in attachments or []:
            links = attachment.get("links") or {}
            download_url = links.get("download") or links.get("file")
            if download_url and download_url.lower().endswith(".pdf"):
                return download_url
        return None

    def _fetch_document_detail(self, doc_id: str) -> Optional[Dict[str, Any]]:
        if doc_id in self._document_detail_cache:
            return self._document_detail_cache[doc_id]
        resp = regs_backoff_get(
            f"documents/{doc_id}",
            api_key=self.api_key,
            base_url=self.base_url,
            params={"include": "attachments"},
            user_agent=next_user_agent(),
            rate_limiter=self.rate_limiter,
            raise_for_status=False,
        )
        if resp is None or resp.status_code != 200:
            self._document_detail_cache[doc_id] = None
            return None
        payload = resp.json() or {}
        self._document_detail_cache[doc_id] = payload
        return payload

    def _build_docmeta(
        self,
        collection_id: str,
        item: Dict,
        attachments_index: Dict[str, Dict],
        *,
        is_comment: bool,
    ) -> Optional[DocMeta]:
        attrs = item.get("attributes") or {}
        relationships = item.get("relationships") or {}
        attachments_meta: List[Dict] = []
        for rel in relationships.get("attachments", {}).get("data", []):
            att_info = attachments_index.get(rel.get("id"))
            if att_info:
                attachments_meta.append(att_info)
        comment_text = attrs.get("comment") if is_comment else None
        urls: Dict[str, str] = {}
        links = item.get("links") or {}
        if links.get("self"):
            urls["json"] = links["self"]
        doc_id = item.get("id") or attrs.get("documentId") or attrs.get("commentId")
        pdf_url = self._extract_pdf_url(attrs.get("fileFormats"))
        if not pdf_url:
            pdf_url = self._extract_pdf_url_from_attachments(attachments_meta)
        needs_detail = (not pdf_url) or (is_comment and not attachments_meta)
        if needs_detail and doc_id:
            detail_payload = self._fetch_document_detail(doc_id)
            if detail_payload:
                detail_data = detail_payload.get("data")
                if isinstance(detail_data, list):
                    detail_data = next(
                        (entry for entry in detail_data if entry.get("id") == doc_id),
                        detail_data[0] if detail_data else {},
                    )
                detail_data = detail_data or {}
                detail_attrs = detail_data.get("attributes") or {}
                if not pdf_url:
                    pdf_url = self._extract_pdf_url(detail_attrs.get("fileFormats"))
                detail_attachments = self._hydrate_attachments(detail_payload.get("included", []))
                if not attachments_meta:
                    attachments_meta = detail_attachments
                elif detail_attachments:
                    attachments_meta.extend(att for att in detail_attachments if att not in attachments_meta)
        if not attachments_meta:
            attachments_meta = []
        if not pdf_url:
            pdf_url = self._extract_pdf_url_from_attachments(attachments_meta)
        if pdf_url:
            urls["pdf"] = pdf_url
        return DocMeta(
            source=self.name,
            collection_id=collection_id,
            doc_id=item.get("id") or attrs.get("documentId") or attrs.get("commentId"),
            title=attrs.get("title"),
            submitter=attrs.get("submitterName") if is_comment else attrs.get("agencyId"),
            submitter_type=attrs.get("submitterType") if is_comment else None,
            org=attrs.get("organization") if is_comment else attrs.get("agencyId"),
            submitted_at=attrs.get("postedDate"),
            language=attrs.get("language"),
            urls=urls,
            extra={
                "raw_attributes": attrs,
                "attachments": attachments_meta,
                "comment_text": comment_text,
            },
        )

    # Fetch ---------------------------------------------------------
    def fetch(self, doc: DocMeta, out_dir: Path, **kwargs) -> Dict[str, object]:
        result: Dict[str, object] = {"doc_id": doc.doc_id}
        out_dir.mkdir(parents=True, exist_ok=True)

        # Persist the API payload comment text if present.
        skip_existing = kwargs.get("skip_existing", True)
        comment_html = doc.extra.get("comment_text")
        attachments: List[Dict] = doc.extra.get("attachments", [])
        is_comment_doc = (doc.kind or "response") != "call"
        if is_comment_doc and (not comment_html or not attachments) and self.api_key:
            detail = regs_backoff_get(
                f"comments/{doc.doc_id}",
                api_key=self.api_key,
                base_url=self.base_url,
                params={"include": "attachments"},
                user_agent=next_user_agent(),
                rate_limiter=self.rate_limiter,
                raise_for_status=False,
            )
            if detail is not None and detail.status_code == 200:
                payload = detail.json() or {}
                data = payload.get("data") or {}
                attrs = data.get("attributes") or {}
                if not comment_html:
                    comment_html = attrs.get("comment")
                if not attachments:
                    attachments = self._hydrate_attachments(payload.get("included", []))

        if comment_html:
            html_path = out_dir / f"{doc.doc_id}.html"
            if not (skip_existing and html_path.exists() and html_path.stat().st_size > 0):
                html_path.write_text(comment_html, encoding="utf-8")
            result["html"] = str(html_path)

        # Download attachments if available.
        stored_paths: List[str] = []
        for idx, attachment in enumerate(attachments, start=1):
            att_links = attachment.get("links") or {}
            download_url = att_links.get("download") or att_links.get("file")
            if not download_url:
                # Fallback to constructing from known pattern.
                attachment_id = attachment.get("id")
                if attachment_id and doc.doc_id:
                    download_url = f"{self.download_base}/{doc.doc_id}/attachment_{idx}.pdf"
            if not download_url:
                continue
            suffix = Path(download_url).suffix or ".bin"
            path = out_dir / f"{doc.doc_id}_attachment_{idx}{suffix}"
            if skip_existing and path.exists() and path.stat().st_size > 0:
                stored_paths.append(str(path))
                if suffix.lower() == ".pdf" and "pdf" not in result:
                    result["pdf"] = str(path)
                continue
            response = backoff_get(
                download_url,
                rate_limiter=self.rate_limiter,
                headers={"User-Agent": next_user_agent()},
                session=self.session,
                raise_for_status=False,
            )
            if response is None or response.status_code != 200:
                continue
            path.write_bytes(response.content)
            stored_paths.append(str(path))
            if suffix.lower() == ".pdf" and "pdf" not in result:
                result["pdf"] = str(path)
        if stored_paths:
            result["attachments"] = stored_paths
        if "pdf" not in result:
            pdf_url = doc.urls.get("pdf")
            if pdf_url:
                pdf_path = out_dir / f"{doc.doc_id}.pdf"
                if skip_existing and pdf_path.exists() and pdf_path.stat().st_size > 0:
                    result["pdf"] = str(pdf_path)
                else:
                    response = backoff_get(
                        pdf_url,
                        rate_limiter=self.rate_limiter,
                        headers={"User-Agent": next_user_agent()},
                        session=self.session,
                        raise_for_status=False,
                    )
                    if response is not None and response.status_code == 200:
                        pdf_path.write_bytes(response.content)
                        result["pdf"] = str(pdf_path)
        return result

    def _hydrate_attachments(self, included: List[Dict]) -> List[Dict]:
        hydrated: List[Dict] = []
        for entry in included or []:
            if entry.get("type") != "attachments":
                continue
            hydrated.append(entry)
        return hydrated

    def get_call_document(
        self,
        collection_id: str,
        *,
        preferred_types: Optional[List[str]] = None,
        preferred_subtypes: Optional[List[str]] = None,
        max_pages: int = 3,
        **kwargs,
    ) -> Optional[DocMeta]:
        """
        Attempt to locate the originating call/notice for a docket by querying
        the documents endpoint and selecting the best candidate.
        """
        preferred_types = [t.lower() for t in (preferred_types or ["Notice", "Proposed Rule", "Supporting & Related Material"])]
        preferred_subtypes = [
            s.lower()
            for s in (
                preferred_subtypes
                or [
                    "Request for Information",
                    "Request for Comments",
                    "Request for Public Comment",
                    "Notice",
                ]
            )
        ]
        documents = list(
            self.list_documents(
                collection_id=collection_id,
                document_type="documents",
                page_size=50,
                max_pages=max_pages,
                **kwargs,
            )
        )
        if not documents:
            return None

        def _score(doc: DocMeta) -> int:
            attrs = doc.extra.get("raw_attributes", {})
            doc_type = (attrs.get("documentType") or "").lower()
            doc_subtype = (attrs.get("documentSubtype") or "").lower()
            score = 0
            if doc_type in preferred_types:
                score += 2
            if doc_subtype in preferred_subtypes:
                score += 3
            if "request for" in doc_subtype:
                score += 1
            if "request for" in doc_type:
                score += 1
            return score

        best = max(documents, key=_score)
        if _score(best) == 0 and len(documents) > 1:
            best = min(
                documents,
                key=lambda doc: (doc.submitted_at or "", doc.doc_id),
            )

        best.kind = "call"
        best.extra.setdefault("document_role", "call")
        return best

    # ------------------------------------------------------------------ rules API
    def iter_rule_versions(
        self,
        collection_id: str,
        *,
        history_types: Optional[List[str]] = None,
        history_limit: Optional[int] = None,
        **_,
    ) -> Iterable[Dict[str, Any]]:
        """
        Export all Federal Register documents associated with `collection_id`.
        """

        docket_id = collection_id
        types = [t.strip().upper() for t in (history_types or ["PRORULE", "RULE", "NOTICE"]) if t and t.strip()]
        if not types:
            types = ["PRORULE", "RULE", "NOTICE"]
        try:
            docs = self._fr_fetch_docket_documents(docket_id, types)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch Federal Register history for %s: %s", docket_id, exc)
            return []
        if not docs:
            return []
        docs.sort(
            key=lambda item: (
                self._parse_iso_date(item.get("publication_date")) or date.min,
                item.get("document_number") or "",
            )
        )
        parent_docnum = next((item.get("document_number") for item in docs if item.get("document_number")), None)
        rows: List[Dict[str, Any]] = []
        for idx, doc in enumerate(docs, start=1):
            if history_limit and idx > history_limit:
                break
            rows.append(self._build_rule_row_from_fr(docket_id, doc, idx, parent_docnum))
        return rows

    def _fr_fetch_docket_documents(self, docket_id: str, include_types: List[str]) -> List[Dict[str, Any]]:
        params = [
            ("conditions[term]", docket_id),
            ("per_page", 500),
            ("order", "oldest"),
        ]
        for t in include_types:
            params.append(("conditions[type][]", t))
        for field in FR_DOC_FIELDS:
            params.append(("fields[]", field))
        url = f"{self.fr_base}/documents.json"
        results: List[Dict[str, Any]] = []
        page = 1
        while True:
            page_params = list(params)
            page_params.append(("page", page))
            response = backoff_get(
                url,
                params=page_params,
                rate_limiter=self.fr_rate_limiter,
                headers={"Accept": "application/json", "User-Agent": self.user_agent},
                raise_for_status=False,
            )
            if response is None or response.status_code != 200:
                break
            payload = response.json() or {}
            batch = payload.get("results") or []
            if not batch:
                break
            results.extend(batch)
            meta = payload.get("meta") or {}
            total_pages = meta.get("total_pages") or meta.get("totalPages") or 1
            if page >= total_pages:
                break
            page += 1
        return results

    def _fr_extract_supplementary(self, xml_url: Optional[str]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        if not xml_url:
            return None, None
        response = backoff_get(
            xml_url,
            headers={"Accept": "application/xml", "User-Agent": self.user_agent},
            rate_limiter=self.fr_rate_limiter,
            raise_for_status=False,
        )
        if response is None or response.status_code != 200:
            return None, None
        xml_dict = None
        if xmltodict is not None:
            try:
                xml_dict = xmltodict.parse(response.content)
            except Exception:
                xml_dict = None
        try:
            root = ET.fromstring(response.content)
            nodes = root.findall(".//SUPLINF")
            if not nodes:
                return xml_dict, None
            text = "\n\n".join("".join(node.itertext()) for node in nodes).strip()
            return xml_dict, text or None
        except Exception:
            return xml_dict, None

    def _parse_iso_date(self, value: Optional[str]) -> Optional[date]:
        if not value:
            return None
        try:
            return date.fromisoformat(value[:10])
        except Exception:
            return None

    def _compute_comment_window(
        self,
        publication_date: Optional[str],
        close_on: Optional[str],
    ) -> Tuple[str, str, str, bool]:
        start = self._parse_iso_date(publication_date)
        due = self._parse_iso_date(close_on)
        today = date.today()
        status = "unknown"
        if start and due:
            if today < start:
                status = "scheduled"
            elif today > due:
                status = "closed"
            else:
                status = "open"
        elif due:
            status = "open" if today <= due else "closed"
        elif start:
            status = "open" if today >= start else "scheduled"
        active = status == "open"
        return (
            start.isoformat() if start else "",
            due.isoformat() if due else "",
            status,
            active,
        )

    def _build_rule_row_from_fr(
        self,
        docket_id: str,
        doc: Dict[str, Any],
        rank: int,
        parent_docnum: Optional[str],
    ) -> Dict[str, Any]:
        agencies = doc.get("agency_names") or [a.get("name") for a in (doc.get("agencies") or []) if isinstance(a, dict) and a.get("name")]
        xml_dict, supp = self._fr_extract_supplementary(doc.get("full_text_xml_url"))
        start, due, status, active = self._compute_comment_window(doc.get("publication_date"), doc.get("comments_close_on"))
        regs_info = doc.get("regulations_dot_gov_info") or {}
        regs_doc = regs_info.get("document_id")
        row: Dict[str, Any] = {
            "scrape_mode": "connector_rule_history",
            "source": self.name,
            "fr_document_number": doc.get("document_number") or "",
            "title": doc.get("title") or "",
            "type": doc.get("type") or "",
            "publication_date": doc.get("publication_date") or "",
            "agency": "; ".join(a for a in agencies if a),
            "fr_url": doc.get("html_url") or "",
            "docket_id": docket_id,
            "docket_ids": doc.get("docket_ids") or [],
            "regs_url": doc.get("regulations_dot_gov_url") or "",
            "regs_document_id": regs_doc or "",
            "regs_object_id": regs_info.get("object_id") or "",
            "comment_start_date": start,
            "comment_due_date": due,
            "comment_status": status,
            "comment_active": active,
            "details": "",
            "abstract": doc.get("abstract") or "",
            "action": doc.get("action") or "",
            "dates": doc.get("dates") or "",
            "supplementary_information": supp or "",
            "xml_dict": xml_dict or {},
        }
        mentions, snippet = detect_comment_citations(
            row["supplementary_information"],
            row["abstract"],
            row["action"],
        )
        row["mentions_comment_response"] = mentions
        row["comment_citation_snippet"] = snippet or ""
        row["history_parent_docket"] = docket_id
        row["history_parent_fr_doc"] = parent_docnum or row["fr_document_number"]
        row["history_stage"] = row["type"]
        row["history_relationship"] = "seed" if rank == 1 else "related"
        row["history_rank"] = str(rank)
        title_lower = (row["title"] or "").lower()
        if "request for information" in title_lower:
            row["is_rfi_rfc"] = "TRUE"
            row["rfi_rfc_label"] = "RFI"
        elif "request for comment" in title_lower:
            row["is_rfi_rfc"] = "TRUE"
            row["rfi_rfc_label"] = "RFC"
        else:
            row["is_rfi_rfc"] = "FALSE"
            row["rfi_rfc_label"] = ""
        return row


__all__ = ["RegulationsGovConnector"]
