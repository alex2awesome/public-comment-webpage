"""
Connector for the NITRD/OSTP AI Action Plan RFI comment directory.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from bs4 import BeautifulSoup  # type: ignore

from ai_corpus.connectors.base import Collection, DocMeta, docmeta_to_rule_row
from ai_corpus.utils.http import RateLimiter, backoff_get, get_http_session, next_user_agent


class NitrdAiRfiConnector:
    name = "nitrd_ai_rfi"

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
        self.rate_limiter = RateLimiter(min_interval=0.25)
        self.base_url = self.config.get("base_url", "https://files.nitrd.gov/90-fr-9088/").rstrip("/") + "/"
        self.seed_files = self.config.get("seed_files") or []
        self.max_workers = int(self.config.get("max_workers", 4))
        self._document_cache: Dict[str, List[Tuple[DocMeta, bool]]] = {}

    def discover(self, **_) -> Iterable[Collection]:
        extra = {"source_url": self.base_url}
        if self.seed_files:
            extra["seed_files"] = list(self.seed_files)
        yield Collection(
            source=self.name,
            collection_id="90-FR-9088",
            title="NITRD / OSTP AI Action Plan RFI Responses",
            url=self.base_url,
            jurisdiction="US-Federal",
            topic="AI",
            extra=extra,
        )

    def list_documents(self, collection_id: str, **_) -> Iterable[DocMeta]:
        for doc, is_call in self._enumerate_documents(collection_id):
            if not is_call:
                yield doc

    def get_call_document(self, collection_id: str, **_) -> Optional[DocMeta]:
        for doc, is_call in self._enumerate_documents(collection_id):
            if is_call:
                call_doc = DocMeta(
                    source=doc.source,
                    collection_id=doc.collection_id,
                    doc_id=doc.doc_id,
                    title=doc.title,
                    submitter="NITRD",
                    submitter_type="agency",
                    org="Networking & Information Technology R&D",
                    submitted_at=None,
                    language=doc.language,
                    urls=doc.urls,
                    extra={**doc.extra, "document_role": "call"},
                    kind="call",
                )
                return call_doc
        return None

    def _enumerate_documents(self, collection_id: str) -> List[Tuple[DocMeta, bool]]:
        cached = self._document_cache.get(collection_id)
        if cached is not None:
            return cached

        documents: List[Tuple[DocMeta, bool]] = []
        response = backoff_get(
            self.base_url,
            session=self.session,
            rate_limiter=self.rate_limiter,
            headers={"User-Agent": self.global_config.get("user_agent")},
            raise_for_status=False,
        )
        if response is None or response.status_code != 200:
            documents = []
        else:
            text = response.text or ""
            if not text.strip():
                for filename in self.seed_files:
                    url = f"{self.base_url}{filename}"
                    doc = DocMeta(
                        source=self.name,
                        collection_id=collection_id,
                        doc_id=filename,
                        title=filename,
                        submitter=None,
                        submitter_type=None,
                        org=None,
                        submitted_at=None,
                        language=None,
                        urls={"pdf": url},
                        extra={},
                    )
                    documents.append((doc, self._is_call_candidate(filename, filename)))
            else:
                soup = BeautifulSoup(text, "html.parser")
                for anchor in soup.select("a[href]"):
                    href = anchor.get("href") or ""
                    if href.startswith("../") or href.endswith("/"):
                        continue
                    if not re.search(r"\.(pdf|docx|zip)$", href, re.IGNORECASE):
                        continue
                    url = href if href.startswith("http") else f"{self.base_url}{href}"
                    doc_id = href.rsplit("/", 1)[-1]
                    doc = DocMeta(
                        source=self.name,
                        collection_id=collection_id,
                        doc_id=doc_id,
                        title=doc_id,
                        submitter=None,
                        submitter_type=None,
                        org=None,
                        submitted_at=None,
                        language=None,
                        urls={"pdf": url},
                        extra={},
                    )
                    documents.append((doc, self._is_call_candidate(anchor.get_text() or "", href)))

        self._document_cache[collection_id] = documents
        return documents

    def _is_call_candidate(self, text: str, href: str) -> bool:
        haystack = f"{text} {href}".lower()
        if not haystack:
            return False
        register_triggers = [
            "federal register",
            "90-fr-9088",
            "90fr9088",
            "fr-9088",
            "fr_doc",
        ]
        if any(trigger in haystack for trigger in register_triggers):
            return True
        if "notice" in haystack and "rfi" in haystack:
            return True
        if "request for information" in haystack and ("ostp" in haystack or "ai action plan" in haystack):
            return True
        return False

    def fetch(self, doc: DocMeta, out_dir: Path, **kwargs) -> Dict[str, object]:
        pdf_url = doc.urls.get("pdf")
        if not pdf_url:
            return {"doc_id": doc.doc_id}
        skip_existing = kwargs.get("skip_existing", True)
        result: Dict[str, object] = {"doc_id": doc.doc_id}
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", doc.doc_id)
        path = out_dir / safe_name
        if not path.suffix:
            path = path.with_suffix(".pdf")
        if skip_existing and path.exists() and path.stat().st_size > 0:
            result["pdf"] = str(path)
            return result
        response = backoff_get(
            pdf_url,
            session=self.session,
            rate_limiter=self.rate_limiter,
            headers={"User-Agent": next_user_agent()},
            raise_for_status=False,
        )
        if response is None or response.status_code != 200:
            return result
        path.write_bytes(response.content)
        result["pdf"] = str(path)
        return result

    def iter_rule_versions(self, collection_id: str, **_) -> Iterable[Dict[str, Any]]:
        call_doc = self.get_call_document(collection_id)
        if not call_doc:
            return []
        return [docmeta_to_rule_row(self.name, call_doc, history_rank=1)]


__all__ = ["NitrdAiRfiConnector"]
