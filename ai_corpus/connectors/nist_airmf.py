"""
Connector for NIST AI Risk Management Framework public comment PDFs.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, Optional

from bs4 import BeautifulSoup  # type: ignore

from ai_corpus.connectors.base import Collection, DocMeta, docmeta_to_rule_row
from ai_corpus.utils.http import RateLimiter, backoff_get, get_http_session, next_user_agent


class NistAirmfConnector:
    name = "nist_airmf"

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
        self.rate_limiter = RateLimiter(min_interval=0.5)
        self.index_url = self.config.get("index_url")
        self.base_url = self.config.get("base_url", "https://www.nist.gov").rstrip("/")
        self.max_workers = int(self.config.get("max_workers", 4))

    def discover(self, **_) -> Iterable[Collection]:
        if not self.index_url:
            return []
        yield Collection(
            source=self.name,
            collection_id="AI-RMF-2ND-DRAFT-2022",
            title="NIST AI RMF 2nd Draft Comments",
            url=self.index_url,
            jurisdiction="US-Federal",
            topic="AI",
            extra={"source_url": self.index_url},
        )

    def list_documents(self, collection_id: str, **_) -> Iterable[DocMeta]:
        if not self.index_url:
            return []
        response = backoff_get(
            self.index_url,
            session=self.session,
            rate_limiter=self.rate_limiter,
            headers={"User-Agent": self.global_config.get("user_agent")},
            raise_for_status=False,
        )
        if response is None or response.status_code != 200:
            return []
        soup = BeautifulSoup(response.text, "html.parser")
        anchors = soup.select("a[data-file-url], main a[href$='.pdf']")
        seen: set[str] = set()
        for anchor in anchors:
            file_url = anchor.get("data-file-url") or anchor.get("href")
            if not file_url:
                continue
            if not file_url.lower().endswith(".pdf"):
                continue
            if file_url.startswith("/"):
                pdf_url = f"{self.base_url.rstrip('/')}{file_url}"
            elif file_url.startswith("http"):
                pdf_url = file_url
            else:
                pdf_url = f"{self.base_url.rstrip('/')}/{file_url}"
            if pdf_url in seen:
                continue
            seen.add(pdf_url)
            title = anchor.get_text(strip=True)
            doc_id = self._doc_id_from_url(pdf_url)
            yield DocMeta(
                source=self.name,
                collection_id=collection_id,
                doc_id=doc_id,
                title=title or doc_id,
                submitter=title or None,
                submitter_type=None,
                org=None,
                submitted_at=None,
                language="en",
                urls={"pdf": pdf_url},
                extra={"source_url": pdf_url},
            )

    def get_call_document(self, collection_id: str, **_) -> Optional[DocMeta]:
        if not self.index_url or collection_id != "AI-RMF-2ND-DRAFT-2022":
            return None
        return DocMeta(
            source=self.name,
            collection_id=collection_id,
            doc_id="nist_airmf_call.html",
            title="NIST AI RMF Second Draft Call for Comments",
            submitter="NIST",
            submitter_type="agency",
            org="National Institute of Standards and Technology",
            submitted_at=None,
            language="en",
            urls={"html": self.index_url},
            extra={"source_url": self.index_url, "document_role": "call"},
            kind="call",
        )

    def _doc_id_from_url(self, url: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]", "_", url.rsplit("/", 1)[-1])

    def fetch(self, doc: DocMeta, out_dir: Path, **kwargs) -> Dict[str, object]:
        pdf_url = doc.urls.get("pdf")
        html_url = doc.urls.get("html") if not pdf_url else None
        if not pdf_url and not html_url:
            return {"doc_id": doc.doc_id}
        skip_existing = kwargs.get("skip_existing", True)
        result: Dict[str, object] = {"doc_id": doc.doc_id}
        out_dir.mkdir(parents=True, exist_ok=True)
        if pdf_url:
            path = out_dir / doc.doc_id
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

        # HTML fallback for call documents
        html_path = out_dir / (doc.doc_id if doc.doc_id.endswith(".html") else f"{doc.doc_id}.html")
        if skip_existing and html_path.exists() and html_path.stat().st_size > 0:
            result["html"] = str(html_path)
            return result
        response = backoff_get(
            html_url,
            session=self.session,
            rate_limiter=self.rate_limiter,
            headers={"User-Agent": next_user_agent()},
            raise_for_status=False,
        )
        if response is None or response.status_code != 200:
            return result
        html_path.write_text(response.text or "", encoding="utf-8")
        result["html"] = str(html_path)
        return result

    def iter_rule_versions(self, collection_id: str, **_) -> Iterable[Dict[str, Any]]:
        call_doc = self.get_call_document(collection_id)
        if not call_doc:
            return []
        return [docmeta_to_rule_row(self.name, call_doc, history_rank=1)]


__all__ = ["NistAirmfConnector"]
