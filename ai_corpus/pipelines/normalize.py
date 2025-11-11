"""
Normalization helpers for mapping raw connector metadata into the shared schema.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Optional

from ai_corpus.connectors.base import DocMeta


@dataclass(slots=True)
class NormalizedDocument:
    uid: str
    source: str
    collection_id: str
    doc_id: str
    title: Optional[str]
    submitter_name: Optional[str]
    submitter_type: Optional[str]
    org: Optional[str]
    submitted_at: Optional[str]
    language: Optional[str]
    url_html: Optional[str]
    url_pdf: Optional[str]
    url_json: Optional[str]
    sha256_pdf: Optional[str]
    sha256_text: Optional[str]
    bytes_pdf: Optional[int]
    bytes_text: Optional[int]
    text_path: Optional[str]
    pdf_path: Optional[str]
    raw_meta: Dict


def normalize_doc(meta: DocMeta) -> NormalizedDocument:
    """
    Produce a deterministic identifier and shared set of metadata fields.
    """
    uid = hashlib.sha256(
        "|".join([meta.source, meta.collection_id, meta.doc_id]).encode("utf-8")
    ).hexdigest()
    urls = meta.urls or {}
    return NormalizedDocument(
        uid=uid,
        source=meta.source,
        collection_id=meta.collection_id,
        doc_id=meta.doc_id,
        title=meta.title,
        submitter_name=meta.submitter,
        submitter_type=meta.submitter_type,
        org=meta.org,
        submitted_at=meta.submitted_at,
        language=meta.language,
        url_html=urls.get("html"),
        url_pdf=urls.get("pdf"),
        url_json=urls.get("json"),
        sha256_pdf=None,
        sha256_text=None,
        bytes_pdf=None,
        bytes_text=None,
        text_path=None,
        pdf_path=None,
        raw_meta={**meta.extra, "document_kind": meta.kind},
    )
