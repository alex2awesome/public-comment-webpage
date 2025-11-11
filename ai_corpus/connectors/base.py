"""
Core connector interfaces and shared dataclasses for AI regulatory comment harvesting.

Every data source implements the `BaseConnector` interface so the rest of the
pipeline (discovery, harvesting, storage) can remain source-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Protocol, Any


@dataclass(slots=True)
class Collection:
    """
    Represents a docket, consultation, or other grouping of public comments
    exposed by a source system.
    """

    source: str
    collection_id: str
    title: str
    url: str
    jurisdiction: str
    topic: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DocMeta:
    """
    Normalized metadata about an individual comment or submission before it is
    downloaded and extracted.
    """

    source: str
    collection_id: str
    doc_id: str
    title: Optional[str]
    submitter: Optional[str]
    submitter_type: Optional[str]
    org: Optional[str]
    submitted_at: Optional[str]
    language: Optional[str]
    urls: Dict[str, str]
    extra: Dict[str, Any] = field(default_factory=dict)
    kind: str = "response"


class BaseConnector(Protocol):
    """
    All connectors must expose discovery, document listing, and fetch routines.
    Implementations may choose to support additional keyword arguments (passed
    through from the CLI/pipeline layer).
    """

    name: str

    def discover(self, **kwargs) -> Iterable[Collection]:
        """
        Discover available collections in the remote system. Implementations
        should accept optional kwargs such as `start_date`, `end_date`, or
        source-specific filters to support listing dockets within a date range.
        """

    def list_documents(self, collection_id: str, **kwargs) -> Iterable[DocMeta]:
        """
        Enumerate documents/comments belonging to `collection_id`. Returned
        metadata instances should include all URLs necessary for fetching raw
        content (HTML, PDF, attachments).
        """

    def get_call_document(self, collection_id: str, **kwargs) -> Optional[DocMeta]:
        """
        Return metadata describing the originating call (e.g., RFI/RFC notice)
        for `collection_id`. Connectors may return None if no dedicated call
        artifact is available.
        """

    def fetch(self, doc: DocMeta, out_dir: Path, **kwargs) -> Dict[str, Any]:
        """
        Download the raw assets for `doc` into `out_dir`. Returns a dictionary
        describing the local artifacts (e.g., {"pdf": "/path", "html": "/path"}).
        """

    def iter_rule_versions(self, collection_id: str, **kwargs) -> Iterable[Dict[str, Any]]:
        """
        Optional: yield normalized rule-version rows (matching
        `ai_corpus.rules.schema.RULE_VERSION_COLUMNS`) for the specified
        collection/docket.
        """


def docmeta_to_rule_row(
    connector_name: str,
    doc: DocMeta,
    *,
    history_rank: int = 1,
    relationship: str = "seed",
) -> Dict[str, Any]:
    """
    Convenience helper for connectors whose rule history is derived directly
    from `DocMeta` entries (e.g., single-call consultations).
    """

    urls = doc.urls or {}
    primary_url = urls.get("html") or urls.get("pdf") or urls.get("json") or ""
    return {
        "scrape_mode": "connector_rule_history",
        "source": connector_name,
        "fr_document_number": doc.doc_id,
        "title": doc.title or doc.doc_id,
        "type": doc.kind or "",
        "publication_date": doc.submitted_at or "",
        "agency": doc.org or doc.submitter or "",
        "fr_url": primary_url,
        "docket_id": doc.collection_id,
        "docket_ids": [doc.collection_id],
        "regs_url": urls.get("json") or "",
        "comment_start_date": doc.extra.get("comment_start_date") if doc.extra else "",
        "comment_due_date": doc.extra.get("comment_due_date") if doc.extra else "",
        "comment_status": doc.extra.get("comment_status") if doc.extra else "",
        "comment_active": False,
        "is_rfi_rfc": "FALSE",
        "rfi_rfc_label": doc.extra.get("document_role") if doc.extra else "",
        "details": doc.extra.get("snippet") if doc.extra else "",
        "abstract": "",
        "action": "",
        "dates": "",
        "supplementary_information": "",
        "history_parent_docket": doc.collection_id,
        "history_parent_fr_doc": doc.doc_id,
        "history_stage": doc.kind or "",
        "history_relationship": relationship,
        "history_rank": str(history_rank),
        "mentions_comment_response": False,
        "comment_citation_snippet": "",
    }
