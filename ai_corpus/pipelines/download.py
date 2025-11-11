"""
Pipeline helpers for fetching raw documents via connectors.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from ai_corpus.connectors.base import BaseConnector, DocMeta
from ai_corpus.storage.db import Database


def _download_documents(
    connector: BaseConnector,
    documents: List[DocMeta],
    out_dir: Path,
    *,
    database: Optional[Database] = None,
    max_workers: Optional[int] = None,
    use_cache: bool = True,
    **kwargs,
) -> List[Dict[str, object]]:
    """Internal helper used by the public download entrypoints.

    Mirrors the original ``download_documents`` behaviour while tagging each
    payload with ``document_kind`` so downstream components can distinguish
    call documents from responses.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, object]] = []
    if not documents:
        return results

    effective_workers = max_workers or getattr(connector, "max_workers", 4)
    effective_workers = max(1, effective_workers)

    def _fetch(doc: DocMeta) -> Dict[str, object]:
        return connector.fetch(doc=doc, out_dir=out_dir, skip_existing=use_cache, **kwargs)

    pending: List[DocMeta] = []
    def _needs_refetch(doc: DocMeta, payload: Dict[str, object]) -> bool:
        if not doc.extra.get("is_bundle"):
            return False
        letters = payload.get("letters")
        if not letters or not isinstance(letters, list):
            return True
        has_valid_path = False
        for entry in letters:
            if not isinstance(entry, dict):
                continue
            text_path = entry.get("text_path")
            if not text_path:
                continue
            has_valid_path = True
            if not Path(text_path).exists():
                return True
        return not has_valid_path

    if database and use_cache:
        for doc in documents:
            record = database.get_download(doc.doc_id)
            if record and record.get("payload"):
                payload = record["payload"].copy()
                if _needs_refetch(doc, payload):
                    pending.append(doc)
                    continue
                payload.setdefault("doc_id", doc.doc_id)
                payload.setdefault("collection_id", doc.collection_id)
                payload.setdefault("source", doc.source)
                payload.setdefault("document_kind", getattr(doc, "kind", "response"))
                results.append(payload)
            else:
                pending.append(doc)
    else:
        pending = list(documents)

    progress = tqdm(
        total=len(documents),
        desc="Downloading",
        unit="doc",
        disable=False,
        dynamic_ncols=True,
    )
    progress.update(len(results))

    if not pending:
        progress.close()
        return results

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        future_map = {executor.submit(_fetch, doc): doc for doc in pending}
        for future in as_completed(future_map):
            doc = future_map[future]
            try:
                result = future.result() or {}
            except Exception as exc:  # noqa: BLE001
                progress.write(f"Download failed for {doc.doc_id}: {exc}")
                progress.update(1)
                continue
            result.setdefault("doc_id", doc.doc_id)
            result.setdefault("collection_id", doc.collection_id)
            result.setdefault("source", doc.source)
            result.setdefault("document_kind", getattr(doc, "kind", "response"))
            results.append(result)
            if database:
                database.record_download(
                    connector_name=doc.source,
                    collection_id=doc.collection_id,
                    doc_id=doc.doc_id,
                    payload=result,
                )
            progress.update(1)

    progress.close()
    return results


def download_responses(
    connector: BaseConnector,
    documents: List[DocMeta],
    out_dir: Path,
    *,
    database: Optional[Database] = None,
    max_workers: Optional[int] = None,
    use_cache: bool = True,
    **kwargs,
) -> List[Dict[str, object]]:
    """Download response documents (the original behaviour)."""
    for doc in documents:
        if getattr(doc, "kind", None) is None:
            doc.kind = "response"  # type: ignore[attr-defined]
    return _download_documents(
        connector,
        documents,
        out_dir,
        database=database,
        max_workers=max_workers,
        use_cache=use_cache,
        **kwargs,
    )


def download_call(
    connector: BaseConnector,
    document: DocMeta,
    out_dir: Path,
    *,
    database: Optional[Database] = None,
    use_cache: bool = True,
    **kwargs,
) -> Dict[str, object]:
    """Download the originating call (e.g., RFI/RFC notice) for a collection."""
    document.kind = "call"
    results = _download_documents(
        connector,
        [document],
        out_dir,
        database=database,
        max_workers=1,
        use_cache=use_cache,
        **kwargs,
    )
    return results[0] if results else {}


def download_documents(
    connector: BaseConnector,
    documents: List[DocMeta],
    out_dir: Path,
    *,
    database: Optional[Database] = None,
    max_workers: Optional[int] = None,
    use_cache: bool = True,
    **kwargs,
) -> List[Dict[str, object]]:
    """
    Backwards-compatible shim that delegates to ``download_responses``.

    Planned for deprecation once callers migrate to ``download_responses``.
    """
    return download_responses(
        connector,
        documents,
        out_dir,
        database=database,
        max_workers=max_workers,
        use_cache=use_cache,
        **kwargs,
    )
