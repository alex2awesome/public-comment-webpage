"""
Harvest pipeline helpers for listing document metadata from a collection.
"""

from __future__ import annotations

from typing import Iterable, List

from ai_corpus.connectors.base import BaseConnector, DocMeta


def harvest_documents(
    connector: BaseConnector,
    collection_id: str,
    **kwargs,
) -> List[DocMeta]:
    """
    Materialize document metadata from the connector. Harvesting is purely about
    metadata enumeration; downloading is handled separately.
    """
    return list(connector.list_documents(collection_id=collection_id, **kwargs))


def harvest_call_document(
    connector: BaseConnector,
    collection_id: str,
    **kwargs,
) -> List[DocMeta]:
    """
    Request the originating call metadata from the connector. Returns an empty
    list if the connector does not expose a dedicated call document.
    """
    getter = getattr(connector, "get_call_document", None)
    if not callable(getter):
        return []
    doc = getter(collection_id=collection_id, **kwargs)
    if doc is None:
        return []
    doc.kind = "call"
    return [doc]
