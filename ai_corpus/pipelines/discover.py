"""
Discovery pipeline utilities for enumerating collections from configured sources.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

from ai_corpus.connectors.base import BaseConnector, Collection


def discover_collections(
    connector: BaseConnector,
    **kwargs,
) -> List[Collection]:
    """
    Helper that materializes the discovered collections into a list so callers
    can serialize or further process them.
    """
    return list(connector.discover(**kwargs))

