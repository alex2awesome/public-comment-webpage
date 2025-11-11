"""
Filesystem storage utilities for managing content-addressed blobs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional


class BlobStore:
    """
    Content-addressed blob store that organizes files by SHA256 prefix.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for_hash(self, digest: str) -> Path:
        return self.root / digest[:2] / digest

    def store_bytes(self, data: bytes, suffix: Optional[str] = None) -> Path:
        sha = hashlib.sha256(data).hexdigest()
        dest = self._path_for_hash(sha)
        if suffix:
            dest = dest.with_suffix(suffix if suffix.startswith(".") else f".{suffix}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_bytes(data)
        return dest

    def store_file(self, source: Path, suffix: Optional[str] = None) -> Path:
        data = source.read_bytes()
        return self.store_bytes(data, suffix=suffix or source.suffix)

