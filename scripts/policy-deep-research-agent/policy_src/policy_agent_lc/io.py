"""Simple JSONL helpers with file locks for concurrent writers."""

from __future__ import annotations

import json
from pathlib import Path

from filelock import FileLock


def append_jsonl(path: str, record: dict) -> None:
    """Append a JSON record to a file, protecting writes with a lock."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(output_path) + ".lock")
    with lock:
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
