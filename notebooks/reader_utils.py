"""Utilities for loading AI corpus datasets used in the notebook."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

__all__ = [
    "resolve_project_root",
    "get_project_paths",
    "connect_export_db",
    "load_text_df",
    "merge_label_results",
    "read_text_blob",
    "load_api_key",
]


def resolve_project_root(start: Optional[Path] = None) -> Path:
    """Return the repository root based on the current working directory."""
    current = (start or Path.cwd()).resolve()
    if (current / "data").exists():
        return current
    if (current.parent / "data").exists():
        return current.parent
    raise FileNotFoundError(
        "Run this notebook from the repository root or provide a project root with a 'data' directory."
    )


def get_project_paths(project_root: Path) -> dict[str, Path]:
    """Return commonly used data paths."""
    comments_root = project_root / "data" / "comments"
    app_data_root = project_root / "data" / "app_data"
    return {
        "project_root": project_root,
        "comments_root": comments_root,
        "app_data_root": app_data_root,
        "blob_root": comments_root / "blobs",
        "download_db_path": comments_root / "ai_pipeline.sqlite",
        "export_db_path": app_data_root / "ai_corpus.db",
    }


def connect_export_db(export_db_path: Path) -> sqlite3.Connection:
    """Return a SQLite connection to the export database."""
    return sqlite3.connect(export_db_path)


def _resolve_blob_path(project_root: Path, raw: str) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = project_root / p
    if p.exists():
        return p
    if "data/blobs/" in raw:
        alt = raw.replace("data/blobs/", "data/comments/blobs/")
        alt_path = project_root / Path(alt) if not Path(alt).is_absolute() else Path(alt)
        if alt_path.exists():
            return alt_path
    return p


def read_text_blob(project_root: Path, raw_path: str) -> str:
    """Read a blob file relative to the project."""
    p = _resolve_blob_path(project_root, raw_path)
    try:
        return p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"<missing blob: {p}>"
    except Exception as exc:  # pragma: no cover
        return f"<error reading {p}: {exc}>"


def load_text_df(export_db_path: Path, project_root: Path) -> pd.DataFrame:
    """Load the document table and attach raw text content."""
    with connect_export_db(export_db_path) as conn:
        text_sql = """
        SELECT doc_id, source, collection_id, text_path
        FROM documents
        WHERE text_path IS NOT NULL
        ORDER BY source, collection_id, doc_id
        """
        df = pd.read_sql(text_sql, conn)
    df["text"] = df["text_path"].apply(lambda raw: read_text_blob(project_root, raw))
    return df


def merge_label_results(
    text_df: pd.DataFrame, label_results: Iterable[dict]
) -> pd.DataFrame:
    """Attach model label outputs to the document dataframe."""
    label_df = pd.DataFrame(label_results)
    return pd.concat([text_df.reset_index(drop=True), label_df.reset_index(drop=True)], axis=1)


def load_api_key(path: str) -> None:
    """Read an API key from disk and put it in OPENAI_API_KEY."""
    from pathlib import Path as _Path

    key = _Path(path).read_text().strip()
    os.environ["OPENAI_API_KEY"] = key
