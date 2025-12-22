"""SQLite-backed cache used to replicate Semantic Scholar interactions."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional


class CacheDB:
    """Simple SQLite cache that stores queries, results, and paper snapshots."""

    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS queries (
              query_id TEXT PRIMARY KEY,
              query_text TEXT NOT NULL,
              params_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS query_results (
              query_id TEXT NOT NULL,
              rank INTEGER NOT NULL,
              paper_id TEXT NOT NULL,
              PRIMARY KEY (query_id, rank)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS papers (
              paper_id TEXT PRIMARY KEY,
              title TEXT,
              year INTEGER,
              venue TEXT,
              abstract TEXT,
              url TEXT,
              raw_json TEXT,
              fetched_at TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
              run_id TEXT PRIMARY KEY,
              task_id TEXT NOT NULL,
              use_cached INTEGER NOT NULL,
              cache_path TEXT NOT NULL,
              created_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def record_run(self, run_id: str, task_id: str, use_cached: bool) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO runs (run_id, task_id, use_cached, cache_path, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                run_id,
                task_id,
                1 if use_cached else 0,
                self.path,
                _ts(),
            ),
        )
        self._conn.commit()

    def cached_or_fetch_semantic_scholar_search(
        self,
        use_cached: bool,
        query: str,
        params: Dict[str, Any],
        fetch_fn: Callable[[], Iterable[Dict[str, Any]]],
        top_k: int,
    ) -> Dict[str, Any]:
        """Return cached results or fetch from Semantic Scholar."""
        query_id = self._query_hash(query, params)
        cached = self._load_query_results(query_id)
        need_fetch = (not use_cached) or (not cached)

        if need_fetch:
            data = list(fetch_fn())
            self._store_query(query_id, query, params, data)
            cached = self._load_query_results(query_id)

        return {
            "query_id": query_id,
            "query": query,
            "results": cached[:top_k],
            "cached": not need_fetch,
        }

    def cached_or_fetch_semantic_scholar_paper(
        self,
        use_cached: bool,
        paper_id: str,
        fields: str,
        fetch_fn: Callable[[], Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Return cached paper metadata or fetch it."""
        cached = self.get_paper(paper_id)
        need_fetch = (not use_cached) or (cached is None)

        if need_fetch:
            raw = fetch_fn()
            self._store_paper(raw)
            cached = self.get_paper(paper_id)

        if not cached:
            raise ValueError(f"Unable to load paper_id={paper_id}")
        cached["cached"] = not need_fetch
        cached["fields"] = fields
        return cached

    def get_paper(self, paper_id: str) -> Optional[Dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT paper_id, title, year, venue, abstract, url, raw_json, fetched_at FROM papers WHERE paper_id = ?",
            (paper_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        payload = {
            "paperId": row["paper_id"],
            "title": row["title"],
            "year": row["year"],
            "venue": row["venue"],
            "abstract": row["abstract"],
            "url": row["url"],
            "fetched_at": row["fetched_at"],
        }
        if row["raw_json"]:
            try:
                payload["raw"] = json.loads(row["raw_json"])
            except json.JSONDecodeError:
                payload["raw"] = row["raw_json"]
        return payload

    # --------------------------- Internal helpers --------------------------- #
    def _store_query(
        self,
        query_id: str,
        query: str,
        params: Dict[str, Any],
        items: List[Dict[str, Any]],
    ) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO queries (query_id, query_text, params_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                query_id,
                query,
                json.dumps(params, sort_keys=True),
                _ts(),
            ),
        )
        cur.execute("DELETE FROM query_results WHERE query_id = ?", (query_id,))

        for rank, item in enumerate(items):
            paper_id = item.get("paperId")
            if not paper_id:
                continue
            cur.execute(
                "INSERT OR REPLACE INTO query_results (query_id, rank, paper_id) VALUES (?, ?, ?)",
                (query_id, rank, paper_id),
            )
            self._store_paper(item)
        self._conn.commit()

    def _store_paper(self, payload: Dict[str, Any]) -> None:
        paper_id = payload.get("paperId")
        if not paper_id:
            return
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO papers (paper_id, title, year, venue, abstract, url, raw_json, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                paper_id,
                payload.get("title"),
                payload.get("year"),
                payload.get("venue"),
                payload.get("abstract"),
                payload.get("url") or (payload.get("openAccessPdf") or {}).get("url"),
                json.dumps(payload, sort_keys=True),
                _ts(),
            ),
        )
        self._conn.commit()

    def _load_query_results(self, query_id: str) -> List[Dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT qr.rank,
                   qr.paper_id,
                   p.title,
                   p.year,
                   p.venue,
                   p.abstract,
                   p.url,
                   p.raw_json
            FROM query_results qr
            LEFT JOIN papers p ON qr.paper_id = p.paper_id
            WHERE qr.query_id = ?
            ORDER BY qr.rank ASC
            """,
            (query_id,),
        )
        results: List[Dict[str, Any]] = []
        for row in cur.fetchall():
            payload = {
                "rank": row["rank"],
                "paperId": row["paper_id"],
                "title": row["title"],
                "year": row["year"],
                "venue": row["venue"],
                "abstract": row["abstract"],
                "url": row["url"],
            }
            if row["raw_json"]:
                try:
                    payload["raw"] = json.loads(row["raw_json"])
                except json.JSONDecodeError:
                    payload["raw"] = row["raw_json"]
            results.append(payload)
        return results

    @staticmethod
    def _query_hash(query: str, params: Dict[str, Any]) -> str:
        payload = {
            "query": query,
            "params": params or {},
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
