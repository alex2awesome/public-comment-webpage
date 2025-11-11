#!/usr/bin/env python3
"""Run the GOV.UK harvesting pipeline entirely in process."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import logging
from pathlib import Path
from typing import Dict, Iterable, List

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_corpus.cli.main import _docmeta_from_dict, _load_sources  # type: ignore[attr-defined]
from ai_corpus.config.loader import load_config
from ai_corpus.connectors.factory import build_connector
from ai_corpus.pipelines.discover import discover_collections
from ai_corpus.pipelines.download import download_call, download_responses
from ai_corpus.pipelines.extract import extract_text_from_artifacts
from ai_corpus.pipelines.harvest import harvest_call_document, harvest_documents
from ai_corpus.pipelines.normalize import normalize_doc
from ai_corpus.storage.db import Database
from ai_corpus.storage.fs import BlobStore


logger = logging.getLogger("gov_uk_pipeline")


DEFAULT_FALLBACKS: Dict[str, List[str]] = {
    "gov_uk": ["ai-white-paper-2023"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ai_corpus pipeline steps for a connector.")
    parser.add_argument("--connector", default="gov_uk", help="Connector name to run (default: gov_uk).")
    parser.add_argument("--collection-id", help="Specific collection ID to process; skips discovery when provided.")
    parser.add_argument("--start-date", help="Optional start date for discovery (YYYY-MM-DD).")
    parser.add_argument("--end-date", help="Optional end date for discovery (YYYY-MM-DD).")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/comments/gov_uk"),
        help="Directory for data artifacts (default: data/comments/gov_uk).",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/comments/ai_pipeline.sqlite"),
        help="SQLite database path for downloads (default: data/comments/ai_pipeline.sqlite).",
    )
    parser.add_argument(
        "--database-url",
        default="sqlite:///data/app_data/ai_corpus.db",
        help="Destination database URL for export (default: sqlite:///data/app_data/ai_corpus.db).",
    )
    parser.add_argument(
        "--blob-dir",
        type=Path,
        default=Path("data/comments/blobs"),
        help="Blob storage directory (default: data/comments/blobs).",
    )
    parser.add_argument("--max-workers", type=int, default=1, help="Worker count for the download stage (default: 1).")
    parser.add_argument("--verbose", action="store_true", help="Enable additional pipeline logging.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for the connector (equivalent to --debug-connector).",
    )
    return parser.parse_args()


def write_meta(meta_path: Path, docs: Iterable) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as handle:
        for doc in docs:
            payload = dataclasses.asdict(doc)
            payload["kind"] = doc.kind
            handle.write(json.dumps(payload) + "\n")


def export_meta(meta_path: Path, download_db: Database, destination_db: Database) -> None:
    buffer: List = []
    with meta_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            meta = _docmeta_from_dict(json.loads(line))
            download_record = download_db.get_download(meta.doc_id)
            if not download_record or not download_record.get("text_path"):
                continue
            normalized = normalize_doc(meta)
            payload = download_record.get("payload") or {}
            normalized.pdf_path = payload.get("pdf")
            normalized.text_path = download_record.get("text_path")
            normalized.sha256_text = download_record.get("sha256_text")
            buffer.append(normalized)
            if len(buffer) >= 50:
                destination_db.upsert_documents(buffer)
                buffer.clear()
    if buffer:
        destination_db.upsert_documents(buffer)


def extract_pending(download_db: Database, blob_store: BlobStore) -> None:
    pending = list(download_db.iter_downloads(extracted=False))
    if not pending:
        if sys.stdout.isatty():
            tqdm.write("No pending downloads to extract.")
        return

    progress = tqdm(total=len(pending), desc="Extracting", unit="doc", dynamic_ncols=True)
    for record in pending:
        artifact = record.get("payload") or {}
        text = extract_text_from_artifacts(artifact)
        if text:
            stored_path = blob_store.store_bytes(text.encode("utf-8"), suffix="txt")
            download_db.mark_extracted(
                record["doc_id"],
                text_path=str(stored_path),
                sha256_text=stored_path.name,
            )
        progress.update(1)
    progress.close()


def main() -> int:
    args = parse_args()

    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s %(filename)s:%(lineno)d - %(message)s",
        force=True,
    )
    logger.setLevel(log_level)

    data_dir = args.data_dir.expanduser().resolve()
    blob_dir = args.blob_dir.expanduser().resolve()
    database_path = args.database.expanduser().resolve()

    data_dir.mkdir(parents=True, exist_ok=True)
    blob_dir.mkdir(parents=True, exist_ok=True)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    config = load_config()
    global_cfg, sources_cfg = _load_sources(config)

    if args.debug:
        connector_cfg = sources_cfg.setdefault(args.connector, {})
        connector_cfg["debug"] = True
        connector_cfg["headless"] = False

    def build() -> object:
        return build_connector(args.connector, sources_cfg.get(args.connector, {}), global_cfg)

    if args.collection_id:
        collections: List[Dict[str, str | None]] = [{"collection_id": args.collection_id, "title": None}]
    else:
        connector = build()
        discover_kwargs: Dict[str, object] = {}
        if args.start_date:
            discover_kwargs["start_date"] = args.start_date
        if args.end_date:
            discover_kwargs["end_date"] = args.end_date
        discovered = discover_collections(connector, **discover_kwargs)
        if discovered:
            collections = [{"collection_id": item.collection_id, "title": item.title} for item in discovered]
        else:
            fallback_ids = DEFAULT_FALLBACKS.get(args.connector, [])
            if not fallback_ids:
                raise RuntimeError(f"No collections discovered for connector '{args.connector}'.")
            logger.warning(
                "Discovery returned no results; using fallback collections: %s",
                ", ".join(fallback_ids),
                stacklevel=2,
            )
            collections = [{"collection_id": cid, "title": None} for cid in fallback_ids]

    manifest_path = data_dir / "collections_manifest.json"
    manifest_path.write_text(json.dumps(collections, indent=2), encoding="utf-8")

    logger.info(
        "Processing collections: %s",
        [item["collection_id"] for item in collections],
        stacklevel=1,
    )

    download_db = Database(f"sqlite:///{database_path}")
    destination_db = Database(args.database_url)
    blob_store = BlobStore(blob_dir)

    for entry in collections:
        collection_id = entry.get("collection_id")
        if not collection_id:
            continue

        logger.info("Harvesting '%s'", collection_id, stacklevel=1)
        response_docs = list(harvest_documents(build(), collection_id=collection_id))
        call_docs = list(harvest_call_document(build(), collection_id=collection_id))
        all_docs = response_docs + call_docs
        if not all_docs:
            logger.warning(
                "No documents harvested for %s:%s. Skipping.",
                args.connector,
                collection_id,
                stacklevel=1,
            )
            continue

        collection_path = collection_id.lstrip("/").replace("/", "_")
        collection_dir = data_dir / collection_path
        meta_path = collection_dir / f"{collection_path}.meta.jsonl"
        write_meta(meta_path, all_docs)

        raw_dir = collection_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        if call_docs:
            logger.info(
                "Downloading call document(s) for '%s' into %s",
                meta_path,
                raw_dir,
                stacklevel=1,
            )
            call_connector = build()
            for doc in call_docs:
                download_call(
                    call_connector,
                    doc,
                    raw_dir,
                    database=download_db,
                    use_cache=True,
                )
        if response_docs:
            logger.info(
                "Downloading response documents for '%s' into %s",
                meta_path,
                raw_dir,
                stacklevel=1,
            )
            download_responses(
                build(),
                response_docs,
                raw_dir,
                database=download_db,
                max_workers=args.max_workers,
                use_cache=True,
            )

        logger.info("Extracting text for '%s'", collection_id, stacklevel=1)
        extract_pending(download_db, blob_store)

        logger.info("Exporting normalized documents for '%s'", collection_id, stacklevel=1)
        export_meta(meta_path, download_db, destination_db)

    print(f"[pipeline] Completed {len(collections)} collection(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
