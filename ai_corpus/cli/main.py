"""
CLI entrypoint for the AI corpus harvester.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from ai_corpus.config.loader import load_config
from ai_corpus.connectors.base import Collection, DocMeta
from ai_corpus.connectors.factory import build_connector, CONNECTOR_MAP
from ai_corpus.pipelines.discover import discover_collections
from ai_corpus.pipelines.download import download_call, download_responses
from ai_corpus.pipelines.harvest import harvest_call_document, harvest_documents
from ai_corpus.pipelines.normalize import NormalizedDocument, normalize_doc
from ai_corpus.pipelines.extract import extract_text_from_artifacts
from ai_corpus.rules import RULE_VERSION_COLUMNS, normalize_rule_row
from ai_corpus.storage.db import Database, DEFAULT_DB_URL
from ai_corpus.storage.fs import BlobStore
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-corpus",
        description="Harvest public comments about AI regulation from multiple sources.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to an alternative YAML config file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--debug-connector",
        action="append",
        dest="debug_connectors",
        default=None,
        help="Enable debug mode for specified connector(s); repeat flag to target multiple connectors.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover_parser = subparsers.add_parser("discover", help="List available collections.")
    discover_parser.add_argument(
        "--connector",
        action="append",
        dest="connectors",
        help="Connector(s) to run discovery against (defaults to all).",
    )
    discover_parser.add_argument(
        "--start-date",
        dest="start_date",
        help="ISO date (YYYY-MM-DD) or timestamp to filter collections on or after this date.",
    )
    discover_parser.add_argument(
        "--end-date",
        dest="end_date",
        help="ISO date (YYYY-MM-DD) or timestamp to filter collections on or before this date.",
    )

    crawl_parser = subparsers.add_parser("crawl", help="Harvest document metadata for a collection.")
    crawl_parser.add_argument("--connector", required=True)
    crawl_parser.add_argument("--collection-id", required=True)
    crawl_parser.add_argument("--output", type=Path, required=True, help="Path to write JSONL metadata.")
    crawl_parser.add_argument(
        "--target",
        choices=["responses", "call", "all"],
        default="responses",
        help="Select which metadata to harvest: individual responses, the originating call, or both.",
    )

    download_call_parser = subparsers.add_parser(
        "download-call",
        help="Download the originating call (RFI/RFC) listed in metadata.",
    )
    download_call_parser.add_argument("--connector", required=True)
    download_call_parser.add_argument("--collection-id", required=True)
    download_call_parser.add_argument("--meta-file", type=Path, required=True, help="JSONL produced by crawl.")
    download_call_parser.add_argument("--out-dir", type=Path, required=True, help="Directory for downloaded files.")
    download_call_parser.add_argument("--database", type=Path, required=True, help="SQLite database to store download records.")
    download_call_parser.add_argument("--no-cache", action="store_true", help="Force re-download even if cached.")

    download_responses_parser = subparsers.add_parser(
        "download-responses",
        help="Download response documents listed in metadata.",
    )
    download_responses_parser.add_argument("--connector", required=True)
    download_responses_parser.add_argument("--collection-id", required=True)
    download_responses_parser.add_argument("--meta-file", type=Path, required=True, help="JSONL produced by crawl.")
    download_responses_parser.add_argument("--out-dir", type=Path, required=True, help="Directory for downloaded files.")
    download_responses_parser.add_argument("--database", type=Path, required=True, help="SQLite database to store download records.")
    download_responses_parser.add_argument("--max-workers", type=int, default=None, help="Parallel download workers (default connector preference).")
    download_responses_parser.add_argument("--no-cache", action="store_true", help="Force re-download even if cached.")

    extract_parser = subparsers.add_parser("extract", help="Extract text from downloaded artifacts.")
    extract_parser.add_argument("--database", type=Path, required=True, help="SQLite database with download records.")
    extract_parser.add_argument("--blob-dir", type=Path, required=True, help="Directory for blob storage.")
    extract_parser.add_argument("--max-workers", type=int, default=1, help="Parallel extraction workers.")

    export_parser = subparsers.add_parser("export", help="Normalize metadata and persist to the database.")
    export_parser.add_argument("--meta-file", type=Path, required=True)
    export_parser.add_argument("--database", type=Path, required=True)
    export_parser.add_argument("--database-url", type=str, default=DEFAULT_DB_URL)
    export_parser.add_argument(
        "--blob-dir",
        type=Path,
        default=Path("data/comments/blobs"),
        help="Blob storage directory for normalized text content (default: data/comments/blobs).",
    )

    rules_parser = subparsers.add_parser("rules", help="Export rule-version timelines to CSV.")
    rules_parser.add_argument(
        "--connector",
        action="append",
        dest="connectors",
        help="Connector(s) to include (defaults to all connectors with rule support).",
    )
    rules_parser.add_argument(
        "--collection-id",
        action="append",
        dest="collection_ids",
        help="Limit to specific collections using connector:collection_id (repeat flag to add more).",
    )
    rules_parser.add_argument(
        "--start-date",
        dest="rules_start_date",
        help="ISO date to include dockets last modified on/after this date (passed to discovery).",
    )
    rules_parser.add_argument(
        "--end-date",
        dest="rules_end_date",
        help="ISO date to include dockets last modified on/before this date (passed to discovery).",
    )
    rules_parser.add_argument(
        "--query",
        dest="rules_query",
        help="Optional search term passed to connector discovery (e.g., keyword for Regulations.gov).",
    )
    rules_parser.add_argument(
        "--cache-db",
        type=Path,
        default=Path("data/harvest_cache.sqlite"),
        help="Path to the shared harvest cache database (default: data/harvest_cache.sqlite).",
    )
    rules_parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore cached harvest entries and re-pull collections even when they exist.",
    )
    rules_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination CSV path for rule-version rows.",
    )

    return parser


def _serialize_collections(collections: Iterable[Collection]) -> List[Dict]:
    return [dataclasses.asdict(collection) for collection in collections]


def _serialize_docs(docs: Iterable[DocMeta]) -> List[Dict]:
    return [dataclasses.asdict(doc) for doc in docs]


def _load_sources(config: Dict) -> Tuple[Dict, Dict]:
    global_cfg = config.get("global", {})
    sources_cfg = {k: v for k, v in config.items() if k != "global"}
    return global_cfg, sources_cfg


def _ensure_connectors(requested: List[str] | None, sources_cfg: Dict, global_cfg: Dict):
    names = requested or [name for name in sources_cfg if name in CONNECTOR_MAP]
    for name in names:
        cfg = sources_cfg.get(name, {})
        yield name, build_connector(name, cfg, global_cfg)


def _parse_collection_overrides(values: List[str] | None) -> Dict[str, List[str]]:
    overrides: Dict[str, List[str]] = {}
    if not values:
        return overrides
    for item in values:
        if ":" not in item:
            raise SystemExit(f"Invalid --collection-id value '{item}'. Expected format connector:collection_id.")
        connector, collection_id = item.split(":", 1)
        connector = connector.strip()
        collection_id = collection_id.strip()
        if not connector or not collection_id:
            raise SystemExit(f"Invalid --collection-id value '{item}'.")
        overrides.setdefault(connector, []).append(collection_id)
    return overrides


def _docmeta_from_dict(data: Dict) -> DocMeta:
    return DocMeta(
        source=data["source"],
        collection_id=data["collection_id"],
        doc_id=data["doc_id"],
        title=data.get("title"),
        submitter=data.get("submitter"),
        submitter_type=data.get("submitter_type"),
        org=data.get("org"),
        submitted_at=data.get("submitted_at"),
        language=data.get("language"),
        urls=data.get("urls") or {},
        extra=data.get("extra") or {},
        kind=data.get("kind") or data.get("document_kind") or "response",
    )


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in (
        "pdfminer",
        "pdfminer.psparser",
        "pdfminer.pdfdocument",
        "PIL",
        "PIL.PngImagePlugin",
        "pytesseract",
    ):
        noise_logger = logging.getLogger(noisy)
        noise_logger.setLevel(logging.WARNING)
        noise_logger.propagate = False

    config = load_config(args.config)
    global_cfg, sources_cfg = _load_sources(config)
    debug_connectors = set(args.debug_connectors or [])
    for connector_name in debug_connectors:
        connector_cfg = sources_cfg.setdefault(connector_name, {})
        connector_cfg["debug"] = True
        connector_cfg["headless"] = False
    global_flags = {"verbose": args.verbose}

    if args.command == "discover":
        results = {}
        discover_kwargs = {}
        if args.start_date:
            discover_kwargs["start_date"] = args.start_date
        if args.end_date:
            discover_kwargs["end_date"] = args.end_date
        for name, connector in _ensure_connectors(args.connectors, sources_cfg, global_cfg):
            collections = discover_collections(connector, **discover_kwargs)
            results[name] = _serialize_collections(collections)
        json.dump(results, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    if args.command == "rules":
        overrides = _parse_collection_overrides(args.collection_ids)
        discover_kwargs = {}
        if args.rules_start_date:
            discover_kwargs["start_date"] = args.rules_start_date
        if args.rules_end_date:
            discover_kwargs["end_date"] = args.rules_end_date
        if args.rules_query:
            discover_kwargs["query"] = args.rules_query
        if not overrides and not discover_kwargs:
            tqdm.write(
                "[rules] No collection ids or discovery filters provided; defaulting to each "
                "connector's configured seeds."
            )
        cache_db_path = args.cache_db.expanduser().resolve()
        cache_db_path.parent.mkdir(parents=True, exist_ok=True)
        cache_db = Database(f"sqlite:///{cache_db_path}")
        artifact_type = "rules"
        total_rows = 0
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=RULE_VERSION_COLUMNS)
            writer.writeheader()
            for name, connector in _ensure_connectors(args.connectors, sources_cfg, global_cfg):
                iterator = getattr(connector, "iter_rule_versions", None)
                if not callable(iterator):
                    logging.warning("Connector %s does not implement iter_rule_versions; skipping.", name)
                    continue
                collections = overrides.get(name)
                if not collections:
                    discovered = discover_collections(connector, **discover_kwargs)
                    collections = [c.collection_id for c in discovered]
                if not collections:
                    logging.warning("No collections found for connector %s; skipping.", name)
                    continue
                for collection_id in collections:
                    cached = cache_db.get_harvest(name, collection_id, artifact_type)
                    if cached and not args.refresh_cache:
                        tqdm.write(
                            f"[rules] Skipping {name}:{collection_id} "
                            f"(cached {cached['item_count']} rows at {cached['last_run_at']} -> {cached['output_path']})"
                        )
                        continue
                    new_rows = 0
                    last_rank = None
                    file_offset = total_rows
                    try:
                        for row in iterator(collection_id=collection_id):
                            payload = dict(row)
                            payload.setdefault("source", name)
                            payload.setdefault("docket_id", collection_id)
                            writer.writerow(normalize_rule_row(payload))
                            total_rows += 1
                            new_rows += 1
                            last_rank = row.get("history_rank")
                        fh.flush()
                    except Exception as exc:  # noqa: BLE001
                        logging.warning("Failed to collect rule versions for %s:%s (%s)", name, collection_id, exc)
                    finally:
                        cache_db.record_harvest(
                            connector=name,
                            collection_id=collection_id,
                            artifact_type=artifact_type,
                            item_count=new_rows,
                            output_path=str(args.output),
                            metadata={
                                "last_history_rank": last_rank,
                                "file_row_offset": file_offset,
                            },
                        )
        tqdm.write(f"[rules] Wrote {total_rows} rows to {args.output}")
        return 0

    if args.command == "crawl":
        connector = build_connector(args.connector, sources_cfg.get(args.connector, {}), global_cfg)
        docs: List[DocMeta] = []
        if args.target in {"responses", "all"}:
            docs.extend(harvest_documents(connector, collection_id=args.collection_id))
        if args.target in {"call", "all"}:
            docs.extend(harvest_call_document(connector, collection_id=args.collection_id))
        if not docs:
            tqdm.write(
                f"[crawl] No documents harvested for {args.connector}:{args.collection_id} "
                f"(target={args.target}). "
                "Check the connector logic or upstream source; aborting."
            )
            return 1
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as fh:
            for doc in docs:
                payload = dataclasses.asdict(doc)
                payload["kind"] = doc.kind
                fh.write(json.dumps(payload) + "\n")
        return 0

    if args.command == "download-call":
        connector = build_connector(args.connector, sources_cfg.get(args.connector, {}), global_cfg)
        meta_items: List[DocMeta] = []
        with args.meta_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                item = _docmeta_from_dict(json.loads(line))
                if item.kind == "call":
                    meta_items.append(item)
        if not meta_items:
            tqdm.write(
                f"[download-call] No call documents found in {args.meta_file} "
                f"for {args.connector}:{args.collection_id}."
            )
            return 0
        db_path = args.database.expanduser().resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        database = Database(f"sqlite:///{db_path}")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for doc in meta_items:
            download_call(
                connector,
                doc,
                args.out_dir,
                database=database,
                use_cache=not args.no_cache,
            )
        return 0

    if args.command == "download-responses":
        connector = build_connector(args.connector, sources_cfg.get(args.connector, {}), global_cfg)
        meta_items: List[DocMeta] = []
        with args.meta_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    item = _docmeta_from_dict(json.loads(line))
                    if item.kind != "call":
                        meta_items.append(item)
        db_path = args.database.expanduser().resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        database = Database(f"sqlite:///{db_path}")
        download_responses(
            connector,
            meta_items,
            args.out_dir,
            database=database,
            max_workers=args.max_workers,
            use_cache=not args.no_cache,
        )
        return 0

    if args.command == "extract":
        db_path = args.database.expanduser().resolve()
        database = Database(f"sqlite:///{db_path}")
        blob_store = BlobStore(args.blob_dir)
        pending = list(database.iter_downloads(extracted=False))
        if not pending:
            tqdm.write("No pending downloads to extract.")
            return 0
        progress = tqdm(
            total=len(pending),
            desc="Extracting",
            unit="doc",
            disable=False,
            dynamic_ncols=True,
        )
        for record in pending:
            artifact = record.get("payload") or {}
            text = extract_text_from_artifacts(artifact)
            if text:
                stored_path = blob_store.store_bytes(text.encode("utf-8"), suffix="txt")
                database.mark_extracted(
                    record["doc_id"],
                    text_path=str(stored_path),
                    sha256_text=stored_path.name,
                )
            progress.update(1)
        progress.close()
        return 0

    if args.command == "export":
        download_db_path = args.database.expanduser().resolve()
        download_db_path.parent.mkdir(parents=True, exist_ok=True)
        download_db = Database(f"sqlite:///{download_db_path}")
        blob_store = BlobStore(args.blob_dir)
        destination_db = Database(args.database_url)
        buffer: List[NormalizedDocument] = []

        def _flush():
            if buffer:
                destination_db.upsert_documents(buffer)
                buffer.clear()

        with args.meta_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                meta = _docmeta_from_dict(json.loads(line))
                download_record = download_db.get_download(meta.doc_id)
                if not download_record:
                    continue
                payload = download_record.get("payload") or {}
                letters = payload.get("letters") if isinstance(payload, dict) else None
                if letters:
                    base = normalize_doc(meta)
                    base_dict = dataclasses.asdict(base)
                    for index, letter in enumerate(letters, start=1):
                        text_path_value = letter.get("text_path") if isinstance(letter, dict) else None
                        if not text_path_value:
                            continue
                        text_path = Path(text_path_value)
                        if not text_path.exists():
                            continue
                        text_bytes = text_path.read_bytes()
                        stored_blob = blob_store.store_bytes(text_bytes, suffix="txt")
                        letter_doc_id = f"{meta.doc_id}#L{index:03d}"
                        letter_payload = dict(base_dict)
                        letter_payload.update(
                            {
                                "uid": hashlib.sha256(
                                    "|".join([base.source, base.collection_id, letter_doc_id]).encode("utf-8")
                                ).hexdigest(),
                                "doc_id": letter_doc_id,
                                "title": letter.get("submitter_name") or base.title,
                                "submitter_name": letter.get("submitter_name"),
                                "submitted_at": letter.get("submitted_at"),
                                "text_path": str(stored_blob),
                                "sha256_text": stored_blob.name,
                                "bytes_text": len(text_bytes),
                                "pdf_path": payload.get("pdf"),
                                "raw_meta": {
                                    "bundle_meta": base.raw_meta,
                                    "letter_meta": letter,
                                    "bundle_doc_id": meta.doc_id,
                                    "bundle_pdf_path": payload.get("pdf"),
                                    "letter_index": index,
                                    "original_letter_path": str(text_path),
                                },
                            }
                        )
                        buffer.append(NormalizedDocument(**letter_payload))
                        if len(buffer) >= 50:
                            _flush()
                    continue

                if not download_record.get("text_path"):
                    continue
                normalized = normalize_doc(meta)
                normalized.pdf_path = payload.get("pdf")
                normalized.text_path = download_record.get("text_path")
                normalized.sha256_text = download_record.get("sha256_text")
                buffer.append(normalized)
                if len(buffer) >= 50:
                    _flush()

        _flush()
        return 0

    parser.error(f"Unhandled command {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
