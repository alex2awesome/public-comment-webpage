#!/usr/bin/env python3
"""
Merge inline CSV comments with parsed attachment text under downloaded_content/
into a single combined CSV containing canonical_text.

This mirrors the logic from notebooks/2026-02-09__bulk_data_explorer.ipynb (Cell 35).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None  # type: ignore[assignment]

if tqdm is None:  # pragma: no cover - tqdm is optional
    def tqdm(iterable: Iterable, **kwargs):
        return iterable


DOWNLOAD_DIR_NAME = "downloaded_content"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_DIR = SCRIPT_DIR.parent
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "bulk_downloads_all_text.csv"
YEAR_PATTERN = re.compile(r"(?:19|20)\d{2}")


def discover_csv_paths(base_dir: Path) -> List[Path]:
    csv_paths: List[Path] = []
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if d != DOWNLOAD_DIR_NAME]
        for file_name in files:
            if not file_name.endswith(".csv"):
                continue
            if file_name.endswith("_all_text.csv") or file_name.endswith("_1.csv"):
                continue
            csv_paths.append(Path(root) / file_name)
    return sorted(csv_paths)


def _extract_years(path: Path) -> List[str]:
    return YEAR_PATTERN.findall(str(path))


def _matches_years(path: Path, years: List[str]) -> bool:
    if not years:
        return True
    found = _extract_years(path)
    if not found:
        return False
    return any(year in found for year in years)


def _resolve_doc_id(doc_dir: Path) -> str:
    metadata_path = doc_dir / "metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
        except json.JSONDecodeError:
            metadata = {}
        if isinstance(metadata, dict):
            for key in [
                "Document ID",
                "DocumentId",
                "document_id",
                "doc_id",
                "DocumentID",
            ]:
                value = metadata.get(key)
                if value:
                    value_str = str(value).strip()
                    if value_str:
                        return value_str
    return doc_dir.name


def load_download_records(base_dir: Path) -> pd.DataFrame:
    records = []
    for agency_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        for collection_dir in sorted(p for p in agency_dir.iterdir() if p.is_dir()):
            download_root = collection_dir / DOWNLOAD_DIR_NAME
            if not download_root.is_dir():
                continue
            for csv_dir in sorted(p for p in download_root.iterdir() if p.is_dir()):
                for doc_dir in sorted(p for p in csv_dir.iterdir() if p.is_dir()):
                    doc_id = _resolve_doc_id(doc_dir)
                    for file_path in sorted(doc_dir.iterdir()):
                        if not file_path.is_file():
                            continue
                        name = file_path.name
                        if name == "metadata.json":
                            continue
                        if name.endswith((".processing", ".partial", ".stats.json")):
                            continue
                        records.append(
                            {
                                "agency": agency_dir.name,
                                "collection": collection_dir.name,
                                "csv_name": csv_dir.name,
                                "doc_id": doc_id,
                                "file_name": name,
                                "file_path": str(file_path),
                            }
                        )
    return pd.DataFrame(records)


def build_canonical_texts(
    csv_path: Path,
    csv_downloads: pd.DataFrame,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False).fillna("")
    if "Document ID" not in df.columns and "DocumentId" in df.columns:
        df = df.rename(columns={"DocumentId": "Document ID"})
    if "Document ID" not in df.columns:
        raise ValueError(f"{csv_path} is missing a Document ID column")

    comment_cols = [
        col
        for col in df.columns
        if ("comment" in col.lower()) and ("comment on" not in col.lower())
    ]
    inline_series = (
        df[comment_cols]
        .fillna("")
        .agg(
            lambda row: "\n\n".join(
                [text.strip() for text in row if str(text).strip()]
            ),
            axis=1,
        )
        if comment_cols
        else pd.Series("", index=df.index)
    )

    derived_texts = (
        csv_downloads.loc[lambda d: d["file_name"].str.endswith(".txt", na=False)]
        .assign(
            text=lambda pdf: pdf["file_path"].apply(
                lambda p: Path(p).read_text(encoding="utf-8", errors="ignore").strip()
                if Path(p).exists()
                else ""
            )
        )
        .loc[lambda d: d["text"].str.len() > 0]
    )
    doc_text_map: Dict[str, List[str]] = (
        derived_texts.groupby("doc_id")["text"].apply(list).to_dict()
        if not derived_texts.empty
        else {}
    )

    def assemble(row: pd.Series) -> str:
        pieces: List[str] = []
        inline_val = str(row["inline_text"]).strip()
        if inline_val:
            pieces.append(inline_val)
        for extra in doc_text_map.get(row.get("Document ID", ""), []):
            cleaned = str(extra).strip()
            if cleaned:
                pieces.append(cleaned)
        if not pieces:
            return ""
        return "\n\n".join(
            f"<<COMMENT {idx}>>\n{chunk}"
            for idx, chunk in enumerate(pieces, start=1)
        )

    return (
        df.assign(inline_text=inline_series)
        .assign(canonical_text=lambda pdf: pdf.apply(assemble, axis=1))
        .drop(columns="inline_text")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge inline CSV comments with parsed attachment text into one combined CSV."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help=f"Root directory containing agency folders (default: {DEFAULT_BASE_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Combined CSV output path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of CSVs to process (for debugging).",
    )
    parser.add_argument(
        "--year",
        action="append",
        default=[],
        help=(
            "Filter to CSVs whose path includes the given 4-digit year. "
            "Repeatable (e.g., --year 2023 --year 2024)."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    base_dir = args.base_dir
    if not base_dir.exists():
        raise SystemExit(f"Base directory {base_dir} does not exist")

    csv_paths = discover_csv_paths(base_dir)
    if not csv_paths:
        raise SystemExit(f"No CSV files found under {base_dir}")
    if tqdm is None:
        logging.info("Install tqdm for progress bars (pip install tqdm).")

    if args.year:
        csv_paths = [p for p in csv_paths if _matches_years(p, args.year)]
        logging.info("Filtered to %s CSVs after year filter(s): %s", len(csv_paths), ", ".join(args.year))

    if not csv_paths:
        raise SystemExit("No CSVs matched the year filter(s).")

    logging.info("Discovered %s CSV files under %s", len(csv_paths), base_dir)

    download_records_df = load_download_records(base_dir)
    logging.info("Loaded %s download records.", len(download_records_df))
    combined_frames: List[pd.DataFrame] = []

    processed = 0
    for csv_path in tqdm(csv_paths, desc="Merging CSVs"):
        if args.limit is not None and processed >= args.limit:
            break
        rel_parts = csv_path.relative_to(base_dir).parts
        agency = rel_parts[0] if len(rel_parts) > 0 else ""
        collection = rel_parts[1] if len(rel_parts) > 1 else ""
        csv_name = csv_path.stem
        logging.debug("Processing %s (agency=%s collection=%s csv=%s)", csv_path, agency, collection, csv_name)
        subset = download_records_df.loc[
            lambda d: (d["agency"] == agency)
            & (d["collection"] == collection)
            & (d["csv_name"] == csv_name)
        ]
        try:
            enriched = build_canonical_texts(csv_path, subset)
        except Exception as exc:
            logging.warning("Skipping %s: %s", csv_path, exc)
            continue
        combined_frames.append(enriched)
        processed += 1

    if not combined_frames:
        raise SystemExit("No CSVs were processed successfully.")

    combined_df = pd.concat(combined_frames, ignore_index=True)
    combined_df.to_csv(args.output, index=False)
    logging.info("Wrote %s (%s rows).", args.output, len(combined_df))


if __name__ == "__main__":
    main()
