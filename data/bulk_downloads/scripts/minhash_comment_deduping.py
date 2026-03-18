from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from comment_dedup import CommentDeduplicator
from tqdm import tqdm


REQUIRED_COLUMNS = ["Document ID", "Docket ID", "canonical_text", "Agency ID"]


def iter_input_files(base_dir: Path) -> Iterable[Path]:
    return base_dir.glob("*/*/public_submission_all_text.csv")


def load_comments(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    df = df.dropna(subset=["Document ID", "Docket ID", "canonical_text"])
    df["Document ID"] = df["Document ID"].astype(str)
    df["Docket ID"] = df["Docket ID"].astype(str)
    df["Agency ID"] = df["Agency ID"].astype(str)
    return df


def agency_id_for_file(df: pd.DataFrame, csv_path: Path) -> str:
    agency_series = df["Agency ID"].dropna()
    if agency_series.empty:
        raise ValueError(f"No Agency ID values found in {csv_path}")
    return str(agency_series.iloc[0])


def build_clusters(dedup: CommentDeduplicator) -> list[list[str]]:
    clusters: list[list[str]] = [list(c.member_ids) for c in dedup.clusters]
    for sid in sorted(dedup._singletons):
        clusters.append([sid])
    return clusters


def dedup_docket(docket_df: pd.DataFrame) -> list[list[str]]:
    docket_df = docket_df[["Document ID", "canonical_text"]]
    input_records = docket_df.rename(
        columns={"Document ID": "id", "canonical_text": "text"}
    ).to_dict(orient="records")

    dedup = CommentDeduplicator(threshold=0.8)
    dedup.add_comments(input_records)
    dedup.cluster()

    return build_clusters(dedup)


def process_file(csv_path: Path, overwrite: bool) -> None:
    logging.info("Processing %s", csv_path)
    df = load_comments(csv_path)
    agency_id = agency_id_for_file(df, csv_path)

    output_dir = csv_path.parent
    clusters_path = output_dir / "public_submission_all_text__dedup_clusters.jsonl"
    mapper_path = output_dir / "public_submission_all_text__dedup_mapper.csv"
    if not overwrite and (clusters_path.exists() or mapper_path.exists()):
        logging.info("Skipping %s (outputs already exist)", csv_path)
        return

    mapper_rows: list[dict[str, str | int]] = []

    with clusters_path.open("w", encoding="utf-8") as clusters_file:
        docket_count = df["Docket ID"].nunique(dropna=False)
        for docket_id, docket_df in tqdm(
            df.groupby("Docket ID", dropna=False),
            total=docket_count,
            desc=f"Dockets {csv_path.parent.name}",
            leave=False,
        ):
            clusters = dedup_docket(docket_df)

            record = {
                "agency": agency_id,
                "docket_id": docket_id,
                "clusters": clusters,
            }
            clusters_file.write(json.dumps(record) + "\n")

            for cluster_id, members in enumerate(clusters):
                cluster_uid = f"{docket_id}::{cluster_id}"
                for document_id in members:
                    mapper_rows.append(
                        {
                            "agency_id": agency_id,
                            "docket_id": docket_id,
                            "document_id": document_id,
                            "cluster_id": cluster_id,
                            "cluster_uid": cluster_uid,
                        }
                    )

    mapper_df = pd.DataFrame(
        mapper_rows,
        columns=[
            "agency_id",
            "docket_id",
            "document_id",
            "cluster_id",
            "cluster_uid",
        ],
    )
    mapper_df.to_csv(mapper_path, index=False)
    logging.info(
        "Wrote %s and %s (%s rows)",
        clusters_path,
        mapper_path,
        len(mapper_df),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite outputs if they already exist.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    base_dir = Path("data/bulk_downloads")
    csv_files = list(iter_input_files(base_dir))
    if not csv_files:
        logging.warning("No input files found under %s", base_dir)
        return

    for csv_path in tqdm(csv_files, desc="Files"):
        try:
            process_file(csv_path, overwrite=args.overwrite)
        except Exception:
            logging.exception("Failed processing %s", csv_path)


if __name__ == "__main__":
    main()
