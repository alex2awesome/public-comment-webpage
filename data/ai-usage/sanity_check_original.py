#!/usr/bin/env python3
"""Sanity check: run the ORIGINAL Liang et al. code on our CMS data.

Prepares data in the format expected by the original repo, then calls
their estimation.py and MLE.py directly.

Usage:
    python3 scripts/sanity_check_original.py --agency CMS --doc-types rule notice proposed_rule
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Our utilities for loading data
from ai_detection_utils import (
    ensure_nltk_punkt,
    is_pre_chatgpt,
    iter_input_files,
    tokenize_text,
)

# Original paper's code — resolve from project root
_script_dir = Path(__file__).resolve().parent  # ai-usage/
_data_dir = _script_dir.parent                 # data/
_project_root = _data_dir.parent               # regulations-demo/
ORIG_REPO = _project_root / "LLM-widespread-adoption-impact"
sys.path.insert(0, str(ORIG_REPO / "src"))


def prepare_human_data(base_dir: Path, doc_type: str, agency: str) -> pd.DataFrame:
    """Load pre-ChatGPT human data for one agency, tokenize into sentence lists."""
    all_sentences = []
    for csv_path in tqdm(
        list(iter_input_files(base_dir, doc_type)), desc=f"human {doc_type}"
    ):
        try:
            df = pd.read_csv(csv_path, low_memory=False)
            if "canonical_text" not in df.columns:
                continue
            df = df.dropna(subset=["canonical_text", "Posted Date"])
            df = df[df["Agency ID"] == agency]
            df = df[df["Posted Date"].apply(lambda d: is_pre_chatgpt(d))]
            df = df[df["canonical_text"].str.len() >= 100]

            for text in df["canonical_text"]:
                for sent in tokenize_text(text):
                    if len(sent) > 10:
                        all_sentences.append(sent)
        except Exception:
            continue

    print(f"  {doc_type} human: {len(all_sentences)} sentences")
    return pd.DataFrame({"human_sentence": all_sentences})


def prepare_ai_data(ai_corpus_dir: Path, doc_type: str, agency: str) -> pd.DataFrame:
    """Load AI corpus for one agency, tokenize into sentence lists."""
    ai_path = ai_corpus_dir / f"ai_corpus_{doc_type}.parquet"
    if not ai_path.exists():
        print(f"  WARNING: {ai_path} not found")
        return pd.DataFrame({"ai_sentence": []})

    ai_df = pd.read_parquet(ai_path)
    ai_df = ai_df[ai_df["agency_id"] == agency]

    all_sentences = []
    for text in ai_df["ai_text"].fillna(""):
        for sent in tokenize_text(text):
            if len(sent) > 10:
                all_sentences.append(sent)

    print(f"  {doc_type} AI: {len(all_sentences)} sentences")
    return pd.DataFrame({"ai_sentence": all_sentences})


def prepare_inference_data(
    base_dir: Path, doc_type: str, agency: str
) -> dict[str, pd.DataFrame]:
    """Load post-ChatGPT data by quarter for inference."""
    records = []
    for csv_path in tqdm(
        list(iter_input_files(base_dir, doc_type)), desc=f"infer {doc_type}"
    ):
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception:
            continue
        if "canonical_text" not in df.columns:
            continue
        df = df.dropna(subset=["canonical_text", "Posted Date"])
        df = df[df["Agency ID"] == agency]
        df = df[df["canonical_text"].str.len() >= 100]

        for _, row in df.iterrows():
            date_str = str(row["Posted Date"])[:10]
            try:
                dt = pd.Timestamp(date_str)
                quarter = f"{dt.year}Q{dt.quarter}"
                year = dt.year
            except Exception:
                continue

            sents = []
            for sent in tokenize_text(row["canonical_text"]):
                if len(sent) > 10:
                    sents.append(sent)
            if sents:
                records.append({
                    "quarter": quarter,
                    "year": year,
                    "sentences": sents,
                })

    # Group by quarter
    rec_df = pd.DataFrame(records)
    quarter_data = {}
    for quarter, grp in rec_df.groupby("quarter"):
        all_sents = []
        for sent_list in grp["sentences"]:
            all_sents.extend(sent_list)
        quarter_data[quarter] = pd.DataFrame({"inference_sentence": all_sents})

    return quarter_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agency", default="CMS", help="Agency ID to test")
    parser.add_argument("--base-dir", default=".", help="Base data directory")
    parser.add_argument(
        "--ai-corpus-dir", default="../ai-usage-generations/v3", help="AI corpus directory"
    )
    parser.add_argument(
        "--doc-types",
        nargs="+",
        default=["rule", "notice", "proposed_rule"],
    )
    parser.add_argument(
        "--output-dir", default="data/sanity_check", help="Output directory"
    )
    args = parser.parse_args()

    ensure_nltk_punkt()

    base_dir = Path(args.base_dir)
    ai_corpus_dir = Path(args.ai_corpus_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import original paper's code
    from estimation import estimate_text_distribution
    from MLE import MLE

    for doc_type in args.doc_types:
        print(f"\n{'='*60}")
        print(f"  {doc_type} — {args.agency}")
        print(f"{'='*60}")

        dt_dir = output_dir / doc_type
        dt_dir.mkdir(parents=True, exist_ok=True)

        # 1. Prepare human + AI data
        human_df = prepare_human_data(base_dir, doc_type, args.agency)
        ai_df = prepare_ai_data(ai_corpus_dir, doc_type, args.agency)

        if human_df.empty or ai_df.empty:
            print(f"  SKIPPING {doc_type}: insufficient data")
            continue

        human_path = dt_dir / "human_data.parquet"
        ai_path = dt_dir / "ai_data.parquet"
        dist_path = dt_dir / "distribution.parquet"

        human_df.to_parquet(human_path, index=False)
        ai_df.to_parquet(ai_path, index=False)

        # 2. Run original estimation
        print(f"\n  Running original estimate_text_distribution...")
        estimate_text_distribution(
            str(human_path), str(ai_path), str(dist_path)
        )
        dist = pd.read_parquet(dist_path)
        print(f"  Distribution: {len(dist)} words")
        print(f"  Columns: {list(dist.columns)}")

        # 3. Run original MLE inference per quarter
        print(f"\n  Running original MLE inference...")
        mle = MLE(str(dist_path))

        quarter_data = prepare_inference_data(base_dir, doc_type, args.agency)

        results = []
        for quarter in sorted(quarter_data.keys()):
            infer_df = quarter_data[quarter]
            if len(infer_df) < 50:
                continue

            # Save temp parquet for the original code
            infer_path = dt_dir / f"infer_{quarter}.parquet"
            infer_df.to_parquet(infer_path, index=False)

            try:
                solution, half_width = mle.inference(str(infer_path))
                results.append({
                    "quarter": quarter,
                    "alpha": solution,
                    "half_width": half_width,
                    "n_sentences": len(infer_df),
                })
                print(
                    f"    {quarter}: α = {solution:.4f} ± {half_width:.4f}"
                    f"  ({len(infer_df)} sents)"
                )
            except Exception as e:
                print(f"    {quarter}: ERROR — {e}")

        if results:
            results_df = pd.DataFrame(results)
            out_path = dt_dir / "results.csv"
            results_df.to_csv(out_path, index=False)
            print(f"\n  Wrote {out_path}")

            # Summary
            pre = results_df[results_df["quarter"] < "2023"]
            post = results_df[results_df["quarter"] >= "2023"]
            if not pre.empty:
                print(f"  Pre-ChatGPT  mean α: {pre['alpha'].mean():.4f}")
            if not post.empty:
                print(f"  Post-ChatGPT mean α: {post['alpha'].mean():.4f}")


if __name__ == "__main__":
    main()
