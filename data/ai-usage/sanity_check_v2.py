#!/usr/bin/env python3
"""Sanity check v2: Use the ORIGINAL Liang et al. code exactly on CMS data.

Prepares human/AI/inference data in the exact format expected by their code
(parquet with tokenized sentence lists), then calls their estimation.py and MLE.py.

Tokenization: NLTK sent_tokenize + re.findall(r'\b\w+\b', lower) + filter digits
(matching their tokenize_demo.ipynb, but using NLTK instead of spaCy for sentence splitting)

Usage:
    python3 scripts/sanity_check_v2.py \
        --agency CMS \
        --doc-types rule notice proposed_rule \
        --base-dir . \
        --output-dir data/sanity_check_v2
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Tokenization — matches original paper's tokenize_demo.ipynb
# ---------------------------------------------------------------------------
import nltk
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize


def tokenize_text(text: str) -> list[list[str]]:
    """Tokenize text into list of sentences, each a list of lowercase non-digit words.

    Mirrors the original paper's tokenize() function from tokenize_demo.ipynb:
    1. Split on newlines
    2. Sentence-split each chunk (they use spaCy, we use NLTK)
    3. re.findall(r'\\b\\w+\\b', sent.lower())
    4. Filter out pure-digit tokens
    5. Keep non-empty sentences
    """
    sentence_list = []
    # Split on newlines first, like the original
    chunks = [chunk.strip() for chunk in text.split("\n") if chunk.strip()]
    for chunk in chunks:
        for sent in sent_tokenize(chunk):
            words = re.findall(r"\b\w+\b", sent.lower())
            words = [w for w in words if not w.isdigit()]
            if words:
                sentence_list.append(words)
    return sentence_list


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def iter_period_dirs(base_dir: Path, agency: str) -> list[Path]:
    """Find all period directories for an agency."""
    agency_dir = base_dir / agency.lower()
    if not agency_dir.exists():
        return []
    return sorted(agency_dir.iterdir())


def load_human_docs(
    base_dir: Path, doc_type: str, agency: str, cutoff: str = "2022-11-30"
) -> list[list[list[str]]]:
    """Load pre-ChatGPT human data for one agency, tokenize into per-doc sentence lists.

    Returns list of documents, each document = list of sentences, each sentence = list of words.
    This matches the original paper's format: each parquet row = array of sentence arrays.
    """
    all_docs = []
    for period_dir in tqdm(
        iter_period_dirs(base_dir, agency), desc=f"human {doc_type}"
    ):
        # Try both .csv and .csv.gz
        csv_path = period_dir / f"{doc_type}_all_text.csv"
        if not csv_path.exists():
            csv_gz = period_dir / f"{doc_type}_all_text.csv.gz"
            if csv_gz.exists():
                csv_path = csv_gz
            else:
                continue

        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception:
            continue

        if "canonical_text" not in df.columns or "Posted Date" not in df.columns:
            continue

        df = df.dropna(subset=["canonical_text", "Posted Date"])
        # Filter to pre-ChatGPT only
        df = df[df["Posted Date"].apply(lambda d: str(d)[:10] <= cutoff)]
        df = df[df["canonical_text"].str.len() >= 100]

        for text in df["canonical_text"]:
            sents = tokenize_text(str(text))
            if sents:
                all_docs.append(sents)

    return all_docs


def load_ai_docs(
    ai_corpus_paths: list[Path], doc_type: str, agency: str
) -> list[list[list[str]]]:
    """Load AI corpus for one agency from all available versions.

    Returns list of documents, each = list of sentences, each = list of words.
    """
    all_docs = []
    for ai_path in ai_corpus_paths:
        try:
            ai_df = pd.read_parquet(ai_path)
        except Exception as e:
            print(f"  WARNING: cannot read {ai_path}: {e}")
            continue

        # Filter to agency
        if "agency_id" in ai_df.columns:
            ai_df = ai_df[ai_df["agency_id"] == agency]
        elif "Agency ID" in ai_df.columns:
            ai_df = ai_df[ai_df["Agency ID"] == agency]

        text_col = "ai_text" if "ai_text" in ai_df.columns else None
        if text_col is None:
            print(f"  WARNING: no ai_text column in {ai_path}")
            continue

        n_before = len(all_docs)
        for text in ai_df[text_col].fillna(""):
            sents = tokenize_text(str(text))
            if sents:
                all_docs.append(sents)

        n_sents = sum(len(d) for d in all_docs[n_before:])
        print(f"    {ai_path.name}: {len(ai_df)} docs -> {len(all_docs) - n_before} docs, {n_sents} sents")

    return all_docs


def load_inference_docs(
    base_dir: Path, doc_type: str, agency: str
) -> dict[str, list[list[list[str]]]]:
    """Load all data by quarter for inference.

    Returns dict[quarter -> list of documents], each doc = list of sentences.
    """
    records: dict[str, list[list[list[str]]]] = {}
    for period_dir in tqdm(
        iter_period_dirs(base_dir, agency), desc=f"infer {doc_type}"
    ):
        csv_path = period_dir / f"{doc_type}_all_text.csv"
        if not csv_path.exists():
            csv_gz = period_dir / f"{doc_type}_all_text.csv.gz"
            if csv_gz.exists():
                csv_path = csv_gz
            else:
                continue

        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception:
            continue

        if "canonical_text" not in df.columns or "Posted Date" not in df.columns:
            continue

        df = df.dropna(subset=["canonical_text", "Posted Date"])
        df = df[df["canonical_text"].str.len() >= 100]

        for _, row in df.iterrows():
            date_str = str(row["Posted Date"])[:10]
            try:
                dt = pd.Timestamp(date_str)
                quarter = f"{dt.year}Q{dt.quarter}"
            except Exception:
                continue

            sents = tokenize_text(str(row["canonical_text"]))
            if sents:
                records.setdefault(quarter, []).append(sents)

    return records


def docs_to_parquet(docs: list[list[list[str]]], col_name: str, path: Path):
    """Save documents in the exact format the original code expects.

    Each row = one document. The column contains a numpy array of numpy arrays
    (one per sentence, each sentence = array of word strings).
    After .explode(), each row becomes a single sentence array.
    """
    rows = []
    for doc_sents in docs:
        # Array of arrays, matching original paper format
        arr = np.array(
            [np.array(sent, dtype=object) for sent in doc_sents],
            dtype=object
        )
        rows.append(arr)
    df = pd.DataFrame({col_name: rows})
    df.to_parquet(path, index=False)
    # Count total sentences
    n_sents = sum(len(d) for d in docs)
    return len(docs), n_sents


def find_ai_corpus_files(data_dir: Path, doc_type: str) -> list[Path]:
    """Find all AI corpus parquet files for a doc_type across all versions."""
    paths = []
    # Root level
    p = data_dir / f"ai_corpus_{doc_type}.parquet"
    if p.exists():
        paths.append(p)
    # Versioned subdirs
    for subdir in sorted(data_dir.iterdir()):
        if subdir.is_dir():
            p = subdir / f"ai_corpus_{doc_type}.parquet"
            if p.exists():
                paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agency", default="CMS")
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--ai-data-dir", default="../ai-usage-generations",
                        help="Directory containing AI corpus parquets (root + versioned subdirs)")
    parser.add_argument("--doc-types", nargs="+",
                        default=["rule", "notice", "proposed_rule"])
    parser.add_argument("--output-dir", default="data/sanity_check_v2")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    ai_data_dir = Path(args.ai_data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add original paper's src to path
    script_dir = Path(__file__).resolve().parent  # ai-usage/
    data_dir = script_dir.parent                  # data/
    project_root = data_dir.parent                # regulations-demo/
    orig_repo = project_root / "LLM-widespread-adoption-impact"
    sys.path.insert(0, str(orig_repo / "src"))

    from estimation import estimate_text_distribution
    from MLE import MLE

    for doc_type in args.doc_types:
        print(f"\n{'='*60}")
        print(f"  {doc_type} — {args.agency}")
        print(f"{'='*60}")

        dt_dir = output_dir / doc_type
        dt_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load and tokenize human data (pre-ChatGPT)
        human_docs = load_human_docs(base_dir, doc_type, args.agency)
        n_human_sents = sum(len(d) for d in human_docs)
        print(f"  Human: {len(human_docs)} docs, {n_human_sents} sentences")

        # 2. Load and tokenize AI data (all versions)
        ai_corpus_paths = find_ai_corpus_files(ai_data_dir, doc_type)
        print(f"  AI corpus files found: {[str(p) for p in ai_corpus_paths]}")
        ai_docs = load_ai_docs(ai_corpus_paths, doc_type, args.agency)
        n_ai_sents = sum(len(d) for d in ai_docs)
        print(f"  AI: {len(ai_docs)} docs, {n_ai_sents} sentences")

        if not human_docs or not ai_docs:
            print(f"  SKIPPING {doc_type}: no data")
            continue

        # 3. Save in EXACT format expected by original code:
        #    Each row = one document, column = array of sentence arrays
        #    After .explode(), each row = one sentence (array of words)
        human_path = dt_dir / "human_data.parquet"
        ai_path = dt_dir / "ai_data.parquet"
        dist_path = dt_dir / "distribution.parquet"

        n_h_docs, n_h_sents = docs_to_parquet(human_docs, "human_sentence", human_path)
        n_a_docs, n_a_sents = docs_to_parquet(ai_docs, "ai_sentence", ai_path)
        print(f"  Saved human ({n_h_docs} docs, {n_h_sents} sents) and AI ({n_a_docs} docs, {n_a_sents} sents)")

        # 4. Verify format before running
        _verify = pd.read_parquet(human_path)
        _ex = _verify.explode("human_sentence").dropna(subset=["human_sentence"])
        _s = _ex["human_sentence"].iloc[0]
        print(f"  Format check: after explode, type={type(_s).__name__}, "
              f"len={len(_s)}, sample={list(_s)[:5]}")
        assert hasattr(_s, '__iter__') and not isinstance(_s, str), \
            f"Expected array after explode, got {type(_s)}"
        del _verify, _ex, _s

        # 5. Run original estimate_text_distribution
        print(f"\n  Running estimate_text_distribution...")
        estimate_text_distribution(str(human_path), str(ai_path), str(dist_path))

        dist = pd.read_parquet(dist_path)
        print(f"  Distribution: {len(dist)} words")
        print(f"  Columns: {list(dist.columns)}")
        print(f"\n  Top 10 words by log odds ratio (most human-like first):")
        if "Log Odds Ratio" in dist.columns:
            print(dist[["Word", "logP", "logQ"]].head(10).to_string(index=False))
        else:
            print(dist[["Word", "logP", "logQ"]].head(10).to_string(index=False))

        # 5. Run MLE inference per quarter
        print(f"\n  Running MLE inference...")
        mle = MLE(str(dist_path))

        quarter_data = load_inference_docs(base_dir, doc_type, args.agency)

        results = []
        for quarter in sorted(quarter_data.keys()):
            docs = quarter_data[quarter]
            n_sents = sum(len(d) for d in docs)
            if n_sents < 50:
                continue

            # Save inference parquet in same nested format
            infer_path = dt_dir / f"infer_{quarter}.parquet"
            docs_to_parquet(docs, "inference_sentence", infer_path)

            try:
                solution, half_width = mle.inference(str(infer_path))
                results.append({
                    "quarter": quarter,
                    "alpha": solution,
                    "half_width": half_width,
                    "n_sentences": n_sents,
                })
                print(
                    f"    {quarter}: α = {solution:.4f} ± {half_width:.4f}"
                    f"  ({n_sents} sents)"
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
