#!/usr/bin/env python3
"""
Sample regulatory documents and score them with the Pangram AI detection API.

Produces silver labels for calibrating distributional AI detection estimates.

Sampling strategy:
    Pre-ChatGPT:  Light uniform sample (~200/doc_type) for false-positive calibration.
    Post-ChatGPT: Stratified by agency, heavier on public_submission.

Usage:
    # Step 1: Build the sample (no API calls)
    python pangram_ai_detection.py sample

    # Step 2: Score the sample via Pangram API
    python pangram_ai_detection.py score

    # Step 3: Analyze results
    python pangram_ai_detection.py analyze
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_DIR = SCRIPT_DIR.parent / "bulk_downloads"  # data/bulk_downloads/
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parent / "ai-usage-generations" / "pangram"

DOC_TYPES = ["public_submission", "notice", "rule", "proposed_rule"]
CHATGPT_CUTOFF = "2022-11-30"
MAX_TOKENS = 750  # cap per document (chars ≈ tokens * 4)
MAX_CHARS = MAX_TOKENS * 4
MIN_CHARS = 200  # skip very short docs

PANGRAM_API_URL = "https://text.api.pangram.com/v3"

# ---------------------------------------------------------------------------
# Sampling budget
# ---------------------------------------------------------------------------

# Post-ChatGPT: per-agency caps by doc_type
POST_SAMPLE_PER_AGENCY: Dict[str, int] = {
    "public_submission": 1000,
    "notice": 400,
    "rule": 400,
    "proposed_rule": 400,
}
# Pre-ChatGPT: total per doc_type (uniform random, just for FP calibration)
PRE_SAMPLE_TOTAL: Dict[str, int] = {
    "public_submission": 600,
    "notice": 300,
    "rule": 300,
    "proposed_rule": 300,
}


# ---------------------------------------------------------------------------
# Single-document scoring
# ---------------------------------------------------------------------------


def score_one_doc(
    text: str,
    api_key: Optional[str] = None,
    api_key_file: str = "~/.pangram-api-key.txt",
    max_chars: int = MAX_CHARS,
    timeout: int = 60,
    retries: int = 3,
) -> dict:
    """Score a single document via the Pangram API.

    Returns the raw JSON response dict. Retries on timeout/rate-limit.
    """
    import requests

    if api_key is None:
        key_path = Path(api_key_file).expanduser()
        api_key = key_path.read_text().strip()

    for attempt in range(retries):
        try:
            resp = requests.post(
                PANGRAM_API_URL,
                headers={"x-api-key": api_key, "Content-Type": "application/json"},
                json={"text": text[:max_chars]},
                timeout=timeout,
            )
            if resp.status_code == 429:
                import time as _time
                wait = 30 * (attempt + 1)
                logging.warning("Rate limited, waiting %ds...", wait)
                _time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ReadTimeout:
            if attempt < retries - 1:
                logging.warning("Timeout (attempt %d/%d), retrying...", attempt + 1, retries)
                continue
            raise
    raise RuntimeError("All %d retries failed" % retries)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def iter_input_files(base_dir: Path, doc_type: str):
    return sorted(base_dir.glob(f"*/*/{doc_type}_all_text.csv.gz"))


def load_docs(base_dir: Path, doc_type: str) -> pd.DataFrame:
    """Load all documents for a doc_type with minimal columns."""
    frames = []
    for csv_path in iter_input_files(base_dir, doc_type):
        try:
            df = pd.read_csv(
                csv_path,
                usecols=["Document ID", "Agency ID", "Posted Date", "canonical_text"],
                low_memory=False,
            )
        except (ValueError, KeyError):
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.dropna(subset=["canonical_text", "Posted Date"])
    all_df = all_df[all_df["canonical_text"].str.len() >= MIN_CHARS]
    all_df["posted_dt"] = pd.to_datetime(all_df["Posted Date"], errors="coerce")
    all_df = all_df.dropna(subset=["posted_dt"])
    all_df["is_post"] = all_df["posted_dt"] >= CHATGPT_CUTOFF
    return all_df


def truncate_text(text: str) -> str:
    """Truncate to ~750 tokens (≈3000 chars)."""
    return text[:MAX_CHARS]


# ---------------------------------------------------------------------------
# cmd_sample: build the sample without hitting the API
# ---------------------------------------------------------------------------

def cmd_sample(args: argparse.Namespace) -> None:
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    all_samples = []

    for doc_type in DOC_TYPES:
        logging.info("Sampling %s...", doc_type)
        docs = load_docs(base_dir, doc_type)
        if docs.empty:
            logging.warning("No docs for %s", doc_type)
            continue

        pre = docs[~docs["is_post"]]
        post = docs[docs["is_post"]]
        logging.info(
            "  %s: %d pre-ChatGPT, %d post-ChatGPT docs available",
            doc_type, len(pre), len(post),
        )

        # --- Pre-ChatGPT: uniform random sample ---
        n_pre = min(PRE_SAMPLE_TOTAL[doc_type], len(pre))
        if n_pre > 0:
            pre_sample = pre.sample(n=n_pre, random_state=rng)
            pre_sample = pre_sample.assign(doc_type=doc_type, era="pre")
            all_samples.append(pre_sample)
            logging.info("  Pre-ChatGPT: sampled %d docs", n_pre)

        # --- Post-ChatGPT: stratified by agency ---
        cap = POST_SAMPLE_PER_AGENCY[doc_type]
        post_parts = []
        for agency_id, agency_grp in post.groupby("Agency ID"):
            n_take = min(cap, len(agency_grp))
            post_parts.append(agency_grp.sample(n=n_take, random_state=rng))

        if post_parts:
            post_sample = pd.concat(post_parts, ignore_index=True)
            post_sample = post_sample.assign(doc_type=doc_type, era="post")
            all_samples.append(post_sample)
            logging.info(
                "  Post-ChatGPT: sampled %d docs from %d agencies",
                len(post_sample), len(post_parts),
            )

    # --- Combine and prepare output ---
    sample_df = pd.concat(all_samples, ignore_index=True)
    sample_df["text_truncated"] = sample_df["canonical_text"].apply(truncate_text)
    sample_df["text_chars"] = sample_df["text_truncated"].str.len()
    sample_df["text_words"] = sample_df["text_truncated"].str.split().str.len()

    # Estimate credits: 1 credit per 1000 words
    total_words = sample_df["text_words"].sum()
    est_credits = int(np.ceil(total_words / 1000))

    # Keep only what we need
    out = sample_df[[
        "Document ID", "Agency ID", "Posted Date", "doc_type", "era",
        "text_truncated", "text_chars", "text_words",
    ]].copy()
    out = out.rename(columns={"Document ID": "document_id", "Agency ID": "agency_id"})

    output_path = output_dir / "pangram_sample.parquet"
    out.to_parquet(output_path, index=False)

    # Summary
    summary = out.groupby(["doc_type", "era"]).agg(
        n_docs=("document_id", "count"),
        total_words=("text_words", "sum"),
        n_agencies=("agency_id", "nunique"),
    ).reset_index()
    summary["est_credits"] = np.ceil(summary["total_words"] / 1000).astype(int)

    print(f"\nSample saved to {output_path}")
    print(f"Total documents: {len(out):,}")
    print(f"Total words: {total_words:,}")
    print(f"Estimated Pangram credits: {est_credits:,}\n")
    print(summary.to_string(index=False))


# ---------------------------------------------------------------------------
# cmd_score: hit the Pangram API
# ---------------------------------------------------------------------------

def cmd_score(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    sample_path = output_dir / "pangram_sample.parquet"
    results_path = output_dir / "pangram_results.parquet"
    raw_jsonl_path = output_dir / "pangram_raw_responses.jsonl"

    if not sample_path.exists():
        logging.error("Sample not found at %s. Run 'sample' first.", sample_path)
        return

    # Load API key
    api_key_path = Path(args.api_key_file).expanduser()
    api_key = api_key_path.read_text().strip()

    sample = pd.read_parquet(sample_path)
    logging.info("Loaded %d documents to score", len(sample))

    # Resume from existing results if any
    done_ids: Set[str] = set()
    existing_results = []
    if results_path.exists() and not args.overwrite:
        existing = pd.read_parquet(results_path)
        done_ids = set(existing["document_id"].astype(str))
        existing_results.append(existing)
        logging.info("Resuming: %d documents already scored", len(done_ids))

    # Also collect IDs already in the raw JSONL (in case parquet and JSONL diverge)
    if raw_jsonl_path.exists() and not args.overwrite:
        with open(raw_jsonl_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["document_id"])
                except (json.JSONDecodeError, KeyError):
                    pass

    remaining = sample[~sample["document_id"].astype(str).isin(done_ids)]
    logging.info("Documents to score: %d", len(remaining))

    if remaining.empty:
        logging.info("Nothing to do.")
        return

    import requests

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
    }

    results = []
    errors = 0

    for idx, row in tqdm(remaining.iterrows(), total=len(remaining), desc="Scoring"):
        text = row["text_truncated"]
        try:
            resp = requests.post(
                PANGRAM_API_URL,
                headers=headers,
                json={"text": text},
                timeout=30,
            )
            if resp.status_code == 429:
                # Rate limited — wait and retry once
                logging.warning("Rate limited. Waiting 60s...")
                time.sleep(60)
                resp = requests.post(
                    PANGRAM_API_URL,
                    headers=headers,
                    json={"text": text},
                    timeout=30,
                )

            if resp.status_code != 200:
                logging.warning(
                    "API error %d for %s: %s",
                    resp.status_code, row["document_id"], resp.text[:200],
                )
                errors += 1
                if errors > 50:
                    logging.error("Too many errors. Stopping.")
                    break
                continue

            data = resp.json()
            results.append({
                "document_id": row["document_id"],
                "agency_id": row["agency_id"],
                "doc_type": row["doc_type"],
                "era": row["era"],
                "posted_date": row["Posted Date"],
                "text_words": row["text_words"],
                "fraction_ai": data.get("fraction_ai"),
                "fraction_ai_assisted": data.get("fraction_ai_assisted"),
                "fraction_human": data.get("fraction_human"),
                "prediction_short": data.get("prediction_short"),
                "headline": data.get("headline"),
                "n_ai_segments": data.get("num_ai_segments"),
                "n_human_segments": data.get("num_human_segments"),
                "api_version": data.get("version"),
            })

            # Save full API response (includes segment-level detail)
            with open(raw_jsonl_path, "a") as f:
                raw_record = {
                    "document_id": row["document_id"],
                    "response": data,
                }
                f.write(json.dumps(raw_record) + "\n")

        except Exception as e:
            logging.warning("Request failed for %s: %s", row["document_id"], e)
            errors += 1
            if errors > 50:
                logging.error("Too many errors. Stopping.")
                break

        # Save checkpoint every 500 docs
        if len(results) > 0 and len(results) % 500 == 0:
            _save_results(results_path, existing_results, results)
            logging.info("Checkpoint: %d new results saved", len(results))

        # Polite rate limiting
        time.sleep(0.1)

    # Final save
    if results:
        _save_results(results_path, existing_results, results)
    logging.info(
        "Done. %d scored, %d errors. Results: %s",
        len(results), errors, results_path,
    )


def _save_results(
    path: Path,
    existing: List[pd.DataFrame],
    new_results: List[dict],
) -> None:
    parts = list(existing) + [pd.DataFrame(new_results)]
    combined = pd.concat(parts, ignore_index=True)
    combined.to_parquet(path, index=False)


# ---------------------------------------------------------------------------
# cmd_score_custom: score an arbitrary list of documents
# ---------------------------------------------------------------------------

def cmd_score_custom(args: argparse.Namespace) -> None:
    """Score documents from a user-provided CSV.

    The CSV must have columns: document_id, text
    Optional columns: agency_id, doc_type, era, posted_date

    Results are written to a separate output file (not the main pangram_results).
    """
    import requests

    input_path = Path(args.input)
    output_path = Path(args.output)
    raw_jsonl_path = output_path.with_suffix(".jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logging.error("Input file not found: %s", input_path)
        return

    # Load input
    if str(input_path).endswith(".parquet"):
        input_df = pd.read_parquet(input_path)
    else:
        input_df = pd.read_csv(input_path)

    if "document_id" not in input_df.columns or "text" not in input_df.columns:
        logging.error("Input CSV must have 'document_id' and 'text' columns")
        return

    input_df = input_df[input_df["text"].notna() & (input_df["text"].str.len() >= MIN_CHARS)]
    input_df["text_truncated"] = input_df["text"].apply(truncate_text)
    input_df["text_words"] = input_df["text_truncated"].str.split().str.len()
    logging.info("Loaded %d documents from %s", len(input_df), input_path)

    # Estimate credits
    total_words = input_df["text_words"].sum()
    est_credits = int(np.ceil(total_words / 1000))
    logging.info("Estimated Pangram credits: %d", est_credits)

    # Load API key
    api_key_path = Path(args.api_key_file).expanduser()
    api_key = api_key_path.read_text().strip()

    # Resume from existing results
    done_ids: Set[str] = set()
    existing_results = []
    if output_path.exists() and not args.overwrite:
        existing = pd.read_parquet(output_path)
        done_ids = set(existing["document_id"].astype(str))
        existing_results.append(existing)
        logging.info("Resuming: %d documents already scored", len(done_ids))

    if raw_jsonl_path.exists() and not args.overwrite:
        with open(raw_jsonl_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["document_id"])
                except (json.JSONDecodeError, KeyError):
                    pass

    remaining = input_df[~input_df["document_id"].astype(str).isin(done_ids)]
    logging.info("Documents to score: %d", len(remaining))

    if remaining.empty:
        logging.info("Nothing to do.")
        return

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
    }

    results = []
    errors = 0

    for idx, row in tqdm(remaining.iterrows(), total=len(remaining), desc="Scoring"):
        text = row["text_truncated"]
        try:
            resp = requests.post(
                PANGRAM_API_URL,
                headers=headers,
                json={"text": text},
                timeout=30,
            )
            if resp.status_code == 429:
                logging.warning("Rate limited. Waiting 60s...")
                time.sleep(60)
                resp = requests.post(
                    PANGRAM_API_URL,
                    headers=headers,
                    json={"text": text},
                    timeout=30,
                )

            if resp.status_code != 200:
                logging.warning(
                    "API error %d for %s: %s",
                    resp.status_code, row["document_id"], resp.text[:200],
                )
                errors += 1
                if errors > 50:
                    logging.error("Too many errors. Stopping.")
                    break
                continue

            data = resp.json()
            result_row = {
                "document_id": row["document_id"],
                "agency_id": row.get("agency_id", ""),
                "doc_type": row.get("doc_type", ""),
                "era": row.get("era", ""),
                "posted_date": row.get("posted_date", row.get("Posted Date", "")),
                "text_words": row["text_words"],
                "fraction_ai": data.get("fraction_ai"),
                "fraction_ai_assisted": data.get("fraction_ai_assisted"),
                "fraction_human": data.get("fraction_human"),
                "prediction_short": data.get("prediction_short"),
                "headline": data.get("headline"),
                "n_ai_segments": data.get("num_ai_segments"),
                "n_human_segments": data.get("num_human_segments"),
                "api_version": data.get("version"),
            }
            results.append(result_row)

            # Save full API response
            with open(raw_jsonl_path, "a") as f:
                f.write(json.dumps({
                    "document_id": row["document_id"],
                    "response": data,
                }) + "\n")

        except Exception as e:
            logging.warning("Request failed for %s: %s", row["document_id"], e)
            errors += 1
            if errors > 50:
                logging.error("Too many errors. Stopping.")
                break

        if len(results) > 0 and len(results) % 100 == 0:
            _save_results(output_path, existing_results, results)
            logging.info("Checkpoint: %d new results saved", len(results))

        time.sleep(0.1)

    if results:
        _save_results(output_path, existing_results, results)
    logging.info(
        "Done. %d scored, %d errors. Results: %s",
        len(results), errors, output_path,
    )


# ---------------------------------------------------------------------------
# cmd_analyze: summarize Pangram results
# ---------------------------------------------------------------------------

def cmd_analyze(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    results_path = output_dir / "pangram_results.parquet"
    if not results_path.exists():
        logging.error("No results at %s. Run 'score' first.", results_path)
        return

    df = pd.read_parquet(results_path)
    print(f"Loaded {len(df):,} Pangram results\n")

    # Overall summary by doc_type and era
    summary = df.groupby(["doc_type", "era"]).agg(
        n=("document_id", "count"),
        frac_ai_mean=("fraction_ai", "mean"),
        frac_ai_median=("fraction_ai", "median"),
        frac_ai_assisted_mean=("fraction_ai_assisted", "mean"),
        frac_human_mean=("fraction_human", "mean"),
        pct_flagged_ai=("prediction_short", lambda x: (x == "AI").mean() * 100),
        pct_flagged_mixed=("prediction_short", lambda x: (x == "Mixed").mean() * 100),
        n_agencies=("agency_id", "nunique"),
    ).reset_index()

    for col in ["frac_ai_mean", "frac_ai_median", "frac_ai_assisted_mean", "frac_human_mean"]:
        summary[col] = (summary[col] * 100).round(2).astype(str) + "%"
    summary["pct_flagged_ai"] = summary["pct_flagged_ai"].round(1).astype(str) + "%"
    summary["pct_flagged_mixed"] = summary["pct_flagged_mixed"].round(1).astype(str) + "%"

    print("=== Overall Summary ===")
    print(summary.to_string(index=False))

    # Pre-ChatGPT false positive rates
    pre = df[df["era"] == "pre"]
    if len(pre) > 0:
        print(f"\n=== Pre-ChatGPT False Positive Check (n={len(pre):,}) ===")
        print(f"  Flagged as AI:    {(pre['prediction_short'] == 'AI').sum()} ({(pre['prediction_short'] == 'AI').mean()*100:.1f}%)")
        print(f"  Flagged as Mixed: {(pre['prediction_short'] == 'Mixed').sum()} ({(pre['prediction_short'] == 'Mixed').mean()*100:.1f}%)")
        print(f"  Flagged as Human: {(pre['prediction_short'] == 'Human').sum()} ({(pre['prediction_short'] == 'Human').mean()*100:.1f}%)")
        print(f"  Mean fraction_ai: {pre['fraction_ai'].mean()*100:.2f}%")

    # Post-ChatGPT by agency (public_submission)
    post_ps = df[(df["era"] == "post") & (df["doc_type"] == "public_submission")]
    if len(post_ps) > 0:
        print(f"\n=== Post-ChatGPT Public Submission by Agency (n={len(post_ps):,}) ===")
        agency_summary = post_ps.groupby("agency_id").agg(
            n=("document_id", "count"),
            frac_ai_mean=("fraction_ai", "mean"),
            pct_flagged_ai=("prediction_short", lambda x: (x == "AI").mean() * 100),
        ).sort_values("frac_ai_mean", ascending=False)
        agency_summary["frac_ai_mean"] = (agency_summary["frac_ai_mean"] * 100).round(2).astype(str) + "%"
        agency_summary["pct_flagged_ai"] = agency_summary["pct_flagged_ai"].round(1).astype(str) + "%"
        print(agency_summary.to_string())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sample and score regulatory documents with Pangram AI detection"
    )
    parser.add_argument(
        "--base-dir", default=str(DEFAULT_BASE_DIR),
    )
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT_DIR),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- sample ---
    p_sample = sub.add_parser("sample", help="Build stratified sample")
    p_sample.add_argument("--seed", type=int, default=42)

    # --- score ---
    p_score = sub.add_parser("score", help="Score sample via Pangram API")
    p_score.add_argument(
        "--api-key-file", default="~/.pangram-api-key.txt",
    )
    p_score.add_argument("--overwrite", action="store_true")

    # --- score-custom ---
    p_custom = sub.add_parser(
        "score-custom",
        help="Score an arbitrary list of documents from a CSV/parquet. "
             "Must have 'document_id' and 'text' columns. "
             "Optional: agency_id, doc_type, era, posted_date.",
    )
    p_custom.add_argument(
        "input", help="Path to input CSV or parquet with document_id and text columns",
    )
    p_custom.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR / "pangram_custom_results.parquet"),
        help="Output parquet for results (default: %(default)s)",
    )
    p_custom.add_argument(
        "--api-key-file", default="~/.pangram-api-key.txt",
    )
    p_custom.add_argument("--overwrite", action="store_true")

    # --- analyze ---
    sub.add_parser("analyze", help="Summarize Pangram results")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    if args.command == "sample":
        cmd_sample(args)
    elif args.command == "score":
        cmd_score(args)
    elif args.command == "score-custom":
        cmd_score_custom(args)
    elif args.command == "analyze":
        cmd_analyze(args)


if __name__ == "__main__":
    main()
