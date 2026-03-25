#!/usr/bin/env python3
"""Benchmark our refactored code against the Liang et al. (2025) reference implementation.

Loads PR Newswire training/test data from the cloned LLM-widespread-adoption-impact repo,
runs both pipelines (ours and theirs), and compares results side-by-side.

Expected baseline: January 2022 alpha ≈ 0.025 (pre-ChatGPT, should be near zero).

Usage:
    python3 benchmark_liang.py [--months 2022_1 2022_6 2023_1 2023_6 2024_1]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent          # data/ai-usage/
DATA_DIR = SCRIPT_DIR.parent                          # data/
PROJECT_ROOT = DATA_DIR.parent                        # regulations-demo/
LIANG_REPO = PROJECT_ROOT / "LLM-widespread-adoption-impact"
LIANG_SRC = LIANG_REPO / "src"
LIANG_TRAIN = LIANG_REPO / "data" / "training_data" / "prnewswire"
LIANG_TEST = LIANG_REPO / "data" / "test_data" / "prnewswire"
LIANG_DIST = LIANG_REPO / "distribution" / "prnewswire.parquet"


def load_sentences_from_parquet(path: Path, col: str) -> list[list[str]]:
    """Load a Liang-format parquet: explode array-of-arrays column, filter len>10."""
    df = pd.read_parquet(path)
    df = df.explode(col).dropna(subset=[col])
    df = df[df[col].apply(len) > 10]
    return [list(s) for s in df[col]]


# ---------------------------------------------------------------------------
# Our pipeline
# ---------------------------------------------------------------------------
def run_our_pipeline(
    human_sents: list[list[str]],
    ai_sents: list[list[str]],
    test_data: dict[str, list[list[str]]],
) -> tuple[pd.DataFrame | None, dict[str, tuple]]:
    """Build distribution with our code and run inference on test months."""
    from ai_detection_utils import build_distribution, MLEEstimator

    # Count binary word occurrences (same logic as original)
    human_counts = Counter(w for sent in human_sents for w in set(sent))
    ai_counts = Counter(w for sent in ai_sents for w in set(sent))

    dist_df = build_distribution(
        human_counts=human_counts,
        n_human=len(human_sents),
        ai_counts=ai_counts,
        n_ai=len(ai_sents),
        min_human_count=5,
        min_ai_count=3,
    )

    if dist_df is None:
        print("  ERROR: our build_distribution returned None")
        return None, {}

    print(f"  Our distribution: {len(dist_df)} words")

    # Save to temp file for MLEEstimator
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        dist_df.to_parquet(f.name, index=False)
        estimator = MLEEstimator(Path(f.name))

    results = {}
    for month, sents in sorted(test_data.items()):
        if len(sents) < 50:
            continue
        alpha, lo, hi, n_used = estimator.inference(sents, n_bootstrap=1000)
        results[month] = (alpha, lo, hi, n_used)

    return dist_df, results


# ---------------------------------------------------------------------------
# Their pipeline
# ---------------------------------------------------------------------------
def run_their_pipeline(
    human_path: Path,
    ai_path: Path,
    test_data_dir: Path,
    months: list[str],
) -> tuple[pd.DataFrame | None, dict[str, tuple]]:
    """Run the original Liang et al. code."""
    sys.path.insert(0, str(LIANG_SRC))
    from estimation import estimate_text_distribution
    from MLE import MLE

    # Build distribution
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        dist_path = f.name
    estimate_text_distribution(str(human_path), str(ai_path), dist_path)
    dist_df = pd.read_parquet(dist_path)
    print(f"  Their distribution: {len(dist_df)} words")

    # Run MLE inference
    mle = MLE(dist_path)
    results = {}
    for month in months:
        test_path = test_data_dir / f"{month}.parquet"
        if not test_path.exists():
            continue
        try:
            solution, half_width = mle.inference(str(test_path))
            results[month] = (solution, half_width)
        except Exception as e:
            print(f"    {month}: ERROR — {e}")

    return dist_df, results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Benchmark against Liang et al.")
    parser.add_argument(
        "--months", nargs="+",
        default=["2022_1", "2022_6", "2023_1", "2023_6", "2024_1"],
        help="Test months to evaluate (format: YYYY_M)",
    )
    parser.add_argument(
        "--all-months", action="store_true",
        help="Run all available test months (overrides --months)",
    )
    args = parser.parse_args()

    # Validate repo exists
    if not LIANG_REPO.exists():
        print(f"ERROR: Liang et al. repo not found at {LIANG_REPO}")
        print("Clone it with: git clone https://github.com/Weixin-Liang/LLM-widespread-adoption-impact")
        sys.exit(1)

    human_path = LIANG_TRAIN / "human_data.parquet"
    ai_path = LIANG_TRAIN / "ai_data.parquet"

    if not human_path.exists() or not ai_path.exists():
        print(f"ERROR: Training data not found at {LIANG_TRAIN}")
        sys.exit(1)

    # Discover months
    if args.all_months:
        months = sorted(
            p.stem for p in LIANG_TEST.glob("*.parquet")
        )
    else:
        months = args.months

    print("=" * 70)
    print("  Benchmark: Our code vs Liang et al. (2025) reference")
    print("=" * 70)
    print(f"  Training data: {LIANG_TRAIN}")
    print(f"  Test months: {months}")

    # Load training data (shared between both pipelines for our code)
    print("\nLoading training data...")
    human_sents = load_sentences_from_parquet(human_path, "human_sentence")
    ai_sents = load_sentences_from_parquet(ai_path, "ai_sentence")
    print(f"  Human: {len(human_sents)} sentences")
    print(f"  AI:    {len(ai_sents)} sentences")

    # Load test data for our pipeline (we need raw sentences)
    test_data = {}
    for month in months:
        test_path = LIANG_TEST / f"{month}.parquet"
        if test_path.exists():
            test_data[month] = load_sentences_from_parquet(test_path, "inference_sentence")
            print(f"  Test {month}: {len(test_data[month])} sentences")

    # --- Run our pipeline ---
    print("\n" + "-" * 70)
    print("  Running OUR pipeline")
    print("-" * 70)
    our_dist, our_results = run_our_pipeline(human_sents, ai_sents, test_data)

    # --- Run their pipeline ---
    print("\n" + "-" * 70)
    print("  Running THEIR pipeline")
    print("-" * 70)
    their_dist, their_results = run_their_pipeline(
        human_path, ai_path, LIANG_TEST, months
    )

    # --- Compare distributions ---
    if our_dist is not None and their_dist is not None:
        print("\n" + "=" * 70)
        print("  Distribution comparison")
        print("=" * 70)

        our_words = set(our_dist["word"])
        their_words = set(their_dist["Word"])
        overlap = our_words & their_words
        print(f"  Our vocab:    {len(our_words)} words")
        print(f"  Their vocab:  {len(their_words)} words")
        print(f"  Overlap:      {len(overlap)} words ({100*len(overlap)/max(len(their_words),1):.1f}%)")

        # Compare logP/logQ for overlapping words
        our_lookup = {row["word"]: row for _, row in our_dist.iterrows()}
        their_lookup = {row["Word"]: row for _, row in their_dist.iterrows()}

        logP_diffs = []
        logQ_diffs = []
        for w in overlap:
            logP_diffs.append(abs(our_lookup[w]["logP"] - their_lookup[w]["logP"]))
            logQ_diffs.append(abs(our_lookup[w]["logQ"] - their_lookup[w]["logQ"]))

        print(f"  Mean |logP diff|: {np.mean(logP_diffs):.6f}")
        print(f"  Mean |logQ diff|: {np.mean(logQ_diffs):.6f}")
        print(f"  Max  |logP diff|: {np.max(logP_diffs):.6f}")
        print(f"  Max  |logQ diff|: {np.max(logQ_diffs):.6f}")

    # --- Compare results ---
    print("\n" + "=" * 70)
    print("  Inference comparison")
    print("=" * 70)
    print(f"  {'Month':<10} {'Ours α':>10} {'Ours CI':>20} {'Theirs α':>10} {'Theirs ±':>10} {'Δα':>10}")
    print("  " + "-" * 70)

    for month in sorted(set(list(our_results.keys()) + list(their_results.keys()))):
        ours = our_results.get(month)
        theirs = their_results.get(month)

        our_str = f"{ours[0]:.4f}" if ours else "—"
        our_ci = f"[{ours[1]:.4f}, {ours[2]:.4f}]" if ours else "—"
        their_str = f"{theirs[0]:.3f}" if theirs else "—"
        their_hw = f"±{theirs[1]:.3f}" if theirs else "—"

        if ours and theirs:
            delta = abs(ours[0] - theirs[0])
            delta_str = f"{delta:.4f}"
        else:
            delta_str = "—"

        print(f"  {month:<10} {our_str:>10} {our_ci:>20} {their_str:>10} {their_hw:>10} {delta_str:>10}")

    # --- Sanity check ---
    if "2022_1" in our_results:
        alpha_jan22 = our_results["2022_1"][0]
        print(f"\n  Sanity check: Jan 2022 α = {alpha_jan22:.4f} (expected ≈ 0.025)")
        if alpha_jan22 < 0.10:
            print("  PASS: Pre-ChatGPT baseline is low")
        else:
            print("  WARN: Pre-ChatGPT baseline seems high")

    # Check for post-ChatGPT increase
    pre_alphas = [v[0] for k, v in our_results.items() if k.startswith("2022")]
    post_alphas = [v[0] for k, v in our_results.items() if k.startswith(("2023", "2024"))]
    if pre_alphas and post_alphas:
        pre_mean = np.mean(pre_alphas)
        post_mean = np.mean(post_alphas)
        print(f"\n  Pre-ChatGPT  mean α: {pre_mean:.4f} (n={len(pre_alphas)} months)")
        print(f"  Post-ChatGPT mean α: {post_mean:.4f} (n={len(post_alphas)} months)")
        if post_mean > pre_mean:
            print("  PASS: Post-ChatGPT alpha is higher than pre-ChatGPT")
        else:
            print("  WARN: Expected post > pre")

    # Also try with their pre-computed distribution
    print("\n" + "=" * 70)
    print("  Bonus: Our MLEEstimator with THEIR pre-computed distribution")
    print("=" * 70)
    if LIANG_DIST.exists():
        try:
            # Their distribution uses different column names, so we need to adapt
            their_precomputed = pd.read_parquet(LIANG_DIST)
            # Rename columns to match our format
            adapted = their_precomputed.rename(columns={
                "Word": "word",
                "log1-P": "log1mP",
                "log1-Q": "log1mQ",
            })
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
                adapted.to_parquet(f.name, index=False)
                from ai_detection_utils import MLEEstimator
                est = MLEEstimator(Path(f.name))

            for month in sorted(test_data.keys()):
                sents = test_data[month]
                if len(sents) < 50:
                    continue
                alpha, lo, hi, n_used = est.inference(sents, n_bootstrap=1000)
                print(f"  {month}: α = {alpha:.4f} [{lo:.4f}, {hi:.4f}] ({n_used} sents)")
        except Exception as e:
            print(f"  ERROR: {e}")
    else:
        print(f"  Pre-computed distribution not found at {LIANG_DIST}")


if __name__ == "__main__":
    main()
