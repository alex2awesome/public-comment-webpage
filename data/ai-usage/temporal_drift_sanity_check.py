#!/usr/bin/env python3
"""Temporal drift sanity check: is our signal AI or just modern language?

Constructs P from 2016-2022 human text and Q from 2023-2026 human text.
If we see elevated alpha, our method is detecting temporal language drift,
not AI usage. The magnitude of this signal is the baseline to subtract.

Usage:
    python temporal_drift_sanity_check.py \
        --doc-types rule proposed_rule notice \
        --output-dir ../ai-usage-generations/sanity_check/ \
        --workers 4
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from ai_detection_utils import (
    ensure_nltk_punkt,
    tokenize_text,
    sentence_log_probs_raw,
    shrink_p,
    optimize_kappa,
    build_distribution,
    iter_input_files,
    load_dedup_representatives,
    load_and_tokenize_file,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

BASE_DIR = SCRIPT_DIR.parent / "bulk_downloads"


def run_sanity_check(args):
    ensure_nltk_punkt()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for doc_type in args.doc_types:
        logging.info("=== %s ===", doc_type)

        # Load all documents
        input_files = list(iter_input_files(BASE_DIR, doc_type))
        dedup_reps = load_dedup_representatives(BASE_DIR, doc_type)

        records = []
        if args.workers > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(load_and_tokenize_file, f, None, dedup_reps): f
                    for f in input_files
                }
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"load {doc_type}"):
                    records.extend(future.result())
        else:
            for f in tqdm(input_files, desc=f"load {doc_type}"):
                records.extend(load_and_tokenize_file(f, None, dedup_reps))

        rec_df = pd.DataFrame(records)
        logging.info("%s: %d total documents", doc_type, len(rec_df))

        # Split: P = pre-ChatGPT (2016-2022), Q = post-ChatGPT (2023+)
        pre_df = rec_df[rec_df["year"] < 2023]
        post_df = rec_df[rec_df["year"] >= 2023]
        logging.info("P (2016-2022): %d docs, Q (2023+): %d docs", len(pre_df), len(post_df))

        if len(pre_df) < 100 or len(post_df) < 100:
            logging.warning("Not enough data, skipping")
            continue

        # Build P word counts from pre-ChatGPT
        p_counts = Counter()
        n_p = 0
        p_agency_accum: Dict[str, Dict] = {}
        for _, row in tqdm(pre_df.iterrows(), total=len(pre_df), desc="P counts"):
            agency = row["agency_id"]
            for sent in row["sentences"]:
                ws = set(sent)
                p_counts.update(ws)
                n_p += 1
                acc = p_agency_accum.setdefault(agency, {"counts": Counter(), "n_sents": 0})
                acc["counts"].update(ws)
                acc["n_sents"] += 1

        # Build Q word counts from post-ChatGPT HUMAN text (NOT AI)
        q_counts = Counter()
        n_q = 0
        q_agency_accum: Dict[str, Dict] = {}
        for _, row in tqdm(post_df.iterrows(), total=len(post_df), desc="Q counts"):
            agency = row["agency_id"]
            for sent in row["sentences"]:
                ws = set(sent)
                q_counts.update(ws)
                n_q += 1
                acc = q_agency_accum.setdefault(agency, {"counts": Counter(), "n_sents": 0})
                acc["counts"].update(ws)
                acc["n_sents"] += 1

        logging.info("P: %d sentences, Q: %d sentences", n_p, n_q)

        # Build distribution
        dist_df = build_distribution(p_counts, n_p, q_counts, n_q,
                                     min_human_count=5, min_ai_count=3)
        if dist_df is None:
            logging.error("Could not build distribution")
            continue
        logging.info("Vocab: %d words", len(dist_df))

        # Save distribution
        dist_path = output_dir / f"temporal_dist_{doc_type}.parquet"
        dist_df.to_parquet(dist_path, index=False)

        # Compute kappa
        common_vocab = set(dist_df["word"].values)
        p_agency_counts = {
            aid: (acc["counts"], acc["n_sents"])
            for aid, acc in p_agency_accum.items()
        }
        pool_freq = {w: p_counts[w] / n_p for w in common_vocab}
        try:
            kappa = optimize_kappa(pool_freq, p_agency_counts, common_vocab)
        except Exception:
            kappa = 500.0

        # Build estimator data
        vocab_list = list(dist_df["word"].values)
        w2i = {w: i for i, w in enumerate(vocab_list)}
        vocab_set = set(vocab_list)
        mu_p = np.exp(dist_df["logP"].values.astype(float))
        logQ = dist_df["logQ"].values.astype(float)
        log1mQ = dist_df["log1mQ"].values.astype(float)
        delta_q = logQ - log1mQ
        baseline_q = float(log1mQ.sum())

        # Infer: estimate alpha per (agency, half) using hierarchical P
        # Score ALL documents (pre and post) against this temporal P/Q
        results = []
        for (agency_id, half), grp in rec_df.groupby(["agency_id", "half"]):
            all_sents = []
            for sent_list in grp["sentences"]:
                all_sents.extend(sent_list)
            if len(all_sents) < args.min_sentences:
                continue

            filtered = [s for s in all_sents if set(s) & vocab_set]
            if len(filtered) < args.min_sentences:
                continue

            # Agency-specific P via shrinkage
            if agency_id in p_agency_counts:
                wc, n_a = p_agency_counts[agency_id]
                n_aw = np.array([wc.get(w, 0) for w in vocab_list], dtype=float)
            else:
                n_aw = np.zeros(len(vocab_list))
                n_a = 0
            p_shrunk = shrink_p(mu_p, n_aw, n_a, kappa)
            logP_a = np.log(p_shrunk)
            log1mP_a = np.log(1 - p_shrunk)
            delta_p = logP_a - log1mP_a
            baseline_p = float(log1mP_a.sum())

            log_p, log_q = sentence_log_probs_raw(
                filtered, w2i, delta_p, delta_q, baseline_p, baseline_q)

            # Bootstrap MLE
            rng = np.random.RandomState(hash(f"{agency_id}_{half}") % (2**31))
            n = len(log_p)
            alphas = []
            for _ in range(args.bootstrap_n):
                idx = rng.choice(n, size=n, replace=True)
                res = minimize(
                    lambda a, lp=log_p[idx], lq=log_q[idx]: -np.mean(np.log(np.maximum(
                        (1 - a[0]) + a[0] * np.exp(lq - lp), 1e-300))),
                    x0=[0.5], method="L-BFGS-B", bounds=[(0, 1)])
                if res.success:
                    alphas.append(float(res.x[0]))

            if not alphas:
                continue

            ci_lo = float(np.percentile(alphas, 2.5))
            ci_hi = float(np.percentile(alphas, 97.5))
            alpha_point = float(np.mean([ci_lo, ci_hi]))

            year = int(str(half)[:4])
            results.append({
                "doc_type": doc_type,
                "agency_id": agency_id,
                "half": half,
                "year": year,
                "alpha_estimate": round(alpha_point, 6),
                "ci_lower": round(ci_lo, 6),
                "ci_upper": round(ci_hi, 6),
                "n_documents": len(grp),
                "n_sentences": len(filtered),
            })

        results_df = pd.DataFrame(results)
        out_path = output_dir / f"temporal_drift_{doc_type}.csv.gz"
        results_df.to_csv(out_path, index=False, compression="gzip")
        logging.info("Wrote %s (%d strata)", out_path, len(results_df))

        # Summary
        pre_results = results_df[results_df["year"] < 2023]
        post_results = results_df[results_df["year"] >= 2023]

        if len(pre_results) > 0 and pre_results["n_sentences"].sum() > 0:
            pre_alpha = np.average(pre_results["alpha_estimate"], weights=pre_results["n_sentences"])
        else:
            pre_alpha = 0

        if len(post_results) > 0 and post_results["n_sentences"].sum() > 0:
            post_alpha = np.average(post_results["alpha_estimate"], weights=post_results["n_sentences"])
        else:
            post_alpha = 0

        logging.info("TEMPORAL DRIFT RESULT for %s:", doc_type)
        logging.info("  Pre-2023 alpha (should be ~0): %.4f (%.2f%%)", pre_alpha, pre_alpha * 100)
        logging.info("  Post-2023 alpha (temporal drift): %.4f (%.2f%%)", post_alpha, post_alpha * 100)
        logging.info("  If this is similar to our AI detection alpha, the signal is drift not AI.")

        # Top words by LOR (temporal drift words)
        logging.info("\n  Top 15 'modern' words (high in 2023+ vs 2016-2022):")
        for _, r in dist_df.head(15).iterrows():
            logging.info("    %-20s LOR=%+.3f  pre=%d  post=%d",
                        r["word"], r["log_odds_ratio"], r["human_count"], r["ai_count"])

        logging.info("\n  Top 15 'old' words (high in 2016-2022 vs 2023+):")
        for _, r in dist_df.tail(15).iterrows():
            logging.info("    %-20s LOR=%+.3f  pre=%d  post=%d",
                        r["word"], r["log_odds_ratio"], r["human_count"], r["ai_count"])


def main():
    parser = argparse.ArgumentParser(
        description="Temporal drift sanity check: P=pre-2023, Q=post-2023 (both human)")
    parser.add_argument("--doc-types", nargs="+", default=["rule", "proposed_rule", "notice"])
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR.parent / "ai-usage-generations" / "sanity_check"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-sentences", type=int, default=50)
    parser.add_argument("--bootstrap-n", type=int, default=100)
    args = parser.parse_args()
    run_sanity_check(args)


if __name__ == "__main__":
    main()
