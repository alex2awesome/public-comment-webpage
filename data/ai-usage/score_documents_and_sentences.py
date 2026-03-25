#!/usr/bin/env python3
"""Compute document-level and sentence-level AI scores with hierarchical shrinkage.

Uses the best optuna config's distribution + agency word counts to build
per-agency P_a(w) and Q_a(w) distributions, then scores every document and sentence.

Document score: MAP alpha with Beta prior centered on stratum alpha.
Sentence score: P(AI | sentence, alpha) = alpha * Q_a(s) / [(1-alpha)*P_a(s) + alpha*Q_a(s)]

Usage:
    python score_documents_and_sentences.py \
        --doc-types rule proposed_rule notice \
        --output-dir ../ai-usage-generations/best_configs/scores/ \
        --sentence-scores --workers 4
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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
    iter_input_files,
    load_dedup_representatives,
    load_and_tokenize_file,
    build_distribution,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

BASE_DIR = SCRIPT_DIR.parent / "bulk_downloads"
BEST_DIR = SCRIPT_DIR.parent / "ai-usage-generations" / "best_configs"
AI_CORPUS_BASE = SCRIPT_DIR.parent / "ai-usage-generations"

MODEL_SOURCES = {
    "mix-all": ["llama-v4-generations", "llama-v4-8b-generations", "gpt5mini-generations"],
    "llama-3.1-8b": ["llama-v4-8b-generations"],
    "mixture": ["llama-v4-generations", "llama-v4-8b-generations", "gpt5mini-generations"],
}

# Best configs from optuna (half-year)
BEST_CONFIGS = {
    "rule": {
        "model": "mix-all", "matched": True, "min_human_count": 20,
        "min_ai_count": 3, "max_vocab": None, "min_word_len": 0,
        "min_lor": 0.0, "kappa": 100, "stratify_by": "half",
        "min_sentences": 100, "results_file": "best_rule_half.csv.gz",
        "kappa_doc": 10,
    },
    "proposed_rule": {
        "model": "mix-all", "matched": True, "min_human_count": 5,
        "min_ai_count": 5, "max_vocab": None, "min_word_len": 5,
        "min_lor": 0.5, "kappa": 50, "stratify_by": "half",
        "min_sentences": 25, "results_file": "best_proposed_rule_half.csv.gz",
        "kappa_doc": 10,
    },
    "notice": {
        "model": "mixture", "matched": True, "min_human_count": 3,
        "min_ai_count": 3, "max_vocab": None, "min_word_len": 4,
        "min_lor": 0.0, "kappa": 100, "stratify_by": "half",
        "min_sentences": 50, "results_file": "best_notice.csv.gz",
        "kappa_doc": 10,
    },
}


# ---------------------------------------------------------------------------
# MLE / MAP helpers
# ---------------------------------------------------------------------------

def fit_alpha_mle(log_p: np.ndarray, log_q: np.ndarray) -> float:
    result = minimize(
        lambda a, lp=log_p, lq=log_q: -np.mean(np.log(np.maximum(
            (1 - a[0]) + a[0] * np.exp(lq - lp), 1e-300))),
        x0=[0.5], method="L-BFGS-B", bounds=[(0, 1)],
    )
    return float(result.x[0]) if result.success else float("nan")


def fit_alpha_map(log_p, log_q, alpha_prior, kappa_doc):
    a0 = kappa_doc * max(alpha_prior, 1e-6)
    b0 = kappa_doc * max(1 - alpha_prior, 1e-6)

    def neg_log_posterior(alpha):
        alpha = alpha[0]
        ll = np.sum(np.log(np.maximum(
            (1 - alpha) + alpha * np.exp(log_q - log_p), 1e-300)))
        log_prior = (a0 - 1) * np.log(max(alpha, 1e-300)) + \
                    (b0 - 1) * np.log(max(1 - alpha, 1e-300))
        return -(ll + log_prior)

    result = minimize(
        neg_log_posterior,
        x0=[max(0.01, min(0.99, alpha_prior))],
        method="L-BFGS-B", bounds=[(1e-8, 1 - 1e-8)],
    )
    return float(result.x[0]) if result.success else float("nan")


def sentence_ai_probability(log_p_sent, log_q_sent, alpha):
    if alpha <= 0: return 0.0
    if alpha >= 1: return 1.0
    log_num = np.log(alpha) + log_q_sent
    log_den = np.logaddexp(np.log(1 - alpha) + log_p_sent, np.log(alpha) + log_q_sent)
    return float(np.exp(log_num - log_den))


# ---------------------------------------------------------------------------
# Build agency-specific P/Q and estimate
# ---------------------------------------------------------------------------

def build_estimate_with_agency_counts(
    doc_type: str, cfg: Dict,
) -> Tuple[pd.DataFrame, Dict, Dict, float, float]:
    """Build full distribution + per-agency word counts for P and Q.

    Returns: (dist_df, human_agency_counts, ai_agency_counts, kappa_p, kappa_q)
    """
    from ai_detection_utils import optimize_kappa

    model = cfg["model"]
    matched = cfg["matched"]

    # Load AI corpus
    subdirs = MODEL_SOURCES.get(model, [model])
    ai_frames = []
    for subdir in subdirs:
        path = AI_CORPUS_BASE / subdir / f"ai_corpus_{doc_type}.parquet"
        if path.exists():
            ai_frames.append(pd.read_parquet(path))
    if not ai_frames:
        raise FileNotFoundError(f"No AI corpus for {model}/{doc_type}")
    ai_df = pd.concat(ai_frames, ignore_index=True)
    ai_doc_ids = set(ai_df["document_id"].astype(str)) if matched else None

    # Load human documents
    input_files = list(iter_input_files(BASE_DIR, doc_type))
    dedup_reps = load_dedup_representatives(BASE_DIR, doc_type)
    records = []
    for f in tqdm(input_files, desc=f"load human {doc_type}"):
        records.extend(load_and_tokenize_file(f, None, dedup_reps))
    human_df = pd.DataFrame(records)

    # Build human word counts (pooled + per-agency)
    human_counts = Counter()
    n_human = 0
    human_agency_accum: Dict[str, Dict] = {}
    for _, row in tqdm(human_df.iterrows(), total=len(human_df), desc="human counts"):
        doc_id = str(row.get("document_id", ""))
        if ai_doc_ids is not None and doc_id not in ai_doc_ids:
            continue
        agency = row["agency_id"]
        for sent in row["sentences"]:
            ws = set(sent)
            human_counts.update(ws)
            n_human += 1
            acc = human_agency_accum.setdefault(agency, {"counts": Counter(), "n_sents": 0})
            acc["counts"].update(ws)
            acc["n_sents"] += 1

    # Build AI word counts (pooled + per-agency)
    ai_counts = Counter()
    n_ai = 0
    ai_agency_accum: Dict[str, Dict] = {}
    for _, row in tqdm(ai_df.iterrows(), total=len(ai_df), desc="AI counts"):
        text = row.get("ai_text", "")
        agency = row.get("agency_id", "")
        if not isinstance(text, str) or not text:
            continue
        for sent in tokenize_text(text):
            ws = set(sent)
            ai_counts.update(ws)
            n_ai += 1
            acc = ai_agency_accum.setdefault(agency, {"counts": Counter(), "n_sents": 0})
            acc["counts"].update(ws)
            acc["n_sents"] += 1

    # Build distribution (min_count=1, filter later)
    dist_df = build_distribution(human_counts, n_human, ai_counts, n_ai,
                                 min_human_count=1, min_ai_count=1)

    # Compute optimal kappas
    common_vocab = set(dist_df["word"].values)
    human_agency_counts = {
        aid: (acc["counts"], acc["n_sents"])
        for aid, acc in human_agency_accum.items()
    }
    ai_agency_counts = {
        aid: (acc["counts"], acc["n_sents"])
        for aid, acc in ai_agency_accum.items()
    }

    pool_freq_p = {w: human_counts[w] / n_human for w in common_vocab}
    pool_freq_q = {w: ai_counts[w] / n_ai for w in common_vocab}

    kappa_p = optimize_kappa(pool_freq_p, human_agency_counts, common_vocab)
    kappa_q = optimize_kappa(pool_freq_q, ai_agency_counts, common_vocab)

    return dist_df, human_agency_counts, ai_agency_counts, kappa_p, kappa_q, human_df


def filter_dist(dist_df, cfg):
    """Apply vocab filtering from config."""
    df = dist_df.copy()
    if cfg.get("min_human_count", 1) > 1:
        df = df[df["human_count"] >= cfg["min_human_count"]]
    if cfg.get("min_ai_count", 1) > 1:
        df = df[df["ai_count"] >= cfg["min_ai_count"]]
    if cfg.get("min_word_len", 0) > 0:
        df = df[df["word"].str.len() >= cfg["min_word_len"]]
    if cfg.get("min_lor", 0) > 0:
        df = df[df["log_odds_ratio"].abs() >= cfg["min_lor"]]
    if cfg.get("max_vocab"):
        df = df.nlargest(cfg["max_vocab"], "ai_count")
        df = df.sort_values("log_odds_ratio").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_scoring(args):
    ensure_nltk_punkt()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for doc_type in args.doc_types:
        logging.info("=== Scoring %s ===", doc_type)
        cfg = BEST_CONFIGS.get(doc_type)
        if not cfg:
            continue

        # Build estimate with agency counts
        logging.info("Building estimate with per-agency counts...")
        dist_df, human_agency_counts, ai_agency_counts, kappa_p, kappa_q, human_df = \
            build_estimate_with_agency_counts(doc_type, cfg)
        logging.info("kappa_p=%.1f, kappa_q=%.1f, full vocab=%d", kappa_p, kappa_q, len(dist_df))

        # Filter distribution
        filtered = filter_dist(dist_df, cfg)
        logging.info("Filtered vocab: %d words", len(filtered))

        # Build per-agency est_data
        vocab_list = list(filtered["word"].values)
        w2i = {w: i for i, w in enumerate(vocab_list)}
        vocab_set = set(vocab_list)
        mu_p = np.exp(filtered["logP"].values.astype(float))
        mu_q = np.exp(filtered["logQ"].values.astype(float))
        log1mQ_pool = filtered["log1mQ"].values.astype(float)
        logQ_pool = filtered["logQ"].values.astype(float)

        # Use config kappa or empirical
        use_kappa_p = cfg.get("kappa", kappa_p)
        use_kappa_q = kappa_q  # always use empirical for Q

        # Load stratum results for priors
        results_path = BEST_DIR / cfg["results_file"]
        results_df = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
        stratum_lookup = {}
        time_col = None
        for col in ["half", "quarter", "year"]:
            if col in results_df.columns:
                time_col = col
                break
        if time_col and "agency_id" in results_df.columns:
            for _, r in results_df.iterrows():
                stratum_lookup[(r["agency_id"], str(r[time_col]))] = r["alpha_estimate"]

        logging.info("Loaded %d stratum priors", len(stratum_lookup))
        kappa_doc = args.kappa_doc or cfg["kappa_doc"]

        # Score each document with agency-specific P_a and Q_a
        doc_scores = []
        sent_scores = []

        for _, row in tqdm(human_df.iterrows(), total=len(human_df), desc=f"score {doc_type}"):
            sents = row["sentences"]
            filtered_sents = [s for s in sents if set(s) & vocab_set]
            if len(filtered_sents) < 3:
                continue

            agency_id = row["agency_id"]

            # Build agency-specific P_a(w)
            if agency_id in human_agency_counts:
                hc, n_a = human_agency_counts[agency_id]
                n_aw = np.array([hc.get(w, 0) for w in vocab_list], dtype=float)
            else:
                n_aw = np.zeros(len(vocab_list))
                n_a = 0
            p_shrunk = shrink_p(mu_p, n_aw, n_a, use_kappa_p)
            logP_a = np.log(p_shrunk)
            log1mP_a = np.log(1 - p_shrunk)

            # Build agency-specific Q_a(w)
            if agency_id in ai_agency_counts:
                ac, n_ai_a = ai_agency_counts[agency_id]
                n_ai_aw = np.array([ac.get(w, 0) for w in vocab_list], dtype=float)
                q_shrunk = shrink_p(mu_q, n_ai_aw, n_ai_a, use_kappa_q)
                logQ_a = np.log(q_shrunk)
                log1mQ_a = np.log(1 - q_shrunk)
            else:
                logQ_a = logQ_pool
                log1mQ_a = log1mQ_pool

            delta_p = logP_a - log1mP_a
            delta_q = logQ_a - log1mQ_a
            baseline_p = float(log1mP_a.sum())
            baseline_q = float(log1mQ_a.sum())

            log_p, log_q = sentence_log_probs_raw(
                filtered_sents, w2i, delta_p, delta_q, baseline_p, baseline_q,
            )

            # Stratum prior
            time_val = row.get("half") or row.get("quarter") or str(row.get("year", ""))
            alpha_prior = stratum_lookup.get((agency_id, time_val))

            # Document alpha (MAP or MLE)
            if alpha_prior is not None and kappa_doc > 0:
                doc_alpha = fit_alpha_map(log_p, log_q, alpha_prior, kappa_doc)
            else:
                doc_alpha = fit_alpha_mle(log_p, log_q)

            doc_scores.append({
                "document_id": row["document_id"],
                "agency_id": agency_id,
                "doc_type": doc_type,
                "quarter": row.get("quarter", ""),
                "half": row.get("half", ""),
                "year": row.get("year", ""),
                "alpha_doc": round(doc_alpha, 6),
                "alpha_stratum": round(alpha_prior, 6) if alpha_prior is not None else None,
                "n_sentences": len(filtered_sents),
            })

            # Sentence scores
            if args.sentence_scores:
                use_alpha = doc_alpha if doc_alpha > 0 else (alpha_prior or 0)
                for si, (sent, lp, lq) in enumerate(zip(filtered_sents, log_p, log_q)):
                    p_ai = sentence_ai_probability(lp, lq, use_alpha)
                    sent_scores.append({
                        "document_id": row["document_id"],
                        "agency_id": agency_id,
                        "doc_type": doc_type,
                        "sentence_idx": si,
                        "sentence_text": " ".join(sent),
                        "p_ai": round(p_ai, 6),
                        "log_p": round(float(lp), 4),
                        "log_q": round(float(lq), 4),
                        "alpha_doc": round(doc_alpha, 6),
                    })

        # Write
        doc_df = pd.DataFrame(doc_scores)
        doc_path = output_dir / f"doc_scores_{doc_type}.csv.gz"
        doc_df.to_csv(doc_path, index=False, compression="gzip")
        logging.info("Wrote %s (%d docs)", doc_path, len(doc_df))

        if args.sentence_scores and sent_scores:
            sent_df = pd.DataFrame(sent_scores)
            sent_path = output_dir / f"sent_scores_{doc_type}.csv.gz"
            sent_df.to_csv(sent_path, index=False, compression="gzip")
            logging.info("Wrote %s (%d sentences)", sent_path, len(sent_df))

        # Save the filtered distribution for reference
        dist_path = output_dir / f"distribution_{doc_type}.parquet"
        filtered.to_parquet(dist_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-types", nargs="+", default=["rule", "proposed_rule", "notice"])
    parser.add_argument("--output-dir", default=str(BEST_DIR / "scores"))
    parser.add_argument("--kappa-doc", type=float, default=None)
    parser.add_argument("--sentence-scores", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run_scoring(args)


if __name__ == "__main__":
    main()
