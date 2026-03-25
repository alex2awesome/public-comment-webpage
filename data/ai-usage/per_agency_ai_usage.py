#!/usr/bin/env python3
"""Per-agency AI usage estimation (isolated, no hierarchical model).

For each (doc_type, agency) pair with >= min_docs pre-ChatGPT documents,
builds agency-specific P and Q distributions from scratch, then runs
MLE inference per (agency, quarter) stratum.

Agencies below the document threshold are skipped entirely (to be scored
with Pangram or another classifier).

Usage:
    # Estimate per-agency distributions
    python3 per_agency_ai_usage.py estimate \
        --base-dir data/ --ai-corpus-dir data/v3 \
        --output-dir data/v5_per_agency \
        --doc-types rule notice proposed_rule public_submission \
        --min-docs 100 --dedup --workers 8

    # Infer using per-agency distributions
    python3 per_agency_ai_usage.py infer \
        --base-dir data/ --distribution-dir data/v5_per_agency \
        --doc-types rule notice proposed_rule public_submission \
        --output data/v5_per_agency/ai_usage_results.csv.gz \
        --dedup --document-scores \
        --doc-scores-output data/v5_per_agency/ai_usage_doc_scores.csv.gz \
        --workers 8
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Import shared utilities
# ---------------------------------------------------------------------------
from ai_detection_utils import (
    DEFAULT_BOOTSTRAP_N,
    DEFAULT_HUMAN_CUTOFF,
    DEFAULT_MIN_AI_COUNT,
    DEFAULT_MIN_HUMAN_COUNT,
    DEFAULT_MIN_SENTENCES,
    ensure_nltk_punkt,
    load_and_tokenize_file,
    sentence_log_probs_raw,
    is_pre_chatgpt,
    iter_input_files,
    load_dedup_representatives,
    tokenize_text,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


# ===================================================================
# Estimate: build per-agency P/Q distributions
# ===================================================================


def _process_human_csv_per_agency(args_tuple):
    """Process one CSV file: return per-agency word counts.

    Top-level function for multiprocessing pickling.
    Returns Dict[agency_id, {"counts": Counter, "n_sents": int, "n_docs": int}]
    """
    csv_path, human_cutoff, eligible_agencies, dedup_reps = args_tuple
    csv_path = Path(csv_path)

    cols = ["Posted Date", "canonical_text", "Agency ID"]
    if dedup_reps is not None:
        cols.append("Document ID")
    try:
        df = pd.read_csv(csv_path, usecols=cols, low_memory=False)
    except (ValueError, KeyError):
        return {}

    df = df.dropna(subset=["canonical_text", "Posted Date"])
    df = df[df["Posted Date"].apply(lambda d: is_pre_chatgpt(d, human_cutoff))]
    df = df[df["canonical_text"].str.len() >= 100]
    if eligible_agencies:
        df = df[df["Agency ID"].isin(eligible_agencies)]
    if dedup_reps is not None and "Document ID" in df.columns:
        df = df[df["Document ID"].astype(str).isin(dedup_reps)]

    agency_data: Dict[str, Dict] = {}
    for agency_id, text in zip(df["Agency ID"], df["canonical_text"]):
        acc = agency_data.setdefault(
            agency_id, {"counts": Counter(), "n_sents": 0, "n_docs": 0}
        )
        acc["n_docs"] += 1
        for sent in tokenize_text(text):
            word_set = set(sent)
            acc["counts"].update(word_set)
            acc["n_sents"] += 1

    return agency_data


def _count_agency_human_docs(
    base_dir: Path,
    doc_type: str,
    human_cutoff: str,
    dedup_reps: Optional[Set[str]],
) -> Dict[str, int]:
    """Quick pass: count pre-ChatGPT documents per agency."""
    agency_doc_counts: Dict[str, int] = Counter()
    for csv_path in iter_input_files(base_dir, doc_type):
        cols = ["Posted Date", "Agency ID"]
        if dedup_reps is not None:
            cols.append("Document ID")
        try:
            df = pd.read_csv(csv_path, usecols=cols, low_memory=False)
        except (ValueError, KeyError):
            continue
        df = df.dropna(subset=["Posted Date"])
        df = df[df["Posted Date"].apply(lambda d: is_pre_chatgpt(d, human_cutoff))]
        if dedup_reps is not None and "Document ID" in df.columns:
            df = df[df["Document ID"].astype(str).isin(dedup_reps)]
        for aid, cnt in df["Agency ID"].value_counts().items():
            agency_doc_counts[aid] += cnt
    return dict(agency_doc_counts)


def _build_human_counts_per_agency(
    base_dir: Path,
    doc_type: str,
    human_cutoff: str,
    eligible_agencies: Set[str],
    dedup_reps: Optional[Set[str]],
    workers: int = 1,
) -> Dict[str, Tuple[Counter, int]]:
    """Build word counts per agency from pre-ChatGPT human text."""
    input_files = list(iter_input_files(base_dir, doc_type))

    tasks = [
        (str(f), human_cutoff, eligible_agencies, dedup_reps)
        for f in input_files
    ]

    # Merge results from all files
    merged: Dict[str, Dict] = {}

    if workers > 1 and len(tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        logging.info(
            "%s: processing %d files with %d workers",
            doc_type, len(tasks), workers,
        )
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_process_human_csv_per_agency, t): t[0]
                for t in tasks
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc=f"human {doc_type}"
            ):
                file_result = future.result()
                for agency_id, data in file_result.items():
                    if agency_id not in merged:
                        merged[agency_id] = {
                            "counts": Counter(), "n_sents": 0, "n_docs": 0
                        }
                    merged[agency_id]["counts"] += data["counts"]
                    merged[agency_id]["n_sents"] += data["n_sents"]
                    merged[agency_id]["n_docs"] += data["n_docs"]
    else:
        for task in tqdm(tasks, desc=f"human {doc_type}"):
            file_result = _process_human_csv_per_agency(task)
            for agency_id, data in file_result.items():
                if agency_id not in merged:
                    merged[agency_id] = {
                        "counts": Counter(), "n_sents": 0, "n_docs": 0
                    }
                merged[agency_id]["counts"] += data["counts"]
                merged[agency_id]["n_sents"] += data["n_sents"]
                merged[agency_id]["n_docs"] += data["n_docs"]

    return {
        aid: (d["counts"], d["n_sents"])
        for aid, d in merged.items()
        if d["n_sents"] > 0
    }


def _build_ai_counts_per_agency(
    ai_df: pd.DataFrame,
    eligible_agencies: Set[str],
) -> Dict[str, Tuple[Counter, int]]:
    """Build word counts per agency from AI corpus."""
    agency_data: Dict[str, Dict] = {}
    for agency_id, text in zip(
        ai_df["agency_id"], ai_df["ai_text"].fillna("")
    ):
        if agency_id not in eligible_agencies:
            continue
        acc = agency_data.setdefault(
            agency_id, {"counts": Counter(), "n_sents": 0}
        )
        for sent in tokenize_text(text):
            word_set = set(sent)
            acc["counts"].update(word_set)
            acc["n_sents"] += 1

    return {
        aid: (d["counts"], d["n_sents"])
        for aid, d in agency_data.items()
        if d["n_sents"] > 0
    }


def _build_distribution(
    human_counts: Counter,
    n_human: int,
    ai_counts: Counter,
    n_ai: int,
    min_human_count: int,
    min_ai_count: int,
) -> Optional[pd.DataFrame]:
    """Build word-level distribution DataFrame for one agency."""
    common_vocab = set(human_counts.keys()) & set(ai_counts.keys())
    common_vocab = {
        w for w in common_vocab
        if human_counts[w] >= min_human_count
        and ai_counts[w] >= min_ai_count
    }
    if len(common_vocab) < 50:
        return None

    rows = []
    for word in common_vocab:
        logP = np.log(human_counts[word] / n_human)
        logQ = np.log(ai_counts[word] / n_ai)
        log1mP = np.log1p(-np.exp(logP))
        log1mQ = np.log1p(-np.exp(logQ))
        lor = (logP - log1mP) - (logQ - log1mQ)
        if np.isinf(lor) or np.isnan(lor):
            continue
        rows.append({
            "word": word,
            "logP": logP,
            "logQ": logQ,
            "log1mP": log1mP,
            "log1mQ": log1mQ,
            "human_count": human_counts[word],
            "ai_count": ai_counts[word],
            "log_odds_ratio": lor,
        })
    if len(rows) < 50:
        return None
    return pd.DataFrame(rows).sort_values("log_odds_ratio").reset_index(drop=True)


def cmd_estimate(args: argparse.Namespace) -> None:
    """Estimate per-agency P/Q distributions."""
    ensure_nltk_punkt()

    base_dir = Path(args.base_dir)
    ai_corpus_dir = Path(args.ai_corpus_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata: Dict = {
        "min_docs": args.min_docs,
        "min_human_count": args.min_human_count,
        "min_ai_count": args.min_ai_count,
        "human_cutoff": args.human_cutoff,
    }

    for doc_type in args.doc_types:
        logging.info("=== %s ===", doc_type)

        # Dedup
        dedup_reps = None
        if args.dedup and doc_type == "public_submission":
            dedup_reps = load_dedup_representatives(base_dir, doc_type)
            if dedup_reps:
                logging.info("Dedup: %d representative docs", len(dedup_reps))

        # 1. Count docs per agency
        agency_doc_counts = _count_agency_human_docs(
            base_dir, doc_type, args.human_cutoff, dedup_reps,
        )
        logging.info(
            "%s: %d agencies total, %d with >= %d docs",
            doc_type,
            len(agency_doc_counts),
            sum(1 for c in agency_doc_counts.values() if c >= args.min_docs),
            args.min_docs,
        )

        eligible = {
            aid for aid, cnt in agency_doc_counts.items()
            if cnt >= args.min_docs
        }
        skipped = {
            aid for aid, cnt in agency_doc_counts.items()
            if cnt < args.min_docs
        }
        if not eligible:
            logging.warning("%s: no eligible agencies, skipping", doc_type)
            continue

        # 2. Load AI corpus for this doc type
        ai_parquet = ai_corpus_dir / f"ai_corpus_{doc_type}.parquet"
        if not ai_parquet.exists():
            logging.warning("Missing %s; skipping", ai_parquet)
            continue
        ai_df = pd.read_parquet(ai_parquet)
        logging.info("%s: %d AI corpus rows", doc_type, len(ai_df))

        # 3. Build per-agency human counts (parallelized)
        logging.info("%s: building human word counts for %d agencies...", doc_type, len(eligible))
        human_per_agency = _build_human_counts_per_agency(
            base_dir, doc_type, args.human_cutoff, eligible, dedup_reps,
            workers=args.workers,
        )

        # 4. Build per-agency AI counts
        logging.info("%s: building AI word counts...", doc_type)
        ai_per_agency = _build_ai_counts_per_agency(ai_df, eligible)
        del ai_df

        # 5. Build and save per-agency distributions
        doc_type_dir = output_dir / doc_type
        doc_type_dir.mkdir(parents=True, exist_ok=True)

        agency_meta = {}
        for agency_id in sorted(eligible):
            if agency_id not in human_per_agency:
                logging.debug("  %s: no human data after tokenization", agency_id)
                continue
            if agency_id not in ai_per_agency:
                logging.debug("  %s: no AI data", agency_id)
                continue

            h_counts, n_h = human_per_agency[agency_id]
            a_counts, n_a = ai_per_agency[agency_id]

            dist_df = _build_distribution(
                h_counts, n_h, a_counts, n_a,
                args.min_human_count, args.min_ai_count,
            )
            if dist_df is None:
                logging.info("  %s: vocab too small, skipping", agency_id)
                continue

            safe_name = agency_id.replace("/", "_").replace(" ", "_")
            out_path = doc_type_dir / f"distribution_{safe_name}.parquet"
            dist_df.to_parquet(out_path, index=False)

            agency_meta[agency_id] = {
                "vocab_size": len(dist_df),
                "human_sentences": n_h,
                "ai_sentences": n_a,
                "human_docs": agency_doc_counts.get(agency_id, 0),
                "distribution_file": out_path.name,
            }
            logging.info(
                "  %s: vocab=%d, human=%d sents, ai=%d sents",
                agency_id, len(dist_df), n_h, n_a,
            )

        metadata[doc_type] = {
            "eligible_agencies": agency_meta,
            "skipped_agencies": sorted(skipped),
            "n_eligible": len(agency_meta),
            "n_skipped": len(skipped),
        }
        logging.info(
            "%s: wrote distributions for %d agencies, skipped %d",
            doc_type, len(agency_meta), len(skipped),
        )

    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logging.info("Wrote %s", meta_path)


# ===================================================================
# Infer: run MLE per (agency, quarter)
# ===================================================================


def _optimized_log_likelihood(alpha, log_p_values, log_q_values):
    """Negative log likelihood — matches original Liang et al. implementation."""
    alpha = alpha[0]
    ll = np.mean(np.log(np.maximum(
        (1 - alpha) + alpha * np.exp(log_q_values - log_p_values), 1e-300,
    )))
    return -ll


def _fit_alpha(log_p: np.ndarray, log_q: np.ndarray) -> float:
    """MLE for alpha using L-BFGS-B, matching original paper."""
    result = minimize(
        _optimized_log_likelihood,
        x0=[0.5],
        args=(log_p, log_q),
        method="L-BFGS-B",
        bounds=[(0, 1)],
    )
    return float(result.x[0]) if result.success else float("nan")


def _neg_log_posterior(alpha, log_p_values, log_q_values, a0, b0):
    """Negative log posterior for MAP estimation with Beta prior on alpha."""
    alpha = alpha[0]
    ll = np.sum(np.log(np.maximum(
        (1 - alpha) + alpha * np.exp(log_q_values - log_p_values), 1e-300,
    )))
    log_prior = (a0 - 1) * np.log(max(alpha, 1e-300)) + \
                (b0 - 1) * np.log(max(1 - alpha, 1e-300))
    return -(ll + log_prior)


def _fit_alpha_map(
    log_p: np.ndarray, log_q: np.ndarray,
    alpha_prior: float, kappa_doc: float,
) -> float:
    """MAP for alpha with Beta(kappa_doc * alpha_prior, kappa_doc * (1 - alpha_prior)) prior."""
    a0 = kappa_doc * max(alpha_prior, 1e-6)
    b0 = kappa_doc * max(1 - alpha_prior, 1e-6)
    result = minimize(
        _neg_log_posterior,
        x0=[max(0.01, min(0.99, alpha_prior))],
        args=(log_p, log_q, a0, b0),
        method="L-BFGS-B",
        bounds=[(1e-8, 1 - 1e-8)],
    )
    return float(result.x[0]) if result.success else float("nan")


def _infer_stratum_task(task_tuple):
    """Top-level function for parallel stratum inference.

    task_tuple: (key_dict, sentences, est_data, n_bootstrap)
      key_dict: {"doc_type", "agency_id", "quarter", "year", "n_documents"}
      est_data: (w2i, delta_p, delta_q, baseline_p, baseline_q, vocab_set)
    Returns: result dict ready for DataFrame row.
    """
    key_dict, sentences, est_data, n_bootstrap = task_tuple
    w2i, delta_p, delta_q, baseline_p, baseline_q, vocab_set = est_data

    filtered = [s for s in sentences if set(s) & vocab_set]
    n_used = len(filtered)
    if n_used < 10:
        return {
            **key_dict,
            "alpha_estimate": None,
            "ci_lower": None,
            "ci_upper": None,
            "n_sentences": n_used,
            "vocab_size": len(vocab_set),
        }

    log_p, log_q = sentence_log_probs_raw(
        filtered, w2i, delta_p, delta_q, baseline_p, baseline_q,
    )

    # Bootstrap (matching original paper)
    # Seed per-task to avoid correlated samples across forked workers
    seed = hash(tuple(sorted(key_dict.items()))) % (2**31)
    rng = np.random.RandomState(seed)
    n = len(log_p)
    alphas = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        a = _fit_alpha(log_p[idx], log_q[idx])
        if not np.isnan(a):
            alphas.append(a)

    if not alphas:
        return {
            **key_dict,
            "alpha_estimate": None,
            "ci_lower": None,
            "ci_upper": None,
            "n_sentences": n_used,
            "vocab_size": len(vocab_set),
        }

    ci_lo = float(np.percentile(alphas, 2.5))
    ci_hi = float(np.percentile(alphas, 97.5))
    alpha_point = float(np.mean([ci_lo, ci_hi]))

    return {
        **key_dict,
        "alpha_estimate": round(alpha_point, 6),
        "ci_lower": round(ci_lo, 6),
        "ci_upper": round(ci_hi, 6),
        "n_sentences": n_used,
        "vocab_size": len(vocab_set),
    }


def cmd_infer(args: argparse.Namespace) -> None:
    """Run per-agency inference."""
    ensure_nltk_punkt()

    base_dir = Path(args.base_dir)
    dist_dir = Path(args.distribution_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    meta_path = dist_dir / "metadata.json"
    if not meta_path.exists():
        logging.error("Missing %s; run estimate first", meta_path)
        return
    with open(meta_path) as f:
        metadata = json.load(f)

    all_results = []
    all_doc_scores = []

    for doc_type in args.doc_types:
        logging.info("=== infer %s ===", doc_type)

        dt_meta = metadata.get(doc_type)
        if not dt_meta:
            logging.warning("No metadata for %s, skipping", doc_type)
            continue

        eligible_agencies = dt_meta["eligible_agencies"]
        if not eligible_agencies:
            continue

        # Dedup
        dedup_reps = None
        if args.dedup and doc_type == "public_submission":
            dedup_reps = load_dedup_representatives(base_dir, doc_type)

        # Load all documents once (parallelized)
        logging.info("%s: loading documents...", doc_type)
        input_files = list(iter_input_files(base_dir, doc_type))
        records: List[Dict] = []

        if args.workers > 1 and len(input_files) > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            from functools import partial

            load_fn = partial(
                load_and_tokenize_file,
                agencies_filter=None,
                dedup_reps=dedup_reps,
            )
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(load_fn, f): f for f in input_files}
                for future in tqdm(
                    as_completed(futures), total=len(futures),
                    desc=f"load {doc_type}",
                ):
                    records.extend(future.result())
        else:
            for csv_path in tqdm(input_files, desc=f"load {doc_type}"):
                records.extend(
                    load_and_tokenize_file(csv_path, None, dedup_reps)
                )

        if not records:
            logging.warning("No records for %s", doc_type)
            continue

        rec_df = pd.DataFrame(records)
        logging.info("%s: %d documents loaded", doc_type, len(rec_df))

        # Build all stratum inference tasks across agencies
        inference_tasks = []
        agency_doc_map: Dict[str, Tuple] = {}  # agency_id -> (est_data, agency_docs_df)

        for agency_id, agency_info in eligible_agencies.items():
            dist_file = agency_info["distribution_file"]
            dist_path = dist_dir / doc_type / dist_file
            if not dist_path.exists():
                logging.warning("Missing %s", dist_path)
                continue

            dist_df = pd.read_parquet(dist_path)
            vocab_list = list(dist_df["word"].values)
            w2i = {w: i for i, w in enumerate(vocab_list)}
            vocab_set = set(vocab_list)

            logP = dist_df["logP"].values.astype(float)
            logQ = dist_df["logQ"].values.astype(float)
            log1mP = dist_df["log1mP"].values.astype(float)
            log1mQ = dist_df["log1mQ"].values.astype(float)
            delta_p = logP - log1mP
            delta_q = logQ - log1mQ
            baseline_p = float(log1mP.sum())
            baseline_q = float(log1mQ.sum())
            est_data = (w2i, delta_p, delta_q, baseline_p, baseline_q, vocab_set)

            agency_docs = rec_df[rec_df["agency_id"] == agency_id]
            if agency_docs.empty:
                continue

            # Save for doc scoring later
            if args.document_scores:
                agency_doc_map[agency_id] = (est_data, agency_docs)

            # Build stratum tasks
            for quarter, q_grp in agency_docs.groupby("quarter"):
                all_sents = []
                for sent_list in q_grp["sentences"]:
                    all_sents.extend(sent_list)
                if len(all_sents) < args.min_sentences:
                    continue

                key_dict = {
                    "doc_type": doc_type,
                    "agency_id": agency_id,
                    "quarter": quarter,
                    "year": int(str(quarter)[:4]) if quarter else None,
                    "n_documents": len(q_grp),
                }
                inference_tasks.append(
                    (key_dict, all_sents, est_data, args.bootstrap_n)
                )

        logging.info(
            "%s: %d strata to infer (workers=%d)",
            doc_type, len(inference_tasks), args.workers,
        )

        # Run inference (parallelized across strata)
        if args.workers > 1 and len(inference_tasks) > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(_infer_stratum_task, t): t[0]
                    for t in inference_tasks
                }
                for future in tqdm(
                    as_completed(futures), total=len(futures),
                    desc=f"infer {doc_type}",
                ):
                    result = future.result()
                    if result["alpha_estimate"] is not None:
                        all_results.append(result)
        else:
            for task in tqdm(inference_tasks, desc=f"infer {doc_type}"):
                result = _infer_stratum_task(task)
                if result["alpha_estimate"] is not None:
                    all_results.append(result)

        # Per-document scores (MAP with stratum prior, computed AFTER strata)
        if args.document_scores:
            # Build lookup: (doc_type, agency_id, quarter) -> stratum alpha
            stratum_alpha_lookup = {}
            for r in all_results:
                if r["doc_type"] == doc_type and r["alpha_estimate"] is not None:
                    stratum_alpha_lookup[(r["agency_id"], r["quarter"])] = r["alpha_estimate"]

            kappa_doc = getattr(args, "kappa_doc", None)
            if kappa_doc:
                logging.info(
                    "%s: scoring documents with MAP (kappa_doc=%.1f, %d strata as priors)",
                    doc_type, kappa_doc, len(stratum_alpha_lookup),
                )
            else:
                logging.info("%s: scoring individual documents (MLE)...", doc_type)

            for agency_id, (est_data, agency_docs) in agency_doc_map.items():
                w2i, delta_p, delta_q, baseline_p, baseline_q, vocab_set = est_data
                for _, row in agency_docs.iterrows():
                    sents = row["sentences"]
                    filtered = [s for s in sents if set(s) & vocab_set]
                    if len(filtered) < 3:
                        continue
                    log_p, log_q = sentence_log_probs_raw(
                        filtered, w2i, delta_p, delta_q, baseline_p, baseline_q,
                    )
                    # Use MAP with stratum prior if available, else MLE
                    stratum_key = (agency_id, row["quarter"])
                    alpha_prior = stratum_alpha_lookup.get(stratum_key)
                    if kappa_doc and alpha_prior is not None:
                        alpha = _fit_alpha_map(log_p, log_q, alpha_prior, kappa_doc)
                    else:
                        alpha = _fit_alpha(log_p, log_q)
                    all_doc_scores.append({
                        "doc_type": doc_type,
                        "agency_id": agency_id,
                        "document_id": row["document_id"],
                        "quarter": row["quarter"],
                        "year": row["year"],
                        "alpha_score": round(alpha, 6),
                        "n_sentences": len(filtered),
                    })

        logging.info(
            "%s: %d stratum results, %d doc scores",
            doc_type,
            len([r for r in all_results if r["doc_type"] == doc_type]),
            len([r for r in all_doc_scores if r["doc_type"] == doc_type]),
        )

    # Write results
    if all_results:
        results_df = pd.DataFrame(all_results)
        if str(output_path).endswith(".gz"):
            results_df.to_csv(output_path, index=False, compression="gzip")
        else:
            results_df.to_csv(output_path, index=False)
        logging.info("Wrote %s (%d rows)", output_path, len(results_df))
    else:
        logging.warning("No results produced")

    if all_doc_scores and args.document_scores:
        doc_df = pd.DataFrame(all_doc_scores)
        doc_out = Path(args.doc_scores_output)
        doc_out.parent.mkdir(parents=True, exist_ok=True)
        if str(doc_out).endswith(".gz"):
            doc_df.to_csv(doc_out, index=False, compression="gzip")
        else:
            doc_df.to_csv(doc_out, index=False)
        logging.info("Wrote %s (%d rows)", doc_out, len(doc_df))


# ===================================================================
# CLI
# ===================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Per-agency AI usage detection (isolated pipeline)"
    )
    sub = parser.add_subparsers(dest="command")

    # --- estimate ---
    est = sub.add_parser("estimate", help="Build per-agency P/Q distributions")
    est.add_argument("--base-dir", required=True, help="Base data directory")
    est.add_argument("--ai-corpus-dir", required=True, help="AI corpus parquet dir")
    est.add_argument("--output-dir", required=True, help="Output directory")
    est.add_argument(
        "--doc-types", nargs="+",
        default=["rule", "notice", "proposed_rule", "public_submission"],
    )
    est.add_argument("--min-docs", type=int, default=100,
                     help="Min pre-ChatGPT docs per agency (default: 100)")
    est.add_argument("--min-human-count", type=int, default=DEFAULT_MIN_HUMAN_COUNT)
    est.add_argument("--min-ai-count", type=int, default=DEFAULT_MIN_AI_COUNT)
    est.add_argument("--human-cutoff", default=DEFAULT_HUMAN_CUTOFF)
    est.add_argument("--dedup", action="store_true",
                     help="Use minhash dedup for public_submission")
    est.add_argument("--workers", type=int, default=1,
                     help="Number of parallel workers")
    est.set_defaults(func=cmd_estimate)

    # --- infer ---
    inf = sub.add_parser("infer", help="Run per-agency inference")
    inf.add_argument("--base-dir", required=True, help="Base data directory")
    inf.add_argument("--distribution-dir", required=True,
                     help="Directory with per-agency distributions")
    inf.add_argument("--output", required=True, help="Output CSV path")
    inf.add_argument(
        "--doc-types", nargs="+",
        default=["rule", "notice", "proposed_rule", "public_submission"],
    )
    inf.add_argument("--min-sentences", type=int, default=DEFAULT_MIN_SENTENCES)
    inf.add_argument("--bootstrap-n", type=int, default=DEFAULT_BOOTSTRAP_N)
    inf.add_argument("--dedup", action="store_true")
    inf.add_argument("--document-scores", action="store_true",
                     help="Score individual documents")
    inf.add_argument("--kappa-doc", type=float, default=None,
                     help="Concentration for Beta prior on document-level alpha, "
                          "centered on the stratum estimate. E.g. 10 = prior has "
                          "the weight of 10 sentences. If not set, uses plain MLE.")
    inf.add_argument("--doc-scores-output", default="doc_scores.csv.gz",
                     help="Output path for per-document scores")
    inf.add_argument("--workers", type=int, default=1,
                     help="Number of parallel workers")
    inf.set_defaults(func=cmd_infer)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
