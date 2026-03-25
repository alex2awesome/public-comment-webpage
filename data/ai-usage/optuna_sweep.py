#!/usr/bin/env python3
"""Optuna hyperparameter search for distributional AI detection.

In-memory pipeline: loads all data once at startup, then each trial
operates on in-memory dataframes. No subprocess calls for estimate/infer.

Usage:
    python optuna_sweep.py --n-trials 500 --workers 4
    python optuna_sweep.py --analyze --study-name my_study
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import optuna
except ImportError:
    print("pip install optuna")
    sys.exit(1)

# Add ai-usage to path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from ai_detection_utils import (
    DOC_TYPES as ALL_DOC_TYPES,
    DEFAULT_BOOTSTRAP_N,
    DEFAULT_MIN_SENTENCES,
    build_distribution,
    tokenize_text,
    ensure_nltk_punkt,
    iter_input_files,
    load_dedup_representatives,
    is_pre_chatgpt,
    sentence_log_probs_raw,
    MLEEstimator,
    shrink_p,
    optimize_kappa,
    load_agency_word_counts,
    build_agency_est_data,
    load_and_tokenize_file,
    process_human_csv,
    process_ai_texts,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = SCRIPT_DIR.parent / "bulk_downloads"
AI_CORPUS_BASE = SCRIPT_DIR.parent / "ai-usage-generations"
SWEEP_DIR = AI_CORPUS_BASE / "optuna_sweep"
SWEEP_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = SWEEP_DIR / "optuna_study.db"

DOC_TYPES = ["rule", "proposed_rule", "notice", "public_submission"]

MODEL_SOURCES = {
    "llama-3.3-70b": ["llama-v4-generations"],
    "llama-3.1-8b": ["llama-v4-8b-generations"],
    "gpt-5-mini": ["gpt5mini-generations"],
    "mix-weak": ["llama-v4-8b-generations", "gpt5mini-generations"],
    "mix-llama": ["llama-v4-generations", "llama-v4-8b-generations"],
    "mix-gpt-llama70b": ["gpt5mini-generations", "llama-v4-generations"],
    "mix-all": ["llama-v4-generations", "llama-v4-8b-generations", "gpt5mini-generations"],
    "mixture": ["llama-v4-generations", "llama-v4-8b-generations", "gpt5mini-generations"],
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


# ---------------------------------------------------------------------------
# Data loading (done once at startup)
# ---------------------------------------------------------------------------

class DataCache:
    """Holds all loaded data in memory."""

    def __init__(self, doc_types: List[str], workers: int = 4):
        self.doc_types = doc_types
        self.workers = workers

        # AI corpora: {model_source_dir: {doc_type: DataFrame}}
        self.ai_corpora: Dict[str, Dict[str, pd.DataFrame]] = {}

        # Human documents (tokenized): {doc_type: list of record dicts}
        self.human_records: Dict[str, pd.DataFrame] = {}

        # Dedup representatives: {doc_type: set of doc IDs or None}
        self.dedup_reps: Dict[str, Optional[Set[str]]] = {}

    def load_ai_corpora(self):
        """Load all available AI corpus parquets."""
        for subdir_name in set(s for sources in MODEL_SOURCES.values() for s in sources):
            subdir = AI_CORPUS_BASE / subdir_name
            if not subdir.exists():
                continue
            self.ai_corpora[subdir_name] = {}
            for dt in self.doc_types:
                path = subdir / f"ai_corpus_{dt}.parquet"
                if path.exists():
                    df = pd.read_parquet(path)
                    self.ai_corpora[subdir_name][dt] = df
                    logging.info("Loaded AI corpus %s/%s: %d rows", subdir_name, dt, len(df))

    def load_human_records(self):
        """Load and tokenize all human documents."""
        ensure_nltk_punkt()
        for dt in self.doc_types:
            logging.info("Loading human documents for %s...", dt)
            input_files = list(iter_input_files(BASE_DIR, dt))
            dedup_reps = load_dedup_representatives(BASE_DIR, dt)
            self.dedup_reps[dt] = dedup_reps

            records = []
            if self.workers > 1 and len(input_files) > 1:
                from concurrent.futures import ProcessPoolExecutor, as_completed
                with ProcessPoolExecutor(max_workers=self.workers) as pool:
                    futures = {
                        pool.submit(load_and_tokenize_file, f, None, dedup_reps): f
                        for f in input_files
                    }
                    for future in tqdm(
                        as_completed(futures), total=len(futures),
                        desc=f"load {dt}",
                    ):
                        records.extend(future.result())
            else:
                for f in tqdm(input_files, desc=f"load {dt}"):
                    records.extend(load_and_tokenize_file(f, None, dedup_reps))

            if records:
                self.human_records[dt] = pd.DataFrame(records)
                logging.info("%s: %d human documents loaded", dt, len(self.human_records[dt]))
            else:
                logging.warning("%s: no human documents found", dt)

    def get_ai_corpus(self, model: str, doc_type: str,
                      min_doc_words: int = 0, max_doc_tokens: int = 0) -> Optional[pd.DataFrame]:
        """Get a filtered AI corpus for a model + doc_type."""
        subdirs = MODEL_SOURCES.get(model)
        if not subdirs:
            return None

        frames = []
        for subdir in subdirs:
            if subdir in self.ai_corpora and doc_type in self.ai_corpora[subdir]:
                frames.append(self.ai_corpora[subdir][doc_type])
        if not frames:
            return None

        df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

        if min_doc_words > 0:
            word_lens = df["original_text"].str.split().str.len()
            df = df[word_lens >= min_doc_words]

        # Note: max_doc_tokens truncation is expensive (tiktoken) and happens
        # per-trial. We skip it here and let the estimate step work with
        # word-level tokenization which is fast.

        return df


# ---------------------------------------------------------------------------
# In-memory estimate
# ---------------------------------------------------------------------------

def run_estimate_inmemory(
    ai_df: pd.DataFrame,
    human_records_df: pd.DataFrame,
    matched: bool,
) -> Optional[Tuple[pd.DataFrame, Dict, Dict, float]]:
    """Build P/Q distribution in memory.

    Returns: (dist_df, agency_word_counts, agency_ai_counts, optimal_kappa)
    """
    # Get AI doc IDs for matching
    ai_doc_ids = set(ai_df["document_id"].astype(str)) if matched else None

    # Build human word counts
    human_counts = Counter()
    n_human = 0
    agency_accum: Dict[str, Dict] = {}

    for _, row in human_records_df.iterrows():
        doc_id = str(row.get("document_id", ""))
        if ai_doc_ids is not None and doc_id not in ai_doc_ids:
            continue
        agency_id = row["agency_id"]
        for sent in row["sentences"]:
            word_set = set(sent)
            human_counts.update(word_set)
            n_human += 1
            acc = agency_accum.setdefault(agency_id, {"counts": Counter(), "n_sents": 0})
            acc["counts"].update(word_set)
            acc["n_sents"] += 1

    # Build AI word counts
    ai_counts = Counter()
    n_ai = 0
    ai_agency_accum: Dict[str, Dict] = {}

    for _, row in ai_df.iterrows():
        text = row.get("ai_text", "")
        agency_id = row.get("agency_id", "")
        if not isinstance(text, str) or not text:
            continue
        for sent in tokenize_text(text):
            word_set = set(sent)
            ai_counts.update(word_set)
            n_ai += 1
            acc = ai_agency_accum.setdefault(agency_id, {"counts": Counter(), "n_sents": 0})
            acc["counts"].update(word_set)
            acc["n_sents"] += 1

    if n_human == 0 or n_ai == 0:
        return None

    # Build distribution with min_count=1 (filtering happens later)
    dist_df = build_distribution(human_counts, n_human, ai_counts, n_ai,
                                 min_human_count=1, min_ai_count=1)
    if dist_df is None:
        return None

    # Compute optimal kappa
    common_vocab = set(dist_df["word"].values)
    agency_word_counts = {
        aid: (acc["counts"], acc["n_sents"])
        for aid, acc in agency_accum.items()
    }
    pool_freq = {w: human_counts[w] / n_human for w in common_vocab}

    try:
        optimal_kappa = optimize_kappa(pool_freq, agency_word_counts, common_vocab)
    except Exception:
        optimal_kappa = 500.0

    # Compute optimal kappa_q for AI side
    ai_agency_word_counts = {
        aid: (acc["counts"], acc["n_sents"])
        for aid, acc in ai_agency_accum.items()
    }
    pool_freq_q = {w: ai_counts[w] / n_ai for w in common_vocab}
    try:
        optimal_kappa_q = optimize_kappa(pool_freq_q, ai_agency_word_counts, common_vocab)
    except Exception:
        optimal_kappa_q = optimal_kappa

    return dist_df, agency_word_counts, ai_agency_word_counts, optimal_kappa, optimal_kappa_q


# ---------------------------------------------------------------------------
# In-memory infer
# ---------------------------------------------------------------------------

def run_infer_inmemory(
    dist_df: pd.DataFrame,
    human_records_df: pd.DataFrame,
    agency_word_counts: Dict,
    hierarchical: bool,
    kappa: Optional[float],
    stratify_by: str,
    min_sentences: int,
    bootstrap_n: int,
    subsample_frac: Optional[float] = None,
    subsample_seed: Optional[int] = None,
    doc_type: str = "",
    agency_ai_word_counts: Optional[Dict] = None,
    kappa_q: Optional[float] = None,
) -> pd.DataFrame:
    """Run MLE inference in memory. Shrinks both P and Q per-agency when hierarchical."""
    from scipy.optimize import minimize

    df = human_records_df.copy()

    # Subsample
    if subsample_frac and 0 < subsample_frac < 1:
        df = df.sample(frac=subsample_frac, random_state=subsample_seed).reset_index(drop=True)

    # Build estimator data
    vocab_list = list(dist_df["word"].values)
    w2i = {w: i for i, w in enumerate(vocab_list)}
    vocab_set = set(vocab_list)

    if hierarchical and agency_word_counts:
        # Will build per-agency est_data below
        mu_p = np.exp(dist_df["logP"].values.astype(float))
        mu_q = np.exp(dist_df["logQ"].values.astype(float))
        logQ_pool = dist_df["logQ"].values.astype(float)
        log1mQ_pool = dist_df["log1mQ"].values.astype(float)
        use_kappa = kappa if kappa else 500.0
        use_kappa_q = kappa_q if kappa_q else use_kappa
    else:
        logP = dist_df["logP"].values.astype(float)
        logQ = dist_df["logQ"].values.astype(float)
        log1mP = dist_df["log1mP"].values.astype(float)
        log1mQ = dist_df["log1mQ"].values.astype(float)
        delta_p = logP - log1mP
        delta_q = logQ - log1mQ
        baseline_p = float(log1mP.sum())
        baseline_q = float(log1mQ.sum())

    # Determine groupby column
    if stratify_by == "quarter":
        group_col = "quarter"
    elif stratify_by == "half":
        group_col = "half"
    elif stratify_by == "year":
        group_col = "year"
    else:
        group_col = "year"

    results = []
    groups = df.groupby(["agency_id", group_col] if "agency" in stratify_by or hierarchical else [group_col])

    for key, grp in groups:
        if isinstance(key, tuple):
            agency_id = key[0] if len(key) > 1 else None
            time_val = key[-1]
        else:
            agency_id = None
            time_val = key

        all_sents = []
        for sent_list in grp["sentences"]:
            all_sents.extend(sent_list)
        if len(all_sents) < min_sentences:
            continue

        # Filter sentences to those with vocab overlap
        filtered = [s for s in all_sents if set(s) & vocab_set]
        if len(filtered) < min_sentences:
            continue

        # Build est_data
        if hierarchical and agency_word_counts and agency_id:
            # Shrink P toward pool
            if agency_id in agency_word_counts:
                word_counts_a, n_a = agency_word_counts[agency_id]
                n_aw = np.array([word_counts_a.get(w, 0) for w in vocab_list], dtype=float)
            else:
                n_aw = np.zeros(len(vocab_list))
                n_a = 0
            p_shrunk = shrink_p(mu_p, n_aw, n_a, use_kappa)
            _logP = np.log(p_shrunk)
            _log1mP = np.log(1 - p_shrunk)
            _delta_p = _logP - _log1mP
            _baseline_p = float(_log1mP.sum())

            # Shrink Q toward pool
            if agency_ai_word_counts and agency_id in agency_ai_word_counts:
                ai_counts_a, n_ai_a = agency_ai_word_counts[agency_id]
                n_ai_aw = np.array([ai_counts_a.get(w, 0) for w in vocab_list], dtype=float)
                q_shrunk = shrink_p(mu_q, n_ai_aw, n_ai_a, use_kappa_q)
                _logQ = np.log(q_shrunk)
                _log1mQ = np.log(1 - q_shrunk)
            else:
                _logQ = logQ_pool
                _log1mQ = log1mQ_pool
            _delta_q = _logQ - _log1mQ
            _baseline_q = float(_log1mQ.sum())
        else:
            _delta_p = delta_p
            _delta_q = delta_q
            _baseline_p = baseline_p
            _baseline_q = baseline_q

        log_p, log_q = sentence_log_probs_raw(
            filtered, w2i, _delta_p, _delta_q, _baseline_p, _baseline_q,
        )

        # Bootstrap MLE
        rng = np.random.RandomState(hash(str(key)) % (2**31))
        n = len(log_p)
        alphas = []
        for _ in range(bootstrap_n):
            idx = rng.choice(n, size=n, replace=True)
            lp, lq = log_p[idx], log_q[idx]
            from scipy.optimize import minimize as _minimize
            res = _minimize(
                lambda a, lp=lp, lq=lq: -np.mean(np.log(np.maximum(
                    (1 - a[0]) + a[0] * np.exp(lq - lp), 1e-300))),
                x0=[0.5], method="L-BFGS-B", bounds=[(0, 1)],
            )
            if res.success:
                alphas.append(float(res.x[0]))

        if not alphas:
            continue

        ci_lo = float(np.percentile(alphas, 2.5))
        ci_hi = float(np.percentile(alphas, 97.5))
        alpha_point = float(np.mean([ci_lo, ci_hi]))

        row = {
            "doc_type": doc_type,
            "alpha_estimate": round(alpha_point, 6),
            "ci_lower": round(ci_lo, 6),
            "ci_upper": round(ci_hi, 6),
            "n_documents": len(grp),
            "n_sentences": len(filtered),
        }
        if agency_id:
            row["agency_id"] = agency_id
        row[group_col] = time_val
        if group_col in ("quarter", "half"):
            row["year"] = int(str(time_val)[:4])
        elif group_col == "year":
            row["year"] = int(time_val)

        results.append(row)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Post-process distribution (in memory)
# ---------------------------------------------------------------------------

def postprocess_dist(
    dist_df: pd.DataFrame,
    min_human_count: int,
    min_ai_count: int,
    max_vocab: Optional[int],
    min_word_len: int,
    min_lor: float,
) -> pd.DataFrame:
    """Filter distribution DataFrame by vocab thresholds."""
    df = dist_df.copy()
    if min_human_count > 1:
        df = df[df["human_count"] >= min_human_count]
    if min_ai_count > 1:
        df = df[df["ai_count"] >= min_ai_count]
    if min_word_len > 0:
        df = df[df["word"].str.len() >= min_word_len]
    if min_lor > 0:
        df = df[df["log_odds_ratio"].abs() >= min_lor]
    if max_vocab and len(df) > max_vocab:
        df = df.nlargest(max_vocab, "ai_count")
        df = df.sort_values("log_odds_ratio").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate(df: pd.DataFrame, doc_types: List[str]) -> float:
    """Compute objective score from infer results."""
    if df.empty:
        return -1.0

    # Parse time column
    if "quarter" in df.columns:
        df = df.copy()
        df["_dt"] = pd.PeriodIndex(df["quarter"], freq="Q").to_timestamp()
    elif "half" in df.columns:
        df = df.copy()
        df["_dt"] = df["half"].apply(
            lambda h: pd.Timestamp(year=int(h[:4]), month=1 if h[-1] == "1" else 7, day=1)
        )
    elif "year" in df.columns:
        df = df.copy()
        df["_dt"] = pd.to_datetime(df["year"].astype(int), format="%Y")
    else:
        return -1.0

    pre = df[df["_dt"] < "2022-12-01"]
    post = df[df["_dt"] >= "2022-12-01"]

    scores = []
    for dt in doc_types:
        pre_dt = pre[pre["doc_type"] == dt]
        post_dt = post[post["doc_type"] == dt]
        if len(pre_dt) == 0 or len(post_dt) == 0:
            continue
        if pre_dt["n_sentences"].sum() == 0 or post_dt["n_sentences"].sum() == 0:
            continue
        pre_alpha = float(np.average(pre_dt["alpha_estimate"], weights=pre_dt["n_sentences"]))
        post_alpha = float(np.average(post_dt["alpha_estimate"], weights=post_dt["n_sentences"]))
        pre_std = float(pre_dt["alpha_estimate"].std())
        score = post_alpha - 3 * pre_alpha - pre_std
        scores.append(score)

    return float(np.mean(scores)) if scores else -1.0


# ---------------------------------------------------------------------------
# Estimate cache (in-memory)
# ---------------------------------------------------------------------------

class EstimateCache:
    """Caches estimate results keyed by (model, min_doc_words, matched)."""

    def __init__(self):
        # key -> (dist_df, agency_word_counts, agency_ai_counts, optimal_kappa)
        self._cache: Dict[str, Tuple] = {}

    def key(self, model: str, min_doc_words: int, matched: bool, doc_type: str) -> str:
        return f"{model}__{min_doc_words}__{matched}__{doc_type}"

    def get(self, model, min_doc_words, matched, doc_type):
        return self._cache.get(self.key(model, min_doc_words, matched, doc_type))

    def put(self, model, min_doc_words, matched, doc_type, value):
        self._cache[self.key(model, min_doc_words, matched, doc_type)] = value
        logging.info("Estimate cached: %s", self.key(model, min_doc_words, matched, doc_type))


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def create_objective(
    data_cache: DataCache,
    estimate_cache: EstimateCache,
    available_models: List[str],
    doc_types: List[str],
    hierarchical_only: bool = False,
    subsample_frac: Optional[float] = None,
):
    def objective(trial: optuna.Trial) -> float:
        # --- Sample hyperparameters ---
        model = trial.suggest_categorical("model", available_models)
        min_doc_words = trial.suggest_categorical("min_doc_words", [0, 100, 250, 500, 750])
        max_doc_tokens = trial.suggest_categorical(
            "max_doc_tokens", [0, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
        )
        matched = trial.suggest_categorical("matched", [True, False])

        min_human_count = trial.suggest_categorical(
            "min_human_count", [3, 5, 8, 10, 15, 20, 30, 50, 75, 100]
        )
        min_ai_count = trial.suggest_categorical(
            "min_ai_count", [3, 5, 8, 10, 15, 20, 30, 50]
        )
        max_vocab_choice = trial.suggest_categorical(
            "max_vocab", ["none", "500", "1000", "2000", "5000", "10000"]
        )
        max_vocab = None if max_vocab_choice == "none" else int(max_vocab_choice)

        min_word_len = trial.suggest_categorical("min_word_len", [0, 3, 4, 5])
        min_lor = trial.suggest_categorical("min_lor", [0.0, 0.1, 0.3, 0.5])

        if hierarchical_only:
            hierarchical = True
        else:
            hierarchical = trial.suggest_categorical("hierarchical", [True, False])

        kappa_choice = trial.suggest_categorical(
            "kappa", ["none", "50", "100", "200", "300", "500", "700", "1000", "2000", "5000", "10000"]
        )
        kappa = None if (kappa_choice == "none" or not hierarchical) else int(kappa_choice)

        stratify_by = trial.suggest_categorical("stratify_by", ["half", "year"])
        min_sentences = trial.suggest_categorical("min_sentences", [10, 25, 50, 100, 200])

        all_infer_results = []

        for dt in doc_types:
            # --- Step 1: Get or build estimate ---
            cached = estimate_cache.get(model, min_doc_words, matched, dt)
            if cached is None:
                ai_df = data_cache.get_ai_corpus(model, dt, min_doc_words=min_doc_words)
                if ai_df is None or ai_df.empty:
                    continue
                human_df = data_cache.human_records.get(dt)
                if human_df is None or human_df.empty:
                    continue

                logging.info(
                    "RUNNING ESTIMATE: model=%s minw=%d matched=%s dt=%s (%d AI docs)",
                    model, min_doc_words, matched, dt, len(ai_df),
                )
                result = run_estimate_inmemory(ai_df, human_df, matched)
                if result is None:
                    continue
                estimate_cache.put(model, min_doc_words, matched, dt, result)
                cached = result

            dist_df, agency_word_counts, agency_ai_counts, optimal_kappa, optimal_kappa_q = cached

            # --- Step 2: Post-process distribution ---
            filtered_dist = postprocess_dist(
                dist_df, min_human_count, min_ai_count, max_vocab, min_word_len, min_lor,
            )
            if filtered_dist.empty or len(filtered_dist) < 50:
                continue

            # --- Step 3: Infer ---
            human_df = data_cache.human_records.get(dt)
            if human_df is None:
                continue

            use_kappa = kappa if kappa else optimal_kappa
            infer_results = run_infer_inmemory(
                dist_df=filtered_dist,
                human_records_df=human_df,
                agency_word_counts=agency_word_counts if hierarchical else {},
                hierarchical=hierarchical,
                kappa=use_kappa,
                stratify_by=stratify_by,
                min_sentences=min_sentences,
                bootstrap_n=50,
                subsample_frac=subsample_frac,
                subsample_seed=trial.number,
                doc_type=dt,
                agency_ai_word_counts=agency_ai_counts if hierarchical else None,
                kappa_q=optimal_kappa_q if hierarchical else None,
            )
            if not infer_results.empty:
                all_infer_results.append(infer_results)

        if not all_infer_results:
            return -1.0

        results_df = pd.concat(all_infer_results, ignore_index=True)
        score = evaluate(results_df, doc_types)

        logging.info(
            "Trial %d: score=%.6f | model=%s minw=%d maxt=%d matched=%s "
            "h=%d a=%d mv=%s wl=%d lor=%.1f hier=%s k=%s strat=%s minsent=%d",
            trial.number, score, model, min_doc_words, max_doc_tokens, matched,
            min_human_count, min_ai_count, max_vocab_choice,
            min_word_len, min_lor, hierarchical, kappa, stratify_by,
            min_sentences,
        )

        return score

    return objective


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_study(study: optuna.Study):
    print("\n" + "=" * 80)
    print("OPTUNA STUDY ANALYSIS")
    print("=" * 80)

    completed = [t for t in study.trials if t.state.name == "COMPLETE" and t.value is not None and t.value != -1.0]
    print(f"\nTotal trials: {len(study.trials)} ({len(completed)} with valid scores)")

    if not completed:
        print("No valid trials.")
        return

    print(f"Best score: {study.best_value:.6f}")
    print(f"Best params:")
    for k, v in sorted(study.best_params.items()):
        print(f"  {k}: {v}")

    print("\n--- TOP 10 TRIALS ---")
    trials = sorted(completed, key=lambda t: t.value, reverse=True)
    for t in trials[:10]:
        p = t.params
        print(f"  #{t.number:4d}  score={t.value:.6f}  "
              f"model={p.get('model','?'):15s}  "
              f"k={p.get('kappa','?'):>5s}  "
              f"strat={p.get('stratify_by','?'):8s}  "
              f"matched={p.get('matched','?')}")

    print("\n--- HYPERPARAMETER IMPORTANCE (fANOVA) ---")
    try:
        importances = optuna.importance.get_param_importances(study)
        for param, imp in importances.items():
            bar = "#" * int(imp * 50)
            print(f"  {param:20s}  {imp:.4f}  {bar}")
    except Exception as e:
        print(f"  Could not compute importance: {e}")

    results_df = study.trials_dataframe()
    safe_name = study.study_name.replace("/", "_").replace(" ", "_")
    out_path = SWEEP_DIR / f"optuna_results_{safe_name}.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for distributional AI detection (in-memory)"
    )
    parser.add_argument("--n-trials", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--study-name", default="ai_detection_sweep")
    parser.add_argument("--hierarchical-only", action="store_true")
    parser.add_argument("--doc-type", choices=DOC_TYPES, default=None)
    parser.add_argument(
        "--subsample-frac", type=float, default=None,
        help="Subsample fraction for infer (e.g. 0.05). Default: 0.05 for public_submission, None for others.",
    )
    args = parser.parse_args()

    doc_types = [args.doc_type] if args.doc_type else DOC_TYPES

    # Detect available models
    has_70b = (AI_CORPUS_BASE / "llama-v4-generations").exists()
    has_8b = (AI_CORPUS_BASE / "llama-v4-8b-generations").exists()
    has_gpt = (AI_CORPUS_BASE / "gpt5mini-generations").exists()

    available_models = []
    if has_70b: available_models.append("llama-3.3-70b")
    if has_8b: available_models.append("llama-3.1-8b")
    if has_gpt: available_models.append("gpt-5-mini")
    if has_8b and has_gpt: available_models.append("mix-weak")
    if has_70b and has_8b: available_models.append("mix-llama")
    if has_gpt and has_70b: available_models.append("mix-gpt-llama70b")
    if has_70b and has_8b and has_gpt: available_models.append("mix-all")
    if len(available_models) >= 2: available_models.append("mixture")  # backward compat

    if not available_models:
        print("No AI corpus directories found!")
        sys.exit(1)

    logging.info("Available models: %s", available_models)
    logging.info("Doc types: %s", doc_types)

    storage = f"sqlite:///{DB_PATH}"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )

    if args.analyze:
        analyze_study(study)
        return

    # --- Load all data into memory ---
    logging.info("Loading all data into memory...")
    data_cache = DataCache(doc_types, workers=args.workers)
    data_cache.load_ai_corpora()
    data_cache.load_human_records()
    logging.info("Data loading complete.")

    estimate_cache = EstimateCache()

    # Determine subsample
    subsample_frac = args.subsample_frac
    if subsample_frac is None and "public_submission" in doc_types and len(doc_types) == 1:
        subsample_frac = 0.05

    objective = create_objective(
        data_cache, estimate_cache, available_models, doc_types,
        hierarchical_only=args.hierarchical_only,
        subsample_frac=subsample_frac,
    )
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    analyze_study(study)


if __name__ == "__main__":
    main()
