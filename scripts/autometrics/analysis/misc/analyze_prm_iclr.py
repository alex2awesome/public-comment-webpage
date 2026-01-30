#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, sys, re, unicodedata, traceback
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------
# Cluster definitions
# ---------------------------

MAIN_CLUSTERS = [
    "INTRODUCTION",
    "RELATED WORK",
    "BACKGROUND/PRELIMINARIES",
    "METHODS",
    "EXPERIMENTAL SETUP",
    "EXPERIMENTS/RESULTS",
    "DISCUSSION",
    "CONCLUSION",
]
AUX_CLUSTERS = ["ABSTRACT", "ACKNOWLEDGMENTS", "APPENDIX", "MIXED/COMBINED", "OTHER"]
ALL_CLUSTERS_ORDER = MAIN_CLUSTERS + AUX_CLUSTERS

KW_INTRO = ["INTRO", "INTRODUCTION", "OVERVIEW", "MOTIVATION", "MOTIVATIONS"]
KW_RELATED = ["RELATED WORK", "PREVIOUS RELATED WORK", "COMPARISON WITH RELATED WORK"]
KW_BACKGROUND = [
    "BACKGROUND", "PRELIMINARIES", "PRELIMINARY", "THEORY", "THEORETICAL",
    "TASK DESCRIPTION", "PROBLEM STATEMENT", "PROBLEM FORMULATION", "PROBLEM FORMULATION AND ASSUMPTIONS",
    "GENERAL FRAMEWORK"
]
KW_METHODS = [
    "METHOD", "METHODS", "METHODOLOGY", "APPROACH", "PROPOSED ARCHITECTURE", "PROPOSED METHOD",
    "ARCHITECTURE", "ALGORITHM OVERVIEW", "ALGORITHM", "MODEL", "MODELS", "FRAMEWORK",
    "TRAINING"
]
KW_SETUP = [
    "EXPERIMENTAL SETUP", "SETUP", "IMPLEMENTATION DETAILS", "IMPLEMENTATION", "EXPERIMENTAL DETAILS",
    "DATA ", "DATASET", "DATASETS", "DATA SETS", "HYPERPARAMETERS", "HYPER-PARAMETERS", "PARAMETERS",
    "EXPERIMENTAL SECTION", "NETWORK ARCHITECTURES/TRAINING"
]
KW_EXPER = [
    "EXPERIMENT", "EXPERIMENTS", "EVALUATION", "RESULTS", "RESULTS AND DISCUSSION",
    "EXPERIMENTAL EVALUATION", "EXPERIMENTAL RESULTS", "EXPERIMENTS AND RESULTS",
    "CASE STUDIES", "COMPARATIVE CASE STUDIES", "QUALITATIVE", "QUANTITATIVE"
]
KW_DISCUSSION = ["DISCUSSION", "DISCUSSIONS", "ANALYSIS", "LIMITATIONS", "OPEN QUESTIONS", "SUMMARY"]
KW_CONCLUSION = [
    "CONCLUSION", "CONCLUSIONS", "CONCLUSION & FUTURE WORK", "CONCLUSION AND FUTURE WORK",
    "CONCLUSIONS AND FUTURE WORK", "DISCUSSION AND CONCLUSION", "DISCUSSION AND CONCLUSIONS",
    "FUTURE WORK", "FUTURE DIRECTIONS"
]
KW_ABSTRACT = ["ABSTRACT"]
KW_ACK = ["ACKNOWLEDGMENTS", "ACKNOWLEDGEMENTS"]
COMBINERS = [" AND ", "/", "&"]

# ---------------------------
# Helpers
# ---------------------------

def normalize_title(s: str) -> str:
    if not isinstance(s, str): 
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("“", "\"").replace("”", "\"").replace("’", "'").replace("–", "-").replace("—", "-")
    s = re.sub(r'^[\s"\'\.\-:]+|[\s"\'\.\-:]+$', "", s.strip())
    s = re.sub(r"\s+", " ", s)
    return s.upper()

def safe_float_series(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")

def classify_basic(title_norm: str) -> Optional[str]:
    t = title_norm
    if any(t == k or t.startswith(k) for k in KW_ABSTRACT): return "ABSTRACT"
    if any(t == k or t.startswith(k) for k in KW_ACK): return "ACKNOWLEDGMENTS"
    if "APPENDIX" in t or "SUPPLEMENTARY" in t: return "APPENDIX"
    if any(c in t for c in COMBINERS):
        if "CONCLUSION" in t: return "CONCLUSION"
        if "RESULT" in t and "DISCUSSION" in t: return "EXPERIMENTS/RESULTS"
        if "INTRO" in t and "RELATED" in t: return "MIXED/COMBINED"
        return "MIXED/COMBINED"
    if any(k == t or t.startswith(k) for k in KW_INTRO): return "INTRODUCTION"
    if any(k == t or t.startswith(k) for k in KW_RELATED): return "RELATED WORK"
    if any(k == t or t.startswith(k) for k in KW_BACKGROUND): return "BACKGROUND/PRELIMINARIES"
    if any(k == t or t.startswith(k) for k in KW_CONCLUSION): return "CONCLUSION"
    if any(k == t or t.startswith(k) for k in KW_SETUP): return "EXPERIMENTAL SETUP"
    if any(k == t or t.startswith(k) for k in KW_EXPER): return "EXPERIMENTS/RESULTS"
    if any(k == t or t.startswith(k) for k in KW_DISCUSSION): return "DISCUSSION"
    if any(k == t or t.startswith(k) for k in KW_METHODS): return "METHODS"
    return None

def classify_with_context(title_norm: str, doc_titles_norm: List[str], doc_positions: Dict[str,int]) -> str:
    basic = classify_basic(title_norm)
    if basic:
        return basic
    pos_this = doc_positions.get(title_norm)
    intro_pos = min((doc_positions[t] for t in doc_titles_norm if classify_basic(t) == "INTRODUCTION"), default=None)
    exp_pos = min((doc_positions[t] for t in doc_titles_norm if classify_basic(t) == "EXPERIMENTS/RESULTS"), default=None)
    if any(k in title_norm for k in ["FRAMEWORK", "ALGORITHM", "TRAINING", "ARCHITECTURE", "MODEL"]):
        return "METHODS"
    if "THEORY" in title_norm: 
        return "BACKGROUND/PRELIMINARIES"
    if any(k in title_norm for k in ["IMPLEMENTATION", "HYPERPARAMETER", "DETAILS", "DATASET"]):
        if intro_pos is not None and exp_pos is not None and pos_this is not None and intro_pos < pos_this < exp_pos:
            return "EXPERIMENTAL SETUP"
    if any(k in title_norm for k in ["EXPERIMENT", "RESULT", "EVALUATION"]):
        return "EXPERIMENTS/RESULTS"
    if any(k in title_norm for k in ["DISCUSSION", "ANALYSIS", "LIMITATIONS"]):
        return "DISCUSSION"
    return "OTHER"

def sanitize_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))

# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-prefix", default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    in_path = os.path.abspath(args.input)
    print(f"[INFO] CWD: {os.getcwd()}")
    print(f"[INFO] Input CSV: {in_path}")
    if not os.path.exists(in_path):
        print(f"[ERROR] Input file does not exist: {in_path}", file=sys.stderr)
        return 2

    try:
        df = pd.read_csv(in_path)
    except Exception as e:
        print("[ERROR] Failed to read CSV:", e, file=sys.stderr)
        return 2

    if df.empty:
        print("[ERROR] Input CSV has 0 rows", file=sys.stderr)
        return 2

    print(f"[INFO] Loaded {len(df):,} rows")

    df["_title_norm"] = df["section_title"].map(normalize_title)

    docid_col = "doc_id" if "doc_id" in df.columns else "doc_index"
    doc_to_titles, doc_to_pos = {}, {}
    for dk, sub in df.groupby(docid_col):
        titles = list(sub["_title_norm"].values)
        posmap = {}
        for i, t in enumerate(titles):
            if t not in posmap:
                posmap[t] = i
        doc_to_titles[dk] = titles
        doc_to_pos[dk] = posmap

    clusters = []
    iterator = df.itertuples(index=False)
    if tqdm is not None:
        iterator = tqdm(iterator, total=len(df), desc="Clustering", unit="row")

    for row in iterator:
        tnorm = normalize_title(getattr(row, "section_title"))
        docid = getattr(row, docid_col)
        cl = classify_basic(tnorm)
        if cl is None:
            cl = classify_with_context(tnorm, doc_to_titles.get(docid, []), doc_to_pos.get(docid, {}))
        clusters.append(cl)
        if args.verbose:
            print(f"[MAP] {docid} :: '{getattr(row, 'section_title')}' → {cl}")

    df["cluster"] = clusters

    counts = df.groupby("cluster").size().reset_index(name="count").sort_values("count", ascending=False)
    print("\n[INFO] Cluster counts:")
    for _, r in counts.iterrows():
        print(f"  - {r['cluster']:<24} {int(r['count']):>6}")

    for col in ["PRM_min", "PRM_mean", "PRM_max"]:
        if col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = safe_float_series(df[col])

    agg = (
        df.groupby("cluster", dropna=False)[["PRM_min", "PRM_mean", "PRM_max"]]
        .mean(numeric_only=True)
        .reset_index()
    )
    agg["cluster"] = pd.Categorical(agg["cluster"], categories=ALL_CLUSTERS_ORDER, ordered=True)
    agg = agg.sort_values("cluster")

    out_dir = os.path.dirname(in_path)
    base = os.path.splitext(os.path.basename(args.out_prefix or in_path))[0]
    rows_csv = os.path.join(out_dir, f"{base}.clustered_rows.csv")
    agg_csv = os.path.join(out_dir, f"{base}.cluster_aggregates.csv")
    df.to_csv(rows_csv, index=False)
    agg.to_csv(agg_csv, index=False)
    print(f"\n[OK] Wrote clustered rows: {rows_csv}")
    print(f"[OK] Wrote aggregate CSV: {agg_csv}")

    if not args.no_plots:
        try:
            for metric in ["PRM_min", "PRM_mean", "PRM_max"]:
                if not agg[metric].dropna().empty:
                    fig = plt.figure(figsize=(10, 5))
                    plt.bar(agg["cluster"].astype(str), agg[metric].values)
                    plt.xticks(rotation=30, ha="right")
                    plt.ylabel(metric)
                    plt.title(f"{metric} by cluster")
                    plt.tight_layout()
                    out_path = os.path.join(out_dir, f"{base}.bar_{metric}.png")
                    fig.savefig(out_path, dpi=150)
                    plt.close(fig)
                    print(f"[OK] Saved bar: {out_path}")
                        # Histograms for ALL clusters (main + aux), incl. APPENDIX
            present_clusters = [c for c in ALL_CLUSTERS_ORDER if c in set(df["cluster"].unique())]
            for cl in present_clusters:
                sdf = df[df["cluster"] == cl]
                if sdf.empty:
                    continue
                for m in ["PRM_min", "PRM_mean", "PRM_max"]:
                    series = sdf[m].dropna().astype(float)
                    if series.empty:
                        continue
                    fig = plt.figure(figsize=(7, 4))
                    plt.hist(series.values, bins=20)
                    plt.xlabel(m)
                    plt.ylabel("count")
                    plt.title(f"{cl} — {m} histogram")
                    plt.tight_layout()
                    out_path = os.path.join(out_dir, f"{base}.hist_{sanitize_filename(cl)}_{m}.png")
                    fig.savefig(out_path, dpi=150)
                    plt.close(fig)
                    print(f"[OK] Saved histogram: {out_path}")
        except Exception:
            print("[WARN] Plotting failed")
            traceback.print_exc()

    print("[DONE] Finished successfully.")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")
        sys.exit(130)
    except Exception:
        print("[FATAL] Unexpected error:")
        traceback.print_exc()
        sys.exit(1)

# Example Usage:
# python analysis/misc/analyze_prm_iclr.py --input analysis/misc/prm_iclr_results.csv --out-prefix analysis/misc/prm_iclr_results/out/results