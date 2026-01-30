import os
import argparse
import re
import sys
import json
import math
import glob
import hashlib
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


CSV_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "outputs",
        "robustness",
        "csvs",
    )
)
CSV_ROOT = os.path.normpath(CSV_ROOT)

BEST_METRICS_CSV = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "results",
        "best_metrics.csv",
    )
)
BEST_METRICS_CSV = os.path.normpath(BEST_METRICS_CSV)

OUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "outputs")
)

# Global plotting/theme settings
sns.set_theme(style="darkgrid")
plt.rcParams.update({
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

# Reproducible RNG for baselines
_RNG = np.random.default_rng(0)

def _sample_trunc_standard_normal(n: int) -> np.ndarray:
    """Sample from N(0,1) with rejection until all samples fall within [-3, 3]."""
    if n <= 0:
        return np.array([], dtype=float)
    vals = _RNG.standard_normal(n)
    mask = (vals < -3.0) | (vals > 3.0)
    while np.any(mask):
        vals[mask] = _RNG.standard_normal(int(np.sum(mask)))
        mask = (vals < -3.0) | (vals > 3.0)
    return vals

def _simulate_baseline_scores(n: int, which: str) -> np.ndarray:
    """Simulate baseline scores using truncated Normal(0,1) resampled to [-3,3],
    then rescaled linearly to [0,1] via (x + 3) / 6.
    which: 'binary' or 'stability'
    Returns a vector of length n with scores in [0,1].
    """
    if n <= 0:
        return np.array([], dtype=float)
    a = _sample_trunc_standard_normal(n)
    b = _sample_trunc_standard_normal(n)
    a_norm = (a + 3.0) / 6.0
    b_norm = (b + 3.0) / 6.0
    if which == 'binary':
        return (b_norm < a_norm).astype(float)
    # stability
    return 1.0 - np.abs(a_norm - b_norm)


def list_dataset_dirs(root_dir: str) -> List[str]:
    if not os.path.isdir(root_dir):
        return []
    entries = []
    for name in sorted(os.listdir(root_dir)):
        full = os.path.join(root_dir, name)
        if os.path.isdir(full):
            entries.append(full)
    return entries


def read_csv_if_exists(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def infer_dataset_measure_from_folder(folder_name: str) -> Tuple[str, str]:
    """Given a folder like 'simpeval_score', return ('SimpEval','score')-like tokens for matching.
    We'll split on the last underscore. If no underscore, dataset=folder, measure=''.
    """
    base = os.path.basename(folder_name)
    if "_" not in base:
        return base, ""
    parts = base.rsplit("_", 1)
    return parts[0], parts[1]


def normalize_series_minmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    s_min = np.nanmin(values)
    s_max = np.nanmax(values)
    if not np.isfinite(s_min) or not np.isfinite(s_max):
        return np.zeros_like(values, dtype=float)
    denom = s_max - s_min
    if denom <= 0:
        return np.zeros_like(values, dtype=float)
    return (values - s_min) / denom


def select_metric_columns(df: pd.DataFrame, dataset_dir: str, best_df: Optional[pd.DataFrame]) -> Dict[str, str]:
    """Return mapping logical_name -> column_name for the 5 metrics:
    - LLMJudge
    - DNAEval
    - Autometrics (Autometrics_Regression_*)
    - MetaMetrics (metametrics_score)
    - BEST_METRIC (from best_metrics.csv)
    Only include those that can be found in df.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    found: Dict[str, str] = {}

    # LLMJudge: pick the first column that contains 'llmjudge'
    llm_cols = [c for c in df.columns if re.search(r"llm\s*judge|llmjudge", c, flags=re.I)]
    if llm_cols:
        found["LLMJudge"] = llm_cols[0]

    # DNAEval
    dna_cols = [c for c in df.columns if re.search(r"dna\s*eval|dnaeval", c, flags=re.I)]
    if dna_cols:
        found["DNAEval"] = dna_cols[0]

    # Autometrics
    auto_cols = [c for c in df.columns if re.search(r"^autometrics_\s*regression_", c, flags=re.I)]
    if auto_cols:
        found["Autometrics"] = auto_cols[0]

    # MetaMetrics (constant name metametrics_score)
    for cname in df.columns:
        if cname.lower() == "metametrics_score":
            found["MetaMetrics"] = cname
            break

    # BEST_METRIC from best_metrics.csv
    dataset_token, measure_token = infer_dataset_measure_from_folder(dataset_dir)
    best_col = None
    if best_df is not None and {"dataset", "measure", "metric"}.issubset(set(best_df.columns)):
        # try strict match by lowercased tokens (strip non-alnum)
        def _norm(s: str) -> str:
            return re.sub(r"[^a-z0-9]+", "", str(s).lower())

        folder_dataset_n = _norm(dataset_token)
        folder_measure_n = _norm(measure_token)

        candidates = []
        for _, row in best_df.iterrows():
            d = _norm(row.get("dataset", ""))
            m = _norm(row.get("measure", ""))
            if d == folder_dataset_n and m == folder_measure_n:
                candidates.append(row)

        if not candidates:
            # fallback: any row whose normalized dataset+measure equals folder base
            base_n = _norm(os.path.basename(dataset_dir))
            for _, row in best_df.iterrows():
                dm = _norm(str(row.get("dataset", "")) + "_" + str(row.get("measure", "")))
                if dm == base_n:
                    candidates.append(row)

        if candidates:
            # If multiple, pick the first
            metric_name = str(candidates[0].get("metric", "")).strip()
            # Try to find exact (case-insensitive) column match
            if metric_name:
                exact = None
                for c in df.columns:
                    if c.lower() == metric_name.lower():
                        exact = c
                        break
                if exact is None:
                    # maybe metric_class is the column? try that if present
                    metric_class = str(candidates[0].get("metric_class", "")).strip()
                    for c in df.columns:
                        if c.lower() == metric_class.lower():
                            exact = c
                            break
                if exact is None:
                    # partial match, contains metric_name
                    for c in df.columns:
                        if metric_name.lower() in c.lower():
                            exact = c
                            break
                if exact is not None:
                    best_col = exact

    if best_col is not None:
        found["BEST_METRIC"] = best_col

    return found


def compute_per_instance_scores(
    df_all: pd.DataFrame,
    metric_col: str,
    uid_col: str = "uid",
    group_col: str = "group",
    strategy_col: str = "strategy",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (stability_df, sensitivity_df)
    - stability: for each uid and same_obvious row, score = 1 - |orig_norm - same_norm|
    - sensitivity: for each uid and worse_obvious row, score = orig_norm - worse_norm
    All normalization is min-max over all rows for that metric within df_all.
    """
    # Ensure required cols exist
    for c in [uid_col, group_col, metric_col]:
        if c not in df_all.columns:
            return pd.DataFrame(), pd.DataFrame()

    # Strategy may be missing; create if needed
    if strategy_col not in df_all.columns:
        df_all[strategy_col] = df_all[group_col]

    # Normalize metric column across the whole robustness table
    vals = df_all[metric_col].astype(float).values
    norm_vals = normalize_series_minmax(vals)
    df_all = df_all.copy()
    df_all[f"__norm__{metric_col}"] = norm_vals

    # Compute orig per uid (mean if multiple rows)
    df_orig = df_all[df_all[group_col].str.lower() == "original"][[uid_col, f"__norm__{metric_col}"]]
    orig_mean = df_orig.groupby(uid_col, as_index=False)[f"__norm__{metric_col}"].mean()
    orig_mean = orig_mean.rename(columns={f"__norm__{metric_col}": "orig_norm"})

    # Stability: same_obvious rows
    df_same = df_all[df_all[group_col].str.lower() == "same_obvious"][
        [uid_col, strategy_col, f"__norm__{metric_col}"]
    ].rename(columns={f"__norm__{metric_col}": "same_norm"})
    stab = df_same.merge(orig_mean, on=uid_col, how="left")
    stab = stab.dropna(subset=["orig_norm", "same_norm"]).copy()
    stab["score"] = 1.0 - (stab["orig_norm"] - stab["same_norm"]).abs()
    stab["metric"] = metric_col
    stab = stab[[uid_col, strategy_col, "metric", "score"]]

    # Sensitivity: worse_obvious rows
    df_worse = df_all[df_all[group_col].str.lower() == "worse_obvious"][
        [uid_col, strategy_col, f"__norm__{metric_col}"]
    ].rename(columns={f"__norm__{metric_col}": "worse_norm"})
    sens = df_worse.merge(orig_mean, on=uid_col, how="left")
    sens = sens.dropna(subset=["orig_norm", "worse_norm"]).copy()
    sens["score"] = sens["orig_norm"] - sens["worse_norm"]
    sens["metric"] = metric_col
    sens = sens[[uid_col, strategy_col, "metric", "score"]]

    return stab, sens


def compute_binary_drop(
    df_all: pd.DataFrame,
    metric_col: str,
    uid_col: str = "uid",
    group_col: str = "group",
    strategy_col: str = "strategy",
) -> pd.DataFrame:
    """Return a DataFrame with a binary drop indicator per instance for worse_obvious rows.
    score = 1.0 if worse_norm < orig_norm else 0.0
    Normalization uses the same min-max normalization as other robustness metrics.
    Columns: [uid, strategy, metric, score]
    """
    # Ensure required cols exist
    for c in [uid_col, group_col, metric_col]:
        if c not in df_all.columns:
            return pd.DataFrame()

    # Strategy may be missing; create if needed
    if strategy_col not in df_all.columns:
        df_all[strategy_col] = df_all[group_col]

    # Normalize metric column across the whole robustness table (monotonic -> preserves ordering)
    vals = df_all[metric_col].astype(float).values
    norm_vals = normalize_series_minmax(vals)
    df_all = df_all.copy()
    df_all[f"__norm__{metric_col}"] = norm_vals

    # Compute orig per uid (mean if multiple rows)
    df_orig = df_all[df_all[group_col].str.lower() == "original"][[uid_col, f"__norm__{metric_col}"]]
    if df_orig.empty:
        return pd.DataFrame()
    orig_mean = df_orig.groupby(uid_col, as_index=False)[f"__norm__{metric_col}"].mean()
    orig_mean = orig_mean.rename(columns={f"__norm__{metric_col}": "orig_norm"})

    # Worse rows
    df_worse = df_all[df_all[group_col].str.lower() == "worse_obvious"][
        [uid_col, strategy_col, f"__norm__{metric_col}"]
    ].rename(columns={f"__norm__{metric_col}": "worse_norm"})

    if df_worse.empty:
        return pd.DataFrame()

    bin_df = df_worse.merge(orig_mean, on=uid_col, how="left")
    bin_df = bin_df.dropna(subset=["orig_norm", "worse_norm"]).copy()
    # 1 if went down under worse perturbation, else 0
    bin_df["score"] = (bin_df["worse_norm"] < bin_df["orig_norm"]).astype(float)
    bin_df["metric"] = metric_col
    bin_df = bin_df[[uid_col, strategy_col, "metric", "score"]]
    return bin_df


def process_dataset_dir(dataset_dir: str, best_df: Optional[pd.DataFrame], out_dir: str) -> Dict[str, Any]:
    base = os.path.basename(dataset_dir)
    orig_path = os.path.join(dataset_dir, "original_subset.csv")
    same_path = os.path.join(dataset_dir, "perturbed_same_obvious.csv")
    worse_path = os.path.join(dataset_dir, "perturbed_worse_obvious.csv")

    df_o = read_csv_if_exists(orig_path)
    df_s = read_csv_if_exists(same_path)
    df_w = read_csv_if_exists(worse_path)

    if df_o is None:
        return {"dataset": base, "ok": False, "reason": "missing original_subset.csv"}
    frames = []
    if df_o is not None:
        frames.append(df_o)
    if df_s is not None:
        frames.append(df_s)
    if df_w is not None:
        frames.append(df_w)
    if not frames:
        return {"dataset": base, "ok": False, "reason": "no csvs present"}

    df_all = pd.concat(frames, ignore_index=True, sort=False)

    # Ensure group column present; infer from file if needed
    if "group" not in df_all.columns:
        # try to infer by checking which frame it came from based on lengths (fallback)
        gvals = []
        for f, gname in [(df_o, "original"), (df_s, "same_obvious"), (df_w, "worse_obvious")]:
            if f is not None:
                gvals.extend([gname] * len(f))
        if len(gvals) == len(df_all):
            df_all["group"] = gvals
        else:
            df_all["group"] = "unknown"

    # UID column must exist
    uid_col = "uid" if "uid" in df_all.columns else None
    if uid_col is None:
        # create stable uid from input text
        if "input" in df_all.columns:
            df_all["uid"] = (
                df_all["input"].astype(str).str.strip().str.lower().apply(lambda x: hashlib.md5(x.encode()).hexdigest())
            )
            uid_col = "uid"
        else:
            return {"dataset": base, "ok": False, "reason": "missing uid and input columns"}

    metric_map = select_metric_columns(df_all, base, best_df)
    if not metric_map:
        return {"dataset": base, "ok": False, "reason": "no target metric columns found"}

    # Compute per-instance scores for each selected metric
    all_stab = []
    all_sens = []
    all_bin: List[pd.DataFrame] = []
    for logical_name, metric_col in metric_map.items():
        stab_df, sens_df = compute_per_instance_scores(df_all, metric_col, uid_col=uid_col)
        if not stab_df.empty:
            stab_df = stab_df.copy()
            stab_df["metric_logical"] = logical_name
            all_stab.append(stab_df)
        if not sens_df.empty:
            sens_df = sens_df.copy()
            sens_df["metric_logical"] = logical_name
            all_sens.append(sens_df)
        # Binary drop (worse goes down vs orig)
        bin_df = compute_binary_drop(df_all, metric_col, uid_col=uid_col)
        if not bin_df.empty:
            bin_df = bin_df.copy()
            bin_df["metric_logical"] = logical_name
            all_bin.append(bin_df)

    if not all_stab and not all_sens and not all_bin:
        return {"dataset": base, "ok": False, "reason": "no per-instance scores computed"}

    os.makedirs(out_dir, exist_ok=True)
    stab_out = os.path.join(out_dir, f"{base}_stability.csv")
    sens_out = os.path.join(out_dir, f"{base}_sensitivity.csv")
    bin_out = os.path.join(out_dir, f"{base}_binary_robustness.csv")

    stab_all_df = pd.concat(all_stab, ignore_index=True) if all_stab else pd.DataFrame()
    sens_all_df = pd.concat(all_sens, ignore_index=True) if all_sens else pd.DataFrame()
    bin_all_df = pd.concat(all_bin, ignore_index=True) if all_bin else pd.DataFrame()
    if not stab_all_df.empty:
        stab_all_df.to_csv(stab_out, index=False)
    if not sens_all_df.empty:
        sens_all_df.to_csv(sens_out, index=False)
    if not bin_all_df.empty:
        bin_all_df.to_csv(bin_out, index=False)

    # Averages per metric_logical (and metric) for summary
    summaries = []
    if not stab_all_df.empty:
        grp = stab_all_df.groupby(["metric_logical", "metric"])['score'].mean().reset_index()
        for _, r in grp.iterrows():
            summaries.append({
                "dataset": base,
                "type": "stability",
                "metric_logical": r["metric_logical"],
                "metric": r["metric"],
                "avg": float(r["score"]),
            })
    if not sens_all_df.empty:
        grp = sens_all_df.groupby(["metric_logical", "metric"])['score'].mean().reset_index()
        for _, r in grp.iterrows():
            summaries.append({
                "dataset": base,
                "type": "sensitivity",
                "metric_logical": r["metric_logical"],
                "metric": r["metric"],
                "avg": float(r["score"]),
            })
    if not bin_all_df.empty:
        grp = bin_all_df.groupby(["metric_logical", "metric"])['score'].mean().reset_index()
        for _, r in grp.iterrows():
            summaries.append({
                "dataset": base,
                "type": "binary",
                "metric_logical": r["metric_logical"],
                "metric": r["metric"],
                "avg": float(r["score"]),
            })

    # Aggregated = average of stability and sensitivity means when both exist for a (metric_logical, metric)
    if not stab_all_df.empty and not sens_all_df.empty:
        stab_means = stab_all_df.groupby(["metric_logical", "metric"])['score'].mean()
        sens_means = sens_all_df.groupby(["metric_logical", "metric"])['score'].mean()
        common_idx = stab_means.index.intersection(sens_means.index)
        for idx in common_idx:
            mlog, mcol = idx
            agg_val = float((stab_means.loc[idx] + sens_means.loc[idx]) / 2.0)
            summaries.append({
                "dataset": base,
                "type": "aggregated",
                "metric_logical": mlog,
                "metric": mcol,
                "avg": agg_val,
            })

    return {"dataset": base, "ok": True, "summaries": summaries, "stab_df": stab_all_df, "sens_df": sens_all_df, "bin_df": bin_all_df}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute per-instance stability and sensitivity for robustness CSVs.")
    parser.add_argument(
        "--csv-root",
        default=CSV_ROOT,
        help="Root directory containing dataset subfolders (default: discovered relative to this script).",
    )
    parser.add_argument(
        "--best-metrics-csv",
        default=BEST_METRICS_CSV,
        help="Path to results/best_metrics.csv (default: discovered relative to this script).",
    )
    parser.add_argument(
        "--out-dir",
        default=OUT_DIR,
        help="Output directory for per-instance CSVs and averages (default: analysis/robustness/outputs).",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional list of dataset folder basenames to process (e.g., simpeval_score evalgenproduct_grade).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable generation of bar plot figure(s).",
    )
    args = parser.parse_args()

    csv_root = os.path.abspath(args.csv_root)
    out_dir = os.path.abspath(args.out_dir)
    best_metrics_csv = os.path.abspath(args.best_metrics_csv)

    os.makedirs(out_dir, exist_ok=True)
    try:
        best_df = pd.read_csv(best_metrics_csv)
    except Exception:
        best_df = None

    ds_dirs = list_dataset_dirs(csv_root)
    if args.datasets:
        want = set(args.datasets)
        ds_dirs = [d for d in ds_dirs if os.path.basename(d) in want]
    all_summaries: List[Dict[str, Any]] = []
    # For plotting
    per_dataset_frames: Dict[str, Dict[str, pd.DataFrame]] = {}
    results: List[Dict[str, Any]] = []

    for d in ds_dirs:
        res = process_dataset_dir(d, best_df, out_dir)
        results.append(res)
        if res.get("ok") and res.get("summaries"):
            all_summaries.extend(res["summaries"])
        if res.get("ok"):
            per_dataset_frames[res["dataset"]] = {
                "stab": res.get("stab_df", pd.DataFrame()),
                "sens": res.get("sens_df", pd.DataFrame()),
                "bin": res.get("bin_df", pd.DataFrame()),
            }

    # Write summary TXT
    txt_path = os.path.join(out_dir, "robustness_averages.txt")
    lines: List[str] = []
    for r in results:
        if not r.get("ok"):
            lines.append(f"{r.get('dataset')}: ERROR - {r.get('reason')}")
            continue
    # Write averages grouped by dataset
    if all_summaries:
        # order by dataset then type then logical metric
        df_sum = pd.DataFrame(all_summaries)
        df_sum = df_sum.sort_values(by=["dataset", "type", "metric_logical"]).reset_index(drop=True)
        for dataset, df_d in df_sum.groupby("dataset"):
            lines.append(f"Dataset: {dataset}")
            # Always list in order sensitivity, stability, binary, aggregated
            for t in ["sensitivity", "stability", "binary", "aggregated"]:
                df_t = df_d[df_d['type'] == t]
                if df_t.empty:
                    continue
                lines.append(f"  {t}:")
                for _, row in df_t.iterrows():
                    lines.append(
                        f"    {row['metric_logical']} ({row['metric']}): avg={row['avg']:.6f}"
                    )
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # Also dump a JSON for programmatic inspection (optional)
    json_path = os.path.join(out_dir, "robustness_averages.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    # Plotting: 3-wide, 2-tall grid (adjust rows if datasets != 6)
    if not args.no_plot:
        datasets = list(per_dataset_frames.keys())
        if datasets:
            n = len(datasets)
            cols = 3
            rows = max(1, int(np.ceil(n / cols)))

            # Desired order (logical names) and display mapping
            pref_order_logic = ["MetaMetrics", "BEST_METRIC", "DNAEval", "LLMJudge", "Autometrics"]
            display_name = {
                "MetaMetrics": "MetaMetrics",
                "BEST_METRIC": "Best Metric",
                "DNAEval": "DNAEval",
                "LLMJudge": "LLMJudge",
                "Autometrics": "AutoMetrics",
            }

            def method_order(df_primary: pd.DataFrame, df_stab: pd.DataFrame) -> List[str]:
                methods = set()
                if df_primary is not None and not df_primary.empty:
                    methods.update(df_primary["metric_logical"].unique().tolist())
                if df_stab is not None and not df_stab.empty:
                    methods.update(df_stab["metric_logical"].unique().tolist())
                ordered = [m for m in pref_order_logic if m in methods]
                extras = sorted([m for m in methods if m not in pref_order_logic])
                return ordered + extras

            def mean_ci(a: np.ndarray) -> Tuple[float, float]:
                a = a[~np.isnan(a)]
                if a.size == 0:
                    return float("nan"), 0.0
                m = float(np.mean(a))
                if a.size < 2:
                    return m, 0.0
                s = float(np.std(a, ddof=1))
                ci = 1.96 * (s / np.sqrt(a.size))
                return m, ci

            # Pretty titles for panels
            def title_for_dataset(ds_base: str) -> str:
                mapping = {
                    "simpeval_score": "SimpEval (score)",
                    "helpsteer2_helpfulness": "HelpSteer2 (helpfulness)",
                    "cogymtraveloutcome_outcomerating": "CoGym (outcome rating)",
                    "realhumaneval_accepted": "RealHumanEval (accepted)",
                    "primock57_time_sec": "Primock57 (time_sec)",
                    "evalgenproduct_grade": "EvalGenProduct (grade)",
                }
                return mapping.get(ds_base, ds_base)

            # 1) Old style (paired bars at each method position): robustness (binary) + stability
            fig1, axes1 = plt.subplots(rows, cols, figsize=(cols * 5.8, rows * 4.4), squeeze=False)
            legend_shown_pair = False
            # Baselines (compute once per figure)
            baseline_robust = 0.5
            baseline_stab = float(np.mean(_simulate_baseline_scores(1000, 'stability')))
            for idx, dataset in enumerate(datasets):
                r = idx // cols
                c = idx % cols
                ax = axes1[r][c]
                frames = per_dataset_frames[dataset]
                df_bin = frames.get("bin", pd.DataFrame())
                df_stab = frames.get("stab", pd.DataFrame())
                methods = method_order(df_bin, df_stab)
                x = np.arange(len(methods))
                width = 0.35

                robust_means, robust_errs = [], []
                stab_means, stab_errs = [], []
                labels = []
                for mname in methods:
                    labels.append(display_name.get(mname, mname))
                    # robustness (binary)
                    if not df_bin.empty:
                        vals = df_bin[df_bin["metric_logical"] == mname]["score"].astype(float).values
                        m, ci = mean_ci(vals)
                    else:
                        m, ci = float("nan"), 0.0
                    robust_means.append(m)
                    robust_errs.append(ci)
                    # stability
                    if not df_stab.empty:
                        vals = df_stab[df_stab["metric_logical"] == mname]["score"].astype(float).values
                        m, ci = mean_ci(vals)
                    else:
                        m, ci = float("nan"), 0.0
                    stab_means.append(m)
                    stab_errs.append(ci)

                ax.bar(x - width / 2, robust_means, width, yerr=robust_errs, capsize=3, label="Sensitivity", edgecolor="black", linewidth=0.4)
                ax.bar(x + width / 2, stab_means, width, yerr=stab_errs, capsize=3, label="Stability", edgecolor="black", linewidth=0.4)
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=20, ha="right")
                ax.set_title(title_for_dataset(dataset))
                ax.set_ylim(0.0, 1.0)
                # No bottom x-label text ("method") as requested
                if c == 0:
                    ax.set_ylabel("score (mean ± 95% CI)")
                # Baseline lines (partial coverage: left half for robustness, right half for stability)
                x_left = -0.5
                x_right = len(methods) - 0.5
                x_mid = (x_left + x_right) / 2.0
                # Draw after bars, with high zorder, black dashed
                ax.hlines(baseline_robust, xmin=x_left, xmax=x_mid, linestyles='--', colors='black', linewidth=1.2, alpha=0.95, zorder=10, label="Normal Baseline")
                ax.hlines(baseline_stab, xmin=x_mid, xmax=x_right, linestyles='--', colors='black', linewidth=1.2, alpha=0.95, zorder=10, label="_nolegend_")
                if not legend_shown_pair:
                    ax.legend(loc='lower left')
                    legend_shown_pair = True

            for j in range(n, rows * cols):
                r = j // cols
                c = j % cols
                axes1[r][c].axis('off')

            fig1.tight_layout()
            plt.savefig(os.path.join(out_dir, "robustness_bars_pair.png"), dpi=300)
            plt.savefig(os.path.join(out_dir, "robustness_bars_pair.pdf"), bbox_inches="tight")
            plt.close(fig1)

            # 2) Two-cluster style (robustness group, stability group)
            def draw_two_cluster(dlist: List[str], save_name: str) -> None:
                if not dlist:
                    return
                n2 = len(dlist)
                rows2 = max(1, int(np.ceil(n2 / cols)))
                fig2, axes2 = plt.subplots(rows2, cols, figsize=(cols * 6.2, rows2 * 4.4), squeeze=False)
                legend_shown_two = False
                baseline_robust = 0.5
                baseline_stab = float(np.mean(_simulate_baseline_scores(1000, 'stability')))
                for idx, dataset in enumerate(dlist):
                    r = idx // cols
                    c = idx % cols
                    ax = axes2[r][c]
                    frames = per_dataset_frames[dataset]
                    df_bin = frames.get("bin", pd.DataFrame())
                    df_stab = frames.get("stab", pd.DataFrame())
                    methods = method_order(df_bin, df_stab)
                    k = len(methods)
                    gap = 1.0
                    pos_rob = np.arange(k)
                    pos_stab = np.arange(k) + k + gap

                    robust_means, robust_errs = [], []
                    stab_means, stab_errs = [], []
                    labels = []
                    for mname in methods:
                        labels.append(display_name.get(mname, mname))
                        # robustness values (binary)
                        if not df_bin.empty:
                            vals_r = df_bin[df_bin["metric_logical"] == mname]["score"].astype(float).values
                            m_r, ci_r = mean_ci(vals_r)
                        else:
                            m_r, ci_r = float("nan"), 0.0
                        robust_means.append(m_r)
                        robust_errs.append(ci_r)
                        # stability values
                        if not df_stab.empty:
                            vals_t = df_stab[df_stab["metric_logical"] == mname]["score"].astype(float).values
                            m_t, ci_t = mean_ci(vals_t)
                        else:
                            m_t, ci_t = float("nan"), 0.0
                        stab_means.append(m_t)
                        stab_errs.append(ci_t)

                    ax.bar(pos_rob, robust_means, color="#4C78A8", yerr=robust_errs, capsize=3, label="Sensitivity", edgecolor="black", linewidth=0.4)
                    ax.bar(pos_stab, stab_means, color="#F58518", yerr=stab_errs, capsize=3, label="Stability", edgecolor="black", linewidth=0.4)

                    xticks = np.concatenate([pos_rob, pos_stab])
                    xticklabels = labels + labels
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xticklabels, rotation=0, ha="center")
                    ax.set_title(title_for_dataset(dataset))
                    ax.set_ylim(0.0, 1.0)
                    # No bottom x-label text ("method") as requested
                    if c == 0:
                        ax.set_ylabel("score (mean ± 95% CI)")
                    # Baseline lines (partial coverage aligned to clusters)
                    if k > 0:
                        left_xmin = float(np.min(pos_rob)) - 0.5
                        left_xmax = float(np.max(pos_rob)) + 0.5
                        right_xmin = float(np.min(pos_stab)) - 0.5
                        right_xmax = float(np.max(pos_stab)) + 0.5
                        ax.hlines(baseline_robust, xmin=left_xmin, xmax=left_xmax, linestyles='--', colors='black', linewidth=1.2, alpha=0.95, zorder=10, label="Normal Baseline")
                        ax.hlines(baseline_stab, xmin=right_xmin, xmax=right_xmax, linestyles='--', colors='black', linewidth=1.2, alpha=0.95, zorder=10, label="_nolegend_")
                    if not legend_shown_two:
                        ax.legend(loc='lower left')
                        legend_shown_two = True

                for j in range(n2, rows2 * cols):
                    r = j // cols
                    c = j % cols
                    axes2[r][c].axis('off')

                fig2.tight_layout()
                plt.savefig(os.path.join(out_dir, save_name), dpi=300)
                pdf_name = os.path.splitext(save_name)[0] + ".pdf"
                plt.savefig(os.path.join(out_dir, pdf_name), bbox_inches="tight")
                plt.close(fig2)

            # Full set
            draw_two_cluster(datasets, "robustness_bars_two_clusters.png")
            # Core 3 requested order
            core3 = ["simpeval_score", "helpsteer2_helpfulness", "cogymtraveloutcome_outcomerating"]
            core3 = [d for d in core3 if d in datasets]
            draw_two_cluster(core3, "robustness_bars_two_clusters_core3.png")
            # Other 3 (remaining)
            other3 = [d for d in datasets if d not in set(core3)]
            # If more than 3 remain, still plot them in a grid
            draw_two_cluster(other3, "robustness_bars_two_clusters_other.png")

            # 2a) Two-cluster style with abbreviated method labels
            abbrev_name = {
                "MetaMetrics": "MM",
                "BEST_METRIC": "BM",
                "DNAEval": "DE",
                "LLMJudge": "LM",
                "Autometrics": "AM",
            }

            def draw_two_cluster_abbrev(dlist: List[str], save_name: str) -> None:
                if not dlist:
                    return
                n2 = len(dlist)
                rows2 = max(1, int(np.ceil(n2 / cols)))
                fig2, axes2 = plt.subplots(rows2, cols, figsize=(cols * 6.2, rows2 * 4.4), squeeze=False)
                legend_shown_two = False
                baseline_robust = 0.5
                baseline_stab = float(np.mean(_simulate_baseline_scores(1000, 'stability')))
                for idx, dataset in enumerate(dlist):
                    r = idx // cols
                    c = idx % cols
                    ax = axes2[r][c]
                    frames = per_dataset_frames[dataset]
                    df_bin = frames.get("bin", pd.DataFrame())
                    df_stab = frames.get("stab", pd.DataFrame())
                    methods = method_order(df_bin, df_stab)
                    k = len(methods)
                    gap = 1.0
                    pos_rob = np.arange(k)
                    pos_stab = np.arange(k) + k + gap

                    robust_means, robust_errs = [], []
                    stab_means, stab_errs = [], []
                    labels = []
                    for mname in methods:
                        # abbreviated label if available, else fallback to display_name or raw
                        labels.append(abbrev_name.get(mname, display_name.get(mname, mname)))
                        # robustness values (binary)
                        if not df_bin.empty:
                            vals_r = df_bin[df_bin["metric_logical"] == mname]["score"].astype(float).values
                            m_r, ci_r = mean_ci(vals_r)
                        else:
                            m_r, ci_r = float("nan"), 0.0
                        robust_means.append(m_r)
                        robust_errs.append(ci_r)
                        # stability values
                        if not df_stab.empty:
                            vals_t = df_stab[df_stab["metric_logical"] == mname]["score"].astype(float).values
                            m_t, ci_t = mean_ci(vals_t)
                        else:
                            m_t, ci_t = float("nan"), 0.0
                        stab_means.append(m_t)
                        stab_errs.append(ci_t)

                    ax.bar(pos_rob, robust_means, color="#4C78A8", yerr=robust_errs, capsize=3, label="Sensitivity", edgecolor="black", linewidth=0.4)
                    ax.bar(pos_stab, stab_means, color="#F58518", yerr=stab_errs, capsize=3, label="Stability", edgecolor="black", linewidth=0.4)

                    xticks = np.concatenate([pos_rob, pos_stab])
                    xticklabels = labels + labels
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xticklabels, rotation=0, ha="center")
                    ax.set_title(title_for_dataset(dataset))
                    ax.set_ylim(0.0, 1.0)
                    if c == 0:
                        ax.set_ylabel("score (mean ± 95% CI)")
                    # Baseline lines
                    if k > 0:
                        left_xmin = float(np.min(pos_rob)) - 0.5
                        left_xmax = float(np.max(pos_rob)) + 0.5
                        right_xmin = float(np.min(pos_stab)) - 0.5
                        right_xmax = float(np.max(pos_stab)) + 0.5
                        ax.hlines(baseline_robust, xmin=left_xmin, xmax=left_xmax, linestyles='--', colors='black', linewidth=1.2, alpha=0.95, zorder=10, label="Normal Baseline")
                        ax.hlines(baseline_stab, xmin=right_xmin, xmax=right_xmax, linestyles='--', colors='black', linewidth=1.2, alpha=0.95, zorder=10, label="_nolegend_")
                    if not legend_shown_two:
                        ax.legend(loc='lower left')
                        legend_shown_two = True

                for j in range(n2, rows2 * cols):
                    r = j // cols
                    c = j % cols
                    axes2[r][c].axis('off')

                fig2.tight_layout()
                plt.savefig(os.path.join(out_dir, save_name), dpi=300)
                pdf_name = os.path.splitext(save_name)[0] + ".pdf"
                plt.savefig(os.path.join(out_dir, pdf_name), bbox_inches="tight")
                plt.close(fig2)

            draw_two_cluster_abbrev(datasets, "robustness_bars_two_clusters_abbrev.png")
            # Core3 abbreviated
            draw_two_cluster_abbrev(core3, "robustness_bars_two_clusters_abbrev_core3.png")

            # 2b) Two-cluster style, AutoMetrics-only, narrower width
            def draw_two_cluster_autometrics_only(dlist: List[str], save_name: str) -> None:
                if not dlist:
                    return
                n2 = len(dlist)
                rows2 = max(1, int(np.ceil(n2 / cols)))
                # narrower figure width per column since fewer bars
                fig2, axes2 = plt.subplots(rows2, cols, figsize=(cols * 3.2, rows2 * 4.0), squeeze=False)
                legend_shown_two = False
                baseline_robust = 0.5
                baseline_stab = float(np.mean(_simulate_baseline_scores(1000, 'stability')))
                for idx, dataset in enumerate(dlist):
                    r = idx // cols
                    c = idx % cols
                    ax = axes2[r][c]
                    frames = per_dataset_frames[dataset]
                    df_bin = frames.get("bin", pd.DataFrame())
                    df_stab = frames.get("stab", pd.DataFrame())
                    methods = method_order(df_bin, df_stab)
                    # filter to AutoMetrics only
                    methods = [m for m in methods if m == "Autometrics"]
                    k = len(methods)
                    gap = 1.0
                    pos_rob = np.arange(k)
                    pos_stab = np.arange(k) + k + gap

                    robust_means, robust_errs = [], []
                    stab_means, stab_errs = [], []
                    labels = []
                    for mname in methods:
                        labels.append(display_name.get(mname, mname))
                        if not df_bin.empty:
                            vals_r = df_bin[df_bin["metric_logical"] == mname]["score"].astype(float).values
                            m_r, ci_r = mean_ci(vals_r)
                        else:
                            m_r, ci_r = float("nan"), 0.0
                        robust_means.append(m_r)
                        robust_errs.append(ci_r)
                        if not df_stab.empty:
                            vals_t = df_stab[df_stab["metric_logical"] == mname]["score"].astype(float).values
                            m_t, ci_t = mean_ci(vals_t)
                        else:
                            m_t, ci_t = float("nan"), 0.0
                        stab_means.append(m_t)
                        stab_errs.append(ci_t)

                    ax.bar(pos_rob, robust_means, color="#4C78A8", yerr=robust_errs, capsize=3, label="Sensitivity", edgecolor="black", linewidth=0.4)
                    ax.bar(pos_stab, stab_means, color="#F58518", yerr=stab_errs, capsize=3, label="Stability", edgecolor="black", linewidth=0.4)

                    # Show concise labels: Sens and Stab instead of repeating method name
                    xticks = np.concatenate([pos_rob, pos_stab])
                    xticklabels = ["Sens"] * k + ["Stab"] * k
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xticklabels, rotation=0, ha="center")
                    ax.set_title(title_for_dataset(dataset))
                    ax.set_ylim(0.0, 1.0)
                    if c == 0:
                        ax.set_ylabel("score (mean ± 95% CI)")
                    # Small headers over each cluster
                    if k > 0:
                        center_rob = float(np.mean(pos_rob)) if len(pos_rob) > 0 else 0.0
                        center_stab = float(np.mean(pos_stab)) if len(pos_stab) > 0 else 0.0
                        ax.text(center_rob, 0.98, "Sensitivity", ha='center', va='top', fontsize=14)
                        ax.text(center_stab, 0.98, "Stability", ha='center', va='top', fontsize=14)
                    if k > 0:
                        left_xmin = float(np.min(pos_rob)) - 0.5
                        left_xmax = float(np.max(pos_rob)) + 0.5
                        right_xmin = float(np.min(pos_stab)) - 0.5
                        right_xmax = float(np.max(pos_stab)) + 0.5
                        ax.hlines(baseline_robust, xmin=left_xmin, xmax=left_xmax, linestyles='--', colors='black', linewidth=1.2, alpha=0.95, zorder=10, label="Normal Baseline")
                        ax.hlines(baseline_stab, xmin=right_xmin, xmax=right_xmax, linestyles='--', colors='black', linewidth=1.2, alpha=0.95, zorder=10, label="_nolegend_")
                    if not legend_shown_two:
                        ax.legend(loc='lower left')
                        legend_shown_two = True

                for j in range(n2, rows2 * cols):
                    r = j // cols
                    c = j % cols
                    axes2[r][c].axis('off')

                fig2.tight_layout()
                plt.savefig(os.path.join(out_dir, save_name), dpi=300)
                pdf_name = os.path.splitext(save_name)[0] + ".pdf"
                plt.savefig(os.path.join(out_dir, pdf_name), bbox_inches="tight")
                plt.close(fig2)

            draw_two_cluster_autometrics_only(datasets, "robustness_bars_two_clusters_autometrics_only.png")
            # Core3 AutoMetrics-only
            draw_two_cluster_autometrics_only(core3, "robustness_bars_two_clusters_autometrics_only_core3.png")

            # 3) Binary robustness plots (separate set)
            def draw_binary(dlist: List[str], save_name: str) -> None:
                if not dlist:
                    return
                n2 = len(dlist)
                rows2 = max(1, int(np.ceil(n2 / cols)))
                fig2, axes2 = plt.subplots(rows2, cols, figsize=(cols * 5.8, rows2 * 4.4), squeeze=False)
                legend_shown_bin = False
                baseline_bin = 0.5
                for idx, dataset in enumerate(dlist):
                    r = idx // cols
                    c = idx % cols
                    ax = axes2[r][c]
                    frames = per_dataset_frames[dataset]
                    df_bin = frames.get("bin", pd.DataFrame())
                    # Determine methods present in binary df
                    if df_bin is None or df_bin.empty:
                        ax.axis('off')
                        continue
                    methods = []
                    # Preserve preferred order, then extras
                    present = df_bin["metric_logical"].unique().tolist()
                    methods.extend([m for m in pref_order_logic if m in present])
                    methods.extend(sorted([m for m in present if m not in pref_order_logic]))

                    x = np.arange(len(methods))
                    means, errs, labels = [], [], []
                    for mname in methods:
                        labels.append(display_name.get(mname, mname))
                        vals = df_bin[df_bin["metric_logical"] == mname]["score"].astype(float).values
                        m, ci = mean_ci(vals)
                        means.append(m)
                        errs.append(ci)

                    ax.bar(x, means, color="#54A24B", yerr=errs, capsize=3, label="binary drop rate", edgecolor="black", linewidth=0.4)
                    ax.set_xticks(x)
                    ax.set_xticklabels(labels, rotation=0, ha="center")
                    ax.set_title(title_for_dataset(dataset))
                    ax.set_ylim(0.0, 1.0)
                    # No bottom x-label text ("method") as requested
                    if c == 0:
                        ax.set_ylabel("binary drop rate (mean ± 95% CI)")
                    # Baseline line (full width, black)
                    x_left = -0.5
                    x_right = len(methods) - 0.5
                    ax.hlines(baseline_bin, xmin=x_left, xmax=x_right, linestyles='--', colors='black', linewidth=1.2, alpha=0.95, zorder=10, label="Normal Baseline")
                    if not legend_shown_bin:
                        ax.legend(loc='lower left')
                        legend_shown_bin = True

                for j in range(n2, rows2 * cols):
                    r = j // cols
                    c = j % cols
                    axes2[r][c].axis('off')

                fig2.tight_layout()
                plt.savefig(os.path.join(out_dir, save_name), dpi=300)
                pdf_name = os.path.splitext(save_name)[0] + ".pdf"
                plt.savefig(os.path.join(out_dir, pdf_name), bbox_inches="tight")
                plt.close(fig2)

            draw_binary(datasets, "binary_robustness_bars.png")
            draw_binary(core3, "binary_robustness_bars_core3.png")
            draw_binary(other3, "binary_robustness_bars_other.png")
            
            # 3a) Binary robustness with abbreviated method labels
            def draw_binary_abbrev(dlist: List[str], save_name: str) -> None:
                if not dlist:
                    return
                n2 = len(dlist)
                rows2 = max(1, int(np.ceil(n2 / cols)))
                fig2, axes2 = plt.subplots(rows2, cols, figsize=(cols * 5.8, rows2 * 4.4), squeeze=False)
                legend_shown_bin = False
                baseline_bin = 0.5
                for idx, dataset in enumerate(dlist):
                    r = idx // cols
                    c = idx % cols
                    ax = axes2[r][c]
                    frames = per_dataset_frames[dataset]
                    df_bin = frames.get("bin", pd.DataFrame())
                    if df_bin is None or df_bin.empty:
                        ax.axis('off')
                        continue
                    present = df_bin["metric_logical"].unique().tolist()
                    methods = []
                    methods.extend([m for m in pref_order_logic if m in present])
                    methods.extend(sorted([m for m in present if m not in pref_order_logic]))

                    x = np.arange(len(methods))
                    means, errs, labels = [], [], []
                    for mname in methods:
                        labels.append(abbrev_name.get(mname, display_name.get(mname, mname)))
                        vals = df_bin[df_bin["metric_logical"] == mname]["score"].astype(float).values
                        m, ci = mean_ci(vals)
                        means.append(m)
                        errs.append(ci)

                    ax.bar(x, means, color="#54A24B", yerr=errs, capsize=3, label="binary drop rate", edgecolor="black", linewidth=0.4)
                    ax.set_xticks(x)
                    ax.set_xticklabels(labels, rotation=0, ha="center")
                    ax.set_title(title_for_dataset(dataset))
                    ax.set_ylim(0.0, 1.0)
                    if c == 0:
                        ax.set_ylabel("binary drop rate (mean ± 95% CI)")
                    x_left = -0.5
                    x_right = len(methods) - 0.5
                    ax.hlines(baseline_bin, xmin=x_left, xmax=x_right, linestyles='--', colors='black', linewidth=1.2, alpha=0.95, zorder=10, label="Normal Baseline")
                    if not legend_shown_bin:
                        ax.legend(loc='lower left')
                        legend_shown_bin = True

                for j in range(n2, rows2 * cols):
                    r = j // cols
                    c = j % cols
                    axes2[r][c].axis('off')

                fig2.tight_layout()
                plt.savefig(os.path.join(out_dir, save_name), dpi=300)
                pdf_name = os.path.splitext(save_name)[0] + ".pdf"
                plt.savefig(os.path.join(out_dir, pdf_name), bbox_inches="tight")
                plt.close(fig2)

            draw_binary_abbrev(datasets, "binary_robustness_bars_abbrev.png")
            draw_binary_abbrev(core3, "binary_robustness_bars_abbrev_core3.png")
            
            # 3b) Binary robustness, AutoMetrics-only, narrower width
            def draw_binary_autometrics_only(dlist: List[str], save_name: str) -> None:
                if not dlist:
                    return
                n2 = len(dlist)
                rows2 = max(1, int(np.ceil(n2 / cols)))
                fig2, axes2 = plt.subplots(rows2, cols, figsize=(cols * 3.0, rows2 * 4.0), squeeze=False)
                legend_shown_bin = False
                baseline_bin = 0.5
                for idx, dataset in enumerate(dlist):
                    r = idx // cols
                    c = idx % cols
                    ax = axes2[r][c]
                    frames = per_dataset_frames[dataset]
                    df_bin = frames.get("bin", pd.DataFrame())
                    if df_bin is None or df_bin.empty:
                        ax.axis('off')
                        continue
                    # Only Autometrics
                    df_am = df_bin[df_bin["metric_logical"] == "Autometrics"]
                    if df_am.empty:
                        ax.axis('off')
                        continue
                    x = np.arange(1)
                    means, errs = [], []
                    vals = df_am["score"].astype(float).values
                    m, ci = mean_ci(vals)
                    means.append(m)
                    errs.append(ci)
                    ax.bar(x, means, color="#54A24B", yerr=errs, capsize=3, label="binary drop rate", edgecolor="black", linewidth=0.4)
                    ax.set_xticks(x)
                    ax.set_xticklabels(["AM"], rotation=0, ha="center")
                    ax.set_title(title_for_dataset(dataset))
                    ax.set_ylim(0.0, 1.0)
                    if c == 0:
                        ax.set_ylabel("binary drop rate (mean ± 95% CI)")
                    x_left = -0.5
                    x_right = 0.5
                    ax.hlines(baseline_bin, xmin=x_left, xmax=x_right, linestyles='--', colors='black', linewidth=1.2, alpha=0.95, zorder=10, label="Normal Baseline")
                    if not legend_shown_bin:
                        ax.legend(loc='lower left')
                        legend_shown_bin = True

                for j in range(n2, rows2 * cols):
                    r = j // cols
                    c = j % cols
                    axes2[r][c].axis('off')

                fig2.tight_layout()
                plt.savefig(os.path.join(out_dir, save_name), dpi=300)
                pdf_name = os.path.splitext(save_name)[0] + ".pdf"
                plt.savefig(os.path.join(out_dir, pdf_name), bbox_inches="tight")
                plt.close(fig2)

            draw_binary_autometrics_only(datasets, "binary_robustness_bars_autometrics_only.png")
            draw_binary_autometrics_only(core3, "binary_robustness_bars_autometrics_only_core3.png")

            # 4) AutoMetrics-only CORE3 combined figure: top row Sens/Stab, bottom row Binary
            def draw_autometrics_core3_combined(dlist: List[str], save_name: str) -> None:
                if not dlist:
                    return
                n2 = len(dlist)
                cols2 = min(3, max(1, n2))
                rows2 = 2  # two rows: top Sens/Stab, bottom Binary
                fig2, axes2 = plt.subplots(rows2, cols2, figsize=(cols2 * 3.4, rows2 * 3.8), squeeze=False)
                baseline_robust = 0.5
                baseline_stab = float(np.mean(_simulate_baseline_scores(1000, 'stability')))
                baseline_bin = 0.5

                for idx, dataset in enumerate(dlist[:cols2]):
                    c = idx % cols2
                    frames = per_dataset_frames[dataset]
                    df_bin = frames.get("bin", pd.DataFrame())
                    df_stab = frames.get("stab", pd.DataFrame())

                    # Top row: Sens/Stab (AutoMetrics-only)
                    ax_top = axes2[0][c]
                    methods = method_order(df_bin, df_stab)
                    methods = [m for m in methods if m == "Autometrics"]
                    k = len(methods)
                    gap = 1.0
                    pos_rob = np.arange(k)
                    pos_stab = np.arange(k) + k + gap

                    robust_means, robust_errs = [], []
                    stab_means, stab_errs = [], []
                    for mname in methods:
                        if not df_bin.empty:
                            vals_r = df_bin[df_bin["metric_logical"] == mname]["score"].astype(float).values
                            m_r, ci_r = mean_ci(vals_r)
                        else:
                            m_r, ci_r = float("nan"), 0.0
                        robust_means.append(m_r)
                        robust_errs.append(ci_r)
                        if not df_stab.empty:
                            vals_t = df_stab[df_stab["metric_logical"] == mname]["score"].astype(float).values
                            m_t, ci_t = mean_ci(vals_t)
                        else:
                            m_t, ci_t = float("nan"), 0.0
                        stab_means.append(m_t)
                        stab_errs.append(ci_t)

                    ax_top.bar(pos_rob, robust_means, color="#4C78A8", yerr=robust_errs, capsize=3, label="Sensitivity", edgecolor="black", linewidth=0.4)
                    ax_top.bar(pos_stab, stab_means, color="#F58518", yerr=stab_errs, capsize=3, label="Stability", edgecolor="black", linewidth=0.4)

                    xticks = np.concatenate([pos_rob, pos_stab])
                    xticklabels = ["Sens"] * k + ["Stab"] * k
                    ax_top.set_xticks(xticks)
                    ax_top.set_xticklabels(xticklabels, rotation=0, ha="center")
                    ax_top.set_title(title_for_dataset(dataset), fontsize=14)
                    ax_top.set_ylim(0.0, 1.0)
                    if c == 0:
                        ax_top.set_ylabel("score (mean ± 95% CI)")
                    if k > 0:
                        left_xmin = float(np.min(pos_rob)) - 0.5
                        left_xmax = float(np.max(pos_rob)) + 0.5
                        right_xmin = float(np.min(pos_stab)) - 0.5
                        right_xmax = float(np.max(pos_stab)) + 0.5
                        ax_top.hlines(baseline_robust, xmin=left_xmin, xmax=left_xmax, linestyles='--', colors='black', linewidth=1.2, alpha=0.95, zorder=10)
                        ax_top.hlines(baseline_stab, xmin=right_xmin, xmax=right_xmax, linestyles='--', colors='black', linewidth=1.2, alpha=0.95, zorder=10)

                    # Bottom row: Binary (AutoMetrics-only)
                    ax_bot = axes2[1][c]
                    if df_bin is None or df_bin.empty:
                        ax_bot.axis('off')
                        continue
                    df_am = df_bin[df_bin["metric_logical"] == "Autometrics"]
                    if df_am.empty:
                        ax_bot.axis('off')
                        continue
                    x = np.arange(1)
                    vals = df_am["score"].astype(float).values
                    m, ci = mean_ci(vals)
                    ax_bot.bar(x, [m], color="#54A24B", yerr=[ci], capsize=3, label="binary drop rate", edgecolor="black", linewidth=0.4)
                    ax_bot.set_xticks(x)
                    ax_bot.set_xticklabels(["AM"], rotation=0, ha="center")
                    ax_bot.set_ylim(0.0, 1.0)
                    if c == 0:
                        ax_bot.set_ylabel("binary drop rate (mean ± 95% CI)")
                    ax_bot.hlines(baseline_bin, xmin=-0.5, xmax=0.5, linestyles='--', colors='black', linewidth=1.2, alpha=0.95, zorder=10)

                fig2.tight_layout()
                plt.savefig(os.path.join(out_dir, save_name), dpi=300)
                pdf_name = os.path.splitext(save_name)[0] + ".pdf"
                plt.savefig(os.path.join(out_dir, pdf_name), bbox_inches="tight")
                plt.close(fig2)

            draw_autometrics_core3_combined(core3, "autometrics_core3_combined.png")

            # 5) Single-axis combined AutoMetrics-only core3 for Sens/Stab
            def draw_autometrics_core3_single_axis_sens_stab(dlist: List[str], save_name: str) -> None:
                if not dlist:
                    return
                # preserve core3 order present in dlist
                ds = dlist
                fig, ax = plt.subplots(1, 1, figsize=(max(1, len(ds)) * 1.9, 3.8))
                baseline_robust = 0.5
                baseline_stab = float(np.mean(_simulate_baseline_scores(1000, 'stability')))

                x_positions = []
                x_labels = []
                xpos = 0
                width = 0.08  # bar thickness
                pair_gap = 0.09  # distance between pairs (clusters)

                def split_title_and_measure(s: str) -> Tuple[str, str]:
                    s = str(s)
                    if "(" in s and s.endswith(")"):
                        name = s[: s.rfind("(")].strip()
                        meas = s[s.rfind("(") :].strip()
                        return name, meas
                    return s, ""

                for dataset in ds:
                    frames = per_dataset_frames.get(dataset, {})
                    df_bin = frames.get("bin", pd.DataFrame())
                    df_stab = frames.get("stab", pd.DataFrame())
                    # compute AutoMetrics-only means and CIs
                    if df_bin is not None and not df_bin.empty:
                        vals_r = df_bin[df_bin["metric_logical"] == "Autometrics"]["score"].astype(float).values
                        m_r, ci_r = mean_ci(vals_r)
                    else:
                        m_r, ci_r = float("nan"), 0.0
                    if df_stab is not None and not df_stab.empty:
                        vals_t = df_stab[df_stab["metric_logical"] == "Autometrics"]["score"].astype(float).values
                        m_t, ci_t = mean_ci(vals_t)
                    else:
                        m_t, ci_t = float("nan"), 0.0

                    # Plot two adjacent bars: Sens then Stab
                    # Tight pair (no gap within pair): centers separated by exactly bar width
                    pos_sens = xpos - width / 2.0
                    pos_stab = xpos + width / 2.0
                    ax.bar(pos_sens, m_r, width, yerr=ci_r, capsize=3, color="#4C78A8", edgecolor="black", linewidth=0.4, label="Sensitivity" if xpos==0 else "_nolegend_")
                    ax.bar(pos_stab, m_t, width, yerr=ci_t, capsize=3, color="#F58518", edgecolor="black", linewidth=0.4, label="Stability" if xpos==0 else "_nolegend_")
                    # Dataset title and measure above cluster (axes fraction for y)
                    full_title = title_for_dataset(dataset)
                    name_str, meas_str = split_title_and_measure(full_title)
                    if name_str:
                        ax.text(xpos, 1.12, name_str, transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=15, fontweight='bold')
                    if meas_str:
                        ax.text(xpos, 1.04, meas_str, transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=12)
                    x_positions.extend([pos_sens, pos_stab])
                    x_labels.extend(["Sens", "Stab"]) 

                    # Baseline segments per bar
                    ax.hlines(baseline_robust, xmin=pos_sens - width/2.0, xmax=pos_sens + width/2.0, linestyles='--', colors='black', linewidth=1.2, alpha=0.95, zorder=10, label="Normal Baseline" if xpos==0 else "_nolegend_")
                    ax.hlines(baseline_stab, xmin=pos_stab - width/2.0, xmax=pos_stab + width/2.0, linestyles='--', colors='black', linewidth=1.2, alpha=0.95, zorder=10, label="_nolegend_")
                    # Advance center by total pair width (2*width) plus inter-pair gap
                    xpos += (2.0 * width) + pair_gap

                ax.set_xticks(x_positions)
                ax.set_xticklabels(x_labels, rotation=0, ha="center")
                ax.set_ylim(0.0, 1.0)
                ax.set_ylabel("score (mean ± 95% CI)")
                ax.legend(loc='lower left')
                fig.tight_layout()
                plt.savefig(os.path.join(out_dir, save_name), dpi=300)
                pdf_name = os.path.splitext(save_name)[0] + ".pdf"
                plt.savefig(os.path.join(out_dir, pdf_name), bbox_inches="tight")
                plt.close(fig)

            draw_autometrics_core3_single_axis_sens_stab(core3, "autometrics_core3_single_axis_sens_stab.png")

            # 6) Single-axis combined AutoMetrics-only core3 for Binary
            def draw_autometrics_core3_single_axis_binary(dlist: List[str], save_name: str) -> None:
                if not dlist:
                    return
                ds = dlist
                fig, ax = plt.subplots(1, 1, figsize=(max(1, len(ds)) * 2.6, 3.6))
                baseline_bin = 0.5

                x_positions = []
                x_labels = []
                xpos = 0
                gap = 0.7
                width = 0.42
                for dataset in ds:
                    frames = per_dataset_frames.get(dataset, {})
                    df_bin = frames.get("bin", pd.DataFrame())
                    if df_bin is None or df_bin.empty:
                        continue
                    df_am = df_bin[df_bin["metric_logical"] == "Autometrics"]
                    if df_am.empty:
                        continue
                    vals = df_am["score"].astype(float).values
                    m, ci = mean_ci(vals)
                    ax.bar(xpos, m, width=width, color="#54A24B", yerr=[ci], capsize=3, label="binary drop rate" if xpos==0 else "_nolegend_", edgecolor="black", linewidth=0.4)
                    # Dataset title at top in bold (axes fraction for y)
                    ax.text(xpos, 1.06, title_for_dataset(dataset), transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=15, fontweight='bold')
                    x_positions.append(xpos)
                    x_labels.append("AM")
                    # Baseline segment under bar
                    ax.hlines(baseline_bin, xmin=xpos - width/2, xmax=xpos + width/2, linestyles='--', colors='black', linewidth=1.2, alpha=0.95, zorder=10, label="Normal Baseline" if len(x_positions)==1 else "_nolegend_")
                    xpos += gap

                ax.set_xticks(x_positions)
                ax.set_xticklabels(x_labels, rotation=0, ha="center")
                ax.set_ylim(0.0, 1.0)
                ax.set_ylabel("binary drop rate (mean ± 95% CI)")
                ax.legend(loc='lower left')
                fig.tight_layout()
                plt.savefig(os.path.join(out_dir, save_name), dpi=300)
                pdf_name = os.path.splitext(save_name)[0] + ".pdf"
                plt.savefig(os.path.join(out_dir, pdf_name), bbox_inches="tight")
                plt.close(fig)

            draw_autometrics_core3_single_axis_binary(core3, "autometrics_core3_single_axis_binary.png")

    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()

