import os
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Defaults inspired by analysis/robustness/analyze_results.py
sns.set_theme(style="darkgrid")
plt.rcParams.update({
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_RESULTS_DIR = os.path.join(
    REPO_ROOT,
    "results",
    "data_scaling",
    "autometrics",
    "results",
)


def default_csv_path(metric: str) -> str:
    name = metric.strip().lower()
    if name not in {"pearson", "spearman", "kendall"}:
        name = "kendall"
    return os.path.join(DEFAULT_RESULTS_DIR, f"{name}.csv")


def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # Normalize and clean expected columns
    expected = ["dataset", "train_size", "model", "correlation"]
    for c in expected:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {csv_path}")
    # Drop NaN correlations and non-positive train sizes
    df = df.copy()
    df = df.replace({"correlation": {np.inf: np.nan, -np.inf: np.nan}})
    df = df.dropna(subset=["correlation", "train_size"]).reset_index(drop=True)
    df = df[df["train_size"].astype(float) > 0]
    return df


def fit_log_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Fit y = a + b * ln(x). Returns (a, b)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if not np.any(mask):
        return float("nan"), float("nan")
    lx = np.log(x[mask])
    ly = y[mask]
    # Linear regression on ln(x)
    b, a = np.polyfit(lx, ly, deg=1)  # returns slope, intercept for lx -> ly
    # Convert to y = a' + b' * ln(x) form
    a_prime = a
    b_prime = b
    return float(a_prime), float(b_prime)


def evaluate_log_curve(a: float, b: float, x: np.ndarray) -> np.ndarray:
    return a + b * np.log(x)


def title_for_dataset(ds_key: str) -> str:
    mapping = {
        "SimpEval": "SimpEval (score)",
        "HelpSteer2": "HelpSteer2 (helpfulness)",
        "RealHumanEval": "RealHumanEval (accepted)",
    }
    return mapping.get(ds_key, ds_key)


def plot_data_scaling(
    df: pd.DataFrame,
    metric_name: str,
    out_dir: str,
    alpha_points: float = 0.45,
) -> str:
    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Order of subplots
    datasets_order = ["SimpEval", "HelpSteer2", "RealHumanEval"]

    # Colors for model categories
    color_map: Dict[str, str] = {
        "gen_only": "#4C78A8",  # blue-ish
        "full": "#F58518",      # orange
    }
    label_map: Dict[str, str] = {
        "gen_only": "Generated Only",
        "full": "Full",
    }

    # Global y-limits across selected datasets (same scale for all panels)
    sel = df[df["dataset"].astype(str).isin(datasets_order)]
    yvals = sel["correlation"].astype(float).values if not sel.empty else np.array([])
    if yvals.size > 0:
        y_lo = float(np.nanmin(yvals))
        y_hi = float(np.nanmax(yvals))
        if not np.isfinite(y_lo) or not np.isfinite(y_hi):
            y_lo, y_hi = -1.0, 1.0
        y_pad = 0.05 * max(1e-6, (y_hi - y_lo))
        global_ymin = y_lo - y_pad
        global_ymax = y_hi + y_pad
        if global_ymin == global_ymax:
            global_ymin -= 0.05
            global_ymax += 0.05
    else:
        global_ymin, global_ymax = -1.0, 1.0

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.6), squeeze=False, sharey=True)
    axes = axes[0]

    for i, ds in enumerate(datasets_order):
        ax = axes[i]
        sub = df[df["dataset"].astype(str) == ds]
        if sub.empty:
            ax.set_title(title_for_dataset(ds))
            ax.set_xlabel("Train Size")
            if i == 0:
                ax.set_ylabel(f"{metric_name} correlation")
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            ax.tick_params(axis='x', labelsize=14)
            continue

        # Two groups: gen_only and full
        for model_key in ["gen_only", "full"]:
            grp = sub[sub["model"].astype(str) == model_key]
            if grp.empty:
                continue
            x = grp["train_size"].astype(float).values
            y = grp["correlation"].astype(float).values
            ax.scatter(
                x,
                y,
                color=color_map.get(model_key, "C0"),
                alpha=alpha_points,
                s=28,
                label=None,
                edgecolor="black",
                linewidths=0.2,
            )

            # Plot prominent mean dot per train_size for this series
            grp_means = (
                grp.groupby("train_size")["correlation"].mean().reset_index().sort_values("train_size")
            )
            if not grp_means.empty:
                xm = grp_means["train_size"].astype(float).values
                ym = grp_means["correlation"].astype(float).values
                ax.scatter(
                    xm,
                    ym,
                    color=color_map.get(model_key, "C0"),
                    alpha=1.0,
                    s=60,
                    label=label_map.get(model_key, model_key) if i == 0 else None,
                    edgecolor="black",
                    linewidths=0.6,
                    zorder=3,
                )

        ax.set_title(title_for_dataset(ds))
        ax.set_xlabel("Train Size")
        if i == 0:
            ax.set_ylabel(f"{metric_name} correlation")

        # Log x-scale (base 2 aligns with doubling train sizes)
        ax.set_xscale("log", base=2)
        # Nice x ticks at 5 * 2^k within data range
        all_x = sub["train_size"].astype(float).values
        if all_x.size > 0:
            lo, hi = float(np.min(all_x)), float(np.max(all_x))
            ticks: List[float] = []
            v = 5.0
            while v <= hi * 1.01:
                if v >= lo * 0.99:
                    ticks.append(v)
                v *= 2.0
            if ticks:
                ax.set_xticks(ticks)
                ax.set_xticklabels([str(int(t)) for t in ticks], fontsize=14)
                ax.get_xaxis().set_minor_locator(plt.NullLocator())
            else:
                ax.tick_params(axis='x', labelsize=14)
        else:
            ax.tick_params(axis='x', labelsize=14)
        # Shared y-limits
        ax.set_ylim(global_ymin, global_ymax)

    # Legend inside the first panel to avoid extra outer whitespace
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="lower right", frameon=True)

    # Reduce outer whitespace
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.13, top=0.92, wspace=0.18)

    base = f"data_scaling_scatter_{metric_name.lower()}"
    png_path = os.path.join(out_dir, f"{base}.png")
    pdf_path = os.path.join(out_dir, f"{base}.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze data scaling CSVs and plot scatter + log-fit for gen-only vs full."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to a consolidated results CSV (columns: dataset, train_size, model, correlation).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="kendall",
        choices=["kendall", "spearman", "pearson"],
        help="Metric CSV to use when --csv is not provided (default: kendall).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "outputs"),
        help="Directory to save figures (default: analysis/data_scaling/outputs).",
    )
    args = parser.parse_args()

    csv_path = args.csv if args.csv else default_csv_path(args.metric)
    df = load_data(csv_path)
    out_dir = os.path.abspath(args.out_dir)

    metric_label = args.metric.capitalize()
    out_png = plot_data_scaling(df, metric_label, out_dir)
    print(f"Wrote figure to: {out_png}")


if __name__ == "__main__":
    main()


