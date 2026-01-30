#!/usr/bin/env python3
import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")  # headless backend for servers
import matplotlib.pyplot as plt
try:
    # Optional dependency for cleaner broken-axis plots
    from brokenaxes import brokenaxes as BrokenAxes
except Exception:
    BrokenAxes = None


def find_log_files(root_dirs: Iterable[str]) -> List[Tuple[str, str]]:
    """
    Recursively find all log_*.json files under the provided root directories.

    Returns a list of (root_dir, log_file_path) tuples so we can attribute
    results to the selected root directory even if logs live in subfolders.
    """
    log_files: List[Tuple[str, str]] = []
    for root in root_dirs:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for filename in filenames:
                if filename.startswith("log_") and filename.endswith(".json"):
                    log_files.append((root, os.path.join(dirpath, filename)))
    return log_files


def classify_metric(metric_name: str, model_suffixes: List[str]) -> str:
    """
    Classify metric according to user rules:
    - optimized_judge: contains "_optimized_"
    - examples_judge: endswith "_examples"
    - rubric: endswith "_Rubric"
    - llm_judge: endswith one of "_<model_suffix>" (e.g., _Qwen3-32B)
    - named_metric: everything else
    """
    if "_optimized_" in metric_name:
        return "optimized_judge"
    if metric_name.endswith("_examples"):
        return "examples_judge"
    if metric_name.endswith("_Rubric"):
        return "rubric"
    for suffix in model_suffixes:
        if metric_name.endswith(f"_{suffix}"):
            return "llm_judge"
    return "named_metric"


def load_top_metrics(log_path: str) -> List[str]:
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        top_metrics = data.get("top_metrics")
        if isinstance(top_metrics, list):
            # Ensure string values only
            return [str(m) for m in top_metrics]
        return []
    except Exception:
        return []


def tally_metrics(
    log_files: List[Tuple[str, str]],
    model_suffixes: List[str],
) -> Dict[Tuple[str, str, str], int]:
    """
    Returns a dict keyed by (dataset, metric_name, category) -> count.
    - dataset is derived from the basename of the selected root directory.
    """
    counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
    for root, log_path in log_files:
        dataset = os.path.basename(os.path.normpath(root))
        for metric in load_top_metrics(log_path):
            category = classify_metric(metric, model_suffixes)
            counts[(dataset, metric, category)] += 1
    return counts


def write_csv(
    counts: Dict[Tuple[str, str, str], int],
    output_csv: str,
) -> None:
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    rows = [
        {"dataset": ds, "metric_name": m, "category": cat, "count": c}
        for (ds, m, cat), c in counts.items()
    ]
    # Sort for readability: dataset, then descending count, then metric name
    rows.sort(key=lambda r: (r["dataset"], -r["count"], r["metric_name"]))
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "metric_name", "category", "count"])
        writer.writeheader()
        writer.writerows(rows)


def aggregate_by_type(
    counts: Dict[Tuple[str, str, str], int]
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Collapse dataset dimension and aggregate by category, except for named_metric
    which is kept per-metric name.

    Returns:
      - category_counts: category -> total count (excluding named_metric)
      - named_metric_counts: metric_name -> total count (for named_metric only)
    """
    category_counts: Dict[str, int] = defaultdict(int)
    named_metric_counts: Dict[str, int] = defaultdict(int)
    for (dataset, metric_name, category), c in counts.items():
        if category == "named_metric":
            named_metric_counts[metric_name] += c
        else:
            category_counts[category] += c
    return category_counts, named_metric_counts


def write_types_csv(
    category_counts: Dict[str, int],
    named_metric_counts: Dict[str, int],
    output_csv: str,
) -> None:
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    rows: List[Dict[str, object]] = []
    for category, count in category_counts.items():
        rows.append({"category": category, "metric_name": "", "count": count})
    for metric_name, count in named_metric_counts.items():
        rows.append({"category": "named_metric", "metric_name": metric_name, "count": count})
    # Sort: first by category (named_metric last), then by descending count, then name
    def sort_key(r: Dict[str, object]):
        cat = r["category"]
        cat_rank = 1 if cat == "named_metric" else 0
        return (cat_rank, -int(r["count"]), r["metric_name"])  # type: ignore[arg-type]
    rows.sort(key=sort_key)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "metric_name", "count"])
        writer.writeheader()
        writer.writerows(rows)


def _pretty_category_name(cat: str) -> str:
    mapping = {
        "llm_judge": "Single Criteria",
        "rubric": "Rubric",
        "examples_judge": "Examples Judge",
        "optimized_judge": "Optimized Judge",
        "named_metric": "Named Metric",
    }
    return mapping.get(cat, cat.replace("_", " ").title())


def _palette_for_label(label: str, is_named: bool) -> str:
    """Two-color scheme consistent with other scripts: Generated=blue, Existing=orange."""
    generated_color = "#4C78A8"   # blue
    existing_color = "#F58518"    # orange
    return existing_color if is_named else generated_color


def plot_type_bars(
    category_counts: Dict[str, int],
    named_metric_counts: Dict[str, int],
    out_dir: str,
    broken_axis: bool = False,
    compact: bool = False,
) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)

    # Theme and fonts similar to other analysis scripts
    sns.set_theme(style="darkgrid")
    if compact:
        plt.rcParams.update({
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        })
    else:
        plt.rcParams.update({
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.labelsize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        })

    # Build a combined DataFrame with labels and counts
    rows: List[Dict[str, object]] = []
    for cat, cnt in category_counts.items():
        label = _pretty_category_name(cat)
        rows.append({"label": label, "count": int(cnt), "is_named": False})
    for name, cnt in named_metric_counts.items():
        rows.append({"label": str(name), "count": int(cnt), "is_named": True})

    if not rows:
        # Nothing to plot
        base_png = os.path.join(out_dir, "top_metrics_tally_by_type_barh.png")
        base_pdf = os.path.join(out_dir, "top_metrics_tally_by_type_barh.pdf")
        return base_png, base_pdf

    df = pd.DataFrame(rows)
    # Sort descending by count, then label
    df = df.sort_values(by=["count", "label"], ascending=[False, True]).reset_index(drop=True)

    # Colors per row (two-color scheme)
    colors = [_palette_for_label(label=row["label"], is_named=bool(row["is_named"])) for _, row in df.iterrows()]

    # Legend entries
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#4C78A8", edgecolor="black", label="Generated Metrics"),
        Patch(facecolor="#F58518", edgecolor="black", label="Existing Metrics"),
    ]

    # Dynamic height; use a more compact per-row height if requested
    n = len(df)
    if compact:
        height = max(2.6, 0.36 * n + 0.9)
    else:
        height = max(3.8, 0.45 * n + 1.6)

    if broken_axis and _pretty_category_name("llm_judge") in df["label"].tolist() and len(df) > 1:
        # Determine split based on second-largest value
        llm_label = _pretty_category_name("llm_judge")
        llm_val = float(df.loc[df["label"] == llm_label, "count"].iloc[0])
        second_max = float(df[df["label"] != llm_label]["count"].max())
        use_broken = second_max > 0 and llm_val >= 1.6 * second_max
    else:
        use_broken = False

    if use_broken and BrokenAxes is not None and not compact:
        # Compute split ranges
        left_max = float(second_max * 1.15)
        right_min = float(max(second_max * 0.95, llm_val * 0.8))
        right_max = float(llm_val * 1.05 + 1.0)

        # Use brokenaxes for robust broken x-axis
        fig = plt.figure(figsize=(12.0, height))
        brax = BrokenAxes(xlims=((0, left_max), (right_min, right_max)), hspace=0.05, despine=False)

        bars = brax.barh(df["label"], df["count"], color=colors, edgecolor="black", linewidth=0.5)

        # Invert y on each sub-axes to keep longest on top
        for ax in brax.axs:
            ax.invert_yaxis()
            ax.set_ylabel("")

        brax.set_xlabel("Count")
        brax.set_title("Top Metrics by Type")

        # Annotate counts at bar ends by choosing appropriate sub-axes
        pad_l = max(1, 0.01 * left_max)
        pad_r = max(1, 0.01 * (right_max - right_min))
        # brokenaxes returns a BarContainer directly; iterate its children patches
        patches = getattr(bars, "patches", [])
        for rect in patches:
            width = float(rect.get_width())
            y = rect.get_y() + rect.get_height() / 2.0
            if width <= left_max:
                ax_txt = brax.axs[0]
                x = min(width + pad_l, left_max)
            else:
                ax_txt = brax.axs[-1]
                x = min(width + pad_r, right_max)
            ax_txt.text(x, y, str(int(width)), va="center", fontsize=12)

        # Legend (explicit labels to satisfy brokenaxes legend behavior)
        brax.legend(
            handles=legend_handles,
            labels=["Generated Metrics", "Existing Metrics"],
            loc="lower right",
        )

        # Layout
        fig.subplots_adjust(left=0.25, right=0.98, bottom=0.1, top=0.9)

        # Optional: add wavy break glyphs near the seam for nicer styling
        try:
            left_ax, right_ax = brax.axs[0], brax.axs[-1]
            lp, rp = left_ax.get_position(), right_ax.get_position()
            seam_x = (lp.x1 + rp.x0) / 2.0
            for dy in (0.25, 0.75):
                fig.text(seam_x, dy, "~~", ha="center", va="center", fontsize=16, color="black")
        except Exception:
            pass

    elif use_broken and BrokenAxes is None:
        # Fallback to single axis if brokenaxes is unavailable
        fig, ax = plt.subplots(figsize=(10.8, height))
        bars = ax.barh(df["label"], df["count"], color=colors, edgecolor="black", linewidth=0.5)
        ax.invert_yaxis()
        ax.set_xlabel("Count")
        ax.set_ylabel("")
        ax.set_title("Top Metrics by Type")
        pad = max(1, 0.01 * float(df["count"].max()))
        for rect, cnt in zip(bars, df["count"].tolist()):
            y = rect.get_y() + rect.get_height() / 2.0
            x = rect.get_width() + pad
            ax.text(x, y, str(int(cnt)), va="center", fontsize=12)
        ax.legend(handles=legend_handles, loc="lower right")
        fig.tight_layout()
    else:
        fig_width = 4.2 if compact else 10.8
        fig, ax = plt.subplots(figsize=(fig_width, height))
        bars = ax.barh(df["label"], df["count"], color=colors, edgecolor="black", linewidth=0.5)
        ax.invert_yaxis()  # largest on top
        ax.set_xlabel("Count")
        ax.set_ylabel("")
        ax.set_title("Top Metrics by Type")

        # Annotate counts at bar ends using actual bar positions
        pad = max(1, 0.01 * float(df["count"].max()))
        for rect, cnt in zip(bars, df["count"].tolist()):
            y = rect.get_y() + rect.get_height() / 2.0
            if compact and rect.get_width() > 2:
                # Place label inside the bar to save horizontal space
                x = rect.get_width() - 0.5
                ax.text(x, y, str(int(cnt)), va="center", ha="right", fontsize=10, color="white")
            else:
                x = rect.get_width() + pad
                ax.text(x, y, str(int(cnt)), va="center", fontsize=12)

        # Smaller, in-plot legend for compact layout
        if compact:
            ax.legend(handles=legend_handles, loc="lower right", frameon=True, framealpha=0.8)
        else:
            ax.legend(handles=legend_handles, loc="lower right")

    if compact:
        # Extra left margin for ytick labels to avoid cutoffs in narrow figures
        fig.subplots_adjust(left=0.42, right=0.98, bottom=0.14, top=0.88)
    else:
        fig.tight_layout()
    png_path = os.path.join(out_dir, "top_metrics_tally_by_type_barh.png")
    pdf_path = os.path.join(out_dir, "top_metrics_tally_by_type_barh.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tally Autometrics top_metrics across selected directories and write CSV."
    )
    parser.add_argument(
        "--roots",
        nargs="*",
        default=[
            "/nlp/scr2/nlp/personal-rm/autometrics/results/main_runs/autometrics/qwen/CoGymTravelOutcome_outcomeRating",
            "/nlp/scr2/nlp/personal-rm/autometrics/results/main_runs/autometrics/qwen/EvalGenProduct_grade",
            "/nlp/scr2/nlp/personal-rm/autometrics/results/main_runs/autometrics/qwen/HelpSteer2_helpfulness",
            "/nlp/scr2/nlp/personal-rm/autometrics/results/main_runs/autometrics/qwen/RealHumanEval_accepted",
            "/nlp/scr2/nlp/personal-rm/autometrics/results/main_runs/autometrics/qwen/SimpEval_score",
        ],
        help="Root directories to scan (defaults set to five standard datasets).",
    )
    parser.add_argument(
        "--output",
        default="/nlp/scr2/nlp/personal-rm/autometrics/analysis/stats/top_metrics_tally.csv",
        help="Output CSV file path.",
    )
    parser.add_argument(
        "--output-types",
        default="/nlp/scr2/nlp/personal-rm/autometrics/analysis/stats/top_metrics_tally_by_type.csv",
        help="Output CSV file path for type-only aggregation.",
    )
    parser.add_argument(
        "--fig-dir",
        default="/nlp/scr2/nlp/personal-rm/autometrics/analysis/stats/outputs",
        help="Directory to save figures (PNG/PDF) for type-only bar plot.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="If set, do not generate the horizontal bar plot.",
    )
    parser.add_argument(
        "--broken-axis",
        action="store_true",
        help="Use brokenaxes for a broken x-axis (useful when one bar dominates).",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Generate a compact single-column friendly figure (smaller fonts, tighter layout).",
    )
    parser.add_argument(
        "--model-suffixes",
        default="Qwen3-32B",
        help=(
            "Comma-separated list of model suffixes for LLM judge detection. "
            "Example: 'Qwen3-32B,Llama3-8B,GPT-4o'"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_suffixes = [s.strip() for s in args.model_suffixes.split(",") if s.strip()]
    log_files = find_log_files(args.roots)
    counts = tally_metrics(log_files, model_suffixes)
    write_csv(counts, args.output)
    category_counts, named_metric_counts = aggregate_by_type(counts)
    write_types_csv(category_counts, named_metric_counts, args.output_types)
    png_path, pdf_path = "", ""
    if not args.no_plot:
        png_path, pdf_path = plot_type_bars(
            category_counts,
            named_metric_counts,
            args.fig_dir,
            broken_axis=args.broken_axis,
            compact=args.compact,
        )
    msg = (
        f"Wrote tally to {args.output} ({len(counts)} unique metric entries); "
        f"and type-only tally to {args.output_types} "
        f"({len(category_counts)} categories + {len(named_metric_counts)} named metrics)"
    )
    if png_path and pdf_path:
        msg += f"; figures: {png_path}, {pdf_path}"
    print(msg)


if __name__ == "__main__":
    main()


# Example usage:
# python /nlp/scr2/nlp/personal-rm/autometrics/analysis/stats/analyze_metrics.py --compact