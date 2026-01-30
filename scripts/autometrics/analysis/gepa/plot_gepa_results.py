import os
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Match global plotting/theme settings from other analysis scripts
sns.set_theme(style="darkgrid")
plt.rcParams.update({
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})


DEF_CSV = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "combined_eval_with_ci_manual.csv")
)
DEF_OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs"))

# Consistent colors used across repository scripts
COLOR_MAP: Dict[str, str] = {
    "Verifiable Reward": "#4C78A8",     # blue
    "AutoMetrics": "#F58518",           # orange
    "AutoMetrics Filtered": "#54A24B",  # green
}

LABEL_MAP: Dict[str, str] = {
    # Sheet/group name -> Plot label
    "Standard": "Verifiable Reward",
    "Metric 2": "AutoMetrics",
    "Human": "AutoMetrics Filtered",
}

# Alternate labeling: use the AutoMetrics Filtered series but label it as "AutoMetrics"
LABEL_MAP_ALT: Dict[str, str] = {
    "Standard": "Verifiable Reward",
    "Human": "AutoMetrics",
}


def infer_group_from_path(path_value: str) -> str:
    """Map a CSV path entry to one of the sheet groups: Standard, Metric 2, Human.
    Returns an empty string for rows that are not one of the three groups (e.g., baseline).
    """
    p = str(path_value).lower()
    if p == "baseline":
        return ""
    # Order matters: metric2_human_* must be detected as Human first
    if "metric2_human" in p or "/human_" in p or p.endswith("_human.json"):
        return "Human"
    if "standard" in p:
        return "Standard"
    if "metric2" in p:
        return "Metric 2"
    return ""


def _y_at_x(rollouts: List[int], means: List[float], x: int) -> float:
    """Return the y value (mean) active at integer rollout x using piecewise-constant segments."""
    idx = 0
    for i in range(len(rollouts)):
        if rollouts[i] <= x:
            idx = i
        else:
            break
    return float(means[idx])


def build_polyline_points(
    rollouts: List[int],
    means: List[float],
    x_end: int,
    interval: int,
    baseline_val: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct polyline points sampled at fixed intervals and change points.

    Special handling for rollout 0: if baseline_val is provided, force the point at x=0
    to sit on the baseline, and let the first divergence happen at x=interval.

    Returns:
      - xs_all, ys_all: full polyline points (including change points)
      - xs_mark, ys_mark: marker points at fixed intervals only
    """
    if not rollouts:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Ensure increasing and within bounds
    rr: List[int] = []
    mm: List[float] = []
    for r, m in zip(rollouts, means):
        r_i = int(max(0, min(int(r), x_end)))
        if not rr or r_i > rr[-1]:
            rr.append(r_i)
            mm.append(float(m))
        else:
            rr[-1] = r_i
            mm[-1] = float(m)

    # Start from 0 to enable baseline anchoring
    start_x = 0

    # Interval markers (every interval rollouts), include 0
    xs_mark: List[int] = list(range(0, x_end + 1, interval))

    # Include change points and endpoint in the sampling set
    xs_set = set(xs_mark) | set(rr) | {x_end}
    xs_all_sorted = sorted([x for x in xs_set if x >= start_x])

    # Compute y with baseline override at x==0 (and only there)
    ys_all: List[float] = []
    for x in xs_all_sorted:
        if x == 0 and baseline_val is not None:
            ys_all.append(float(baseline_val))
        else:
            ys_all.append(_y_at_x(rr, mm, x))

    ys_mark: List[float] = []
    for x in xs_mark:
        if x < start_x:
            continue
        if x == 0 and baseline_val is not None:
            ys_mark.append(float(baseline_val))
        else:
            ys_mark.append(_y_at_x(rr, mm, x))

    return (
        np.asarray(xs_all_sorted, dtype=float),
        np.asarray(ys_all, dtype=float),
        np.asarray([x for x in xs_mark if x >= start_x], dtype=float),
        np.asarray(ys_mark, dtype=float),
    )


def plot_gepa(
    csv_path: str,
    out_dir: str,
    x_end: int = 3325,
    y_min: Optional[float] = 0.5,
    y_max: Optional[float] = 0.8,
) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # Expect columns: path, avg, ci95_halfwidth, rollout
    required = {"path", "avg", "ci95_halfwidth", "rollout"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Extract baseline for dashed horizontal line
    baseline_val: Optional[float] = None
    if (df["path"].astype(str) == "baseline").any():
        baseline_val = float(df.loc[df["path"].astype(str) == "baseline", "avg"].iloc[0])

    # Build group -> rows
    df = df.copy()
    df["group"] = df["path"].apply(infer_group_from_path)
    df = df[df["group"] != ""].reset_index(drop=True)

    # Sort by rollout within groups
    df["rollout"] = df["rollout"].astype(int)
    df = df.sort_values(["group", "rollout", "path"]).reset_index(drop=True)

    # Map group names to plot labels
    df["label"] = df["group"].map(LABEL_MAP)

    # Prepare figure
    fig, ax = plt.subplots(figsize=(8.8, 5.0))

    line_width = 1.6
    marker_edge_width = line_width + 0.2   # slightly thicker than the line
    marker_size_pts = 2.0                   # slightly smaller markers

    # Plot each label as polyline with interval markers
    for group_name, gdf in df.groupby("group"):
        label = LABEL_MAP.get(group_name, group_name)
        color = COLOR_MAP.get(label, None)

        rollouts = gdf["rollout"].astype(int).tolist()
        means = gdf["avg"].astype(float).tolist()

        xs_line, ys_line, xs_mark, ys_mark = build_polyline_points(
            rollouts, means, x_end=x_end, interval=25, baseline_val=baseline_val
        )
        if xs_line.size == 0:
            continue

        # Polyline
        ax.plot(
            xs_line, ys_line,
            color=color if color else "C0",
            linewidth=line_width,
            label=label,
        )

        # Interval markers every 25 rollouts (solid small circles)
        if xs_mark.size > 0:
            ax.plot(
                xs_mark, ys_mark,
                linestyle='None',
                marker='o',
                markersize=marker_size_pts,
                markerfacecolor=color if color else "C0",
                markeredgecolor=color if color else "C0",
                markeredgewidth=marker_edge_width,
                zorder=3,
            )

    # Baseline dashed line
    if baseline_val is not None:
        ax.hlines(
            baseline_val,
            xmin=0,
            xmax=x_end,
            colors='black',
            linestyles='--',
            linewidth=1.2,
            label="Unoptimized Baseline",
        )

    # Axes and labels
    pad_right = max(10, int(0.02 * x_end))
    ax.set_xlim(0, x_end + pad_right)
    if y_min is not None and y_max is not None:
        ax.set_ylim(float(y_min), float(y_max))
    else:
        ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Number of Rollouts")
    ax.set_ylabel("Average Pass^1 Test Score")
    ax.set_title("Tau Bench, Qwen3 32B")

    # Legend in lower right to match other plots
    ax.legend(loc='lower right')

    # Reduce outer whitespace, keep a small right padding
    fig.subplots_adjust(left=0.1, right=0.985, bottom=0.12, top=0.92)

    png_path = os.path.join(out_dir, "gepa_rollout_line_plot.png")
    pdf_path = os.path.join(out_dir, "gepa_rollout_line_plot.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def plot_gepa_filtered_as_autometrics(
    csv_path: str,
    out_dir: str,
    x_end: int = 3325,
    y_min: Optional[float] = 0.5,
    y_max: Optional[float] = 0.8,
) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # Expect columns: path, avg, ci95_halfwidth, rollout
    required = {"path", "avg", "ci95_halfwidth", "rollout"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Extract baseline for dashed horizontal line
    baseline_val: Optional[float] = None
    if (df["path"].astype(str) == "baseline").any():
        baseline_val = float(df.loc[df["path"].astype(str) == "baseline", "avg"].iloc[0])

    # Build group -> rows
    df = df.copy()
    df["group"] = df["path"].apply(infer_group_from_path)
    # Keep only Standard and Human (exclude unfiltered Metric 2)
    df = df[df["group"].isin(["Standard", "Human"])].reset_index(drop=True)

    # Sort by rollout within groups
    df["rollout"] = df["rollout"].astype(int)
    df = df.sort_values(["group", "rollout", "path"]).reset_index(drop=True)

    # Map group names to alternate plot labels
    df["label"] = df["group"].map(LABEL_MAP_ALT)

    # Prepare figure
    fig, ax = plt.subplots(figsize=(8.8, 5.0))

    line_width = 1.6
    marker_edge_width = line_width + 0.2
    marker_size_pts = 2.0

    # Plot each label as polyline with interval markers
    for group_name, gdf in df.groupby("group"):
        label = LABEL_MAP_ALT.get(group_name, group_name)
        color = COLOR_MAP.get(label, None)

        rollouts = gdf["rollout"].astype(int).tolist()
        means = gdf["avg"].astype(float).tolist()

        xs_line, ys_line, xs_mark, ys_mark = build_polyline_points(
            rollouts, means, x_end=x_end, interval=25, baseline_val=baseline_val
        )
        if xs_line.size == 0:
            continue

        # Polyline
        ax.plot(
            xs_line, ys_line,
            color=color if color else "C0",
            linewidth=line_width,
            label=label,
        )

        # Interval markers every 25 rollouts (solid small circles)
        if xs_mark.size > 0:
            ax.plot(
                xs_mark, ys_mark,
                linestyle='None',
                marker='o',
                markersize=marker_size_pts,
                markerfacecolor=color if color else "C0",
                markeredgecolor=color if color else "C0",
                markeredgewidth=marker_edge_width,
                zorder=3,
            )

    # Baseline dashed line
    if baseline_val is not None:
        ax.hlines(
            baseline_val,
            xmin=0,
            xmax=x_end,
            colors='black',
            linestyles='--',
            linewidth=1.2,
            label="Unoptimized Baseline",
        )

    # Axes and labels
    pad_right = max(10, int(0.02 * x_end))
    ax.set_xlim(0, x_end + pad_right)
    if y_min is not None and y_max is not None:
        ax.set_ylim(float(y_min), float(y_max))
    else:
        ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Number of Rollouts")
    ax.set_ylabel("Average Pass^1 Test Score")
    ax.set_title("Tau Bench, Qwen3 32B")

    # Legend in lower right to match other plots
    ax.legend(loc='lower right')

    # Reduce outer whitespace, keep a small right padding
    fig.subplots_adjust(left=0.1, right=0.985, bottom=0.12, top=0.92)

    png_path = os.path.join(out_dir, "gepa_rollout_line_plot_filtered_as_autometrics.png")
    pdf_path = os.path.join(out_dir, "gepa_rollout_line_plot_filtered_as_autometrics.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GEPA rollout curves with interval markers and diagonal connectors.")
    parser.add_argument(
        "--csv",
        default=DEF_CSV,
        help="Path to combined_eval_with_ci_manual.csv",
    )
    parser.add_argument(
        "--out-dir",
        default=DEF_OUT_DIR,
        help="Directory to save PNG/PDF outputs",
    )
    parser.add_argument(
        "--x-end",
        type=int,
        default=3325,
        help="Maximum number of rollouts on the x-axis (default: 3325)",
    )
    parser.add_argument(
        "--y-min",
        type=float,
        default=0.5,
        help="Lower bound of y-axis (default: 0.5). Use together with --y-max.",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=0.8,
        help="Upper bound of y-axis (default: 0.9). Use together with --y-min.",
    )
    args = parser.parse_args()

    # Validate y bounds
    y_min: Optional[float] = args.y_min
    y_max: Optional[float] = args.y_max
    if y_min is not None and y_max is not None:
        if not (0.0 <= y_min < y_max <= 1.0):
            raise ValueError(f"Invalid y range: y_min={y_min}, y_max={y_max}. Must satisfy 0.0 <= y_min < y_max <= 1.0")

    png_path, pdf_path = plot_gepa(args.csv, args.out_dir, x_end=args.x_end, y_min=y_min, y_max=y_max)
    print(f"Wrote figures (original): {png_path}, {pdf_path}")

    png_path2, pdf_path2 = plot_gepa_filtered_as_autometrics(args.csv, args.out_dir, x_end=args.x_end, y_min=y_min, y_max=y_max)
    print(f"Wrote figures (filtered-as-AutoMetrics): {png_path2}, {pdf_path2}")


if __name__ == "__main__":
    main()

# python /nlp/scr2/nlp/personal-rm/autometrics/analysis/gepa/plot_gepa_results.py --csv /nlp/scr2/nlp/personal-rm/autometrics/analysis/gepa/combined_eval_with_ci_manual.csv --out-dir /nlp/scr2/nlp/personal-rm/autometrics/analysis/gepa/outputs --x-end 2000