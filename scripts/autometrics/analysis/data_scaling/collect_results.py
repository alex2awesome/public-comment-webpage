import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def infer_output_dir_from_input(input_dir: Path) -> Path:
    """Infer the output directory as results/data_scaling/autometrics/{model}.

    We detect the model name from the segment after "autometrics" in the input path
    (e.g., .../results/data_scaling/autometrics/qwen → model = "qwen"). We also anchor
    under the same top-level "results" directory detected from the input.
    """
    parts = list(input_dir.parts)
    try:
        autometrics_idx = parts.index("autometrics")
        model_name = parts[autometrics_idx + 1]
    except (ValueError, IndexError):
        # Fallback: use last directory name as model
        model_name = input_dir.name

    # Find the base "results" directory to anchor outputs beside inputs
    try:
        results_idx = parts.index("results")
        results_base = Path(*parts[: results_idx + 1])
    except ValueError:
        # If not found, default to repo-root-ish relative path
        results_base = Path("results")

    return results_base / "data_scaling" / "autometrics" / model_name


def parse_train_size_from_path(path: Path) -> Optional[int]:
    """Attempt to parse the train size from any path segment like sz{N}_..."""
    size_pattern = re.compile(r"^.*_sz(\d+)_.*$")
    for part in path.parts[::-1]:
        match = size_pattern.match(part)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
    return None


def parse_model_label(json_data: Dict, path: Path) -> str:
    """Return "gen_only" or "full".

    Priority:
    1) Use json field "generated_only" when present.
    2) Fallback to directory name hints: contains "genonly" → gen_only, contains "fullbank" → full
    3) Default to "full".
    """
    generated_only = json_data.get("generated_only")
    if isinstance(generated_only, bool):
        return "gen_only" if generated_only else "full"

    path_str_lower = str(path).lower()
    if "genonly" in path_str_lower:
        return "gen_only"
    if "fullbank" in path_str_lower:
        return "full"
    return "full"


def infer_dataset_name(json_data: Dict, input_dir: Path, log_path: Path) -> str:
    """Infer dataset name.

    Prefer json field "dataset_name" if present. Otherwise, use the first path
    component under the input_dir root (e.g., SimpEval_score, RealHumanEval_accepted).
    """
    dataset_name = json_data.get("dataset_name")
    if isinstance(dataset_name, str) and dataset_name.strip():
        return dataset_name.strip()

    try:
        rel = log_path.parent.relative_to(input_dir)
        # The top-level folder under input_dir corresponds to dataset grouping
        first = rel.parts[0] if rel.parts else log_path.parent.name
        return first
    except Exception:
        return log_path.parent.name


def collect_rows(input_dir: Path) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Walk input_dir, parse all log_*.json files, and return rows for three metrics.

    Returns three lists of rows (pearson_rows, spearman_rows, kendall_rows).
    Each row: {"dataset", "train_size", "model", "correlation"}
    """
    pearson_rows: List[Dict] = []
    spearman_rows: List[Dict] = []
    kendall_rows: List[Dict] = []

    for root, _dirs, files in os.walk(input_dir):
        for fname in files:
            if not (fname.startswith("log_") and fname.endswith(".json")):
                continue

            fpath = Path(root) / fname
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                # Skip malformed or unreadable files
                continue

            # Extract fields with fallbacks
            dataset = infer_dataset_name(data, input_dir, fpath)

            train_size = data.get("train_size")
            if not isinstance(train_size, int):
                train_size = parse_train_size_from_path(fpath)

            model_label = parse_model_label(data, fpath)

            scores = data.get("test_scores") or {}

            def add_row(metric_name: str, rows: List[Dict]):
                val = scores.get(metric_name)
                if isinstance(val, (int, float)):
                    rows.append(
                        {
                            "dataset": dataset,
                            "train_size": train_size,
                            "model": model_label,
                            "correlation": val,
                        }
                    )

            add_row("pearson", pearson_rows)
            add_row("spearman", spearman_rows)
            add_row("kendall", kendall_rows)

    return pearson_rows, spearman_rows, kendall_rows


def write_csv(rows: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "train_size", "model", "correlation"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect correlation results into CSVs (pearson/spearman/kendall)."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help=(
            "Path to the model directory under results/data_scaling/autometrics/ (e.g., "
            ".../results/data_scaling/autometrics/qwen)"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Optional path to write outputs. If omitted, inferred as "
            "results/data_scaling/autometrics/{model} beside the input results."
        ),
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir.resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    out_dir: Path = (args.out_dir.resolve() if args.out_dir else infer_output_dir_from_input(input_dir))

    pearson_rows, spearman_rows, kendall_rows = collect_rows(input_dir)

    # Sort for stable output
    def sort_key(r: Dict) -> Tuple:
        return (
            str(r.get("dataset")),
            int(r.get("train_size") or -1),
            str(r.get("model")),
        )

    pearson_rows.sort(key=sort_key)
    spearman_rows.sort(key=sort_key)
    kendall_rows.sort(key=sort_key)

    pearson_csv = out_dir / "pearson.csv"
    spearman_csv = out_dir / "spearman.csv"
    kendall_csv = out_dir / "kendall.csv"

    write_csv(pearson_rows, pearson_csv)
    write_csv(spearman_rows, spearman_csv)
    write_csv(kendall_rows, kendall_csv)

    print(f"Wrote {len(pearson_rows)} pearson rows to: {pearson_csv}")
    print(f"Wrote {len(spearman_rows)} spearman rows to: {spearman_csv}")
    print(f"Wrote {len(kendall_rows)} kendall rows to: {kendall_csv}")


if __name__ == "__main__":
    main()


# Example usage:
# python analysis/data_scaling/collect_results.py --input-dir /nlp/scr2/nlp/personal-rm/autometrics/results/data_scaling/autometrics/qwen

