#!/usr/bin/env python3
import argparse
import json
import math
import sys
from typing import List, Dict

import numpy as np
from scipy import stats


def compute_statistics(values: List[float], alpha: float = 0.05) -> Dict[str, float]:
    valid_values = [v for v in values if not (v is None or (isinstance(v, float) and math.isnan(v)))]
    n = len(valid_values)

    if n == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "ci_range": float("nan"),
            "num_successful_runs": 0,
        }

    mean_val = float(np.mean(valid_values))

    if n == 1:
        return {
            "mean": mean_val,
            "std": 0.0,
            "ci_lower": mean_val,
            "ci_upper": mean_val,
            "ci_range": 0.0,
            "num_successful_runs": n,
        }

    std_val = float(np.std(valid_values, ddof=1))
    t_value = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    margin_error = t_value * std_val / math.sqrt(n)

    return {
        "mean": mean_val,
        "std": std_val,
        "ci_lower": mean_val - margin_error,
        "ci_upper": mean_val + margin_error,
        "ci_range": margin_error,  # half-width
        "num_successful_runs": n,
    }


def parse_numbers_from_text(text: str) -> List[float]:
    tokens = []
    for line in text.strip().splitlines():
        for part in line.replace(",", " ").split():
            tokens.append(part)
    numbers = []
    for tok in tokens:
        try:
            numbers.append(float(tok))
        except ValueError:
            pass
    return numbers


def main():
    parser = argparse.ArgumentParser(
        description="Compute 95% confidence interval (t-distribution) from a list of floats."
    )
    parser.add_argument("values", nargs="*", type=float, help="Values as positional floats")
    parser.add_argument("--file", "-f", type=str, help="Path to file with numbers (whitespace or comma-separated)")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level (default: 0.05 for 95% CI)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--precision", type=int, default=6, help="Decimal places in text output")
    args = parser.parse_args()

    values: List[float] = []
    if args.file:
        with open(args.file, "r", encoding="utf-8") as fh:
            values.extend(parse_numbers_from_text(fh.read()))
    if args.values:
        values.extend(args.values)
    if not values and not sys.stdin.isatty():
        values.extend(parse_numbers_from_text(sys.stdin.read()))

    stats_dict = compute_statistics(values, alpha=args.alpha)

    if args.json:
        print(json.dumps(stats_dict, indent=2))
    else:
        p = args.precision
        def fmt(x):
            return "nan" if isinstance(x, float) and math.isnan(x) else f"{x:.{p}f}"
        print(f"n: {stats_dict['num_successful_runs']}")
        print(f"mean: {fmt(stats_dict['mean'])}")
        print(f"std: {fmt(stats_dict['std'])}")
        print(f"ci_lower: {fmt(stats_dict['ci_lower'])}")
        print(f"ci_upper: {fmt(stats_dict['ci_upper'])}")
        print(f"ci_half_width: {fmt(stats_dict['ci_range'])}")
        # LaTeX-friendly single-line summary (defaults: mean 3 dp, CI half-width 2 dp)
        print(f"{stats_dict['mean']:.3f} {{\\scriptsize $\\pm$ {stats_dict['ci_range']:.2f}}}")


if __name__ == "__main__":
    main()