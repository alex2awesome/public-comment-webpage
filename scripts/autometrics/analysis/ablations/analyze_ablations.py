#!/usr/bin/env python3
"""
analysis/ablations/analyze_ablations.py

Aggregate ablation results across seeds and produce LaTeX tables.

Features:
- Recursively scan a results directory (e.g., results/ablations/qwen_remote).
- Parse .txt and .csv result files to extract correlations for a selected
  correlation type (kendall|spearman|pearson; default kendall).
- Aggregate across seeds/trials; compute mean and 95% CI using sample std dev.
- Fill a main LaTeX table with a fixed set of dataset/target columns.
- Emit an auxiliary LaTeX table for any remaining dataset/target pairs.

Notes on CI: we use 95% CI = 1.96 * (sample_std / sqrt(n)), where n
is the number of trials aggregated for a given (method, dataset, target).
If n == 1, we render ± 0.000 for layout preview (and log a warning).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ───────────────────────── SETTINGS ───────────────────────── #

# The main table fixed columns (dataset, target) and display names.
MAIN_COLUMNS: List[Tuple[str, str]] = [
    ("SimpEval", "score"),
    ("Primock57", "time_sec"),
    ("HelpSteer2", "helpfulness"),
    ("EvalGenProduct", "grade"),
    ("RealHumanEval", "accepted"),
    ("CoGymTravel", "outcomeRating"),
]

# Normalization for dataset names appearing in paths/files.
DATASET_SYNONYMS: Dict[str, List[str]] = {
    "SimpEval": ["simpeval", "simp_eval", "simp-eval"],
    "Primock57": ["primock57", "primock", "primock-57"],
    "HelpSteer2": ["helpsteer2", "helpsteer", "help_steer", "help-steer"],
    "EvalGenProduct": ["evalgenproduct", "evalgen_product", "evalgen-product", "evalgen"],
    "RealHumanEval": ["realhumaneval", "rhe", "real-human-eval", "real_human_eval"],
    "CoGymTravel": [
        "cogymtravel",
        "cogym_travel",
        "cogym-travel",
        "cogym",
        "cogymtraveloutcome",
        "cogym_travel_outcome",
        "cogym-travel-outcome",
    ],
}

# Normalization for target/measure names.
TARGET_SYNONYMS: Dict[str, List[str]] = {
    "score": ["score", "scores"],
    "time_sec": ["time_sec", "time", "time_s", "seconds", "runtime_s"],
    "helpfulness": ["helpfulness", "helpful", "helpfulness_score"],
    # Put outcomeRating BEFORE grade to avoid generic 'rating' stealing matches
    "outcomeRating": ["outcomerating", "outcome_rating", "outcome", "outcome_score"],
    "grade": ["grade", "grades", "rating", "ratings"],
    "accepted": ["accepted", "accept", "accept_rate", "acceptance"],
}

# Ablation rows grouped by section; each row maps exact directory names to display names.
# Order matters: more specific patterns should come before more general ones.
ABLATION_GROUPS: List[Tuple[str, List[Tuple[str, List[str]]]]] = [
    (
        "MetricBank Ablations",
        [
            ("Existing Metrics Only", ["existing_only"]),
            ("Generated Metrics Only", ["generated_only"]),
            ("Full MetricBank", ["full_k30", "full_k30_n5"]),
        ],
    ),
    (
        "Retrieval Ablations",
        [
            ("Retrieve k=5", ["full_k5"]),
            ("Retrieve k=10", ["full_k10"]),
            ("No Metric Cards (k=20)", ["full_k20_desc"]),
            ("Retrieve k=20", ["full_k20"]),
            ("Retrieve k=30", ["full_k30", "full_k30_n5"]),
        ],
    ),
    (
        "Regression Ablations",
        [
            ("No Regression (n=1)", ["full_k30_n1"]),
            ("Regress n=3", ["full_k30_n3"]),
            ("Regress n=5", ["full_k30_n5", "full_k30"]),
            ("Regress n=10", ["full_k30_n10"]),
            ("Regress n=20", ["full_k30_n20"]),
        ],
    ),
]

# For labeling/metadata.
def extract_model_name(results_path: Path) -> Tuple[str, str]:
    path_str = str(results_path)
    if "qwen" in path_str.lower():
        return "Qwen3 32B", "qwen3_32b"
    if "gpt4o" in path_str.lower() or "gpt-4o" in path_str.lower():
        return "GPT-4o Mini", "gpt4o_mini"
    # Fallback to last directory name
    tail = results_path.name.lower().replace("-", "_").replace(" ", "_") or "unknown"
    return tail, tail


# ───────────────────────── DATA STRUCTURES ───────────────────────── #

@dataclass
class ResultRecord:
    method_row: Optional[str]
    group_name: Optional[str]
    dataset: str
    target: str
    corr_type: str
    value: float
    seed: Optional[str]
    source_file: Path


# ───────────────────────── UTILITIES ───────────────────────── #

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def match_dataset(name: str) -> Optional[str]:
    key = normalize_text(name)
    for canonical, syns in DATASET_SYNONYMS.items():
        for s in [canonical] + syns:
            if s in key:
                return canonical
    return None


def match_target(name: str) -> Optional[str]:
    key = normalize_text(name)
    tokens = set([t for t in re.split(r"[^a-z0-9]+", key) if t])

    # Prefer higher token overlap; on tie, prefer shorter candidate (fewer extra tokens)
    best_score: int = 0
    best_len: int = 10**9
    best_canonical: Optional[str] = None

    for canonical, syns in TARGET_SYNONYMS.items():
        candidates = [normalize_text(canonical)] + [normalize_text(s) for s in syns]
        for cand in set(candidates):
            cand_tokens = [t for t in re.split(r"[^a-z0-9]+", cand) if t]
            if not cand_tokens:
                continue
            score = sum(1 for t in cand_tokens if t in tokens)
            if score > 0:
                cand_len = len(cand)
                if score > best_score or (score == best_score and cand_len < best_len):
                    best_score = score
                    best_len = cand_len
                    best_canonical = canonical
    if best_canonical is not None:
        return best_canonical

    # Conservative fallback: require full-word containment (space-padded) to reduce accidental substring matches
    spaced_key = f" {key} "
    for canonical, syns in TARGET_SYNONYMS.items():
        candidates = [normalize_text(canonical)] + [normalize_text(s) for s in syns]
        for cand in sorted(set(candidates), key=len):  # prefer shorter
            if not cand:
                continue
            if f" {cand} " in spaced_key:
                return canonical
    return None


def detect_seed_from_path(path: Path) -> Optional[str]:
    s = str(path)
    m = re.search(r"seed[-_]?([0-9]+)", s, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"(?:^|[^a-zA-Z])s([0-9]+)(?:[^0-9]|$)", s)
    if m:
        return m.group(1)
    return None


def map_path_to_method_rows(path: Path) -> List[Tuple[str, str]]:
    """Return all (group_name, method_row_name) pairs whose exact directory name matches the parent dir."""
    parent_dir = path.parent.name
    matches: List[Tuple[str, str]] = []
    for group_name, rows in ABLATION_GROUPS:
        for method_row_name, dir_names in rows:
            for dir_name in dir_names:
                if parent_dir == dir_name:
                    matches.append((group_name, method_row_name))
    return matches

# Backwards-compatible: keep single-match finder (first match if multiple)
def map_path_to_method_row(path: Path) -> Tuple[Optional[str], Optional[str]]:
    """Return (group_name, method_row_name) if exact directory name is found in parent directory, else (None, None)."""
    all_matches = map_path_to_method_rows(path)
    if all_matches:
        g, m = all_matches[0]
        return g, m
    return None, None


def extract_dataset_target_from_path(path: Path) -> Tuple[Optional[str], Optional[str]]:
    # Work with structured parts (directories and filename)
    path_parts = list(path.parts)

    # Helper to normalize and test a string chunk for dataset/target
    def find_in_chunk(chunk: str) -> Tuple[Optional[str], Optional[str]]:
        tokens = [t for t in re.split(r"[\\/_.-]", chunk) if t]
        joined = " ".join(tokens)
        return match_dataset(joined), match_target(joined)

    dataset: Optional[str] = None
    target: Optional[str] = None

    # 1) Look for a single directory part encoding both dataset and target (e.g., EvalGenProduct_grade)
    for part in path_parts[:-1]:  # exclude filename
        ds, tg = find_in_chunk(part)
        if ds and tg:
            dataset = dataset or ds
            target = target or tg
            if dataset and target:
                break

    # 2) If still missing, scan directories from top to bottom and pick first matches
    if not dataset or not target:
        for part in path_parts[:-1]:  # exclude filename
            ds, tg = find_in_chunk(part)
            if not dataset and ds:
                dataset = ds
            if not target and tg:
                target = tg
            if dataset and target:
                break

    # 3) As a last resort, allow picking from filename, but only if not already set
    if not dataset or not target:
        ds, tg = find_in_chunk(path_parts[-1])
        if not dataset and ds:
            dataset = ds
        if not target and tg:
            target = tg

    # 4) If target still missing, default to canonical main-table targets for known datasets
    if dataset and not target:
        default_targets = {
            "SimpEval": "score",
            "Primock57": "time_sec",
            "HelpSteer2": "helpfulness",
            "EvalGenProduct": "grade",
            "RealHumanEval": "accepted",
            "CoGymTravel": "outcomeRating",
        }
        if dataset in default_targets:
            target = default_targets[dataset]

    return dataset, target


def parse_txt_file(path: Path, corr_type: str) -> List[ResultRecord]:
    records: List[ResultRecord] = []
    text = path.read_text(errors="ignore")
    matches = map_path_to_method_rows(path)
    if not matches:
        matches = [(None, None)]
    seed = detect_seed_from_path(path)

    # Try to extract (dataset, target) from path first; else from content.
    path_dataset, path_target = extract_dataset_target_from_path(path)

    # Regex patterns for correlations and potential dataset/target lines.
    corr_patterns = [
        re.compile(rf"\b{corr_type}\b[\w\s'`-]*?:\s*(-?\d+\.\d+|-?\d+)", re.IGNORECASE),
        re.compile(rf"\b{corr_type}[\w'`\s-]*?=\s*(-?\d+\.\d+|-?\d+)", re.IGNORECASE),
        re.compile(rf"\b{corr_type}[\w'`\s-]*?\(.*?\):\s*(-?\d+\.\d+|-?\d+)", re.IGNORECASE),
    ]

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # If dataset/target known from path, try to parse correlation values. Do not mix corr types.
    if path_dataset and path_target:
        found_any = False
        for ln in lines:
            for pat in corr_patterns:
                m = pat.search(ln)
                if m:
                    try:
                        val = float(m.group(1))
                    except Exception:
                        continue
                    for group_name, method_row in matches:
                        records.append(
                            ResultRecord(
                                method_row=method_row,
                                group_name=group_name,
                                dataset=path_dataset,
                                target=path_target,
                                corr_type=corr_type,
                                value=val,
                                seed=seed,
                                source_file=path,
                            )
                        )
                    found_any = True
        if not found_any:
            # Fallback: bare numeric value(s) with no corr type keyword — only if filename indicates corr_type
            # Example: score_kendall_44.txt vs score_pearson_44.txt
            stem_toks = [t for t in re.split(r"[^a-z0-9]+", path.stem.lower()) if t]
            if corr_type.lower() not in stem_toks:
                return records
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
            for tok in nums:
                try:
                    val = float(tok)
                except Exception:
                    continue
                for group_name, method_row in matches:
                    records.append(
                        ResultRecord(
                            method_row=method_row,
                            group_name=group_name,
                            dataset=path_dataset,
                            target=path_target,
                            corr_type=corr_type,
                            value=val,
                            seed=seed,
                            source_file=path,
                        )
                    )
                # Only take the first number to avoid accidental multiples
                break
        return records

    # Otherwise, search for lines that include a dataset and target nearby.
    for i, ln in enumerate(lines):
        ds = match_dataset(ln) or (match_dataset(lines[i - 1]) if i > 0 else None)
        tg = match_target(ln) or (match_target(lines[i + 1]) if i + 1 < len(lines) else None)
        if not ds or not tg:
            continue
        window = " ".join(lines[max(0, i - 2) : min(len(lines), i + 3)])
        for pat in corr_patterns:
            m = pat.search(window)
            if m:
                try:
                    val = float(m.group(1))
                except Exception:
                    continue
                for group_name, method_row in matches:
                    records.append(
                        ResultRecord(
                            method_row=method_row,
                            group_name=group_name,
                            dataset=ds,
                            target=tg,
                            corr_type=corr_type,
                            value=val,
                            seed=seed,
                            source_file=path,
                        )
                    )
                break

    return records


def parse_csv_file(path: Path, corr_type: str) -> List[ResultRecord]:
    records: List[ResultRecord] = []
    matches = map_path_to_method_rows(path)
    if not matches:
        matches = [(None, None)]
    seed = detect_seed_from_path(path)

    try:
        df = pd.read_csv(path)
    except Exception as e:
        logging.warning(f"Failed to read CSV {path}: {e}")
        return records

    # Try to normalize columns
    cols = {c.lower(): c for c in df.columns}

    def get_col(*cands: str) -> Optional[str]:
        for c in cands:
            if c in cols:
                return cols[c]
        return None

    dataset_col = get_col("dataset")
    target_col = get_col("measure", "target", "label", "field")

    # Correlation value may be in a dedicated column or generic 'value'
    corr_col = get_col(corr_type)
    if not corr_col:
        corr_col = get_col("corr", "correlation", f"{corr_type}_corr", "value")

    # Some exports may include all corr types in a single 'mean_±_ci' field; skip those
    if not dataset_col or not target_col or not corr_col:
        # Fallback: attempt row-wise dict parsing
        for _, row in df.iterrows():
            try:
                ds_guess, tg_guess = extract_dataset_target_from_path(path)
                ds = row.get(dataset_col, ds_guess) if dataset_col else ds_guess
                tg = row.get(target_col, tg_guess) if target_col else tg_guess
                if not ds or not tg:
                    continue
                val = None
                if corr_type in row:
                    val = float(row[corr_type])
                elif "correlation" in row:
                    val = float(row["correlation"])  # type: ignore[index]
                elif "value" in row:
                    val = float(row["value"])  # type: ignore[index]
                else:
                    continue
            except Exception:
                continue
            for group_name, method_row in matches:
                records.append(
                    ResultRecord(
                        method_row=method_row,
                        group_name=group_name,
                        dataset=str(ds),
                        target=str(tg),
                        corr_type=corr_type,
                        value=float(val),
                        seed=seed,
                        source_file=path,
                    )
                )
        return records

    for _, row in df.iterrows():
        try:
            ds = str(row[dataset_col])
            tg = str(row[target_col])
            val = float(row[corr_col])
        except Exception:
            continue
        # Normalize dataset/target
        ds_norm = match_dataset(ds) or ds
        tg_norm = match_target(tg) or tg
        for group_name, method_row in matches:
            records.append(
                ResultRecord(
                    method_row=method_row,
                    group_name=group_name,
                    dataset=ds_norm,
                    target=tg_norm,
                    corr_type=corr_type,
                    value=val,
                    seed=seed,
                    source_file=path,
                )
            )

    return records


def scan_results(results_root: Path, corr_type: str) -> List[ResultRecord]:
    def name_tokens(p: Path) -> List[str]:
        stem = p.stem.lower()
        return [t for t in re.split(r"[^a-z0-9]+", stem) if t]

    # Collect and filter files so we only parse the requested correlation type.
    candidate_paths: List[Path] = []
    for ext in ("*.txt", "*.csv"):
        candidate_paths.extend(results_root.rglob(ext))

    paths: List[Path] = []
    for p in candidate_paths:
        if p.suffix.lower() == ".txt":
            # Only include .txt files if their name explicitly contains the corr type token
            toks = name_tokens(p)
            if corr_type.lower() in toks:
                paths.append(p)
        else:
            # CSVs may contain multiple corr columns; include and let parser pick the right column
            paths.append(p)

    # Enforce resized-only directories for specific dataset/target pairs by excluding
    # files under the non-resized directories. This ensures we pull from *_resized only.
    # Targets:
    # - EvalGenProduct_grade -> include only EvalGenProduct_grade_resized
    # - CoGymTravelOutcome_outcomeRating -> include only CoGymTravelOutcome_outcomeRating_resized
    exclude_non_resized_dirs = {"evalgenproduct_grade", "cogymtraveloutcome_outcomerating"}
    if exclude_non_resized_dirs:
        filtered_paths: List[Path] = []
        for p in paths:
            parts = {part.lower() for part in p.parts}
            # Drop any file that lives under a non-resized directory name
            if any(dir_name in parts for dir_name in exclude_non_resized_dirs):
                continue
            filtered_paths.append(p)
        paths = filtered_paths

    logging.info(f"Scanning {len(paths)} files under {results_root}")
    all_records: List[ResultRecord] = []
    for p in paths:
        try:
            if p.suffix.lower() == ".txt":
                recs = parse_txt_file(p, corr_type)
            else:
                recs = parse_csv_file(p, corr_type)
            if recs:
                all_records.extend(recs)
            else:
                logging.debug(f"No parseable records found in {p}")
        except Exception as e:
            logging.warning(f"Failed to parse {p}: {e}")

    logging.info(f"Parsed {len(all_records)} correlation records")
    return all_records


def compute_mean_ci(values: List[float]) -> Tuple[float, Optional[float]]:
    if not values:
        return math.nan, None
    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    if len(arr) < 2:
        # Layout preview: render ± 0.000 when only one run exists
        logging.warning("Only one run found for a cell; rendering ± 0.000 as placeholder CI.")
        return mean, 0.0
    std = float(np.std(arr, ddof=1))
    ci = 1.96 * std / math.sqrt(len(arr))
    return mean, ci


def format_value(mean: float, ci: Optional[float]) -> str:
    if math.isnan(mean):
        return "---"
    if ci is None:
        return f"{mean:.2f}"
    return f"{mean:.3f} {{\\scriptsize $\\pm$ {ci:.2f}}}"


def build_aggregates(records: List[ResultRecord]) -> Dict[str, Dict[Tuple[str, str], List[float]]]:
    """Aggregate values keyed by method_row over (dataset,target). Unknown methods are grouped under their own inferred names for the auxiliary table."""
    aggregates: Dict[str, Dict[Tuple[str, str], List[float]]] = {}

    def key_for_record(r: ResultRecord) -> str:
        if r.method_row:
            return r.method_row
        # Fallback method name: derive from nearest parent directory name
        parent = r.source_file.parent.name or "unknown"
        return parent

    for r in records:
        method_key = key_for_record(r)
        ds_tg = (r.dataset, r.target)
        aggregates.setdefault(method_key, {}).setdefault(ds_tg, []).append(r.value)
    return aggregates


def split_main_vs_aux(aggregates: Dict[str, Dict[Tuple[str, str], List[float]]]) -> Tuple[
    Dict[str, Dict[Tuple[str, str], List[float]]],
    Dict[str, Dict[Tuple[str, str], List[float]]],
]:
    main_set = set(MAIN_COLUMNS)
    main: Dict[str, Dict[Tuple[str, str], List[float]]] = {}
    aux: Dict[str, Dict[Tuple[str, str], List[float]]] = {}
    for method, ds_map in aggregates.items():
        for k, vals in ds_map.items():
            (main if k in main_set else aux).setdefault(method, {})[k] = vals
    return main, aux


def make_main_table(main_aggr: Dict[str, Dict[Tuple[str, str], List[float]]], corr_type: str, model_name: str) -> str:
    # Prepare rows in the order of ABLATION_GROUPS
    method_to_group: Dict[str, str] = {}
    for group_name, rows in ABLATION_GROUPS:
        for method_row_name, _ in rows:
            method_to_group[method_row_name] = group_name

    # Build LaTeX table lines
    header = r"""\begin{table*}[h]
    \centering
    \resizebox{\textwidth}{!}{%
    \begin{tabular}{lccc|ccc}
        \toprule
        \rowcolor[gray]{0.9}
         & \multicolumn{3}{c|}{\textbf{In-Distribution}} & \multicolumn{3}{c}{\textbf{Out-of-Distribution}} \\
        \rowcolor[gray]{0.9}
        Method & SimpEval & Primock57 & HelpSteer & EvalGen & RealHumanEval & CoGym \\
        \midrule
    """

    lines: List[str] = [header]

    def get_cell(method: str, ds_tg: Tuple[str, str]) -> str:
        values = main_aggr.get(method, {}).get(ds_tg, [])
        mean, ci = compute_mean_ci(values)
        return format_value(mean, ci)

    # Render groups and their rows in declared order.
    for group_index, (group_name, rows) in enumerate(ABLATION_GROUPS):
        lines.append(rf"        \rowcolor[gray]{{0.85}} \multicolumn{{7}}{{l}}{{\textbf{{{group_name}}}}} \\")
        # Compute per-column top and second-best within this section, with underline expanded to all visually equivalent values (rounded to 3 decimals)
        group_methods = [method_row_name for method_row_name, _ in rows]
        main_ds_pairs: List[Tuple[str, str]] = [
            ("SimpEval", "score"),
            ("Primock57", "time_sec"),
            ("HelpSteer2", "helpfulness"),
            ("EvalGenProduct", "grade"),
            ("RealHumanEval", "accepted"),
            ("CoGymTravel", "outcomeRating"),
        ]

        # Map each column to sets of top and visually-second-best methods (handle ties and visible equivalence)
        column_ranks: Dict[Tuple[str, str], Tuple[set, set]] = {}
        for ds_tg in main_ds_pairs:
            means_by_method: Dict[str, float] = {}
            for m in group_methods:
                vals = main_aggr.get(m, {}).get(ds_tg, [])
                mean, _ci = compute_mean_ci(vals)
                if not math.isnan(mean):
                    means_by_method[m] = mean
            if not means_by_method:
                column_ranks[ds_tg] = (set(), set())
                continue
            # Determine exact top set (bolding remains based on exact best means)
            top_val = max(means_by_method.values())
            top_set = {m for m, v in means_by_method.items() if v == top_val}

            # Determine second-best by displayed mean (rounded to 3 decimals), then underline all methods visually equal to that
            displayed_top = round(top_val, 3)
            displayed_values_below_top = sorted(
                {round(v, 3) for v in means_by_method.values() if round(v, 3) < displayed_top},
                reverse=True,
            )
            if not displayed_values_below_top:
                column_ranks[ds_tg] = (top_set, set())
                continue
            second_displayed = displayed_values_below_top[0]
            second_set = {m for m, v in means_by_method.items() if round(v, 3) == second_displayed}
            column_ranks[ds_tg] = (top_set, second_set)

        for method_row_name, _ in rows:
            cells: List[str] = []
            # The display columns must follow MAIN_COLUMNS order, but we display short dataset names in the header; here only values.
            def style_cell(value: str, ds_tg: Tuple[str, str]) -> str:
                top_set, second_set = column_ranks.get(ds_tg, (set(), set()))
                if method_row_name in top_set:
                    return f"\\textbf{{{value}}}"
                if method_row_name in second_set:
                    return f"\\underline{{{value}}}"
                return value

            d1 = style_cell(get_cell(method_row_name, ("SimpEval", "score")), ("SimpEval", "score"))
            d2 = style_cell(get_cell(method_row_name, ("Primock57", "time_sec")), ("Primock57", "time_sec"))
            d3 = style_cell(get_cell(method_row_name, ("HelpSteer2", "helpfulness")), ("HelpSteer2", "helpfulness"))
            d4 = style_cell(get_cell(method_row_name, ("EvalGenProduct", "grade")), ("EvalGenProduct", "grade"))
            d5 = style_cell(get_cell(method_row_name, ("RealHumanEval", "accepted")), ("RealHumanEval", "accepted"))
            d6 = style_cell(get_cell(method_row_name, ("CoGymTravel", "outcomeRating")), ("CoGymTravel", "outcomeRating"))
            cells.extend([d1, d2, d3, d4, d5, d6])
            line = f"        {method_row_name:<30} & " + " & ".join(cells) + " \\\\"
            lines.append(line)
        if group_index < len(ABLATION_GROUPS) - 1:
            lines.append(r"        \midrule")

    footer = rf"""
        \bottomrule
    \end{{tabular}}
    }}
    \caption{{Performance ({corr_type.title()} correlation) with 95\% confidence intervals on in-distribution and out-of-distribution datasets. Model: {model_name}.}}
    \label{{tab:ablations_{corr_type}}}
\end{{table*}}"""

    lines.append(footer)
    return "\n".join(lines)


def make_main_table_ci_overlap(main_aggr: Dict[str, Dict[Tuple[str, str], List[float]]], corr_type: str, model_name: str) -> str:
    """Variant of the main table that bolds any method whose 95% CI overlaps the best method's CI in each column. No underlines."""
    # Prepare rows in the order of ABLATION_GROUPS
    method_to_group: Dict[str, str] = {}
    for group_name, rows in ABLATION_GROUPS:
        for method_row_name, _ in rows:
            method_to_group[method_row_name] = group_name

    header = r"""\begin{table*}[h]
    \centering
    \resizebox{\textwidth}{!}{%
    \begin{tabular}{lccc|ccc}
        \toprule
        \rowcolor[gray]{0.9}
         & \multicolumn{3}{c|}{\textbf{In-Distribution}} & \multicolumn{3}{c}{\textbf{Out-of-Distribution}} \\
        \rowcolor[gray]{0.9}
        Method & SimpEval & Primock57 & HelpSteer & EvalGen & RealHumanEval & CoGym \\
        \midrule
    """

    lines: List[str] = [header]

    # Convenience accessors
    def get_values(method: str, ds_tg: Tuple[str, str]) -> List[float]:
        return main_aggr.get(method, {}).get(ds_tg, [])

    def get_mean_ci(method: str, ds_tg: Tuple[str, str]) -> Tuple[float, Optional[float]]:
        values = get_values(method, ds_tg)
        return compute_mean_ci(values)

    main_ds_pairs: List[Tuple[str, str]] = [
        ("SimpEval", "score"),
        ("Primock57", "time_sec"),
        ("HelpSteer2", "helpfulness"),
        ("EvalGenProduct", "grade"),
        ("RealHumanEval", "accepted"),
        ("CoGymTravel", "outcomeRating"),
    ]

    # Render groups and their rows in declared order.
    for group_index, (group_name, rows) in enumerate(ABLATION_GROUPS):
        lines.append(rf"        \rowcolor[gray]{{0.85}} \multicolumn{{7}}{{l}}{{\textbf{{{group_name}}}}} \\")

        group_methods = [method_row_name for method_row_name, _ in rows]

        # Pre-compute the best CI interval per column (handle ties by taking the union of best intervals)
        best_intervals: Dict[Tuple[str, str], Optional[Tuple[float, float]]] = {}
        for ds_tg in main_ds_pairs:
            # Compute means for group methods in this column
            means_by_method: Dict[str, float] = {}
            mcis_by_method: Dict[str, Tuple[float, Optional[float]]] = {}
            for m in group_methods:
                mean, ci = get_mean_ci(m, ds_tg)
                mcis_by_method[m] = (mean, ci)
                if not math.isnan(mean):
                    means_by_method[m] = mean
            if not means_by_method:
                best_intervals[ds_tg] = None
                continue
            top_val = max(means_by_method.values())
            top_methods = [m for m, v in means_by_method.items() if v == top_val]
            lowers: List[float] = []
            uppers: List[float] = []
            for m in top_methods:
                mean, ci = mcis_by_method[m]
                if math.isnan(mean) or ci is None:
                    continue
                lowers.append(mean - ci)
                uppers.append(mean + ci)
            if not lowers or not uppers:
                # If CI is None for the best (e.g., no variance), treat as a point interval
                # Use 0.0 margin in that case
                for m in top_methods:
                    mean, ci = mcis_by_method[m]
                    if not math.isnan(mean):
                        margin = ci if ci is not None else 0.0
                        lowers.append(mean - margin)
                        uppers.append(mean + margin)
                if not lowers:
                    best_intervals[ds_tg] = None
                    continue
            best_intervals[ds_tg] = (min(lowers), max(uppers))

        for method_row_name, _ in rows:
            cells: List[str] = []

            def style_cell(ds_tg: Tuple[str, str]) -> str:
                best_iv = best_intervals.get(ds_tg)
                mean, ci = get_mean_ci(method_row_name, ds_tg)
                value_str = format_value(mean, ci)
                if best_iv is None or math.isnan(mean):
                    return value_str
                margin = ci if ci is not None else 0.0
                lower = mean - margin
                upper = mean + margin
                best_lower, best_upper = best_iv
                # Overlap if intervals intersect
                if not (upper < best_lower or lower > best_upper):
                    return f"\\textbf{{{value_str}}}"
                return value_str

            d1 = style_cell(("SimpEval", "score"))
            d2 = style_cell(("Primock57", "time_sec"))
            d3 = style_cell(("HelpSteer2", "helpfulness"))
            d4 = style_cell(("EvalGenProduct", "grade"))
            d5 = style_cell(("RealHumanEval", "accepted"))
            d6 = style_cell(("CoGymTravel", "outcomeRating"))
            cells.extend([d1, d2, d3, d4, d5, d6])
            line = f"        {method_row_name:<30} & " + " & ".join(cells) + " \\\\"
            lines.append(line)
        if group_index < len(ABLATION_GROUPS) - 1:
            lines.append(r"        \midrule")

    footer = rf"""
        \bottomrule
    \end{{tabular}}
    }}
    \caption{{Performance ({corr_type.title()} correlation) with 95\% confidence intervals. Cells are \textbf{{bold}} when the method's 95\% CI overlaps the best method's 95\% CI for that column. Model: {model_name}.}}
    \label{{tab:ablations_ci_overlap_{corr_type}}}
\end{{table*}}"""

    lines.append(footer)
    return "\n".join(lines)

def make_main_table_best_vs_second_ci(main_aggr: Dict[str, Dict[Tuple[str, str], List[float]]], corr_type: str, model_name: str) -> str:
    """Variant where only the best method is textbf{bold} if its 95% CI excludes the second-best mean; otherwise the best is underline{underlined}. No styling for non-best methods."""
    header = r"""\begin{table*}[h]
    \centering
    \resizebox{\textwidth}{!}{%
    \begin{tabular}{lccc|ccc}
        \toprule
        \rowcolor[gray]{0.9}
         & \multicolumn{3}{c|}{\textbf{In-Distribution}} & \multicolumn{3}{c}{\textbf{Out-of-Distribution}} \\
        \rowcolor[gray]{0.9}
        Method & SimpEval & Primock57 & HelpSteer & EvalGen & RealHumanEval & CoGym \\
        \midrule
    """

    lines: List[str] = [header]

    main_ds_pairs: List[Tuple[str, str]] = [
        ("SimpEval", "score"),
        ("Primock57", "time_sec"),
        ("HelpSteer2", "helpfulness"),
        ("EvalGenProduct", "grade"),
        ("RealHumanEval", "accepted"),
        ("CoGymTravel", "outcomeRating"),
    ]

    def get_mean_ci(method: str, ds_tg: Tuple[str, str]) -> Tuple[float, Optional[float]]:
        values = main_aggr.get(method, {}).get(ds_tg, [])
        return compute_mean_ci(values)

    for group_index, (group_name, rows) in enumerate(ABLATION_GROUPS):
        lines.append(f"        \\rowcolor[gray]{{0.85}} \\multicolumn{{7}}{{l}}{{\\textbf{{{group_name}}}}} \\\\")

        group_methods = [method_row_name for method_row_name, _ in rows]

        # Pre-compute per-column top set and second-best mean
        per_column_info: Dict[Tuple[str, str], Tuple[set, Optional[float]]] = {}
        for ds_tg in main_ds_pairs:
            means_by_method: Dict[str, float] = {}
            for m in group_methods:
                mean, _ci = get_mean_ci(m, ds_tg)
                if not math.isnan(mean):
                    means_by_method[m] = mean
            if not means_by_method:
                per_column_info[ds_tg] = (set(), None)
                continue
            top_val = max(means_by_method.values())
            top_set = {m for m, v in means_by_method.items() if v == top_val}
            lower_vals = [v for v in means_by_method.values() if v < top_val]
            second_mean = max(lower_vals) if lower_vals else None
            per_column_info[ds_tg] = (top_set, second_mean)

        for method_row_name, _ in rows:
            def style_cell(ds_tg: Tuple[str, str]) -> str:
                top_set, second_mean = per_column_info.get(ds_tg, (set(), None))
                mean, ci = get_mean_ci(method_row_name, ds_tg)
                value_str = format_value(mean, ci)
                if method_row_name not in top_set or math.isnan(mean) or second_mean is None:
                    return value_str
                margin = ci if ci is not None else 0.0
                lower = mean - margin
                # Bold if second-best mean is strictly below the lower bound of top's 95% CI
                if second_mean < lower:
                    return f"\\textbf{{{value_str}}}"
                # Otherwise underline to denote best but not significantly better
                return f"\\underline{{{value_str}}}"

            d1 = style_cell(("SimpEval", "score"))
            d2 = style_cell(("Primock57", "time_sec"))
            d3 = style_cell(("HelpSteer2", "helpfulness"))
            d4 = style_cell(("EvalGenProduct", "grade"))
            d5 = style_cell(("RealHumanEval", "accepted"))
            d6 = style_cell(("CoGymTravel", "outcomeRating"))
            line = f"        {method_row_name:<30} & " + " & ".join([d1, d2, d3, d4, d5, d6]) + " \\\\" 
            lines.append(line)
        if group_index < len(ABLATION_GROUPS) - 1:
            lines.append(r"        \\midrule")

    footer = rf"""
        \bottomrule
    \end{{tabular}}
    }}
    \caption{{Performance ({corr_type.title()} correlation). Cells are \textbf{{bold}} when the best method's 95\% CI excludes the second-best mean (significantly best), and \underline{{underlined}} when best but not significantly better. Model: {model_name}.}}
    \label{{tab:ablations_best_vs_second_{corr_type}}}
\end{{table*}}"""

    lines.append(footer)
    return "\n".join(lines)

def make_aux_table(aux_aggr: Dict[str, Dict[Tuple[str, str], List[float]]], corr_type: str, model_name: str) -> Optional[str]:
    if not aux_aggr:
        return None

    # Collect all (dataset,target) pairs present in auxiliary
    all_pairs: List[Tuple[str, str]] = sorted({k for ds_map in aux_aggr.values() for k in ds_map.keys()})
    # Simple wide table: methods as rows, (dataset,target) as columns
    col_headers = [f"{ds} ({tg})" for ds, tg in all_pairs]

    header = [
        r"\begin{table*}[h]",
        r"    \centering",
        r"    \small",
        r"    \setlength{\tabcolsep}{6pt}",
        r"    \renewcommand{\arraystretch}{1.1}",
        "    \\resizebox{\\textwidth}{!}{%",
        "    \\begin{tabular}{l" + ("c" * len(col_headers)) + "}",
        r"        \toprule",
        r"        \rowcolor[gray]{0.9}",
        "        Method & " + " & ".join(col_headers) + r" \\",
        r"        \midrule",
    ]

    lines: List[str] = header

    for method, ds_map in sorted(aux_aggr.items()):
        row_vals: List[str] = []
        for pair in all_pairs:
            vals = ds_map.get(pair, [])
            mean, ci = compute_mean_ci(vals)
            row_vals.append(format_value(mean, ci))
        lines.append("        " + method + " & " + " & ".join(row_vals) + " \\\\")

    footer = [
        r"        \bottomrule",
        r"    \end{tabular}%",
        r"    }",
        rf"    \caption{{Auxiliary ablation results ({corr_type.title()} correlation) for additional datasets/targets. Model: {model_name}.}}",
        rf"    \label{{tab:ablations_aux_{corr_type}}}",
        r"\end{table*}",
    ]

    lines.extend(footer)
    return "\n".join(lines)


# ───────────────────────── CLI ───────────────────────── #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze ablation results and create LaTeX tables")
    p.add_argument(
        "--results_root",
        type=Path,
        required=True,
        help="Root directory under which ablation results are stored (e.g., results/ablations/qwen_remote)",
    )
    p.add_argument(
        "--corr_type",
        choices=["kendall", "spearman", "pearson"],
        default="kendall",
        help="Correlation type to extract (default: kendall)",
    )
    p.add_argument(
        "--out_root",
        type=Path,
        default=Path("results/ablations/ablations_analysis"),
        help="Directory to write outputs (tables, CSVs)",
    )
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    args.out_root.mkdir(parents=True, exist_ok=True)
    model_display_name, model_safe_name = extract_model_name(args.results_root)
    logging.info(f"Detected model: {model_display_name} (safe: {model_safe_name})")

    # Scan and parse
    records = scan_results(args.results_root, args.corr_type)
    if not records:
        logging.error("No results found. Ensure the results_root is correct and contains .txt or .csv files.")
        return

    # Aggregate
    aggregates = build_aggregates(records)
    main_aggr, aux_aggr = split_main_vs_aux(aggregates)

    # Save aggregated CSV for inspection
    aggr_rows = []
    for method, ds_map in aggregates.items():
        for (ds, tg), vals in ds_map.items():
            mean, ci = compute_mean_ci(vals)
            aggr_rows.append({
                "method": method,
                "dataset": ds,
                "target": tg,
                "n": len(vals),
                "mean": mean,
                "ci_95": ci if ci is not None else "",
            })
    aggr_df = pd.DataFrame(aggr_rows)
    aggr_csv_path = args.out_root / f"ablations_aggregates_{args.corr_type}_{model_safe_name}.csv"
    aggr_df.to_csv(aggr_csv_path, index=False)
    logging.info(f"Aggregates CSV → {aggr_csv_path}")

    # Build LaTeX tables
    main_table_tex = make_main_table(main_aggr, args.corr_type, model_display_name)
    main_tex_path = args.out_root / f"ablations_main_{args.corr_type}_{model_safe_name}.tex"
    main_tex_path.write_text(main_table_tex)
    logging.info(f"Main LaTeX table → {main_tex_path}")

    # Also build CI-overlap variant of the main table
    main_overlap_tex = make_main_table_ci_overlap(main_aggr, args.corr_type, model_display_name)
    main_overlap_path = args.out_root / f"ablations_main_overlap_{args.corr_type}_{model_safe_name}.tex"
    main_overlap_path.write_text(main_overlap_tex)
    logging.info(f"Main (CI-overlap) LaTeX table → {main_overlap_path}")

    # Build best-vs-second CI significance variant
    main_best_vs_second_tex = make_main_table_best_vs_second_ci(main_aggr, args.corr_type, model_display_name)
    main_best_vs_second_path = args.out_root / f"ablations_main_best_vs_second_{args.corr_type}_{model_safe_name}.tex"
    main_best_vs_second_path.write_text(main_best_vs_second_tex)
    logging.info(f"Main (best-vs-second CI) LaTeX table → {main_best_vs_second_path}")

    aux_table_tex = make_aux_table(aux_aggr, args.corr_type, model_display_name)
    if aux_table_tex:
        aux_tex_path = args.out_root / f"ablations_aux_{args.corr_type}_{model_safe_name}.tex"
        aux_tex_path.write_text(aux_table_tex)
        logging.info(f"Aux LaTeX table → {aux_tex_path}")
    else:
        logging.info("No auxiliary datasets detected; skipping aux table.")

    # Console summary
    print(f"\nAblation Analysis ({args.corr_type.title()} correlation, {model_display_name})")
    print("=" * 80)
    print("Methods detected:", ", ".join(sorted(aggregates.keys())))
    print(f"Output directory: {args.out_root}")
    print(f"Main LaTeX: {main_tex_path}")
    if aux_table_tex:
        print(f"Aux LaTeX: {aux_tex_path}")
    print(f"Main (CI-overlap) LaTeX: {main_overlap_path}")
    print(f"Main (best-vs-second CI) LaTeX: {main_best_vs_second_path}")


if __name__ == "__main__":
    main()

# Example Usage:
# python analysis/ablations/analyze_ablations.py --results_root /nlp/scr2/nlp/personal-rm/autometrics/results/ablations/qwen_remote --corr_type kendall --out_root /nlp/scr2/nlp/personal-rm/autometrics/results/ablations/ablations_analysis