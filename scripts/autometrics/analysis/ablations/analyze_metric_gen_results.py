#!/usr/bin/env python3
# analysis/ablations/analyze_metric_gen_results.py
"""
Analyze metric generation benchmark results by correlation type.

Filters CSV files by correlation type (kendall, spearman, pearson),
aggregates results into a single CSV, and creates a pivot table
with dataset/measure as rows and generator types as columns.
Also computes totals across all datasets.
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import re
from typing import Dict, List, Tuple, Optional

# ───────────────────────── SETTINGS ───────────────────────── #
# Dataset categorization and ordering
IN_DISTRIBUTION_DATASETS = [
    "SummEval",
    "Primock57", 
    "SimpEval",
    "SimpDA",
    "HelpSteer",
    "HelpSteer2"
]

OUT_DISTRIBUTION_DATASETS = [
    "EvalGenProduct",
    "EvalGenMedical", 
    "RealHumanEval",
    "CoGymTravelProcess",
    "CoGymTravelOutcome",
    "CoGymTabularProcess", 
    "CoGymTabularOutcome",
    "CoGymLessonProcess",
    "CoGymLessonOutcome"
]

# Generator type display names
GENERATOR_DISPLAY_NAMES = {
    "llm_judge": "LLM-Judge",
    "rubric_prometheus": "Rubric (Prometheus)",
    "rubric_dspy": "Rubric (DSPy)",
    "geval": "G-Eval",
    "codegen": "Code Gen",
    "llm_judge_optimized": "MIPROv2",
    "finetune": "Finetune",
    "llm_judge_examples": "Examples"
}

# ───────────────────────── CLI ──────────────────────────── #
def parse_args():
    p = argparse.ArgumentParser(description="Analyze metric generation results by correlation type")
    p.add_argument("--results_path", required=True, type=Path, 
                   help="Path to directory containing metric generation CSV files")
    p.add_argument("--corr_type", required=True, choices=["kendall", "spearman", "pearson"],
                   help="Correlation type to filter files by")
    p.add_argument("--out_root", type=Path, default=Path("results/ablations/metric_generation_analysis"),
                   help="Output directory for results (default: results/ablations/metric_generation_analysis)")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()

def extract_model_name(results_path: Path) -> tuple[str, str]:
    """Extract model name from results path. Returns (display_name, safe_name)."""
    path_str = str(results_path)
    
    # Look for common model patterns in the path
    if "qwen" in path_str.lower():
        return "Qwen3 32B", "qwen3_32b"
    elif "gpt4o" in path_str.lower() or "gpt-4o" in path_str.lower():
        return "GPT-4o Mini", "gpt4o_mini"
    else:
        # Try to extract from path components
        parts = path_str.split('/')
        for part in parts:
            if "qwen" in part.lower():
                return "Qwen3 32B", "qwen3_32b"
            elif "gpt" in part.lower():
                return "GPT-4o Mini", "gpt4o_mini"
        
        # Default fallback
        return "unknown", "unknown"

# ───────────────────────── UTILITIES ────────────────────── #
def filter_files_by_correlation_type(results_path: Path, corr_type: str) -> List[Path]:
    """Filter CSV files to only those matching the specified correlation type."""
    pattern = f"*{corr_type}*.csv"
    files = list(results_path.glob(pattern))
    logging.info(f"Found {len(files)} files matching pattern '{pattern}'")
    return files

def parse_mean_ci(mean_ci_str: str) -> Tuple[float, float]:
    """Parse mean ± CI string into (mean, ci) tuple."""
    if pd.isna(mean_ci_str) or mean_ci_str == "" or mean_ci_str == "N/A":
        return None, None
    
    # Handle different formats
    if "±" in str(mean_ci_str):
        parts = str(mean_ci_str).split("±")
        mean = float(parts[0].strip())
        ci = float(parts[1].strip())
        return mean, ci
    else:
        # Try to parse as just a number
        try:
            mean = float(mean_ci_str)
            return mean, None
        except:
            return None, None



def load_and_aggregate_data(files: List[Path]) -> pd.DataFrame:
    """Load all CSV files and aggregate into a single DataFrame."""
    all_data = []
    
    for file_path in files:
        try:
            df = pd.read_csv(file_path)
            # Extract dataset and measure from filename if not in data
            if 'dataset' not in df.columns or 'measure' not in df.columns:
                # Parse from filename: metric_generation_benchmark_*_*_*_corrtype_dataset_measure.csv
                filename = file_path.stem
                parts = filename.split('_')
                # Find the correlation type and extract dataset/measure
                for i, part in enumerate(parts):
                    if part in ['kendall', 'spearman', 'pearson']:
                        if i + 2 < len(parts):
                            dataset = parts[i + 1]
                            measure = parts[i + 2]
                            df['dataset'] = dataset
                            df['measure'] = measure
                        break
            
            # Keep only essential columns
            essential_cols = ['dataset', 'measure', 'generator_type', 'mean_±_ci']
            available_cols = [col for col in essential_cols if col in df.columns]
            df_subset = df[available_cols].copy()
            
            all_data.append(df_subset)
            logging.debug(f"Loaded {len(df_subset)} rows from {file_path.name}")
            
        except Exception as e:
            logging.warning(f"Failed to load {file_path}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No valid data files found")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    logging.info(f"Combined {len(combined_df)} total rows from {len(files)} files")
    return combined_df

def create_pivot_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create pivot table with dataset/measure as rows and generator types as columns."""
    # Parse mean and CI from mean_±_ci column
    parsed_data = []
    for _, row in df.iterrows():
        mean, ci = parse_mean_ci(row['mean_±_ci'])
        parsed_data.append({
            'dataset': row['dataset'],
            'measure': row['measure'],
            'generator_type': row['generator_type'],
            'mean': mean,
            'ci': ci
        })
    
    parsed_df = pd.DataFrame(parsed_data)
    
    # Create pivot tables for mean and CI separately
    mean_pivot = parsed_df.pivot_table(
        index=['dataset', 'measure'],
        columns='generator_type',
        values='mean',
        aggfunc='first'
    ).fillna('--')
    
    ci_pivot = parsed_df.pivot_table(
        index=['dataset', 'measure'],
        columns='generator_type',
        values='ci',
        aggfunc='first'
    ).fillna('--')
    
    # Combine mean and CI into formatted strings
    combined_pivot = mean_pivot.astype(object).copy()
    for idx in combined_pivot.index:
        for col in combined_pivot.columns:
            mean_val = mean_pivot.loc[idx, col]
            ci_val = ci_pivot.loc[idx, col]
            
            if mean_val != '--' and ci_val != '--':
                combined_pivot.loc[idx, col] = f"{mean_val:.4f} ± {ci_val:.4f}"
            elif mean_val != '--':
                combined_pivot.loc[idx, col] = f"{mean_val:.4f}"
            else:
                combined_pivot.loc[idx, col] = '--'
    
    # Add totals row (average across datasets)
    totals_data = {}
    for col in combined_pivot.columns:
        values = []
        for val in combined_pivot[col]:
            if val != '--' and val is not None:
                try:
                    # Extract just the mean value for averaging
                    if "±" in str(val):
                        mean_str = str(val).split("±")[0].strip()
                        values.append(float(mean_str))
                    else:
                        values.append(float(val))
                except:
                    continue
        
        if values:
            avg = np.mean(values)
            totals_data[col] = f"{avg:.4f}"
        else:
            totals_data[col] = '--'
    
    # Add totals row (average across datasets)
    totals_df = pd.DataFrame([totals_data], index=[('AVERAGE', '')])
    combined_pivot = pd.concat([combined_pivot, totals_df])
    
    return combined_pivot

def format_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Format the pivot table for better display."""
    # The pivot table already has properly formatted values, just return as is
    return df.copy()

def prettify_generator_name(name: str) -> str:
    """Convert generator type to display name."""
    return GENERATOR_DISPLAY_NAMES.get(name, name.replace("_", " ").title())

def escape_latex_text(text: str) -> str:
    """Escape special LaTeX characters in text."""
    # Replace underscores with escaped underscores
    text = text.replace("_", r"\_")
    # Replace other special characters that might cause issues
    text = text.replace("%", r"\%")
    text = text.replace("&", r"\&")
    text = text.replace("#", r"\#")
    text = text.replace("$", r"\$")
    text = text.replace("{", r"\{")
    text = text.replace("}", r"\}")
    text = text.replace("~", r"\textasciitilde{}")
    text = text.replace("^", r"\textasciicircum{}")
    return text

def categorize_dataset(dataset: str) -> str:
    """Categorize dataset as in-distribution or out-of-distribution."""
    if dataset in IN_DISTRIBUTION_DATASETS:
        return "in-distribution"
    elif dataset in OUT_DISTRIBUTION_DATASETS:
        return "out-of-distribution"
    else:
        return "unknown"

def sort_datasets_by_category(datasets: List[str]) -> List[str]:
    """Sort datasets by category (in-distribution first, then out-of-distribution)."""
    in_dist = [d for d in IN_DISTRIBUTION_DATASETS if d in datasets]
    out_dist = [d for d in OUT_DISTRIBUTION_DATASETS if d in datasets]
    return in_dist + out_dist

def format_correlation_value(value: str) -> str:
    """Format correlation value for LaTeX with confidence intervals."""
    if value == '--' or value is None:
        return "---"
    
    # Parse mean ± CI format
    if "±" in str(value):
        parts = str(value).split("±")
        mean = float(parts[0].strip())
        ci = float(parts[1].strip())
        return f"{mean:.3f} {{\\scriptsize $\\pm$ {ci:.3f}}}"
    else:
        # Just a mean value
        try:
            mean = float(value)
            return f"{mean:.3f}"
        except:
            return "---"

def create_latex_table(pivot_df: pd.DataFrame, corr_type: str, model_name: str) -> str:
    """Create LaTeX table with proper formatting and sections."""
    
    # Define column order: cheap methods first, then expensive methods
    cheap_methods = ["codegen", "geval", "llm_judge", "rubric_dspy", "rubric_prometheus"]
    expensive_methods = ["finetune", "llm_judge_examples", "llm_judge_optimized"]
    
    # Get available generator types in desired order
    generator_types = []
    for gen_type in cheap_methods + expensive_methods:
        if gen_type in pivot_df.columns:
            generator_types.append(gen_type)
    
    # Create column specification with vertical divider
    col_spec = "l" + "c" * len([g for g in cheap_methods if g in pivot_df.columns]) + "|" + "c" * len([g for g in expensive_methods if g in pivot_df.columns])
    
    # Get all datasets and sort them
    all_datasets = []
    for idx in pivot_df.index:
        if idx[0] != 'AVERAGE':
            all_datasets.append(idx[0])
    
    sorted_datasets = sort_datasets_by_category(all_datasets)
    
    # Build table body
    body_lines = []
    
    # Track current category for section headers
    current_category = None
    
    for dataset in sorted_datasets:
        category = categorize_dataset(dataset)
        
        # Add section header if category changes
        if category != current_category:
            if current_category is not None:
                body_lines.append(r"        \midrule")
            
            category_text = "In-Distribution" if category == "in-distribution" else "Out-of-Distribution"
            body_lines.append(f"        \\rowcolor{{gray!10}}")
            multicol = "        \\multicolumn{" + str(len(generator_types) + 1) + "}{l}{\\textit{\\textbf{" + category_text + " Tasks}: " + get_category_description(category) + "}} \\\\"
            body_lines.append(multicol)
            body_lines.append(r"        \midrule")
            current_category = category
        
        # Get dataset measures
        dataset_measures = []
        for idx in pivot_df.index:
            if idx[0] == dataset:
                dataset_measures.append(idx[1])
        
        # Add dataset rows
        for measure in dataset_measures:
            row_data = []
            # Use dataset and measure names
            escaped_dataset = escape_latex_text(dataset)
            escaped_measure = escape_latex_text(measure)
            row_data.append(f"{escaped_dataset} ({escaped_measure})")
            
            # Get values for this row
            row_values = []
            for col in generator_types:
                value = pivot_df.loc[(dataset, measure), col]
                row_values.append(value)
            
            # Find best values in each category
            cheap_values = [parse_mean_ci(v)[0] for v in row_values[:len([g for g in cheap_methods if g in pivot_df.columns])] if v != '--']
            expensive_values = [parse_mean_ci(v)[0] for v in row_values[len([g for g in cheap_methods if g in pivot_df.columns]):] if v != '--']
            
            cheap_best = max(cheap_values) if cheap_values else -1
            expensive_best = max(expensive_values) if expensive_values else -1
            
            # Format values with bolding
            for i, col in enumerate(generator_types):
                value = pivot_df.loc[(dataset, measure), col]
                formatted_value = format_correlation_value(value)
                
                # Bold if it's the best in its category
                if value != '--':
                    mean_val, _ = parse_mean_ci(value)
                    if (i < len([g for g in cheap_methods if g in pivot_df.columns]) and mean_val == cheap_best) or \
                       (i >= len([g for g in cheap_methods if g in pivot_df.columns]) and mean_val == expensive_best):
                        # Extract the numeric part and bold it
                        if '±' in formatted_value:
                            parts = formatted_value.split(' ± ')
                            formatted_value = f"\\textbf{{{parts[0]}}} ± {parts[1]}"
                        else:
                            formatted_value = f"\\textbf{{{formatted_value}}}"
                
                row_data.append(formatted_value)
            
            body_lines.append("        " + " & ".join(row_data) + r" \\")
    
    # Add average row
    body_lines.append(r"        \midrule")
    avg_row = ["\\textbf{Average}"]
    for col in generator_types:
        value = pivot_df.loc[('AVERAGE', ''), col]
        formatted_value = format_correlation_value(value)
        avg_row.append(formatted_value)
    body_lines.append("        " + " & ".join(avg_row) + r" \\")
    
    # Create multi-column headers
    cheap_header = " & ".join([GENERATOR_DISPLAY_NAMES[gen] for gen in cheap_methods if gen in pivot_df.columns])
    expensive_header = " & ".join([GENERATOR_DISPLAY_NAMES[gen] for gen in expensive_methods if gen in pivot_df.columns])
    
    # Create caption and label
    caption = f"Metric generation performance ({corr_type.title()}'s Tau) with 95\\% confidence intervals over 5 independent runs. Each generator produces metrics using persistent train sets, then correlation with human annotations is measured on persistent validation sets. Cheap methods (left) generate 10 metrics per trial, expensive methods (right) generate 1 metric per trial (except finetune which generates 10). Results show correlation between generated metrics and ground-truth human annotations across diverse tasks using the {model_name} model."
    
    # Create safe label (no spaces or special chars)
    safe_model_name = model_name.lower().replace(" ", "_").replace("-", "_")
    label = f"tab:metric_gen_{corr_type}_{safe_model_name}"
    
    # Build complete table
    table = rf"""\begin{{table*}}[h]
  \centering
  \small
  \setlength{{\tabcolsep}}{{6pt}}
  \renewcommand{{\arraystretch}}{{1.1}}
  \resizebox{{\textwidth}}{{!}}{{%
  \begin{{tabular}}{{{col_spec}}}
    \toprule
    \rowcolor{{gray!30}}
    & \multicolumn{{{len([g for g in cheap_methods if g in pivot_df.columns])}}}{{c|}}{{\textbf{{Cheap to Produce}}}} & \multicolumn{{{len([g for g in expensive_methods if g in pivot_df.columns])}}}{{c}}{{\textbf{{Expensive to Produce}}}} \\
    \rowcolor{{gray!30}}
    Task (Measure) & {cheap_header} & {expensive_header} \\
    \midrule
{chr(10).join(body_lines)}
    \bottomrule
  \end{{tabular}}%
  }}
  \caption{{{caption}}}
  \label{{{label}}}
\end{{table*}}"""
    
    return table

def get_category_description(category: str) -> str:
    """Get description for dataset category."""
    if category == "in-distribution":
        return "some metrics in our bank were designed to directly evaluate these tasks."
    elif category == "out-of-distribution":
        return "no metric is specifically designed for these -- tests generalization and metric generation."
    else:
        return "unknown category."

# ─────────────────────────── MAIN ──────────────────────── #
def main():
    args = parse_args()
    
    # Create output directory
    args.out_root.mkdir(parents=True, exist_ok=True)
    
    # Extract model name from results path
    model_display_name, model_safe_name = extract_model_name(args.results_path)
    logging.info(f"Detected model: {model_display_name} (safe: {model_safe_name})")
    
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s | %(message)s"
    )
    
    # Filter files by correlation type
    files = filter_files_by_correlation_type(args.results_path, args.corr_type)
    if not files:
        logging.error(f"No files found matching correlation type '{args.corr_type}'")
        return
    
    # Load and aggregate data
    combined_df = load_and_aggregate_data(files)
    
    # Save combined CSV
    combined_csv_path = args.out_root / f"combined_{args.corr_type}_{model_safe_name}.csv"
    combined_df.to_csv(combined_csv_path, index=False)
    logging.info(f"Combined CSV → {combined_csv_path}")
    
    # Create pivot table
    pivot_df = create_pivot_table(combined_df)
    
    # Format for display
    display_df = format_for_display(pivot_df)
    
    # Save pivot table CSV
    pivot_csv_path = args.out_root / f"pivot_{args.corr_type}_{model_safe_name}.csv"
    display_df.to_csv(pivot_csv_path)
    logging.info(f"Pivot CSV → {pivot_csv_path}")
    
    # Generate LaTeX table
    latex_table = create_latex_table(pivot_df, args.corr_type, model_display_name)
    latex_path = args.out_root / f"table_{args.corr_type}_{model_safe_name}.tex"
    latex_path.write_text(latex_table)
    logging.info(f"LaTeX table → {latex_path}")
    
    # Display the table
    print(f"\nMetric Generation Results ({args.corr_type.title()} correlation, {model_display_name}):")
    print("=" * 80)
    print(display_df.to_string())
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"- Model: {model_display_name}")
    print(f"- Total datasets: {len(display_df) - 1}")  # Exclude average row
    print(f"- Generator types: {len(display_df.columns)}")
    print(f"- Files processed: {len(files)}")
    print(f"- LaTeX table saved to: {latex_path}")

if __name__ == "__main__":
    main()

# Example usage:
# python analysis/ablations/analyze_metric_gen_results.py \
#   --results_path results/ablations/metric_generation/qwen3_32b/sub_results \
#   --corr_type kendall \
#   --out_root results/ablations/metric_generation_analysis
#
# python analysis/ablations/analyze_metric_gen_results.py --results_path results/ablations/metric_generation/qwen3_32b/sub_results --corr_type kendall --out_root results/ablations/metric_generation_analysis
# 
# Output files will be:
# - combined_kendall_qwen.csv
# - pivot_kendall_qwen.csv
# - table_kendall_qwen.tex (LaTeX table)
