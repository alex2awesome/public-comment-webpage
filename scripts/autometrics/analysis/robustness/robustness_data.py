import argparse
import hashlib
import json
import importlib
import os
import sys
from typing import Optional

import pandas as pd
import dspy

from autometrics.experiments.robustness.perturb import ProducePerturbations
try:
    # Prefer the canonical loader used in main experiments
    from analysis.main_experiments.run_main_autometrics import load_dataset as load_dataset_by_name
except Exception:
    load_dataset_by_name = None  # Fallback to import-path loader


def _resolve_dataset(import_path: str, init_kwargs: Optional[dict] = None):
    module_path, _, class_name = import_path.rpartition(".")
    if not module_path or not class_name:
        raise ValueError(f"Invalid dataset import path: {import_path}. Use 'module.submodule.ClassName'.")
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    init_kwargs = init_kwargs or {}
    return cls(**init_kwargs)


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _subset_dataframe(dataset, max_examples: Optional[int], seed: int):
    if max_examples is None:
        return dataset
    df = dataset.get_dataframe()
    if max_examples >= len(df):
        return dataset
    return dataset.get_subset(max_examples, seed=seed)


def _compute_uid(text: str) -> str:
    try:
        norm = str(text).strip().lower()
    except Exception:
        norm = ""
    return hashlib.md5(norm.encode()).hexdigest()


def _write_original_subset_csv(dataset, out_dir: str, dimension: str, filename: str = "original_subset.csv", include_references_json: bool = False):
    # Build unified schema matching perturbed files
    input_col = dataset.get_input_column()
    output_col = dataset.get_output_column()
    ref_cols = dataset.get_reference_columns()
    df = dataset.get_dataframe().copy()

    inputs = df[input_col].tolist()
    outputs = df[output_col].tolist()
    refs_per_col = {c: df[c].tolist() for c in ref_cols}
    uids = [ _compute_uid(inp) for inp in inputs ]

    rows = []
    for i in range(len(df)):
        # Build list of references in canonical order
        refs_list = []
        for c in ref_cols:
            val = refs_per_col[c][i]
            try:
                is_missing = pd.isna(val)
            except Exception:
                is_missing = False
            if not is_missing and val is not None and str(val) != "":
                refs_list.append(val)
        row = {
            "dimension": dimension,
            "group": "original",
            "strategy": "original",
            "uid": uids[i],
            "input": inputs[i],
            "original_output": outputs[i],
            "model_output": outputs[i],
        }
        if include_references_json:
            row["references"] = json.dumps(refs_list, ensure_ascii=False)
        for c in ref_cols:
            row[c] = refs_per_col[c][i]
        rows.append(row)

    out_df = pd.DataFrame(rows)

    # Write with stable column order: base columns first, then references
    base_cols = ["dimension", "group", "strategy", "uid", "input", "original_output", "model_output"]
    if include_references_json:
        base_cols.append("references")
    ordered_cols = base_cols + [c for c in ref_cols if c in out_df.columns]
    out_path = os.path.join(out_dir, filename)
    out_df.to_csv(out_path, index=False, columns=ordered_cols)
    return out_path


def _build_perturbed_tables(dataset, perturbations, dimension: str, include_references_json: bool = False):
    # Only use obvious perturbations as requested
    worse_obvious = perturbations["perturbed_worse_obvious"]  # List[List[str]] per record per strategy
    same_obvious = perturbations["perturbed_same_obvious"]    # List[str] per record
    strategies = perturbations["strategies"]                   # List[str]

    input_col = dataset.get_input_column()
    output_col = dataset.get_output_column()
    ref_cols = dataset.get_reference_columns()

    df = dataset.get_dataframe()
    inputs = df[input_col].tolist()
    outputs = df[output_col].tolist()
    refs_per_col = {c: df[c].tolist() for c in ref_cols}
    uids = [ _compute_uid(inp) for inp in inputs ]

    # Expand rows for worse_obvious: one row per strategy per original record
    rows_worse = []
    for i, strat_outputs in enumerate(worse_obvious):
        for s_idx, strategy_name in enumerate(strategies):
            model_output = strat_outputs[s_idx] if s_idx < len(strat_outputs) else None
            # Build references list in canonical order for metrics expecting list[str]
            refs_list = []
            for c in ref_cols:
                val = refs_per_col[c][i]
                try:
                    is_missing = pd.isna(val)
                except Exception:
                    is_missing = False
                if not is_missing and val is not None and str(val) != "":
                    refs_list.append(val)
            row = {
                "dimension": dimension,
                "group": "worse_obvious",
                "strategy": strategy_name,
                "uid": uids[i],
                "input": inputs[i],
                "original_output": outputs[i],
                "model_output": model_output,
            }
            if include_references_json:
                row["references"] = json.dumps(refs_list, ensure_ascii=False)
            for c in ref_cols:
                row[c] = refs_per_col[c][i]
            rows_worse.append(row)

    # One row per original record for same_obvious
    rows_same = []
    for i, pert_out in enumerate(same_obvious):
        # Build references list
        refs_list = []
        for c in ref_cols:
            val = refs_per_col[c][i]
            try:
                is_missing = pd.isna(val)
            except Exception:
                is_missing = False
            if not is_missing and val is not None and str(val) != "":
                refs_list.append(val)
        row = {
            "dimension": dimension,
            "group": "same_obvious",
            "strategy": "same_obvious",
            "uid": uids[i],
            "input": inputs[i],
            "original_output": outputs[i],
            "model_output": pert_out,
        }
        if include_references_json:
            row["references"] = json.dumps(refs_list, ensure_ascii=False)
        for c in ref_cols:
            row[c] = refs_per_col[c][i]
        rows_same.append(row)

    return pd.DataFrame(rows_worse), pd.DataFrame(rows_same)


def main():
    parser = argparse.ArgumentParser(description="Generate robustness CSVs: original subset, perturbed_worse_obvious, perturbed_same_obvious")
    # Dataset selection: by canonical name (preferred) or by import path (fallback)
    parser.add_argument("--dataset_name", type=str, default=None, help="Dataset name (e.g., SimpDA, SimpEval, Primock57, HelpSteer2, EvalGenProduct, RealHumanEval, CoGymTravelOutcome, ...)")
    parser.add_argument("--dataset", type=str, default=None, help="Import path to Dataset class, e.g., autometrics.dataset.datasets.simplification.simplification.SimpDA")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test", help="Dataset split to use (default: test)")
    # Perturbation target/dimension
    parser.add_argument("--dimension", type=str, default=None, help="Target dimension/column to perturb (must be in dataset.get_target_columns())")
    parser.add_argument("--target_name", type=str, default=None, help="Alias for --dimension; uses target/measure name as perturbation dimension")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write CSVs")
    parser.add_argument("--lm", type=str, default=None, help="LM spec for dspy (e.g., openai/gpt-4o-mini)")
    parser.add_argument("--api-base", type=str, default=None, help="API base URL for local models (e.g., http://localhost:7450/v1)")
    parser.add_argument("--max_examples", type=int, default=30, help="Max examples from dataset to process (default: 30)")
    parser.add_argument("--num_examples", type=int, default=3, help="Number of demo examples used to induce strategies")
    parser.add_argument("--max_workers", type=int, default=24, help="Max workers for perturbation generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling")
    parser.add_argument("--dataset_kwargs", type=str, default=None, help="Optional Python dict literal for dataset init kwargs")
    parser.add_argument("--references-json", dest="references_json", action="store_true", help="Include a JSON 'references' column (default: off)")
    args = parser.parse_args()

    # Configure LM if provided
    if args.lm:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            print("WARNING: OPENAI_API_KEY not set; proceeding but LM calls may fail.")
        dspy.configure(lm=dspy.LM(args.lm, api_key=api_key, api_base=args.api_base))

    # Resolve perturbation dimension
    perturb_dimension = args.dimension or args.target_name
    if not perturb_dimension:
        print("ERROR: You must provide --dimension (or --target_name as an alias).")
        sys.exit(1)

    # Load dataset by name if available, else by import path
    dataset = None
    if args.dataset_name and load_dataset_by_name is not None:
        try:
            dataset = load_dataset_by_name(args.dataset_name)
        except Exception as e:
            print(f"Failed to load dataset by name '{args.dataset_name}': {e}")
            sys.exit(1)
    else:
        if not args.dataset:
            print("ERROR: Provide --dataset_name (preferred) or --dataset import path.")
            sys.exit(1)
        init_kwargs = None
        if args.dataset_kwargs:
            try:
                init_kwargs = eval(args.dataset_kwargs, {"__builtins__": {}}, {})  # simple, controlled eval of literal
                if not isinstance(init_kwargs, dict):
                    raise ValueError("dataset_kwargs must evaluate to a dict")
            except Exception as e:
                print(f"Failed to parse dataset_kwargs: {e}")
                sys.exit(1)
        dataset = _resolve_dataset(args.dataset, init_kwargs)

    # Subset if requested
    # Use persistent splits if available
    try:
        train_ds, val_ds, test_ds = dataset.load_permanent_splits()
        if args.split == "train":
            dataset = train_ds
        elif args.split == "val":
            dataset = val_ds
        else:
            dataset = test_ds
    except Exception:
        # Fallback to full dataset if splits unsupported
        pass

    dataset = _subset_dataframe(dataset, args.max_examples, args.seed)

    # Prepare output dir and write original subset
    _ensure_dir(args.output_dir)
    orig_csv = _write_original_subset_csv(
        dataset,
        args.output_dir,
        dimension=perturb_dimension,
        filename="original_subset.csv",
        include_references_json=args.references_json,
    )
    print(f"Wrote original subset to: {orig_csv}")

    # Produce perturbations for the specified dimension
    producer = ProducePerturbations(num_examples=args.num_examples, max_workers=args.max_workers)

    with dspy.settings.context(lm=dspy.settings.lm):
        perturbations = producer.forward(
            task=dataset.get_task_description(),
            dimension=perturb_dimension,
            dataset=dataset,
        )

    worse_df, same_df = _build_perturbed_tables(
        dataset,
        perturbations,
        perturb_dimension,
        include_references_json=args.references_json,
    )

    # Ensure stable column order with references appended after base columns
    ref_cols = dataset.get_reference_columns()
    base_cols = ["dimension", "group", "strategy", "uid", "input", "original_output", "model_output"]
    if args.references_json:
        base_cols.append("references")
    ordered_cols = base_cols + [c for c in ref_cols if c in worse_df.columns]
    worse_df = worse_df[ordered_cols]
    ordered_cols_same = base_cols + [c for c in ref_cols if c in same_df.columns]
    same_df = same_df[ordered_cols_same]

    worse_path = os.path.join(args.output_dir, "perturbed_worse_obvious.csv")
    same_path = os.path.join(args.output_dir, "perturbed_same_obvious.csv")

    worse_df.to_csv(worse_path, index=False)
    same_df.to_csv(same_path, index=False)

    print(f"Wrote perturbed worse (obvious) to: {worse_path}")
    print(f"Wrote perturbed same (obvious) to: {same_path}")


if __name__ == "__main__":
    main()

# EXAMPLE USAGE:
# python -m analysis.robustness.robustness_data --dataset_name SimpEval --target_name score --split test --output_dir ./outputs/robustness/csvs/simpeval_score/ --lm litellm_proxy/Qwen/Qwen3-32B --api-base http://sphinx3.stanford.edu:8544/v1
# python -m analysis.robustness.robustness_data --dataset_name HelpSteer2 --target_name helpfulness --split test --output_dir ./outputs/robustness/csvs/helpsteer2_helpfulness/ --lm litellm_proxy/Qwen/Qwen3-32B --api-base http://sphinx3.stanford.edu:8544/v1
# python -m analysis.robustness.robustness_data --dataset_name CoGymTravelOutcome --target_name outcomeRating --split test --output_dir ./outputs/robustness/csvs/cogymtraveloutcome_outcomerating/ --lm litellm_proxy/Qwen/Qwen3-32B --api-base http://sphinx3.stanford.edu:8544/v1
# python -m analysis.robustness.robustness_data --dataset_name EvalGenProduct --target_name grade --split test --output_dir ./outputs/robustness/csvs/evalgenproduct_grade/ --lm litellm_proxy/Qwen/Qwen3-32B --api-base http://sphinx3.stanford.edu:8544/v1
# python -m analysis.robustness.robustness_data --dataset_name RealHumanEval --target_name accepted --split test --output_dir ./outputs/robustness/csvs/realhumaneval_accepted/ --lm litellm_proxy/Qwen/Qwen3-32B --api-base http://sphinx3.stanford.edu:8544/v1
# python -m analysis.robustness.robustness_data --dataset_name Primock57 --target_name time_sec --split test --output_dir ./outputs/robustness/csvs/primock57_time_sec/ --lm litellm_proxy/Qwen/Qwen3-32B --api-base http://sphinx3.stanford.edu:8544/v1
