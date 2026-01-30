#!/usr/bin/env python3
"""
Run an AutoMetrics Metric or Aggregator on a CSV and write results back.

Examples:
  - Reference-free metric (needs input/output):
      python analysis/robustness/run_metric_on_csv.py \
        --csv path/to/data.csv \
        --class autometrics.metrics.reference_based.BLEU.BLEU \
        --input-column source --output-column prediction \
        --reference-columns reference

  - Aggregator with embedded dependencies (e.g., exported static regression):
      python analysis/robustness/run_metric_on_csv.py \
        --csv path/to/data.csv \
        --class some_exported.StaticRegression_MyTask \
        --input-column prompt --output-column response

  - Aggregator with explicit dependencies:
      python analysis/robustness/run_metric_on_csv.py \
        --csv data.csv \
        --class autometrics.aggregator.regression.Regression.Regression \
        --dependency autometrics.metrics.reference_based.BLEU.BLEU \
        --dependency autometrics.metrics.reference_free.INFORMRewardModel.INFORMRewardModel \
        --input-column src --output-column hyp --reference-columns ref

Notes:
  - For reference-based metrics, you must specify --reference-columns (comma-separated for multiple).
  - You can pass initialization kwargs using repeated --param flags like: --param use_cache=true --param seed=42
    Values are parsed loosely: "true/false" -> bool, numeric -> int/float, JSON for lists/dicts, else string.
  - By default, updates the CSV in-place. Use --output to write to a different file instead.

Python API:
  from analysis.robustness.run_metric_on_csv import run_metric_or_aggregator_on_dataframe

  df_out = run_metric_or_aggregator_on_dataframe(
      metric_or_aggregator=my_metric_or_agg,  # instance, class, or import path string
      df=df,
      input_column="prompt",
      output_column="response",
      reference_columns=["reference"],  # optional
      metric_name=None,                  # optional rename
      with_feedback=True,                # metrics only
      dependencies=[dep_metric1, dep_metric2],  # aggregators only, optional
      init_params={"use_cache": True},        # used if a class or string is provided
      cache_dir="/tmp/am_cache"               # optional
  )
"""

import argparse
import importlib
import json
import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd

# Ensure project root is importable
PROJECT_ROOT = "/nlp/scr2/nlp/personal-rm/autometrics"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from autometrics.dataset.Dataset import Dataset


def parse_kv_params(pairs: List[str]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for p in pairs or []:
        if "=" not in p:
            raise ValueError(f"--param must be key=value, got: {p}")
        k, v = p.split("=", 1)
        k = k.strip()
        v = v.strip()
        # Try to coerce value
        if v.lower() in ("true", "false"):
            coerced: Any = v.lower() == "true"
        else:
            try:
                if v.startswith("[") or v.startswith("{"):
                    coerced = json.loads(v)
                else:
                    # numeric?
                    if "." in v:
                        coerced = float(v)
                        # if it was like "1.0" but is int-like, keep float anyway
                    else:
                        coerced = int(v)
            except Exception:
                coerced = v
        params[k] = coerced
    return params


def import_by_path(class_path: str):
    if "." not in class_path:
        raise ValueError("--class must be a full module path like pkg.mod.ClassName")
    module_path, cls_name = class_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, cls_name)
    return cls


def instantiate_class(cls, kwargs: Dict[str, Any]):
    try:
        return cls(**kwargs)
    except TypeError as e:
        # Helpful hint when user passed unknown kwargs
        raise TypeError(f"Failed to construct {cls.__module__}.{cls.__name__} with kwargs {kwargs}: {e}")


def build_dataset(df: pd.DataFrame,
                  input_column: Optional[str],
                  output_column: Optional[str],
                  reference_columns: Optional[List[str]],
                  name: str) -> Dataset:
    # Basic validation: columns must exist if provided
    if input_column and input_column not in df.columns:
        raise KeyError(f"input_column '{input_column}' not in CSV columns: {list(df.columns)}")
    if output_column and output_column not in df.columns:
        raise KeyError(f"output_column '{output_column}' not in CSV columns: {list(df.columns)}")
    if reference_columns:
        missing = [c for c in reference_columns if c not in df.columns]
        if missing:
            raise KeyError(f"reference_columns missing from CSV: {missing}")

    dataset = Dataset(
        dataframe=df,
        target_columns=[],
        ignore_columns=[],
        metric_columns=[],
        name=name,
        data_id_column=None,
        model_id_column=None,
        input_column=input_column,
        output_column=output_column,
        reference_columns=reference_columns,
        metrics=[],
        task_description=None,
    )
    return dataset


def ensure_is_metric_or_aggregator(obj) -> str:
    # Returns "metric" or "aggregator"
    try:
        from autometrics.metrics.Metric import Metric
        from autometrics.aggregator.Aggregator import Aggregator
        if issubclass(obj, Metric):
            return "metric"
        if issubclass(obj, Aggregator):
            return "aggregator"
    except Exception:
        pass
    raise TypeError("Provided class is neither a Metric nor an Aggregator.")


def _coerce_to_instance(metric_or_aggregator: Any,
                        init_params: Optional[Dict[str, Any]] = None,
                        dependencies: Optional[List[Any]] = None) -> tuple[Any, str]:
    """
    Accepts an instance, class object, or import path string and returns (instance, kind).
    If an Aggregator and dependencies are provided and the instance has no input_metrics,
    attach them.
    """
    init_params = init_params or {}

    # If string path
    if isinstance(metric_or_aggregator, str):
        cls = import_by_path(metric_or_aggregator)
        kind = ensure_is_metric_or_aggregator(cls)
        if kind == "aggregator" and dependencies:
            # If dependencies are provided as strings/classes/instances, coerce each to instance
            dep_instances = []
            for d in dependencies:
                if isinstance(d, str):
                    d_cls = import_by_path(d)
                    dep_instances.append(instantiate_class(d_cls, {}))
                elif isinstance(d, type):
                    dep_instances.append(instantiate_class(d, {}))
                else:
                    dep_instances.append(d)
            local_params = dict(init_params)
            local_params.setdefault("input_metrics", dep_instances)
            inst = instantiate_class(cls, local_params)
        else:
            inst = instantiate_class(cls, init_params)
        return inst, kind

    # If class object
    if isinstance(metric_or_aggregator, type):
        cls = metric_or_aggregator
        kind = ensure_is_metric_or_aggregator(cls)
        if kind == "aggregator" and dependencies:
            dep_instances = []
            for d in dependencies:
                if isinstance(d, str):
                    d_cls = import_by_path(d)
                    dep_instances.append(instantiate_class(d_cls, {}))
                elif isinstance(d, type):
                    dep_instances.append(instantiate_class(d, {}))
                else:
                    dep_instances.append(d)
            local_params = dict(init_params)
            local_params.setdefault("input_metrics", dep_instances)
            inst = instantiate_class(cls, local_params)
        else:
            inst = instantiate_class(cls, init_params)
        return inst, kind

    # Else assume instance
    inst = metric_or_aggregator
    # Try to determine kind via MRO
    try:
        from autometrics.metrics.Metric import Metric
        from autometrics.aggregator.Aggregator import Aggregator
        if isinstance(inst, Metric):
            kind = "metric"
        elif isinstance(inst, Aggregator):
            kind = "aggregator"
        else:
            raise TypeError
    except Exception:
        raise TypeError("metric_or_aggregator must be an instance of Metric or Aggregator (or class/path thereof)")

    # Optionally attach dependencies to aggregator
    if kind == "aggregator" and dependencies:
        try:
            if getattr(inst, "input_metrics", None) in (None, []):
                dep_instances = []
                for d in dependencies:
                    if isinstance(d, str):
                        d_cls = import_by_path(d)
                        dep_instances.append(instantiate_class(d_cls, {}))
                    elif isinstance(d, type):
                        dep_instances.append(instantiate_class(d, {}))
                    else:
                        dep_instances.append(d)
                setattr(inst, "input_metrics", dep_instances)
        except Exception:
            pass

    return inst, kind


def run_metric_or_aggregator_on_dataframe(
    metric_or_aggregator: Any,
    df: pd.DataFrame,
    *,
    input_column: Optional[str],
    output_column: Optional[str],
    reference_columns: Optional[List[str]] = None,
    metric_name: Optional[str] = None,
    with_feedback: bool = True,
    dependencies: Optional[List[Any]] = None,
    init_params: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Programmatic API to run a Metric or Aggregator on a DataFrame and return an updated DataFrame.
    - metric_or_aggregator: instance, class, or import path string
    - dependencies: only used for Aggregators (list of metrics as instances/classes/paths)
    - init_params: only used if a class or path string is provided
    """
    if cache_dir:
        os.environ["AUTOMETRICS_CACHE_DIR"] = cache_dir
        if init_params is not None and "cache_dir" not in init_params:
            init_params = dict(init_params)
            init_params["cache_dir"] = cache_dir

    inst, kind = _coerce_to_instance(metric_or_aggregator, init_params=init_params, dependencies=dependencies)

    # Optional rename
    if metric_name and hasattr(inst, "name"):
        try:
            inst.name = metric_name
        except Exception:
            pass

    dataset = build_dataset(df, input_column, output_column, reference_columns, name="programmatic")

    if kind == "metric":
        inst.predict(dataset, update_dataset=True, with_feedback=with_feedback)
    else:
        inst.predict(dataset, update_dataset=True)

    return dataset.get_dataframe()


def run_metric_or_aggregator_on_csv(
    metric_or_aggregator: Any,
    csv_path: str,
    *,
    input_column: Optional[str],
    output_column: Optional[str],
    reference_columns: Optional[List[str]] = None,
    metric_name: Optional[str] = None,
    with_feedback: bool = True,
    dependencies: Optional[List[Any]] = None,
    init_params: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
    output_csv: Optional[str] = None,
) -> str:
    df = pd.read_csv(csv_path)
    df_out = run_metric_or_aggregator_on_dataframe(
        metric_or_aggregator,
        df,
        input_column=input_column,
        output_column=output_column,
        reference_columns=reference_columns,
        metric_name=metric_name,
        with_feedback=with_feedback,
        dependencies=dependencies,
        init_params=init_params,
        cache_dir=cache_dir,
    )
    out_path = output_csv or csv_path
    df_out.to_csv(out_path, index=False)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Run an AutoMetrics Metric or Aggregator over a CSV and write results.")
    parser.add_argument("--csv", required=True, help="Path to input CSV")
    parser.add_argument("--class", dest="class_path", required=True, help="Full class path (e.g., autometrics.metrics.reference_based.BLEU.BLEU)")
    parser.add_argument("--input-column", dest="input_column", default=None)
    parser.add_argument("--output-column", dest="output_column", default=None)
    parser.add_argument("--reference-columns", dest="reference_columns", default=None, help="Comma-separated list if multiple")
    parser.add_argument("--metric-name", dest="metric_name", default=None, help="Optional override for the output column name (defaults to .get_name())")
    parser.add_argument("--param", dest="params", action="append", default=[], help="Constructor param as key=value; repeat for multiple")
    parser.add_argument("--dependency", dest="dependencies", action="append", default=[], help="For Aggregators: dependency metric class path; repeatable")
    parser.add_argument("--output", dest="output_csv", default=None, help="Output CSV path. If omitted, overwrite the input CSV.")
    parser.add_argument("--no-feedback", dest="no_feedback", action="store_true", help="For Metrics: do not write feedback columns even if supported")
    parser.add_argument("--cache-dir", dest="cache_dir", default=None, help="Optional override for AUTOMETRICS_CACHE_DIR and metric cache_dir init param")

    args = parser.parse_args()

    in_path = args.csv
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"CSV not found: {in_path}")

    df = pd.read_csv(in_path)

    # Parse reference columns
    ref_cols: Optional[List[str]] = None
    if args.reference_columns:
        ref_cols = [c.strip() for c in args.reference_columns.split(",") if c.strip()]

    dataset = build_dataset(
        df=df,
        input_column=args.input_column,
        output_column=args.output_column,
        reference_columns=ref_cols,
        name=os.path.basename(in_path),
    )

    # Prepare cache directory
    ctor_params = parse_kv_params(args.params)
    if args.cache_dir:
        os.environ["AUTOMETRICS_CACHE_DIR"] = args.cache_dir
        # If user didn't pass cache_dir in params, inject it so Metric init respects it
        ctor_params.setdefault("cache_dir", args.cache_dir)

    # Build instance from CLI spec and run via the shared function
    with_feedback = not args.no_feedback
    deps_for_cli: Optional[List[Any]] = args.dependencies or None

    # Use the flexible API: allows class path now, but later also instances
    out_path = run_metric_or_aggregator_on_csv(
        metric_or_aggregator=args.class_path,
        csv_path=in_path,
        input_column=args.input_column,
        output_column=args.output_column,
        reference_columns=[c.strip() for c in args.reference_columns.split(",")] if args.reference_columns else None,
        metric_name=args.metric_name,
        with_feedback=with_feedback,
        dependencies=deps_for_cli,
        init_params=ctor_params,
        cache_dir=args.cache_dir,
        output_csv=args.output_csv,
    )
    print(f"✅ Wrote results to: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


