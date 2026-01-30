import os
import math
import time
import json
import statistics
from typing import Any, Dict, List, Optional, Tuple, Union

import dspy
import re
import numpy as np
try:
    from scipy.stats import pearsonr, kendalltau
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def _safe_getattr(obj: Any, name: str, default: Any = None):
    try:
        return getattr(obj, name)
    except Exception:
        return default


def _is_multimetric(metric: Any) -> bool:
    try:
        from autometrics.metrics.MultiMetric import MultiMetric
        return isinstance(metric, MultiMetric)
    except Exception:
        return False


def _expand_metric_columns(metric: Any) -> List[str]:
    if _is_multimetric(metric):
        try:
            return list(metric.get_submetric_names() or [])
        except Exception:
            pass
    try:
        return [metric.get_name()] if hasattr(metric, 'get_name') else []
    except Exception:
        return []


def extract_feature_names(regression_metric: Any, metrics: List[Any], dataset: Optional[Any]) -> List[str]:
    # Preferred: ask regression for selected columns
    selected = []
    try:
        if hasattr(regression_metric, 'get_selected_columns'):
            cols = regression_metric.get_selected_columns()
            if cols:
                selected = list(cols)
    except Exception:
        selected = []

    if selected:
        return selected

    # Fallback: derive from metric instances
    names: List[str] = []
    for m in metrics:
        names.extend(_expand_metric_columns(m))
    # As a last resort, consult dataset metric columns
    if not names and dataset is not None:
        try:
            names = list(dataset.get_metric_columns() or [])
        except Exception:
            pass
    return names


def extract_regression_coefficients(regression_metric: Any, feature_names: List[str]) -> List[Tuple[str, float]]:
    # Try scikit-learn style attributes
    model = _safe_getattr(regression_metric, 'model')
    coef = None
    intercept = None
    if model is not None:
        coef = _safe_getattr(model, 'coef_')
        intercept = _safe_getattr(model, 'intercept_')
    else:
        coef = _safe_getattr(regression_metric, 'coef_')
        intercept = _safe_getattr(regression_metric, 'intercept_')

    pairs: List[Tuple[str, float]] = []
    try:
        if coef is not None:
            # handle shape (n_features,) or (1, n_features)
            coef_list: List[float]
            if hasattr(coef, 'tolist'):
                coef_list = coef.tolist()
            else:
                coef_list = list(coef)
            if isinstance(coef_list, list) and len(coef_list) == 1 and isinstance(coef_list[0], list):
                coef_list = coef_list[0]
            for i, name in enumerate(feature_names[:len(coef_list)]):
                try:
                    pairs.append((name, float(coef_list[i])))
                except Exception:
                    pairs.append((name, 0.0))
    except Exception:
        pass

    # Include intercept as Aggregate if useful for chart footer
    if intercept is not None:
        try:
            if isinstance(intercept, (list, tuple)) and len(intercept) == 1:
                intercept_val = float(intercept[0])
            else:
                intercept_val = float(intercept)
            pairs.append(('(intercept)', intercept_val))
        except Exception:
            pass

    return pairs


def ensure_eval_metrics(eval_dataset: Any, metrics: List[Any]) -> Any:
    # Add metrics to eval dataset if missing; do not recompute if columns exist
    if eval_dataset is None:
        return None
    df = eval_dataset.get_dataframe()
    missing: List[Any] = []
    for m in metrics:
        cols = _expand_metric_columns(m)
        if not cols:
            missing.append(m)
            continue
        if not all(c in df.columns for c in cols):
            missing.append(m)
    if not missing:
        return eval_dataset

    # Use dataset.add_metric which internally handles caching/parallel as implemented by the dataset
    for m in missing:
        try:
            eval_dataset.add_metric(m, update_dataset=True)
        except Exception:
            # As a fallback try per-batch compute
            try:
                inputs = df[eval_dataset.get_input_column()].tolist()
                outs = df[eval_dataset.get_output_column()].tolist()
                refs_cols = eval_dataset.get_reference_columns() or []
                refs = [[df[c].iloc[i] for c in refs_cols] for i in range(len(df))] if refs_cols else None
                if hasattr(m, 'calculate_batched'):
                    scores = m.calculate_batched(inputs, outs, refs)
                    col = m.get_name()
                    df[col] = scores
                    eval_dataset.set_dataframe(df)
            except Exception:
                continue
    return eval_dataset


def _clean_xy(x: List[Any], y: List[Any]) -> Tuple[np.ndarray, np.ndarray]:
    # Convert to numeric and filter NaNs
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    return x_arr[mask], y_arr[mask]


def compute_correlation(eval_dataset: Any, feature_names: List[str], target_measure: str, include_regression: bool = True, regression_col_name: Optional[str] = None) -> Dict[str, Any]:
    df = eval_dataset.get_dataframe().copy()
    results: Dict[str, Any] = { 'metrics': [] }
    if include_regression and target_measure in df.columns:
        pass
    y_raw = df[target_measure].tolist()
    # Prepare IDs aligned to dataframe rows (match Examples table behavior)
    if 'sample_id' in df.columns:
        ids_all = df['sample_id'].astype(str).tolist()
    else:
        # Use 1-based row numbers to match Examples table fallback
        ids_all = [str(i+1) for i in range(len(df))]
    # Determine y range for normalization
    y_arr_all = np.array([v for v in y_raw if isinstance(v, (int, float)) or (isinstance(v, np.generic))], dtype=float)
    y_min = float(np.nanmin(y_arr_all)) if y_arr_all.size > 0 else 0.0
    y_max = float(np.nanmax(y_arr_all)) if y_arr_all.size > 0 else 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0
    for name in feature_names:
        if name not in df.columns:
            continue
        x_raw = df[name].tolist()
        # Build mask to filter finite pairs and carry IDs through
        x_arr = np.array(x_raw, dtype=float)
        y_arr = np.array(y_raw, dtype=float)
        mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        x_clean = x_arr[mask]
        y_clean = y_arr[mask]
        ids_clean = [ids_all[i] for i, keep in enumerate(mask) if bool(keep)]
        # Normalize x to y range for plotting (leave correlation on raw values)
        if x_clean.size > 0:
            x_min = float(np.nanmin(x_clean))
            x_max = float(np.nanmax(x_clean))
            if x_max > x_min:
                x_norm = ((x_clean - x_min) / (x_max - x_min)) * (y_max - y_min) + y_min
            else:
                x_norm = np.full_like(x_clean, (y_min + y_max) / 2.0)
        else:
            x_norm = x_clean
        r_val: Optional[float] = None
        tau_val: Optional[float] = None
        if len(x_clean) >= 3:
            if _HAVE_SCIPY:
                try:
                    r_val = float(pearsonr(x_clean, y_clean).statistic)
                except Exception:
                    r_val = None
                try:
                    tau_val = float(kendalltau(x_clean, y_clean).statistic)
                except Exception:
                    tau_val = None
            else:
                # Fallback to numpy corrcoef for Pearson only, leave tau as None
                try:
                    if np.std(x_clean) > 0 and np.std(y_clean) > 0:
                        r_val = float(np.corrcoef(x_clean, y_clean)[0, 1])
                except Exception:
                    r_val = None
        results['metrics'].append({ 'name': name, 'r': r_val, 'tau': tau_val, 'x': x_clean.tolist(), 'x_norm': x_norm.tolist() if x_clean.size > 0 else [], 'y': y_clean.tolist(), 'ids': ids_clean, 'y_min': y_min, 'y_max': y_max })
    # Try regression column if present
    if include_regression:
        reg_cols = [c for c in df.columns if c.lower().startswith('autometrics_regression_')]
        # Fallback: use provided regression column name, if available
        if not reg_cols and regression_col_name and regression_col_name in df.columns:
            reg_cols = [regression_col_name]
        if reg_cols:
            col = reg_cols[0]
            x_raw = df[col].tolist()
            x_arr = np.array(x_raw, dtype=float)
            y_arr = np.array(y_raw, dtype=float)
            mask = np.isfinite(x_arr) & np.isfinite(y_arr)
            x_clean = x_arr[mask]
            y_clean = y_arr[mask]
            ids_clean = [ids_all[i] for i, keep in enumerate(mask) if bool(keep)]
            # Regression metric is already on the same scale as y; for plotting, keep as-is
            x_norm = x_clean
            r_val: Optional[float] = None
            tau_val: Optional[float] = None
            if len(x_clean) >= 3:
                if _HAVE_SCIPY:
                    try:
                        r_val = float(pearsonr(x_clean, y_clean).statistic)
                    except Exception:
                        r_val = None
                    try:
                        tau_val = float(kendalltau(x_clean, y_clean).statistic)
                    except Exception:
                        tau_val = None
                else:
                    try:
                        if np.std(x_clean) > 0 and np.std(y_clean) > 0:
                            r_val = float(np.corrcoef(x_clean, y_clean)[0, 1])
                    except Exception:
                        r_val = None
            results['regression'] = { 'name': col, 'r': r_val, 'tau': tau_val, 'x': x_clean.tolist(), 'x_norm': x_norm.tolist(), 'y': y_clean.tolist(), 'ids': ids_clean, 'y_min': y_min, 'y_max': y_max }
    return results


def run_robustness(
    eval_dataset: Any,
    metrics: List[Any],
    target_measure: str,
    lm: Optional[dspy.LM],
    verbose: bool = False,
    output_dir: Optional[str] = None,
    regression_name: Optional[str] = None,
    regression_coeffs: Optional[List[Tuple[str, float]]] = None,
) -> Dict[str, Any]:
    try:
        from autometrics.experiments.robustness.robustness import RobustnessExperiment
        from autometrics.experiments.results import TabularResult
    except Exception as e:
        if verbose:
            print(f"[ReportCard][Robustness] Import failed: {e}")
        return { 'available': False, 'error': f'robustness experiment not available: {e}' }

    # Build a slim experiment using provided eval dataset and metrics
    try:
        out_dir = output_dir or os.path.join("artifacts", "robustness")
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        out_dir = None
    exp = RobustnessExperiment(
        name="Metric Robustness", description="Compute stability and sensitivity on eval dataset",
        metrics=metrics, output_dir=out_dir, dataset=eval_dataset, seed=42
    )
    exp.kwargs = { 'lm': lm }
    # Limit to "obvious" mode by relying on default configured perturbations; the experiment internally decides strategies
    try:
        prev_lm = None
        try:
            prev_lm = dspy.settings.lm
        except Exception:
            prev_lm = None
        if lm is not None:
            try:
                dspy.configure(lm=lm)
            except Exception:
                pass
        with dspy.settings.context(lm=lm):
            exp.run(
                print_results=False,
                num_demonstration_examples=3,
                max_eval_examples=min(30, len(eval_dataset.get_dataframe())),
                max_workers=32,
            )
        # restore
        try:
            if prev_lm is not None:
                dspy.configure(lm=prev_lm)
        except Exception:
            pass
    except Exception as e:
        if verbose:
            print(f"[ReportCard][Robustness] Run failed: {e}")
        return { 'available': False, 'error': f'robustness run failed: {e}' }

    # Parse results to compute stability/sensitivity. Expect TabularResult at key "<target>/full_table"
    key = f"{target_measure}/full_table"
    if key not in exp.results:
        if verbose:
            print(f"[ReportCard][Robustness] Results missing key: {key}")
        return { 'available': False, 'error': f'robustness results missing at key {key}' }
    table: TabularResult = exp.results[key]
    try:
        df = table.dataframe if hasattr(table, 'dataframe') else table.data
    except Exception as e:
        if verbose:
            print(f"[ReportCard][Robustness] Table extraction failed: {e}")
        return { 'available': False, 'error': f'robustness result table error: {e}' }

    # For each metric column in df (present in eval dataset metrics), compute metrics using paired deltas
    out: Dict[str, Dict[str, float]] = {}

    metric_cols: List[str] = []
    # gather metric columns from input metrics
    for m in metrics:
        metric_cols.extend(_expand_metric_columns(m))

    # If regression coefficients provided, compute aggregate regression column ONLY when no Aggregator present
    has_aggregator = False
    try:
        from autometrics.aggregator.Aggregator import Aggregator as _Agg
        has_aggregator = any(isinstance(m, _Agg) for m in metrics)
    except Exception:
        has_aggregator = False
    if regression_coeffs and not has_aggregator:
        try:
            intercept = 0.0
            terms = []
            for name, coef in regression_coeffs:
                if name == '(intercept)':
                    try:
                        intercept = float(coef)
                    except Exception:
                        intercept = 0.0
                else:
                    if name in df.columns:
                        try:
                            terms.append(float(coef) * df[name].astype(float))
                        except Exception:
                            pass
            if terms:
                agg_series = sum(terms)
                try:
                    agg_series = agg_series + intercept
                except Exception:
                    pass
                reg_col_name = regression_name or 'Autometrics_Aggregate'
                df[reg_col_name] = agg_series
                metric_cols.append(reg_col_name)
        except Exception as e:
            if verbose:
                print(f"[ReportCard][Robustness] Failed to compute regression aggregate: {e}")

    # Ensure we have sample_id for pairing; if not present, derive a hash from input text
    if 'sample_id' not in df.columns:
        try:
            import hashlib as _hashlib
            df['sample_id'] = df['input'].astype(str).str.strip().str.lower().apply(lambda x: _hashlib.md5(x.encode()).hexdigest())
        except Exception:
            return { 'available': False, 'error': 'sample_id missing and could not be derived' }

    # Build paired deltas with normalization to [0,1]:
    # For each metric column, normalize over all rows in this robustness table,
    # then compute per-sample deltas (orig vs worse/same) and average across samples.
    by_id = df.groupby('sample_id')
    for col in metric_cols:
        if col not in df.columns:
            continue
        sens_vals: List[float] = []
        stab_vals: List[float] = []
        # Normalize this metric to [0,1] over the entire robustness table
        try:
            col_series = df[col].astype(float)
            s_min = float(np.nanmin(col_series.values))
            s_max = float(np.nanmax(col_series.values))
            if not np.isfinite(s_min) or not np.isfinite(s_max):
                continue
            denom_all = s_max - s_min
            if denom_all <= 0:
                norm_series = np.zeros(len(col_series), dtype=float)
            else:
                norm_series = (col_series.values - s_min) / denom_all
        except Exception as e:
            if verbose:
                print(f"[ReportCard][Robustness] Normalization failed for {col}: {e}")
            continue
        for sid, g in by_id:
            try:
                g_orig = g[g['group'] == 'original']
                if len(g_orig) == 0:
                    continue
                idx_orig = g_orig.index
                v_orig = float(np.mean(norm_series[idx_orig]))
                # Sensitivity: drop on worse_obvious (average over strategies per sample)
                g_w = g[g['group'] == 'worse_obvious']
                if len(g_w) > 0:
                    idx_w = g_w.index
                    v_w = float(np.mean(norm_series[idx_w]))
                    sens_vals.append(v_orig - v_w)
                # Stability: closeness on same_obvious
                g_s = g[g['group'] == 'same_obvious']
                if len(g_s) > 0:
                    idx_s = g_s.index
                    v_s = float(np.mean(norm_series[idx_s]))
                    stab_vals.append(1.0 - abs(v_orig - v_s))
            except Exception as e:
                if verbose:
                    print(f"[ReportCard][Robustness] Pairing failed for {col}, id={sid}: {e}")
                continue
        sens = float(np.mean(sens_vals)) if len(sens_vals) else None
        stab = float(np.mean(stab_vals)) if len(stab_vals) else None
        out[col] = { 'sensitivity': sens, 'stability': stab }
        if verbose:
            print(f"[ReportCard][Robustness] {col}: paired_sens_n={len(sens_vals)}, paired_stab_n={len(stab_vals)}, sens={sens}, stab={stab}")

    return { 'available': True, 'scores': out }


def measure_runtime(eval_dataset: Any, metrics: List[Any], sample_size: int = 30) -> Dict[str, Any]:
    # Subsample
    df = eval_dataset.get_dataframe()
    n = min(sample_size, len(df))
    sample_df = df.sample(n=n, random_state=42) if len(df) > n else df.copy()

    inputs = sample_df[eval_dataset.get_input_column()].tolist()
    outputs = sample_df[eval_dataset.get_output_column()].tolist()
    ref_cols = eval_dataset.get_reference_columns() or []
    refs_list = [[sample_df[c].iloc[i] for c in ref_cols] for i in range(len(sample_df))] if ref_cols else [None] * len(sample_df)

    per_metric_times: Dict[str, List[float]] = {}
    # Run sequentially per-example
    for m in metrics:
        col_name = m.get_name() if hasattr(m, 'get_name') else type(m).__name__
        per_metric_times[col_name] = []
        for i in range(len(sample_df)):
            start = time.time()
            try:
                if hasattr(m, 'calculate'):
                    m.calculate(inputs[i], outputs[i], refs_list[i])
                elif hasattr(m, 'calculate_batched'):
                    m.calculate_batched([inputs[i]], [outputs[i]], [refs_list[i]] if ref_cols else None)
                else:
                    # Fall back to dataset evaluation for this single row by creating a tiny dataset copy
                    sd = eval_dataset.get_subset(1, seed=42)
                    sd.set_dataframe(sample_df.iloc[[i]].copy())
                    sd.add_metric(m, update_dataset=True)
            except Exception:
                pass
            per_metric_times[col_name].append(max(0.0, time.time() - start))

    # Trim extremes and prepare box data; compute aggregate per-example seq and parallel
    trimmed: Dict[str, List[float]] = {}
    for name, times in per_metric_times.items():
        t = list(times)
        if len(t) >= 6:
            t_sorted = sorted(t)
            t = t_sorted[2:-2]
        trimmed[name] = t

    # Aggregate times per example
    metric_names = list(trimmed.keys())
    seq_times: List[float] = []
    par_times: List[float] = []
    for i in range(len(sample_df)):
        vals = [per_metric_times[m][i] for m in metric_names if i < len(per_metric_times[m])]
        if not vals:
            vals = [0.0]
        seq_times.append(sum(vals))
        par_times.append(max(vals))

    return {
        'per_metric': trimmed,
        'aggregate': {
            'sequence_times': seq_times,
            'parallel_times': par_times,
            'sequence_mean': float(statistics.fmean(seq_times)) if seq_times else 0.0,
            'parallel_mean': float(statistics.fmean(par_times)) if par_times else 0.0,
            # 95% CI using normal approx: 1.96 * std / sqrt(n)
            'sequence_ci': (lambda arr: (1.96 * (float(np.std(arr, ddof=1)) / max(1, int(len(arr) ** 0.5)))) if len(arr) > 1 else 0.0)(seq_times),
            'parallel_ci': (lambda arr: (1.96 * (float(np.std(arr, ddof=1)) / max(1, int(len(arr) ** 0.5)))) if len(arr) > 1 else 0.0)(par_times),
        },
        'sample_size': len(sample_df)
    }


def parse_metric_cards(metric_classes: List[Any]) -> Dict[str, Dict[str, str]]:
    parsed: Dict[str, Dict[str, str]] = {}
    for m in metric_classes:
        name = m.get_name() if hasattr(m, 'get_name') else (m.__name__ if hasattr(m, '__name__') else type(m).__name__)
        doc = _safe_getattr(m, '__doc__') or _safe_getattr(type(m), '__doc__') or ''
        desc = _safe_getattr(m, 'description') or ''
        sections = { 'description': '', 'usage': '', 'limitations': '' }
        if isinstance(doc, str) and len(doc) > 0:
            low = doc.lower()

            def extract_between_any(start_key: str, stop_keys: List[str], stop_on_subheadings: bool = True) -> str:
                """Extract text between a start heading and the next stop heading.
                If stop_on_subheadings is False, ignore '###' boundaries.
                """
                idx = low.find(start_key)
                if idx == -1:
                    return ''
                start = idx + len(start_key)
                end = len(doc)
                # Build an ordered list of stop markers respecting subheading rule
                for nk in stop_keys:
                    if not stop_on_subheadings and nk.strip('#') == '##':
                        # Skip '###' style stop when ignoring subheadings
                        continue
                    j = low.find(nk, start)
                    if j != -1:
                        end = min(end, j)
                return doc[start:end].strip()

            # Description: take the paragraph under "### Metric Description" until next '###'/'##'
            sections['description'] = extract_between_any('metric description', ['###', '##', 'known limitations', 'limitations'], True) or desc or ''

            # Usage: prefer full Intended Use section via regex to be robust
            intended = ''
            try:
                m = re.search(r"^##\s*Intended\s*Use\s*$([\s\S]*?)(^##\s+|\Z)", doc, flags=re.IGNORECASE | re.MULTILINE)
                if m:
                    intended = m.group(1).strip()
            except Exception:
                intended = ''
            if not intended:
                # Fallback: up to next top-level '##' (ignore '###' subsections boundary)
                intended = extract_between_any('intended use', ['##', 'known limitations', 'limitations'], False)
            if intended:
                sections['usage'] = intended
            else:
                sections['usage'] = extract_between_any('usage', ['##', 'known limitations', 'limitations'], False)

            # Limitations
            lim = extract_between_any('known limitations', ['###', '##'], True)
            if not lim:
                lim = extract_between_any('limitations', ['###', '##'], True)
            sections['limitations'] = lim
        else:
            sections['description'] = desc
        parsed[name] = sections
    return parsed


def compute_requirements(metric_classes: List[Any]) -> List[Dict[str, Union[str, float]]]:
    rows: List[Dict[str, Union[str, float]]] = []
    for m in metric_classes:
        name = m.get_name() if hasattr(m, 'get_name') else (m.__name__ if hasattr(m, '__name__') else type(m).__name__)
        gpu = _safe_getattr(m, 'gpu_mem', None)
        cpu = _safe_getattr(m, 'cpu_mem', None)
        rows.append({ 'name': name, 'gpu_mem': '--' if gpu is None else float(gpu), 'cpu_mem': '--' if cpu is None else float(cpu) })
    return rows


def collect_metric_docs(metric_classes: List[Any]) -> Dict[str, str]:
    docs: Dict[str, str] = {}
    for m in metric_classes:
        try:
            name = m.get_name() if hasattr(m, 'get_name') else (m.__name__ if hasattr(m, '__name__') else type(m).__name__)
            doc = _safe_getattr(m, '__doc__') or _safe_getattr(type(m), '__doc__') or ''
            if isinstance(doc, str) and len(doc) > 0:
                # Clean common leading marker if present
                if doc.startswith('---\n'):
                    docs[name] = doc
                else:
                    docs[name] = doc
        except Exception:
            continue
    return docs


def build_examples_table(
    eval_dataset: Any,
    feature_names: List[str],
    target_measure: str,
    include_regression: bool = True,
    max_rows: Optional[int] = None,
    aggregator_feedback_cols: Optional[List[str]] = None,
    include_per_metric_feedback: bool = False,
) -> str:
    df = eval_dataset.get_dataframe()
    # Select columns safely
    cols = [eval_dataset.get_input_column(), eval_dataset.get_output_column()]
    ref_cols = eval_dataset.get_reference_columns() or []
    cols.extend(ref_cols)
    if target_measure not in cols:
        cols.append(target_measure)
    for f in feature_names:
        if f not in cols and f in df.columns:
            cols.append(f)
    if include_regression:
        reg_cols = [c for c in df.columns if c.lower().startswith('autometrics_regression_')]
        cols.extend(reg_cols)
    # Add aggregated regression feedback columns by default (if provided and present)
    if aggregator_feedback_cols:
        for c in aggregator_feedback_cols:
            if c in df.columns and c not in cols:
                cols.append(c)
    # Optionally include per-metric feedback columns (off by default to reduce clutter)
    if include_per_metric_feedback:
        for base in feature_names:
            fb = f"{base}__feedback"
            if fb in df.columns and fb not in cols:
                cols.append(fb)
    sub = df[cols]
    if max_rows is not None:
        sub = sub.head(max_rows)
    # Insert a stable row ID as the first column
    sub = sub.copy()
    if 'sample_id' in df.columns:
        sub.insert(0, 'ID', df['sample_id'])
    else:
        sub.insert(0, 'ID', range(1, len(sub) + 1))
    # Render Bootstrap-styled table and wrap in a scrollable container
    # Render multiline feedback nicely: convert newlines to <br/> for any feedback columns
    try:
        fb_cols = [c for c in sub.columns if isinstance(c, str) and c.endswith('__feedback')]
        for c in fb_cols:
            try:
                sub[c] = sub[c].apply(lambda v: (v if isinstance(v, str) else '').replace('\n', '<br/>'))
            except Exception:
                pass
        html_table = sub.to_html(index=False, escape=False, classes=['table', 'table-striped', 'table-sm', 'w-100'], table_id='examples-table')
    except TypeError:
        # Fallback for older pandas without table_id support
        html_table = sub.to_html(index=False, escape=False, classes=['table', 'table-striped', 'table-sm', 'w-100'])
        html_table = html_table.replace('<table ', '<table id="examples-table" ', 1)
    return (
        '<div style="overflow-x: auto; max-width: 100%;">'
        + html_table +
        '</div>'
    )


class MetricSummary(dspy.Signature):
    """Summarize an aggregate evaluation metric built as a regression over component metrics.

    Context:
    - You are given a target task and a small sample from the evaluation dataset.
    - The aggregate metric is a linear regression over component metrics (with coefficients).
    - Each component includes its name, coefficient, description, usage, and limitations.

    Instructions:
    - Write 5–10 sentences covering: what the aggregate metric measures, how components contribute,
      intended use, strengths, limitations, and caveats about overfitting/calibration.
    - Be concrete and concise. Avoid repeating raw lists; synthesize.
    """
    task_description: str = dspy.InputField(desc="Short description of the evaluation task/dataset.")
    target_column: str = dspy.InputField(desc="Name of the ground-truth rating column (target measure).")
    dataset_sample: List[str] = dspy.InputField(desc="A few representative rows from the evaluation dataset, stringified.")
    components: List[str] = dspy.InputField(desc="List of component metric summaries: name, coefficient, description, usage, limitations.")
    summary: str = dspy.OutputField(desc="5–10 sentence narrative summary of the aggregate regression metric.")


def generate_summary_with_lm(lm: Optional[dspy.LM], task_description: str, dataset_sample: List[str], target_column: str, components_list: List[str]) -> str:
    try:
        if lm is None:
            return ";".join(components_list)
        with dspy.settings.context(lm=lm):
            module = dspy.ChainOfThought(MetricSummary)
            result = module(
                task_description=task_description or "",
                dataset_sample=dataset_sample or [],
                target_column=target_column or "",
                components=components_list or []
            )
        return result.summary if hasattr(result, 'summary') else ""
    except Exception:
        return ";".join(components_list)


def render_html(context: Dict[str, Any]) -> str:
    # Minimal HTML with Bootstrap and Plotly; data injected via token replacement (avoid f-string brace issues)
    coeff_rows = context.get('coefficients', [])
    corr = context.get('correlation', {})
    runtime = context.get('runtime', {})
    robustness = context.get('robustness', {})
    details = context.get('details', {})
    # Full metric docs (markdown-like) for modal display
    try:
        metrics_for_docs = context.get('metrics_for_docs', [])
        all_metric_docs = collect_metric_docs(metrics_for_docs)
        # Build submetric -> parent mapping for MultiMetrics
        docs_map: Dict[str, str] = {}
        for m in metrics_for_docs:
            try:
                base = m.get_name() if hasattr(m, 'get_name') else (m.__name__ if hasattr(m, '__name__') else type(m).__name__)
                if _is_multimetric(m):
                    try:
                        subs = list(m.get_submetric_names() or [])
                    except Exception:
                        subs = []
                    for s in subs:
                        if isinstance(s, str) and len(s) > 0:
                            docs_map[s] = base
                            if not s.startswith(base + '_'):
                                docs_map[f"{base}_{s}"] = base
                            if not s.startswith(base + '-'):
                                docs_map[f"{base}-{s}"] = base
            except Exception:
                continue
    except Exception:
        all_metric_docs = {}
        docs_map = {}
    reqs = context.get('requirements', [])
    examples_html = context.get('examples_html', '')
    lm_summary = context.get('summary', '')
    py_code = context.get('python_code', '')
    py_filename = context.get('python_filename', 'AutoMetricsRegression.py')

    def html_escape(s: str) -> str:
        return (s or '').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    coeff_table_rows = ''.join(
        f"<tr><td><a href=\"#\" class=\"coeff-link\" data-metric=\"{html_escape(name)}\">{html_escape(name)}</a></td><td>{coef:.4f}</td></tr>" for name, coef in coeff_rows if name != '(intercept)'
    )

    req_rows = ''.join(
        f"<tr><td>{html_escape(r['name'])}</td><td>{r['gpu_mem']}</td><td>{r['cpu_mem']}</td></tr>" for r in reqs
    )

    # Details accordion items from parsed cards as markdown-like blocks
    def list_items(key: str) -> str:
        items = []
        for name, sec in details.items():
            text = sec.get(key) or ''
            if text:
                items.append(
                    f"<li><strong>{html_escape(name)}:</strong><div class=\"mt-2\"><pre style=\"white-space: pre-wrap; background:#f8f9fa; padding:8px; border-radius:6px;\">{html_escape(text)}</pre></div></li>"
                )
        return '\n'.join(items) or "<li>No content available.</li>"

    details_desc = list_items('description')
    details_usage = list_items('usage')
    details_limits = list_items('limitations')

    corr_json = json.dumps(corr)
    runtime_json = json.dumps(runtime)
    robustness_json = json.dumps(robustness)
    coeff_list_json = json.dumps(coeff_rows)
    py_code_json = json.dumps(py_code)
    py_filename_json = json.dumps(py_filename)
    page_title = html_escape(str(context.get('target_measure', 'Metric')).replace('_',' ').title())

    # Robustness tooltip content (brief, words only)
    rob_tip_html = (
        "<div style=\"max-width: 320px\">"
        "<strong>Sensitivity</strong>: average drop from the original score to the worse_obvious score for each example, after normalizing scores to [0,1]. Higher means the metric reliably penalizes degraded outputs." \
        "<br/><br/>"
        "<strong>Stability</strong>: average agreement between the original and same_obvious scores (1 minus absolute difference) for each example, after normalization. Higher means the metric is invariant to neutral edits." \
        "</div>"
    )

    template = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__ AutoMetric Report Card</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css">
  <link rel="stylesheet" href="https://cdn.datatables.net/2.0.8/css/dataTables.dataTables.min.css">
  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  
  <style>
    body.dark-mode { background-color: #121212; color: #e0e0e0; }
    body.dark-mode .card { background-color: #1e1e1e; border-color: #333; color: #e0e0e0; }
    body.dark-mode .table, body-dark-mode .table td { background-color: #1e1e1e; color: #e0e0e0; border-color: #333; }
  </style>
  <script>const RC_CORR = __CORR__; const RC_RUNTIME = __RUNTIME__; const RC_ROB = __ROB__; const RC_DOCS = __DOCS__; const RC_DOCS_MAP = __DOCS_MAP__; const RC_PY_CODE = __PY_CODE__; const RC_PY_FILENAME = __PY_FILENAME__;</script>
</head>
<body>
  <div class="container my-5">
    <div class="d-flex justify-content-between align-items-center mb-4">
      <h1>__TITLE__ AutoMetric Report Card</h1>
      <div class="d-flex align-items-center">
        <div class="form-check form-switch me-3">
          <input class="form-check-input" type="checkbox" id="darkModeToggle">
          <label class="form-check-label" for="darkModeToggle">Dark Mode</label>
        </div>
        <button class="btn btn-primary" onclick="window.print()">Export to PDF</button>
        <button class="btn btn-outline-primary ms-2" type="button" onclick="downloadPython()">Export to Python</button>
      </div>
    </div>

    <div class="row g-4">
      <div class="col-md-6">
        <div class="card p-3 h-100">
          <h2>Regression Coefficients</h2>
          <table class="table table-striped"><thead><tr><th>Metric</th><th>Coeff.</th></tr></thead>
            <tbody>__COEFF_ROWS__</tbody>
          </table>
        </div>
      </div>

      <div class="col-md-6">
        <div class="card p-3 h-100">
          <h2>Correlation</h2>
          <div id="correlation-chart" style="height:420px;"></div>
          <div id="correlation-stats" class="mt-2" style="text-align:center; font-size: 1rem; font-weight: 600;"></div>
        </div>
      </div>

      <div class="col-md-6">
        <div class="card p-3 h-100">
          <h2>Robustness <sup><span class="robust-tip text-primary" data-tip-id="robustness-tip-template" style="cursor:pointer; text-decoration: underline; font-size: 0.9rem;">?</span></sup></h2>
          <div id="robustness-sens" style="height:240px;"></div>
          <div id="robustness-stab" style="height:240px;"></div>
        </div>
      </div>

      <div class="col-md-6">
        <div class="card p-3 h-100">
          <h2>Run Time Distribution</h2>
          <div id="runtime-chart" style="height:300px;"></div>
          <p id="runtime-info" class="mt-2"></p>
        </div>
      </div>

      <div class="col-md-6">
        <div class="card p-3 h-100">
          <h2>Metric Details</h2>
          <div class="accordion" id="metricDetails">
            <div class="accordion-item">
              <h2 class="accordion-header" id="descHeader"><button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#descPanel">Descriptions</button></h2>
              <div id="descPanel" class="accordion-collapse collapse"><div class="accordion-body"><ul>__DETAILS_DESC__</ul></div></div>
            </div>
            <div class="accordion-item">
              <h2 class="accordion-header" id="usageHeader"><button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#usagePanel">Usage</button></h2>
              <div id="usagePanel" class="accordion-collapse collapse"><div class="accordion-body"><ul>__DETAILS_USAGE__</ul></div></div>
            </div>
            <div class="accordion-item">
              <h2 class="accordion-header" id="limitsHeader"><button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#limitsPanel">Limitations</button></h2>
              <div id="limitsPanel" class="accordion-collapse collapse"><div class="accordion-body"><ul>__DETAILS_LIMITS__</ul></div></div>
            </div>
          </div>
        </div>
      </div>

      <div class="col-md-6">
        <div class="card p-3 h-100">
          <h2>Compute Requirements</h2>
          <table class="table table-striped"><thead><tr><th>Metric</th><th>GPU RAM (MB)</th><th>CPU RAM (MB)</th></tr></thead>
            <tbody>__REQ_ROWS__</tbody>
          </table>
        </div>
      </div>
    </div>

    <div class="mt-5 card p-3">
      <h3>Metric Summary</h3>
      <p>__SUMMARY__</p>
    </div>

    <div class="mt-4 card p-3">
      <div class="d-flex justify-content-between align-items-center mb-2">
        <h3 class="mb-0">Examples</h3>
        <button id="clear-examples-filter" class="btn btn-sm btn-outline-secondary" type="button">Show All</button>
      </div>
      __EXAMPLES__
    </div>
  </div>

  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
  <script src="https://cdn.datatables.net/2.0.8/js/dataTables.min.js"></script>
  <script>
    function getThemeLayout() {
      const color = getComputedStyle(document.body).color;
      return { paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { color } };
    }
    document.getElementById('darkModeToggle').addEventListener('change',e=>{document.body.classList.toggle('dark-mode',e.target.checked); drawAll();});
    // Enable tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (el) {
      const tip = new bootstrap.Tooltip(el, {trigger: 'hover focus', delay: {show: 0, hide: 50}, placement: 'right'});
      el.addEventListener('shown.bs.tooltip', function () {
        try { if (window.MathJax && MathJax.typesetPromise) { MathJax.typesetPromise(); } } catch(_) {}
      });
      return tip;
    });

    // Initialize tooltips; use template content for robustness
    document.addEventListener('DOMContentLoaded', function () {
      document.querySelectorAll('.robust-tip').forEach(function (el) {
        const id = el.getAttribute('data-tip-id');
        let titleHtml = '';
        if (id) {
          const tpl = document.getElementById(id);
          if (tpl) titleHtml = tpl.innerHTML;
        }
        if (!titleHtml) {
          titleHtml = '<div style="max-width: 320px">Robustness tooltip unavailable.</div>';
        }
        const tip = new bootstrap.Tooltip(el, {
          trigger: 'hover focus',
          delay: {show: 0, hide: 50},
          placement: 'right',
          html: true,
          title: titleHtml
        });
      });
    });

    function drawCorrelation() {
      const layout = Object.assign({xaxis:{title:'Metric Score (normalized to target scale)'}, yaxis:{title:'Ground Truth'}}, getThemeLayout());
      layout.legend = layout.legend || {}; layout.legend.font = { size: 9 }; layout.margin = {l:40,r:10,t:30,b:40};
      const traces = [];
      if (RC_CORR.metrics) {
        // Determine top 3 metrics by absolute coefficient if available
        let topNames = [];
        try {
          const coeffPairs = (__COEFF_LIST__);
          const sorted = coeffPairs.filter(p=>p[0] !== '(intercept)').sort((a,b)=>Math.abs(b[1]) - Math.abs(a[1]));
          topNames = sorted.slice(0,3).map(p=>p[0]);
        } catch (e) { topNames = []; }
        for (const m of RC_CORR.metrics) {
          const rlab = (m.r!=null ? (m.r.toFixed ? m.r.toFixed(2) : m.r) : 'NA');
          const tlab = (m.tau!=null ? (m.tau.toFixed ? m.tau.toFixed(2) : m.tau) : 'NA');
          const visible = (topNames.includes(m.name)) ? true : 'legendonly';
          const ids = m.ids || [];
          const text = ids.map(id => 'ID: ' + id);
          traces.push({ x: m.x_norm || m.x || [], y: m.y || [], mode: 'markers', name: (m.name || '') + ' (r=' + rlab + ', \u03C4=' + tlab + ')', visible, text: text, hovertemplate: '%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>' });
        }
      }
      if (RC_CORR.regression) {
        const rlab = (RC_CORR.regression.r!=null ? (RC_CORR.regression.r.toFixed ? RC_CORR.regression.r.toFixed(2) : RC_CORR.regression.r) : 'NA');
        const tlab = (RC_CORR.regression.tau!=null ? (RC_CORR.regression.tau.toFixed ? RC_CORR.regression.tau.toFixed(2) : RC_CORR.regression.tau) : 'NA');
        const ids = RC_CORR.regression.ids || [];
        const text = ids.map(id => 'ID: ' + id);
        traces.push({ x: RC_CORR.regression.x_norm || RC_CORR.regression.x || [], y: RC_CORR.regression.y || [], mode: 'markers', name: (RC_CORR.regression.name || 'Aggregate') + ' (r=' + rlab + ', \u03C4=' + tlab + ')', marker: { size: 8, color: 'black' }, text: text, hovertemplate: '%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>' });
        document.getElementById('correlation-stats').innerText = 'Aggregate metric: r=' + rlab + ', \u03C4=' + tlab;
      }
      Plotly.newPlot('correlation-chart', traces, layout, {displayModeBar: false});
      // Click-to-jump: when a point is clicked, locate its ID in the examples table and jump to it
      const chart = document.getElementById('correlation-chart');
      chart.on('plotly_click', function(data) {
        try {
          if (!data || !data.points || data.points.length === 0) return;
          const pt = data.points[0];
          const idText = (pt.text || '').toString(); // format: 'ID: <val>'
          const id = idText.startsWith('ID: ') ? idText.slice(4) : idText;
          const tblEl = document.getElementById('examples-table');
          if (!tblEl) return;
          // Try DataTables jQuery API first
          if (window.jQuery && jQuery.fn && jQuery.fn.dataTable) {
            const dt = jQuery(tblEl).DataTable();
            // Search by exact match in first column (ID)
            dt.search('');
            dt.columns(0).search('^' + id.replace(/[.*+?^${}()|[\]\\\]/g, '\\\$&') + '$', true, false).draw();
            // Scroll into view first visible row after draw completes
            setTimeout(function(){
              let rowNode = null;
              try {
                const idxs = dt.rows({ search: 'applied' }).indexes();
                if (idxs && idxs.length) rowNode = dt.row(idxs[0]).node();
              } catch(_){ }
              if (!rowNode) {
                try { rowNode = dt.row(0).node(); } catch(_) {}
              }
              if (rowNode && rowNode.scrollIntoView) {
                rowNode.scrollIntoView({behavior:'smooth', block:'center'});
                try { rowNode.classList.add('table-active'); setTimeout(()=>rowNode.classList.remove('table-active'), 1200); } catch(_) {}
              }
            }, 60);
          } else if (typeof DataTable !== 'undefined') {
            // Vanilla DataTables 2 API
            const dt = DataTable.get(tblEl) || new DataTable(tblEl);
            dt.search('');
            // Filter to rows whose first cell (ID) matches
            dt.columns().every(function(idx) {
              if (idx === 0) {
                this.search('^' + id.replace(/[.*+?^${}()|[\]\\\]/g, '\\\$&') + '$', true, false);
              } else {
                this.search('');
              }
            });
            dt.draw();
            setTimeout(function(){
              let firstRow = null;
              try {
                const nodes = dt.rows({ search: 'applied' }).nodes();
                if (nodes && nodes.length) firstRow = nodes[0];
              } catch(_) {}
              if (!firstRow) {
                const body = tblEl.tBodies && tblEl.tBodies[0];
                firstRow = body && body.rows && body.rows[0];
              }
              if (!firstRow) {
                try {
                  const rows = Array.from(tblEl.tBodies[0].rows || []);
                  firstRow = rows.find(r => (r.cells && r.cells[0] && (r.cells[0].textContent||'').trim() === id));
                } catch(_) {}
              }
              if (firstRow && firstRow.scrollIntoView) {
                firstRow.scrollIntoView({behavior:'smooth', block:'center'});
                try { firstRow.classList.add('table-active'); setTimeout(()=>firstRow.classList.remove('table-active'), 1200); } catch(_) {}
              }
            }, 60);
          }
        } catch(e) { try { console.error('[ReportCard] click-jump failed', e); } catch(_){} }
      });
    }

    function drawRuntime() {
      const layout = Object.assign({yaxis:{title:'Time per Sample (s)'}}, getThemeLayout());
      const boxes = [];
      if (RC_RUNTIME.per_metric) {
        for (const [name, arr] of Object.entries(RC_RUNTIME.per_metric)) {
          boxes.push({ y: arr, type: 'box', name });
        }
      }
      Plotly.newPlot('runtime-chart', boxes, layout);
      if (RC_RUNTIME.aggregate) {
        const agg = RC_RUNTIME.aggregate;
        var seq = (agg.sequence_mean||0);
        if (typeof seq === 'number' && seq.toFixed) { seq = seq.toFixed(2); }
        var par = (agg.parallel_mean||0);
        if (typeof par === 'number' && par.toFixed) { par = par.toFixed(2); }
        var seqCI = (agg.sequence_ci||0);
        if (typeof seqCI === 'number' && seqCI.toFixed) { seqCI = seqCI.toFixed(2); }
        var parCI = (agg.parallel_ci||0);
        if (typeof parCI === 'number' && parCI.toFixed) { parCI = parCI.toFixed(2); }
        document.getElementById('runtime-info').innerHTML = 'Avg time/sample (sequence): ' + seq + 's ± ' + seqCI + 's' + '<br/>' + 'Avg time/sample (parallel): ' + par + 's ± ' + parCI + 's (95% CI)';
      }
    }

    function drawRobustness() {
      if (!RC_ROB.available || !RC_ROB.scores) {
        document.getElementById('robustness-sens').innerHTML = '<em>Robustness not available.</em>';
        document.getElementById('robustness-stab').innerHTML = '';
        return;
      }
      const names = Object.keys(RC_ROB.scores);
      const sens = names.map(n => (RC_ROB.scores[n] && RC_ROB.scores[n].sensitivity) || 0);
      const stab = names.map(n => (RC_ROB.scores[n] && RC_ROB.scores[n].stability) || 0);
      Plotly.newPlot('robustness-sens', [{x: names, y: sens, type:'bar', name:'Sensitivity'}], Object.assign({yaxis:{title:'Sensitivity'}}, getThemeLayout()));
      Plotly.newPlot('robustness-stab', [{x: names, y: stab, type:'bar', name:'Stability'}], Object.assign({yaxis:{title:'Stability'}}, getThemeLayout()));
    }

    function downloadPython() {
      try {
        const code = RC_PY_CODE || '';
        if (!code) { return; }
        const blob = new Blob([code], { type: 'text/x-python' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        const name = (RC_PY_FILENAME && typeof RC_PY_FILENAME === 'string' && RC_PY_FILENAME.trim()) ? RC_PY_FILENAME : 'AutoMetricsRegression.py';
        a.download = name;
        document.body.appendChild(a);
        a.click();
        setTimeout(function(){ URL.revokeObjectURL(url); try { a.remove(); } catch(_){} }, 0);
      } catch(_) { }
    }

    function drawAll() { drawCorrelation(); drawRuntime(); drawRobustness(); }
    drawAll();
  </script>
  <!-- Modal for Metric Card -->
  <div class="modal fade" id="metricDocModal" tabindex="-1" aria-hidden="true">
    <div class="modal-dialog modal-xl modal-dialog-scrollable">
      <div class="modal-content">
        <div class="modal-header">
          <h5 class="modal-title" id="metricDocTitle"></h5>
          <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
        </div>
        <div class="modal-body">
          <div id="metricDocBody" style="white-space: normal;"></div>
        </div>
      </div>
    </div>
  </div>
  <script>
    (function() {
      const tbl = document.getElementById('examples-table');
      if (!tbl) return;
      const clearBtn = document.getElementById('clear-examples-filter');
      try {
        if (window.jQuery && jQuery.fn && typeof jQuery.fn.dataTable !== 'undefined') {
          jQuery(tbl).DataTable({
            paging: true,
            pageLength: 5,
            ordering: true,
            searching: true,
            scrollX: true
          });
          if (clearBtn) {
            clearBtn.addEventListener('click', function(){
              try {
                const dt = jQuery(tbl).DataTable();
                dt.search('');
                dt.columns().every(function(){ this.search(''); });
                dt.draw();
              } catch(_) {}
            });
          }
        } else if (typeof DataTable !== 'undefined') {
          new DataTable(tbl, {
            paging: true,
            pageLength: 5,
            ordering: true,
            searching: true,
            scrollX: true
          });
          if (clearBtn) {
            clearBtn.addEventListener('click', function(){
              try {
                const dt = DataTable.get(tbl);
                dt.search('');
                dt.columns().every(function(){ this.search(''); });
                dt.draw();
              } catch(_) {}
            });
          }
        }
      } catch (e) { try { console.error('[ReportCard] DataTables init error:', e); } catch(_){} }
    })();
  </script>
  <script>
    // Click handlers for regression coefficient metric links -> open modal with metric card
    document.addEventListener('click', function(e) {
      const a = e.target.closest && e.target.closest('a.coeff-link');
      if (!a) return;
      e.preventDefault();
      try {
        let metric = a.getAttribute('data-metric');
        // Resolve submetric to parent metric if available
        if (RC_DOCS && !(metric in RC_DOCS) && RC_DOCS_MAP && RC_DOCS_MAP[metric]) {
          metric = RC_DOCS_MAP[metric];
        }
        const doc = (RC_DOCS && RC_DOCS[metric]) ? RC_DOCS[metric] : 'No metric card available.';
        const titleNode = document.getElementById('metricDocTitle');
        const bodyNode = document.getElementById('metricDocBody');
        if (titleNode) titleNode.textContent = metric + ' — Metric Card';
        if (bodyNode) {
          try {
            bodyNode.innerHTML = marked.parse(doc);
          } catch(_) {
            bodyNode.textContent = doc;
          }
        }
        const modalEl = document.getElementById('metricDocModal');
        if (modalEl && bootstrap && bootstrap.Modal) {
          const modal = bootstrap.Modal.getOrCreateInstance(modalEl, {backdrop: true});
          modal.show();
        }
      } catch(_) {}
    });
  </script>
  <div id="robustness-tip-template" class="d-none">
    <div style="max-width: 360px">
      <strong>Sensitivity</strong> (worse_obvious): how much the metric tends to drop when the output is intentionally degraded. For each example, we measure the relative drop from the original to the average worse_obvious score, clip negative values to 0 (no drop), and then average across examples.
      <br/><br/>
      <strong>Stability</strong> (same_obvious): how consistent the metric stays under neutral edits that should not change meaning. For each example, we measure how close the original is to the average same_obvious score (scaled by the original magnitude), clip below 0, and then average across examples. Higher means more stable.
    </div>
  </div>
</body>
</html>
"""

    html = (template
            .replace('__CORR__', corr_json)
            .replace('__RUNTIME__', runtime_json)
            .replace('__ROB__', robustness_json)
            .replace('__COEFF_LIST__', coeff_list_json)
            .replace('__COEFF_ROWS__', coeff_table_rows)
            .replace('__REQ_ROWS__', req_rows)
            .replace('__DETAILS_DESC__', details_desc)
            .replace('__DETAILS_USAGE__', details_usage)
            .replace('__DETAILS_LIMITS__', details_limits)
            .replace('__SUMMARY__', html_escape(lm_summary))
            .replace('__EXAMPLES__', examples_html)
            .replace('__TITLE__', page_title)
            .replace('__ROB_TIP__', rob_tip_html)
            .replace('__DOCS__', json.dumps(all_metric_docs))
            .replace('__DOCS_MAP__', json.dumps(docs_map))
            .replace('__PY_CODE__', py_code_json)
            .replace('__PY_FILENAME__', py_filename_json)
            )
    return html


def generate_metric_report_card(
    regression_metric: Any,
    metrics: List[Any],
    target_measure: str,
    eval_dataset: Optional[Any] = None,
    train_dataset: Optional[Any] = None,
    lm: Optional[dspy.LM] = None,
    output_path: Optional[str] = None,
    verbose: bool = False,
    include_per_metric_feedback: bool = False,
) -> Dict[str, Any]:
    # Determine feature names and coefficients
    feature_names = extract_feature_names(regression_metric, metrics, eval_dataset or train_dataset)
    coeffs = extract_regression_coefficients(regression_metric, feature_names)
    # Sort coefficients descending (keep intercept last if present)
    try:
        non_intercepts = [(n, v) for (n, v) in coeffs if n != '(intercept)']
        intercepts = [(n, v) for (n, v) in coeffs if n == '(intercept)']
        non_intercepts.sort(key=lambda p: (p[1] if isinstance(p[1], (int, float)) else float('-inf')), reverse=True)
        coeffs = non_intercepts + intercepts
    except Exception:
        pass

    # Prepare sections depending on eval availability
    correlation = {}
    robustness = { 'available': False }
    runtime = {}
    examples_html = ''

    if eval_dataset is not None:
        # 1) Measure runtime FIRST on a fresh subsample to avoid warming caches
        try:
            runtime = measure_runtime(eval_dataset, metrics, sample_size=30)
        except Exception:
            runtime = {}
        if verbose:
            print(f"[ReportCard] Runtime sample computed for {runtime.get('sample_size', 0)} rows")

        # 2) Ensure metric columns exist on eval dataset (this may use caches thereafter)
        ensure_eval_metrics(eval_dataset, metrics)
        if verbose:
            print(f"[ReportCard] Ensured metric columns on eval dataset: {len(metrics)} metrics")

        # 3) Ensure regression column exists on eval dataset
        try:
            if hasattr(regression_metric, 'get_name'):
                name = regression_metric.get_name()
            else:
                name = f"Autometrics_Regression_{target_measure}"
            cols = eval_dataset.get_dataframe().columns
            if not any(str(name).lower() in c.lower() for c in cols):
                eval_dataset.add_metric(regression_metric, update_dataset=True)
                if verbose:
                    print(f"[ReportCard] Added regression metric to eval dataset: {name}")
        except Exception:
            pass

        # 4) Correlation (uses eval columns)
        # Try to pass the exact regression column name, if present on the metric
        reg_name = None
        try:
            reg_name = regression_metric.get_name() if hasattr(regression_metric, 'get_name') else None
        except Exception:
            reg_name = None
        correlation = compute_correlation(eval_dataset, feature_names, target_measure, include_regression=True, regression_col_name=reg_name)
        if verbose:
            r = correlation.get('regression', {}).get('r') if isinstance(correlation, dict) else None
            t = correlation.get('regression', {}).get('tau') if isinstance(correlation, dict) else None
            print(f"[ReportCard] Correlation computed (regression): r={r}, tau={t}")

        # 5) Robustness (requires original metric values present on eval dataset)
        try:
            # Include the regression metric itself so robustness evaluates it directly
            metrics_for_robustness = list(metrics)
            if regression_metric is not None:
                metrics_for_robustness.append(regression_metric)
            robustness = run_robustness(
                eval_dataset,
                metrics_for_robustness,
                target_measure,
                lm,
                verbose=verbose,
                output_dir=(os.path.join(os.path.dirname(output_path), 'robustness') if output_path else None),
                regression_name=(regression_metric.get_name() if hasattr(regression_metric, 'get_name') else None),
                regression_coeffs=coeffs
            )
        except Exception as e:
            robustness = { 'available': False, 'error': str(e) }
        if verbose:
            print(f"[ReportCard] Robustness available: {robustness.get('available', False)}")
            if not robustness.get('available', False):
                err = robustness.get('error')
                if err:
                    print(f"[ReportCard] Robustness error: {err}")

        # 6) Examples table (default: include aggregated regression feedback only)
        try:
            # Determine aggregated regression feedback column if present
            agg_fb_cols: List[str] = []
            try:
                if hasattr(regression_metric, 'get_name'):
                    reg_name = regression_metric.get_name()
                    fb_col = f"{reg_name}__feedback"
                    if fb_col in eval_dataset.get_dataframe().columns:
                        agg_fb_cols.append(fb_col)
            except Exception:
                pass
            examples_html = build_examples_table(
                eval_dataset,
                feature_names,
                target_measure,
                include_regression=True,
                max_rows=None,
                aggregator_feedback_cols=agg_fb_cols,
                include_per_metric_feedback=include_per_metric_feedback,
            )
        except Exception:
            examples_html = ''
        if verbose:
            print("[ReportCard] Examples table rendered")

    # Details and requirements are available regardless
    try:
        details = parse_metric_cards(metrics)
    except Exception:
        details = {}
    try:
        requirements = compute_requirements(metrics)
    except Exception:
        requirements = []

    # Prepare components payload for LLM summary
    # Prepare components as list of strings expected by the DSPy signature
    components_list: List[str] = []
    for n, c in coeffs:
        if n == '(intercept)':
            continue
        info = details.get(n, {}) if isinstance(details, dict) else {}
        desc = info.get('description', '')
        use = info.get('usage', '')
        lim = info.get('limitations', '')
        components_list.append(f"name={n}; coef={c:.4f}; description={desc}; usage={use}; limitations={lim}")
    task_desc = ''
    dataset_sample_list: List[str] = []
    try:
        ds = eval_dataset or train_dataset
        if ds is not None:
            task_desc = ds.get_task_description() or ''
            sample_rows = ds.get_dataframe().head(3).to_dict(orient='records')
            dataset_sample_list = [json.dumps(r, ensure_ascii=False) for r in sample_rows]
    except Exception:
        pass

    summary_text = ''
    try:
        summary_text = generate_summary_with_lm(lm, task_desc, dataset_sample_list, target_measure, components_list)
    except Exception:
        summary_text = ''

    # Prepare Python export strings for embedding in the report
    py_code_str = ''
    py_filename = ''
    try:
        # Prefer explicit method to return code string without I/O if available
        if hasattr(regression_metric, 'export_python_code') and callable(getattr(regression_metric, 'export_python_code')):
            py_code_str = regression_metric.export_python_code(inline_generated_metrics=True, name_salt=None)
        # Provide a friendly default filename based on target
        base = str(getattr(regression_metric, 'name', 'AutoMetricsRegression')) or 'AutoMetricsRegression'
        safe_base = base.replace(' ', '_').replace('-', '_')
        py_filename = f"{safe_base}.py"
    except Exception:
        py_code_str = ''
        py_filename = 'AutoMetricsRegression.py'

    html = render_html({
        'coefficients': coeffs,
        'correlation': correlation,
        'robustness': robustness,
        'runtime': runtime,
        'details': details,
        'requirements': requirements,
        'examples_html': examples_html,
        'summary': summary_text,
        'target_measure': target_measure,
        'metrics_for_docs': metrics,
        'python_code': py_code_str,
        'python_filename': py_filename,
    })

    saved_path = None
    if output_path:
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
            saved_path = output_path
        except Exception:
            saved_path = None

    # Extract reusable correlation stats for regression (kendall tau and pearson r)
    reg_tau = None
    reg_r = None
    try:
        if isinstance(correlation, dict) and correlation.get('regression'):
            reg_tau = correlation['regression'].get('tau')
            reg_r = correlation['regression'].get('r')
    except Exception:
        pass

    return {
        'html': html,
        'path': saved_path,
        'artifacts': {
            'coefficients': coeffs,
            'kendall_tau_regression': reg_tau,
            'pearson_r_regression': reg_r,
            'correlation': correlation,
        }
    }


