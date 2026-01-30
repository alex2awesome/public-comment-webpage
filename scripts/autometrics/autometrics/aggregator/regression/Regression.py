import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from autometrics.aggregator.Aggregator import Aggregator
from autometrics.metrics.MultiMetric import MultiMetric
from autometrics.metrics.Metric import MetricResult
from sklearn.preprocessing import StandardScaler

class Regression(Aggregator):
    """
    Class for regression aggregation
    """
    def __init__(self, name, description, input_metrics=None, model=None, dataset=None, **kwargs):
        super().__init__(name, description, input_metrics, dataset, **kwargs)
        self.model = model
        self.scaler = None  # Will store the StandardScaler for consistent scaling
        self._selected_columns = None  # Persist exact feature order used during training

    def learn(self, dataset, target_column=None):
        """
        Learn the regression model with proper feature scaling
        """
        self.ensure_dependencies(dataset)
        df = dataset.get_dataframe()

        input_columns = self.get_input_columns()
        if not target_column:
            target_column = dataset.get_target_columns()[0]

        # Pull out X and y
        X = df[input_columns]
        y = df[target_column]

        # —— clip any +/-inf in X to the finite min/max of each column
        X_clean = X.replace([np.inf, -np.inf], np.nan)
        mins = X_clean.min()
        maxs = X_clean.max()
        X = X.clip(lower=mins, upper=maxs, axis=1).fillna(0)

        # —— same for y (a Series)
        y_clean = y.replace([np.inf, -np.inf], np.nan)
        if y_clean.isna().all():
            # if everything was infinite, just zero out
            y = y.fillna(0)
        else:
            y = y.clip(lower=y_clean.min(), upper=y_clean.max()).fillna(0)

        # Apply StandardScaler to handle scale differences between metrics
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Now safe to fit on scaled data
        self.model.fit(X_scaled, y)

        # Persist exact feature order for downstream export/repro
        self._selected_columns = list(input_columns)

    def _predict_unsafe(self, dataset, update_dataset=True):
        """
        Predict the target column using the same scaling as training
        """
        df = dataset.get_dataframe().copy()
        input_columns = self.get_input_columns()

        # Ensure dependencies are computed (some callers may bypass predict())
        # Note: not calling self.ensure_dependencies here to avoid double-work,
        # but we will fail fast if columns are still missing.
        missing_inputs = [c for c in input_columns if c not in df.columns]
        if missing_inputs:
            # Try once to compute via ensure_dependencies in case predict() caller forgot
            self.ensure_dependencies(dataset)
            df = dataset.get_dataframe().copy()
            missing_inputs = [c for c in input_columns if c not in df.columns]
        if missing_inputs:
            raise KeyError(f"Regression input columns missing from dataset: {missing_inputs}")

        X = df[input_columns]

        # —— clip any +/-inf or too large values in X before predict
        X_clean = X.replace([np.inf, -np.inf], np.nan)
        mins = X_clean.min()
        maxs = X_clean.max()
        X = X.clip(lower=mins, upper=maxs, axis=1).fillna(0)

        # Apply the same scaling used during training
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            # Fallback if no scaler (for backward compatibility)
            X_scaled = X

        y_pred = self.model.predict(X_scaled)

        if update_dataset:
            df.loc[:, self.name] = y_pred
            dataset.set_dataframe(df)
            # Keep dataset.metric_columns metadata in sync so downstream correlation sees this column
            if self.name not in dataset.get_metric_columns():
                dataset.get_metric_columns().append(self.name)

            # Build aggregated feedback column from component metrics, if available
            try:
                # Gather coefficients with feature names in current order
                coef = getattr(self.model, 'coef_', None)
                feature_names = list(self.get_input_columns())
                coef_list = None
                if coef is not None:
                    try:
                        import numpy as _np
                        coef_list = _np.array(coef, dtype=float).reshape(-1)
                    except Exception:
                        try:
                            coef_list = list(coef)[0] if hasattr(coef, '__iter__') and len(coef) == 1 else list(coef)
                        except Exception:
                            coef_list = None

                if coef_list is not None and feature_names:
                    name_to_coef = {}
                    for idx, fname in enumerate(feature_names[:len(coef_list)]):
                        try:
                            name_to_coef[str(fname)] = float(coef_list[idx])
                        except Exception:
                            name_to_coef[str(fname)] = 0.0

                    # Order features by absolute coefficient magnitude (desc)
                    ordered_feats = sorted(name_to_coef.items(), key=lambda p: abs(p[1]), reverse=True)

                    # Collect feedback strings per row following ordered features; include all by default
                    agg_col = f"{self.name}__feedback"
                    feedback_cols = []
                    for fname, _w in ordered_feats:
                        col = f"{fname}__feedback"
                        if col in df.columns:
                            feedback_cols.append(col)

                    if feedback_cols:
                        # Build aggregated feedback; deduplicate consecutive identical strings per row
                        def _combine_feedback(row):
                            seen = set()
                            out_lines = []
                            for c in feedback_cols:
                                try:
                                    txt = row.get(c)
                                except Exception:
                                    txt = None
                                if not isinstance(txt, str) or len(txt.strip()) == 0:
                                    continue
                                key = txt.strip()
                                if key in seen:
                                    continue
                                seen.add(key)
                                out_lines.append(key)
                            return "\n".join(out_lines)

                        df.loc[:, agg_col] = df.apply(_combine_feedback, axis=1)
                        dataset.set_dataframe(df)
            except Exception:
                # Feedback aggregation is best-effort; never fail prediction
                pass

        return y_pred
    
    def identify_important_metrics(self):
        '''
            Identify the most important metrics depending on the model.
            For linear models: Use standardized coefficients (already in standardized form when fit on scaled data).
            For tree-based models: Use feature importances.
        '''
        metric_columns = self.get_input_columns()

        # Linear models (Ridge, Lasso, ElasticNet, PLS)
        if hasattr(self.model, 'coef_'):
            coef = self.model.coef_
            if coef.ndim == 1:
                pairs = zip(coef, metric_columns)
            else:
                pairs = zip(coef[0], metric_columns)
            return sorted(pairs, key=lambda x: abs(x[0]), reverse=True)

        # Tree-based models (RandomForest, GradientBoosting)
        if hasattr(self.model, 'feature_importances_'):
            pairs = zip(self.model.feature_importances_, metric_columns)
            return sorted(pairs, key=lambda x: abs(x[0]), reverse=True)

        raise ValueError(
            "The model does not support extracting feature importances or coefficients."
        )

    # --- Export helpers -------------------------------------------------
    def get_selected_columns(self):
        """
        Return the exact feature column order used during training if available;
        otherwise fall back to current input columns.
        """
        return list(self._selected_columns) if self._selected_columns else self.get_input_columns()

    def export_python_code(self, *, inline_generated_metrics: bool = True, name_salt: Optional[str] = None) -> str:
        """
        Return the generated standalone Python code for this regression without writing to disk.

        Parameters
        ----------
        inline_generated_metrics: bool
            When True, attempt to inline the Python source for any generated metrics
            that support code generation, to produce a single-file export.
        name_salt: Optional[str]
            Optional suffix applied to the internal exported metric name to avoid
            collisions with any existing AutoMetrics caches.

        Returns
        -------
        str
            The full Python module as a string.
        """
        return self._generate_python_code(
            inline_generated_metrics=inline_generated_metrics,
            name_salt=name_salt,
        )

    # --- Dataset-free calculation paths ---------------------------------
    def _require_scaler(self):
        if self.scaler is None:
            raise ValueError("Scaler is not fitted; call learn() before calculate().")

    def _assemble_feature_vector_single(self, input: Any, output: Any, references=None, **kwargs):
        """Compute feature vector and aligned per-feature feedback for one example.

        Returns: (features_np: np.ndarray, feedback_values: List[str])
        """
        cols: List[str] = list(self.get_input_columns())
        feature_values: List[float] = []
        feedback_values: List[str] = []

        for metric in (self.input_metrics or []):
            if isinstance(metric, MultiMetric):
                wrapped = metric.calculate_batched_with_feedback([input], [output], [references] if references is not None else None, **kwargs)
                # wrapped: [submetric][example] MetricResult
                sub_scores = [float(sub_list[0].score) for sub_list in wrapped]
                sub_fb = [str(sub_list[0].feedback) for sub_list in wrapped]
                feature_values.extend(sub_scores)
                feedback_values.extend(sub_fb)
            else:
                mr = metric.calculate_with_feedback(input, output, references, **kwargs)
                feature_values.append(float(mr.score))
                feedback_values.append(str(mr.feedback))

        if len(feature_values) != len(cols):
            raise ValueError("Feature vector length mismatch with get_input_columns()")
        arr = np.array(feature_values, dtype=float)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        arr = np.nan_to_num(arr, nan=0.0)
        if len(feedback_values) != len(cols):
            feedback_values = (feedback_values + [""] * len(cols))[:len(cols)]
        return arr, feedback_values

    def _assemble_feature_matrix_batched(self, inputs: List[Any], outputs: List[Any], references=None, **kwargs):
        """Compute batched features and aligned per-feature feedback in one pass.

        Returns: (X: np.ndarray [n_examples, n_features], feedback_by_feature: List[List[str]]
        where feedback_by_feature has shape [n_features][n_examples]).
        """
        n = len(inputs)
        cols = list(self.get_input_columns())
        X = np.zeros((n, len(cols)), dtype=float)
        feedback_by_feature: List[List[str]] = [[] for _ in range(len(cols))]
        if references is None:
            refs = [None] * n
        else:
            refs = references

        col_cursor = 0
        for metric in (self.input_metrics or []):
            if isinstance(metric, MultiMetric):
                wrapped = metric.calculate_batched_with_feedback(inputs, outputs, refs, **kwargs)
                # wrapped: [submetric][example] MetricResult
                for sub_list in wrapped:
                    if len(sub_list) != n:
                        raise ValueError("Submetric batch length mismatch")
                    X[:, col_cursor] = np.asarray([float(mr.score) for mr in sub_list], dtype=float)
                    feedback_by_feature[col_cursor] = [str(mr.feedback) for mr in sub_list]
                    col_cursor += 1
            else:
                wrapped = metric.calculate_batched_with_feedback(inputs, outputs, refs, **kwargs)
                if len(wrapped) != n:
                    raise ValueError("Scalar metric batch length mismatch")
                X[:, col_cursor] = np.asarray([float(mr.score) for mr in wrapped], dtype=float)
                feedback_by_feature[col_cursor] = [str(mr.feedback) for mr in wrapped]
                col_cursor += 1

        if col_cursor != len(cols):
            raise ValueError("Assembled features do not match expected column count")

        X = np.where(np.isfinite(X), X, np.nan)
        X = np.nan_to_num(X, nan=0.0)
        return X, feedback_by_feature

    def _calculate_impl(self, input, output, references=None, **kwargs):
        self._require_scaler()
        X, _fb = self._assemble_feature_vector_single(input, output, references, **kwargs)
        X_scaled = self.scaler.transform(X.reshape(1, -1)).reshape(-1)
        y = self.model.predict(X_scaled.reshape(1, -1)).reshape(-1)[0]
        return float(y)

    def _calculate_batched_impl(self, inputs, outputs, references=None, **kwargs):
        self._require_scaler()
        X, _fb = self._assemble_feature_matrix_batched(inputs, outputs, references, **kwargs)
        X_scaled = self.scaler.transform(X)
        y = self.model.predict(X_scaled)
        return [float(v) for v in np.asarray(y).reshape(-1)]

    # Feedback is gathered during feature assembly when requested

    def _calculate_with_feedback_impl(self, input, output, references=None, **kwargs):
        self._require_scaler()
        X, fb_values = self._assemble_feature_vector_single(input, output, references, **kwargs)
        X_scaled = self.scaler.transform(X.reshape(1, -1)).reshape(-1)
        y = float(self.model.predict(X_scaled.reshape(1, -1)).reshape(-1)[0])

        feature_names = list(self.get_input_columns())
        coef_list = np.array(getattr(self.model, 'coef_', np.zeros(len(feature_names))), dtype=float).reshape(-1)
        name_to_coef: Dict[str, float] = {}
        for idx, fname in enumerate(feature_names[:len(coef_list)]):
            try:
                name_to_coef[str(fname)] = float(coef_list[idx])
            except Exception:
                name_to_coef[str(fname)] = 0.0
        ordered_feats = [fname for fname, _w in sorted(name_to_coef.items(), key=lambda p: abs(p[1]), reverse=True)]

        seen = set()
        out_lines: List[str] = []
        for fname in ordered_feats:
            try:
                feat_pos = feature_names.index(fname)
            except ValueError:
                continue
            fb_txt = fb_values[feat_pos]
            if not isinstance(fb_txt, str) or len(fb_txt.strip()) == 0:
                continue
            key = fb_txt.strip()
            if key in seen:
                continue
            seen.add(key)
            out_lines.append(key)
        feedback = "\n".join(out_lines)
        return MetricResult(y, feedback)

    def _calculate_batched_with_feedback_impl(self, inputs, outputs, references=None, **kwargs):
        self._require_scaler()
        X, fb_by_feature = self._assemble_feature_matrix_batched(inputs, outputs, references, **kwargs)
        X_scaled = self.scaler.transform(X)
        y = np.asarray(self.model.predict(X_scaled)).reshape(-1)

        results: List[MetricResult] = []
        try:
            feature_names = list(self.get_input_columns())
            coef_list = np.array(getattr(self.model, 'coef_', np.zeros(len(feature_names))), dtype=float).reshape(-1)
            name_to_coef: Dict[str, float] = {}
            for idx, fname in enumerate(feature_names[:len(coef_list)]):
                try:
                    name_to_coef[str(fname)] = float(coef_list[idx])
                except Exception:
                    name_to_coef[str(fname)] = 0.0
            ordered_feats = [fname for fname, _w in sorted(name_to_coef.items(), key=lambda p: abs(p[1]), reverse=True)]

            for row_idx, score in enumerate(y):
                seen = set()
                out_lines: List[str] = []
                for fname in ordered_feats:
                    try:
                        feat_pos = feature_names.index(fname)
                    except ValueError:
                        continue
                    fb_txt = fb_by_feature[feat_pos][row_idx]
                    if not isinstance(fb_txt, str) or len(fb_txt.strip()) == 0:
                        continue
                    key = fb_txt.strip()
                    if key in seen:
                        continue
                    seen.add(key)
                    out_lines.append(key)
                feedback = "\n".join(out_lines)
                results.append(MetricResult(float(score), feedback))
        except Exception:
            results = [MetricResult(float(v), "") for v in y]
        return results

    def _extract_static_params(self):
        """
        Extract coefficients, intercept, and StandardScaler statistics in a
        consistent shape. Coefficients are 1-D length n_features.
        """
        if self.scaler is None:
            raise ValueError("Scaler is not fitted; run learn() before exporting.")
        if not hasattr(self.scaler, 'mean_') or not hasattr(self.scaler, 'scale_'):
            raise ValueError("Scaler is missing mean_/scale_ attributes.")

        # Coefficients/intercept from sklearn-like models
        coef = getattr(self.model, 'coef_', None)
        intercept = getattr(self.model, 'intercept_', 0.0)
        if coef is None:
            raise ValueError("Model is missing coef_.")
        # Flatten shapes (n_features,) or (1, n_features)
        try:
            import numpy as _np
            coef_arr = _np.array(coef, dtype=float).reshape(-1)
        except Exception:
            coef_arr = list(coef)[0] if isinstance(coef, (list, tuple)) and len(coef) == 1 else list(coef)
        # Intercept may be array-like
        try:
            import numpy as _np
            if isinstance(intercept, (list, tuple)):
                intercept_val = float(intercept[0]) if len(intercept) > 0 else 0.0
            else:
                intercept_val = float(_np.array(intercept).reshape(-1)[0])
        except Exception:
            try:
                intercept_val = float(intercept)
            except Exception:
                intercept_val = 0.0

        mean = getattr(self.scaler, 'mean_', None)
        scale = getattr(self.scaler, 'scale_', None)
        if mean is None or scale is None:
            raise ValueError("Scaler statistics are not available (mean_/scale_).")
        return coef_arr, intercept_val, list(mean), list(scale)

    def _generate_python_code(self, inline_generated_metrics: bool = False, name_salt: str | None = None) -> str:
        """
        Generate a standalone Python module that rebuilds this regression as a
        static aggregator using stored coefficients and scaler statistics.
        """
        # Gather static params
        coef_arr, intercept_val, mean_list, scale_list = self._extract_static_params()
        feature_names = self.get_selected_columns()

        # Build import lines and constructor code for input metrics
        from autometrics.aggregator.generated.GeneratedRegressionMetric import generate_metric_constructor_code
        import re as _re

        import_lines = []
        inline_blocks = []
        ctor_exprs = []
        seen_imports = set()

        def _safe_class_name_from_block(code_block: str, fallback: str) -> str:
            m = _re.search(r"class\s+([A-Za-z_]\w*)\(", code_block)
            return m.group(1) if m else fallback

        def _uniquify_constants(code_block: str, class_name: str) -> str:
            # Avoid top-level constant collisions across concatenated blocks
            replacements = [
                (r"\bDEFAULT_MODEL\b", f"DEFAULT_MODEL_{class_name}"),
                (r"\bOPTIMIZED_EXAMPLES\b", f"OPTIMIZED_EXAMPLES_{class_name}"),
                (r"\bOPTIMIZED_PROMPT_DATA\b", f"OPTIMIZED_PROMPT_DATA_{class_name}"),
            ]
            out = code_block
            for pattern, repl in replacements:
                out = _re.sub(pattern, repl, out)
            return out

        for m in (self.input_metrics or []):
            if inline_generated_metrics and hasattr(m, '_generate_python_code'):
                try:
                    raw = m._generate_python_code(include_metric_card=False)
                    # Derive class name from block; fallback to sanitized metric name
                    fallback = str(getattr(m, 'name', m.__class__.__name__)).replace(' ', '_').replace('-', '_')
                    cls_name = _safe_class_name_from_block(raw, fallback)
                    # Uniquify top-level constants inside the block
                    block = _uniquify_constants(raw, cls_name)
                    inline_blocks.append(block)
                    ctor_exprs.append(f"{cls_name}()")
                    continue
                except Exception:
                    # Fallback to import path
                    pass
            imp, ctor = generate_metric_constructor_code(m)
            if imp not in seen_imports:
                import_lines.append(imp)
                seen_imports.add(imp)
            ctor_exprs.append(ctor)

        import_block = "\n".join(import_lines)
        ctor_list = ",\n        ".join(ctor_exprs) if ctor_exprs else ""
        inline_block = ("\n\n".join(inline_blocks) + ("\n\n" if inline_blocks else ""))

        # Build code string
        # Use a single underscore when salting to avoid double underscores in class names
        salted_name = self.name if not name_salt else f"{self.name}_{name_salt}"
        class_base_unsalted = self.name
        def _sanitize(s: str) -> str:
            return s.replace(' ', '_').replace('-', '_')
        class_def_name = f"{_sanitize(class_base_unsalted)}_StaticRegression"
        # Prepare simple metric card text summarizing weights and intercept
        weights_lines = "\n".join([f"- {fname}: {float(coef):.6f}" for fname, coef in zip(feature_names, coef_arr)])
        simple_card = (
            "Regression aggregator over component metrics with a linear model.\n\n"
            f"Components and weights:\n{weights_lines}\n\n"
            f"Intercept: {float(intercept_val):.6f}"
        )
        code = f"""# Auto-generated static regression for {self.name}
from typing import ClassVar
import numpy as np
from autometrics.aggregator.generated.GeneratedRegressionMetric import GeneratedStaticRegressionAggregator

{import_block}

{inline_block}

INPUT_METRICS = [
        {ctor_list}
]

class {class_def_name}(GeneratedStaticRegressionAggregator):
    \"\"\"Regression aggregator over component metrics with a linear model.\n\nComponents and weights:\n{weights_lines}\n\nIntercept: {float(intercept_val):.6f}\"\"\"

    description: ClassVar[str] = {repr(simple_card)}

    def __init__(self):
        super().__init__(
            name={repr(salted_name)},
            description={repr(simple_card)},
            input_metrics=INPUT_METRICS,
            feature_names={repr(list(feature_names))},
            coefficients={repr([float(x) for x in list(coef_arr)])},
            intercept={float(intercept_val)},
            scaler_mean={repr([float(x) for x in list(mean_list)])},
            scaler_scale={repr([float(x) for x in list(scale_list)])},
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(name={{repr(self.name)}})"
"""
        return code

    def export_python(self, output_path: str, *, inline_generated_metrics: bool = True, name_salt: str | None = None) -> str:
        """Write generated code to output_path and return the path.

        If name_salt is provided, the exported static metric's internal name
        will be suffixed with the salt to avoid collisions with any existing
        AutoMetrics caches. If None, no salt is applied.
        """
        code = self.export_python_code(inline_generated_metrics=inline_generated_metrics, name_salt=name_salt)
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(code)
        return output_path
