# HotellingPLS.py
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression

try:
    from scipy.stats import f as f_dist
except Exception:
    f_dist = None  # we'll fall back to a chi-square approximation if SciPy isn't present

from autometrics.aggregator.regression import Regression


class HotellingPLS(Regression):
    """
    PLS with Hotelling's T²-based variable selection (T²-PLS), per:
      Mehmood (2016) "Hotelling T2 based variable selection in partial least squares regression"

    - Tunes number of components A and selection threshold α via inner K-fold CV (default 10-fold),
      minimizing RMSEP on folds (paper's double-CV inner loop).
    - Selects variables by keeping those with T² above C(p, A) * F_{A, p-A}(1 - α).
    - Refits final PLS on the selected variables using the full training set.
    - Predict uses the train-fitted scaler + selected columns only.

    Parameters
    ----------
    name : str
    description : str
    input_metrics : list[Metric | MultiMetric]
        Metrics to consider as candidate predictors (the initial p features).
    dataset : Dataset
    A_grid : List[int]
        Candidate numbers of PLS components to try. (paper uses 1..10)
    alpha_grid : List[float]
        Candidate false-alarm probabilities for the T² threshold (e.g., [0.01, 0.05, 0.10, 0.15, 0.20]).
    inner_cv_folds : int
        K for inner CV.
    selection_mode : {"alpha", "top_n"}
        "alpha": paper's thresholding by α. "top_n": keep exactly N features with highest T².
    top_n : Optional[int]
        If selection_mode == "top_n", keep exactly this many features.
    random_state : Optional[int]
        For reproducible CV splits.
    """
    def __init__(self,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 input_metrics=None,
                 dataset=None,
                 A_grid: Optional[List[int]] = None,
                 alpha_grid: Optional[List[float]] = None,
                 inner_cv_folds: int = 10,
                 selection_mode: str = "alpha",
                 top_n: Optional[int] = None,
                 random_state: Optional[int] = 42,
                 **kwargs):

        # Model will be set after selection (we still pass a placeholder to parent)
        model = PLSRegression(n_components=1)

        if name is None:
            name = "PLS_T2"
        if description is None:
            description = "PLS with Hotelling T² variable selection"

        super().__init__(name, description, input_metrics=input_metrics, model=model, dataset=dataset, **kwargs)

        self.A_grid = A_grid if A_grid is not None else list(range(1, 11))    # paper uses 1..10
        self.alpha_grid = alpha_grid if alpha_grid is not None else [0.01, 0.05, 0.10, 0.15, 0.20]
        self.inner_cv_folds = inner_cv_folds
        self.selection_mode = selection_mode
        self.top_n = top_n
        self.random_state = random_state

        # Learned state
        self.selected_columns_: List[str] = []
        self.t2_scores_: Optional[pd.Series] = None
        self.A_star_: Optional[int] = None
        self.alpha_star_: Optional[float] = None
        self.t2_cutoff_: Optional[float] = None

    # ---------- Utilities ----------

    @staticmethod
    def _rmse(y_true, y_pred) -> float:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def _cov_inv(X: np.ndarray) -> np.ndarray:
        """
        Robust inverse of covariance Σ for columns of X.
        Handles the A=1 case (scalar variance) by upcasting to 1x1,
        adds a tiny ridge, and falls back to pinv if needed.
        """
        X = np.atleast_2d(np.asarray(X))      # ensure (p x A)
        S = np.cov(X, rowvar=False)           # (A x A) or scalar if A==1

        # Ensure 2D covariance matrix
        if np.ndim(S) == 0:                   # scalar variance when A==1
            S = np.array([[float(S)]], dtype=float)
        else:
            S = np.asarray(S, dtype=float)

        # Replace non-finite entries defensively
        if not np.all(np.isfinite(S)):
            S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)

        d = S.shape[0]                        # A
        # Ridge scaled to trace to be scale-aware; fallback to tiny if trace=0
        tr = float(np.trace(S)) if d > 0 else 0.0
        eps = 1e-9 * (tr / d if d > 0 and np.isfinite(tr) and tr > 0 else 1.0)
        S_r = S + eps * np.eye(d)

        try:
            return np.linalg.inv(S_r)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(S_r)

    def _clean_Xy(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        Xc = X.replace([np.inf, -np.inf], np.nan)
        mins = Xc.min()
        maxs = Xc.max()
        Xc = X.clip(lower=mins, upper=maxs, axis=1).fillna(0)

        yc = y.replace([np.inf, -np.inf], np.nan)
        if yc.isna().all():
            y = y.fillna(0)
        else:
            y = y.clip(lower=yc.min(), upper=yc.max()).fillna(0)
        return Xc, y

    def _compute_t2(self, W: np.ndarray) -> np.ndarray:
        """
        Compute T² per variable from W (shape: p x A).
        Each row j is a variable's loading-weight vector across A components.
        Works for both A=1 and A>1.
        """
        W = np.atleast_2d(np.asarray(W))
        if W.ndim != 2:
            W = W.reshape(-1, 1)             # force (p x A)

        w_bar = W.mean(axis=0, keepdims=True)  # (1 x A)
        diffs = W - w_bar                      # (p x A)

        S_inv = self._cov_inv(W)               # (A x A) or (1 x 1)
        # T²_j = (w_j - \bar{w})^T Σ^{-1} (w_j - \bar{w})
        T2 = np.einsum('ij,jk,ik->i', diffs, S_inv, diffs)
        return T2

    def _t2_cutoff(self, p: int, A: int, alpha: float) -> float:
        """
        C(p,A) * F_{A, p-A}(1-alpha); fallback to chi-square approx if SciPy F not available or p-A <= 0.
        """
        if (f_dist is None) or (p - A <= 0):
            # chi-square approximation (paper notes a χ² approximation is possible)
            from scipy.stats import chi2
            return float(chi2.ppf(1.0 - alpha, df=A))
        C = (A * (p - 1)) / (p - A)
        F_quant = float(f_dist.ppf(1.0 - alpha, dfn=A, dfd=(p - A)))
        return C * F_quant

    # ---------- Core training with inner CV ----------

    def learn(self, dataset, target_column: Optional[str] = None):
        """
        Inner-CV tunes (A, alpha) or (A, top_n), selects variables (keep high T² "outliers"),
        then refits final PLS on selected variables using full train set.
        """
        self.ensure_dependencies(dataset)
        df = dataset.get_dataframe()

        input_columns = self.get_input_columns()
        if not target_column:
            target_column = dataset.get_target_columns()[0]

        X = df[input_columns].copy()
        y = df[target_column].copy()
        X, y = self._clean_Xy(X, y)

        p = X.shape[1]
        kf = KFold(n_splits=self.inner_cv_folds, shuffle=True, random_state=self.random_state)

        best = (np.inf, None, None, None)  # (RMSEP, A, alpha, selected_colnames)

        # Grid over A and either alpha or top_n
        alpha_space = self.alpha_grid if self.selection_mode == "alpha" else [None]

        for A in self.A_grid:
            # safety: A must be <= rank-ish and at least 1
            if A < 1 or A > min(X.shape[0] - 1, X.shape[1]):
                continue

            for alpha in alpha_space:
                cv_rmseps = []

                for train_idx, val_idx in kf.split(X):
                    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

                    # Scale on TRAIN fold only; transform VAL with same scaler (paper centers/scales up-front).
                    scaler = StandardScaler()
                    X_tr_s = scaler.fit_transform(X_tr)
                    X_val_s = scaler.transform(X_val)

                    # Fit PLS on TRAIN fold to get W (x_weights_), shape p x A
                    pls = PLSRegression(n_components=A)
                    pls.fit(X_tr_s, y_tr)

                    W = pls.x_weights_  # p x A

                    if W.ndim != 2:
                        print(f"[HotellingPLS] Unexpected W.ndim={W.ndim}, reshaping to 2D")
                        W = np.atleast_2d(W)

                    # T² per variable from W (rows are variables across A comps)
                    T2 = self._compute_t2(W)

                    # Selection
                    if self.selection_mode == "alpha":
                        t2_cut = self._t2_cutoff(p=p, A=A, alpha=alpha)
                        keep_mask = T2 > t2_cut
                        keep_idx = np.where(keep_mask)[0]

                        # fallback if nothing selected: keep top-1 by T²
                        if keep_idx.size == 0:
                            keep_idx = np.argsort(T2)[::-1][:1]
                    else:
                        # fixed-N mode
                        if not self.top_n or self.top_n <= 0:
                            raise ValueError("selection_mode='top_n' requires positive top_n")
                        keep_idx = np.argsort(T2)[::-1][:self.top_n]

                    selected_cols = [input_columns[i] for i in keep_idx]

                    # Refit PLS on TRAIN fold, selected columns only
                    X_tr_sel = X_tr_s[:, keep_idx]
                    X_val_sel = X_val_s[:, keep_idx]

                    pls_sel = PLSRegression(n_components=min(A, X_tr_sel.shape[1]))
                    pls_sel.fit(X_tr_sel, y_tr)
                    y_hat = pls_sel.predict(X_val_sel).ravel()
                    cv_rmseps.append(self._rmse(y_val, y_hat))

                mean_rmsep = float(np.mean(cv_rmseps))
                if mean_rmsep < best[0]:
                    # Remember the *names* of selected columns using full-data fit next
                    # (We recompute selection below on full training to finalize the exact subset.)
                    best = (mean_rmsep, A, alpha, None)

        # Finalize with best (A, alpha)
        best_rmsep, A_star, alpha_star, _ = best
        if A_star is None:
            raise RuntimeError("No valid (A, alpha) combo found; check grids or data.")

        # --- Final selection on FULL training data ---
        # 1) Scale FULL X with a LOCAL scaler (used ONLY to get W_full/T²); do NOT store this scaler
        scaler_full = StandardScaler()
        X_s_full = scaler_full.fit_transform(X)

        # 2) Fit PLS on full scaled X to obtain W_full and compute T²
        pls_full = PLSRegression(n_components=A_star)
        pls_full.fit(X_s_full, y)
        W_full = pls_full.x_weights_
        T2_full = self._compute_t2(W_full)

        # 3) Select variables based on T²
        if self.selection_mode == "alpha":
            t2_cut = self._t2_cutoff(p=p, A=A_star, alpha=alpha_star)
            keep_idx = np.where(T2_full > t2_cut)[0]
            if keep_idx.size == 0:
                keep_idx = np.argsort(T2_full)[::-1][:1]  # safety: keep top-1
            self.t2_cutoff_ = float(t2_cut)
        else:
            keep_idx = np.argsort(T2_full)[::-1][:self.top_n]
            self.t2_cutoff_ = None  # N-based; no α-cut

        self.selected_columns_ = [input_columns[i] for i in keep_idx]
        self.t2_scores_ = pd.Series(T2_full, index=input_columns).sort_values(ascending=False)
        self.A_star_ = int(A_star)
        self.alpha_star_ = float(alpha_star) if alpha_star is not None else None

        # 4) Fit a NEW scaler on the SELECTED columns ONLY (this is the one we keep for predict)
        X_sel_df = X.iloc[:, keep_idx]                 # DataFrame with selected columns
        self.scaler = StandardScaler()
        X_sel_s = self.scaler.fit_transform(X_sel_df)  # fit on selected features only

        # 5) Fit the final PLS model on the selected+scaled features
        self.model = PLSRegression(n_components=min(A_star, X_sel_s.shape[1]))
        self.model.fit(X_sel_s, y)

        # 6) Reduce input_metrics to only those required by the selected columns
        #    so downstream ensure_dependencies() won't compute every candidate metric.
        try:
            if self.input_metrics:
                selected_set = set(self.selected_columns_)
                reduced_metrics = []
                for metric in self.input_metrics:
                    # MultiMetric: include if any submetric selected
                    if hasattr(metric, 'get_submetric_names'):
                        try:
                            subnames = set(metric.get_submetric_names())
                            if selected_set.intersection(subnames):
                                reduced_metrics.append(metric)
                                continue
                        except Exception:
                            pass
                    # Regular metric: include if its name selected
                    try:
                        metric_name = metric.get_name() if hasattr(metric, 'get_name') else None
                        if metric_name and metric_name in selected_set:
                            reduced_metrics.append(metric)
                            continue
                    except Exception:
                        pass
                if reduced_metrics:
                    self.input_metrics = reduced_metrics
        except Exception:
            # Non-fatal; fall back to original input_metrics if mapping fails
            pass

    # ---------- Predict on a (possibly new) dataset ----------

    def _predict_unsafe(self, dataset, update_dataset=True):
        """
        Predict using train-fitted scaler and only the selected columns.
        """
        if not self.selected_columns_:
            raise RuntimeError("Model not trained or no selected columns. Call learn() first.")

        df = dataset.get_dataframe().copy()
        # Ensure dependencies; then re-check we have needed columns
        missing = [c for c in self.selected_columns_ if c not in df.columns]
        if missing:
            self.ensure_dependencies(dataset)
            df = dataset.get_dataframe().copy()
            missing = [c for c in self.selected_columns_ if c not in df.columns]
        if missing:
            raise KeyError(f"Missing selected columns at predict-time: {missing}")

        X = df[self.selected_columns_].copy()
        # clean & scale with train scaler
        Xc, _ = self._clean_Xy(X, pd.Series(np.zeros(len(X))))  # only cleaning X
        Xs = self.scaler.transform(Xc)

        y_pred = self.model.predict(Xs).ravel()

        if update_dataset:
            df.loc[:, self.name] = y_pred
            dataset.set_dataframe(df)
            if self.name not in dataset.get_metric_columns():
                dataset.get_metric_columns().append(self.name)

        return y_pred

    # ---------- Interpretation helpers ----------

    def identify_important_metrics(self):
        """
        Return metrics ranked by T² (highest = most 'outlier' / informative under T²-PLS).
        Falls back to standardized coefficients if T² not available.
        """
        if self.t2_scores_ is not None:
            return list(self.t2_scores_.items())
        return super().identify_important_metrics()

    def get_selected_columns(self) -> List[str]:
        return list(self.selected_columns_)