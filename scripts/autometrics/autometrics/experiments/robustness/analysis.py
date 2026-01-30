import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import MultiComparison
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from autometrics.experiments.results import TabularResult, FigureResult, PlotlyResult

# Bump Matplotlib font sizes globally
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14
})

def _get_metric_cols(metric_objs):
    cols = []
    for m in metric_objs:
        if hasattr(m, 'get_submetric_names'):
            cols.extend(m.get_submetric_names())
        else:
            cols.append(m.get_name())
    return cols

def _normalize_df(df, cols):
    df_norm = df.copy()
    df_norm[cols] = MinMaxScaler().fit_transform(df_norm[cols])
    return df_norm

def _get_tukey_clusters(groups, tukey):
    parent = {g: g for g in groups}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    for row in tukey.summary().data[1:]:
        g1, g2, *_, reject = row
        if not reject:
            union(g1, g2)
    clusters = {}
    for g in groups:
        root = find(g)
        clusters.setdefault(root, []).append(g)
    return clusters

def _do_anova_tukey(df_norm, col, dimension, results):
    model = ols(f"{col} ~ C(group)", data=df_norm).fit()
    anova_tbl = sm.stats.anova_lm(model, typ=2)
    results[f"{dimension}/{col}/anova"] = TabularResult(anova_tbl)
    mc = MultiComparison(df_norm[col], df_norm['group'])
    tukey = mc.tukeyhsd(alpha=0.05)
    tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
    results[f"{dimension}/{col}/tukey"] = TabularResult(tukey_df)
    return tukey

def _plot_bar_mpl(df_norm, col, dimension, tukey, results):
    groups = ['original','same_subtle','same_obvious','worse_subtle','worse_obvious']
    means = df_norm.groupby('group')[col].mean().reindex(groups)
    ci95  = 1.96 * df_norm.groupby('group')[col].sem().reindex(groups)
    clusters = _get_tukey_clusters(groups, tukey)
    roots   = list(clusters.keys())
    cmap    = plt.get_cmap('tab10')
    colors  = [cmap(roots.index(next(r for r,m in clusters.items() if g in m))) for g in groups]

    fig, ax = plt.subplots(figsize=(8,6))
    x = np.arange(len(groups))
    ax.bar(x, means, yerr=ci95, capsize=5, color=colors)
    ax.axhline(means['original'], ls='--', color='gray')
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace('_',' ').title() for g in groups], rotation=30)
    ax.set_ylabel(f"Normalized {col}")
    ax.set_title(f"{col} vs Perturbation Type ({dimension.capitalize()})")
    plt.tight_layout()
    results[f"{dimension}/{col}/bar"] = FigureResult(fig)
    plt.close(fig)

def _plot_bar_plotly(df_norm, col, dimension, tukey, results):
    groups = ['original','same_subtle','same_obvious','worse_subtle','worse_obvious']
    means = df_norm.groupby('group')[col].mean().reindex(groups)
    ci95  = 1.96 * df_norm.groupby('group')[col].sem().reindex(groups)
    clusters = _get_tukey_clusters(groups, tukey)
    roots = list(clusters.keys())
    cmap = plt.get_cmap('tab10')
    group_color = {}
    for idx, root in enumerate(roots):
        for g in clusters[root]:
            r, g_, b, a = [int(255*x) for x in cmap(idx)]
            group_color[g] = f"rgba({r},{g_},{b},{a})"

    fig = go.Figure()
    for g in groups:
        fig.add_trace(go.Bar(
            x=[g.replace('_',' ').title()],
            y=[means[g]],
            error_y=dict(type='data', array=[ci95[g]]),
            marker_color=group_color[g],
            showlegend=False
        ))
    fig.add_shape(dict(
        type='line', x0=-0.5, x1=len(groups)-0.5,
        y0=means['original'], y1=means['original'],
        line=dict(color='gray', dash='dash')
    ))
    fig.update_layout(
        title=f"{col} vs Perturbation Type ({dimension.capitalize()})",
        xaxis_title="Perturbation Type",
        yaxis_title=f"Normalized {col}",
        barmode='group',
        font=dict(size=16),
        showlegend=False
    )
    results[f"{dimension}/{col}/bar_interactive"] = PlotlyResult(fig)

def _build_summary(df_norm, cols):
    rows  = []
    orig  = df_norm[df_norm['group'] == 'original']

    for col in cols:
        row = {'metric': col}

        # sensitivity
        for g in ['worse_subtle', 'worse_obvious']:
            part   = df_norm[df_norm['group'] == g]
            merged = pd.merge(
                part[['sample_id', col]],
                orig[['sample_id', col]],
                on='sample_id',
                suffixes=('_p', '_o')
            )
            row[f'sensitivity_{g}'] = (merged[f'{col}_o'] - merged[f'{col}_p']).mean()

        # stability
        for g in ['same_subtle', 'same_obvious']:
            part   = df_norm[df_norm['group'] == g]
            merged = pd.merge(
                part[['sample_id', col]],
                orig[['sample_id', col]],
                on='sample_id',
                suffixes=('_p', '_o')
            )
            row[f'stability_{g}'] = 1 - np.abs(
                merged[f'{col}_o'] - merged[f'{col}_p']
            ).mean()

        rows.append(row)

    df = pd.DataFrame(rows)
    assert df.filter(like='NaN').empty    # should pass

    return df

def _plot_scatter_mpl(summary_df, dimension, results):
    fig, ax = plt.subplots(figsize=(8,6))
    for _, r in summary_df.iterrows():
        x = np.mean([r['stability_same_subtle'], r['stability_same_obvious']])
        y = np.mean([r['sensitivity_worse_subtle'], r['sensitivity_worse_obvious']])
        ax.scatter(x, y, s=200, marker='o', color='tab:blue', zorder=5)
        ax.annotate(r['metric'], xy=(x,y), xytext=(-4,4),
                    textcoords='offset points', fontsize=14, ha='right', va='bottom')
    ax.set_xlabel("Avg Stability")
    ax.set_ylabel("Avg Sensitivity")
    ax.set_title(f"Stability vs Sensitivity ({dimension.capitalize()})")
    plt.tight_layout()
    results[f"{dimension}/scatter"] = FigureResult(fig)
    plt.close(fig)

def _plot_scatter_plotly(summary_df, dimension, results):
    fig = go.Figure()
    for _, r in summary_df.iterrows():
        x = np.mean([r['stability_same_subtle'], r['stability_same_obvious']])
        y = np.mean([r['sensitivity_worse_subtle'], r['sensitivity_worse_obvious']])
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            name=r['metric'],
            text=[r['metric']],
            textposition='top left',
            marker=dict(size=14)
        ))
    fig.update_layout(
        title=f"Stability vs Sensitivity ({dimension.capitalize()})",
        xaxis_title="Average Stability", yaxis_title="Average Sensitivity",
        font=dict(size=16)
    )
    results[f"{dimension}/scatter_interactive"] = PlotlyResult(fig)

# ────────────────────────────────────────────────────────────────────
# NEW: per-sample |Δ| histogram for every perturbation group
#      – robust to duplicated rows & closes the figure afterward
# ────────────────────────────────────────────────────────────────────
def _plot_abs_diffs(df_norm, metric, dimension, results):
    for pert_group in ["same_subtle",
                       "same_obvious",
                       "worse_subtle",
                       "worse_obvious"]:

        # keep only original + one target group
        tmp = df_norm[df_norm["group"].isin(["original", pert_group])]

        # one value per (sample_id, group): take the mean if duplicates exist
        pairs = (
            tmp.pivot_table(index="sample_id",
                            columns="group",
                            values=metric,
                            aggfunc="mean")
            .dropna()                      # ensure both columns are present
        )

        if pairs.empty:          # safety guard in case a group is missing
            continue

        abs_diffs = (pairs["original"] - pairs[pert_group]).abs()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(abs_diffs, bins=20)
        nice_name = pert_group.replace("_", " ").title().replace(" ", "_")
        ax.set_title(f"|Δ| histogram – {metric} ({nice_name})")
        ax.set_xlabel("|Δ|")
        ax.set_ylabel("# samples")
        plt.tight_layout()

        results[f"{dimension}/{metric}/abs_diff_hist_{pert_group}"] = FigureResult(fig)
        plt.close(fig)  # prevent “too many open figures”

def analyze_and_plot(df: pd.DataFrame, metric_objs: list, dimension: str, results: dict):
    cols    = _get_metric_cols(metric_objs)
    df_norm = _normalize_df(df, cols)
    for col in cols:
        tukey = _do_anova_tukey(df_norm, col, dimension, results)
        _plot_bar_mpl(df_norm, col, dimension, tukey, results)
        _plot_bar_plotly(df_norm, col, dimension, tukey, results)
        _plot_abs_diffs(df_norm, col, dimension, results)
    summary_df = _build_summary(df_norm, cols)
    results[f"{dimension}/sens_stab"] = TabularResult(summary_df)
    _plot_scatter_mpl(summary_df, dimension, results)
    _plot_scatter_plotly(summary_df, dimension, results)