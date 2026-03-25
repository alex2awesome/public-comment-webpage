#!/usr/bin/env python3
"""Analyze per-agency AI usage results: weighted alpha pre/post ChatGPT, per doc type."""

import pandas as pd
import numpy as np
import sys

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "../ai-usage-generations/exp-per-agency/results.csv.gz"

df = pd.read_csv(RESULTS)

# Parse year from quarter string (e.g. "2023Q1" -> 2023)
# The 'year' column already exists
CHATGPT_CUTOFF = 2023  # ChatGPT released Nov 2022; 2023+ is post

print("=" * 80)
print("PER-AGENCY AI USAGE ANALYSIS")
print("=" * 80)
print(f"\nTotal rows: {len(df)}")
print(f"Doc types: {sorted(df['doc_type'].unique())}")
print(f"Agencies: {sorted(df['agency_id'].unique())}")
print(f"N agencies: {df['agency_id'].nunique()}")
print(f"Year range: {df['year'].min()} - {df['year'].max()}")

for doc_type in sorted(df['doc_type'].unique()):
    dt = df[df['doc_type'] == doc_type].copy()
    print(f"\n{'=' * 80}")
    print(f"DOC TYPE: {doc_type}")
    print(f"{'=' * 80}")
    print(f"  Agencies: {sorted(dt['agency_id'].unique())}")
    print(f"  N agencies: {dt['agency_id'].nunique()}")
    print(f"  Total strata: {len(dt)}")

    # Pre/post split
    pre = dt[dt['year'] < CHATGPT_CUTOFF]
    post = dt[dt['year'] >= CHATGPT_CUTOFF]

    # Sentence-weighted alpha
    def weighted_alpha(subset):
        w = subset['n_sentences']
        if w.sum() == 0:
            return 0.0
        return np.average(subset['alpha_estimate'], weights=w)

    def weighted_ci_width(subset):
        w = subset['n_sentences']
        if w.sum() == 0:
            return 0.0
        return np.average(subset['ci_upper'] - subset['ci_lower'], weights=w)

    pre_alpha = weighted_alpha(pre)
    post_alpha = weighted_alpha(post)
    pre_ci = weighted_ci_width(pre)
    post_ci = weighted_ci_width(post)

    print(f"\n  --- Sentence-weighted alpha ---")
    print(f"  Pre-ChatGPT  (< 2023): {pre_alpha:.6f}  (CI width: {pre_ci:.6f}, n_sents: {pre['n_sentences'].sum():,})")
    print(f"  Post-ChatGPT (>= 2023): {post_alpha:.6f}  (CI width: {post_ci:.6f}, n_sents: {post['n_sentences'].sum():,})")
    if pre_alpha > 0:
        print(f"  Post/Pre ratio: {post_alpha / pre_alpha:.2f}x")

    # Per-agency overall weighted alpha (all time)
    agency_stats = []
    for agency in sorted(dt['agency_id'].unique()):
        ag = dt[dt['agency_id'] == agency]
        ag_pre = ag[ag['year'] < CHATGPT_CUTOFF]
        ag_post = ag[ag['year'] >= CHATGPT_CUTOFF]

        overall_alpha = weighted_alpha(ag)
        pre_a = weighted_alpha(ag_pre) if len(ag_pre) > 0 else np.nan
        post_a = weighted_alpha(ag_post) if len(ag_post) > 0 else np.nan
        pre_ci_a = weighted_ci_width(ag_pre) if len(ag_pre) > 0 else np.nan
        post_ci_a = weighted_ci_width(ag_post) if len(ag_post) > 0 else np.nan

        agency_stats.append({
            'agency': agency,
            'n_strata': len(ag),
            'total_sents': ag['n_sentences'].sum(),
            'overall_alpha': overall_alpha,
            'pre_alpha': pre_a,
            'post_alpha': post_a,
            'pre_ci_width': pre_ci_a,
            'post_ci_width': post_ci_a,
            'pre_sents': ag_pre['n_sentences'].sum() if len(ag_pre) > 0 else 0,
            'post_sents': ag_post['n_sentences'].sum() if len(ag_post) > 0 else 0,
        })

    astats = pd.DataFrame(agency_stats).sort_values('pre_alpha', ascending=False)

    print(f"\n  --- Per-agency pre-ChatGPT alpha (sorted desc) ---")
    print(f"  {'Agency':<10} {'Pre-alpha':>12} {'Post-alpha':>12} {'Pre CI wid':>12} {'Pre sents':>12} {'Post sents':>12}")
    for _, r in astats.iterrows():
        pre_str = f"{r['pre_alpha']:.6f}" if not np.isnan(r['pre_alpha']) else "N/A"
        post_str = f"{r['post_alpha']:.6f}" if not np.isnan(r['post_alpha']) else "N/A"
        ci_str = f"{r['pre_ci_width']:.6f}" if not np.isnan(r['pre_ci_width']) else "N/A"
        print(f"  {r['agency']:<10} {pre_str:>12} {post_str:>12} {ci_str:>12} {r['pre_sents']:>12,.0f} {r['post_sents']:>12,.0f}")

    # Top 5 highest spurious (pre-ChatGPT) alpha
    print(f"\n  --- Top 5 HIGHEST pre-ChatGPT alpha (most spurious signal) ---")
    for _, r in astats.head(5).iterrows():
        print(f"    {r['agency']}: pre={r['pre_alpha']:.6f}, post={r['post_alpha']:.6f}, pre_sents={r['pre_sents']:,.0f}")

    # Bottom 5 lowest spurious (pre-ChatGPT) alpha
    astats_asc = astats.sort_values('pre_alpha', ascending=True)
    print(f"\n  --- Top 5 LOWEST pre-ChatGPT alpha (least spurious signal) ---")
    valid = astats_asc[astats_asc['pre_alpha'].notna()]
    for _, r in valid.head(5).iterrows():
        print(f"    {r['agency']}: pre={r['pre_alpha']:.6f}, post={r['post_alpha']:.6f}, pre_sents={r['pre_sents']:,.0f}")

print("\n" + "=" * 80)
print("DONE")
