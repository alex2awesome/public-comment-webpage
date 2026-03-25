# Experiment 10: Stratification Strategy Comparison

**Date:** 2026-03-21
**Server:** sk3
**Distribution:** llama-gen-v2-unmatched (pooled, no hierarchical shrinkage)
**Bootstrap:** 100 resamples, dedup enabled, 4 workers
**Doc types:** rule, proposed_rule

## Setup

The baseline approach stratifies by (agency, quarter), creating 1,909 strata -- many of
which are very small. This experiment tests whether coarser stratification improves
estimation by pooling more data per stratum.

| Experiment | Stratification | N strata (proposed_rule) | N strata (rule) |
|------------|---------------|--------------------------|-----------------|
| Baseline   | agency+quarter | 842                      | 1,067           |
| A          | quarter only   | 41                       | 41              |
| B          | year only      | 11                       | 11              |
| C          | agency only    | 44                       | 44              |

## Results: Cross-Comparison (sentence-weighted alpha)

### proposed_rule

| Stratification     | Pre-ChatGPT alpha | Post-ChatGPT alpha | Post/Pre ratio | Pre CI width (wtd) |
|--------------------|-------------------|---------------------|----------------|---------------------|
| baseline (ag+qtr)  | 0.002134          | 0.004789            | 2.24x          | 0.003270            |
| quarter only       | 0.002606          | 0.005553            | 2.13x          | 0.001399            |
| year only          | 0.002581          | 0.005507            | 2.13x          | 0.000715            |
| agency only        | -- (no time axis) | --                  | --             | 0.000883 (overall)  |

Overall sentence-weighted alpha for agency-only: **0.005143** (blends all years).

### rule

| Stratification     | Pre-ChatGPT alpha | Post-ChatGPT alpha | Post/Pre ratio | Pre CI width (wtd) |
|--------------------|-------------------|---------------------|----------------|---------------------|
| baseline (ag+qtr)  | 0.001222          | 0.001973            | 1.61x          | 0.002034            |
| quarter only       | 0.001006          | 0.001464            | 1.46x          | 0.000716            |
| year only          | 0.000953          | 0.001413            | 1.48x          | 0.000364            |
| agency only        | -- (no time axis) | --                  | --             | 0.000527 (overall)  |

Overall sentence-weighted alpha for agency-only: **0.001235** (blends all years).

## Key Observations

### 1. Coarser stratification dramatically tightens confidence intervals

The sentence-weighted average CI width drops substantially with coarser strata:

- **proposed_rule:** baseline 0.0033 -> quarter 0.0014 -> year 0.0007 (4.7x tighter)
- **rule:** baseline 0.0020 -> quarter 0.0007 -> year 0.0004 (5.6x tighter)

This confirms the hypothesis: (agency, quarter) creates many small strata where bootstrap
CIs are wide and unreliable. Pooling data into fewer, larger strata yields much more
precise estimates.

### 2. Spurious pre-ChatGPT alpha is similar across strategies, but more precisely estimated

All strategies produce roughly the same spurious pre-ChatGPT alpha:
- **proposed_rule:** ~0.21-0.26% regardless of stratification
- **rule:** ~0.095-0.12% regardless of stratification

The baseline's pre-ChatGPT *mean* alpha (0.52% for proposed_rule) is inflated because
unweighted averaging over many small strata overweights noisy small-sample estimates.
The *weighted* pre-ChatGPT alpha (0.21%) matches the coarser strategies. This confirms
the small-strata noise problem rather than a genuine bias difference.

### 3. The post/pre ratio is remarkably stable

- **proposed_rule:** post/pre ~2.1-2.2x across all time-based strategies
- **rule:** post/pre ~1.5-1.6x across all time-based strategies

The signal is consistent: proposed_rules show a ~2x increase in estimated AI prevalence
post-ChatGPT, while rules show a smaller ~1.5x increase. This pattern is robust to
stratification choice.

### 4. Quarter stratification reveals a clear temporal trend for proposed_rule

The quarterly series for proposed_rule shows a monotonic upward drift even pre-ChatGPT:
- 2016-2018: alpha ~0.15-0.20%
- 2019-2020: alpha ~0.17-0.48%
- 2021-2022: alpha ~0.23-0.51%
- 2023-2024: alpha ~0.34-0.76%
- 2025-2026: alpha ~0.34-0.93%

This gradual pre-ChatGPT rise (from ~0.15% to ~0.45%) is concerning -- it suggests either
(a) the detector is picking up stylistic drift over time, or (b) there is genuine pre-
ChatGPT AI-assisted writing growing over time. Either way, the pre-ChatGPT floor is not
zero, and the post-ChatGPT signal must be interpreted against this rising baseline.

### 5. For rules, 2016Q3 is an outlier

Rule alpha for 2016Q3 is 0.54% (vs ~0.02-0.04% for neighboring quarters). This is
likely driven by one agency's large batch of similarly-styled documents. This kind of
outlier is invisible in the baseline's 1,909-stratum approach but becomes apparent
when pooling all agencies per quarter.

### 6. Agency stratification shows high variance

With agency-only stratification (no time dimension):
- proposed_rule: 8/44 agencies above 1%, but most are near zero
- FAA dominates with alpha=2.47% (240K sentences, very tight CI)
- USCIS has alpha=7.6% but tiny sample (131 sentences, huge CI)
- FWS near zero (0.002%) with 188K sentences

This shows the signal is concentrated in a few agencies, not uniform.

### 7. Year stratification is the cleanest for trend analysis

Year-level estimates for proposed_rule show a smooth progression:
```
2016: 0.19%  ->  2019: 0.27%  ->  2022: 0.37%  ->  2023: 0.50%  ->  2024: 0.59%  ->  2026: 0.93%
```
The jump from 2022 (0.37%) to 2023 (0.50%) is noticeable but modest compared to the
steady pre-ChatGPT drift. The 2026 value (0.93%) is the highest but based on only one
partial quarter.

## Recommendation

**For trend analysis, use quarter or year stratification rather than agency+quarter.**

The baseline's (agency, quarter) approach creates too many small strata (median proposed_rule
stratum has ~200 sentences). This causes:
1. Wide, unreliable CIs (mean CI width 1.3% vs 0.1% for quarter-only)
2. Inflated unweighted mean alpha from noisy small strata
3. No meaningful gain: the weighted alpha is the same regardless

Quarter stratification provides a good balance: enough temporal resolution to see trends
while having ~30K-120K sentences per stratum for stable estimation.

## Output Files

- Quarter: `/lfs/skampere3/0/alexspan/regulations-demo/data/ai-usage-generations/exp-stratify/results_quarter.csv.gz` (82 rows)
- Year: `/lfs/skampere3/0/alexspan/regulations-demo/data/ai-usage-generations/exp-stratify/results_year.csv.gz` (22 rows)
- Agency: `/lfs/skampere3/0/alexspan/regulations-demo/data/ai-usage-generations/exp-stratify/results_agency.csv.gz` (88 rows)
