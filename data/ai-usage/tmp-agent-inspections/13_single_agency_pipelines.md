# Single-Agency Pipeline Analysis (EPA, FAA, CMS, SEC)

## Overview

Ran complete end-to-end pipelines for individual agencies to replicate v1 results.
Each agency had its own `estimate` step (agency-specific P and Q distributions) followed
by `infer` stratified by quarter.

**Bug fix applied**: `corpus_ai_usage.py` had a bug where `doc_ids_filter` in matched
mode loaded ALL doc IDs from the AI parquet regardless of `--agencies` filter. This caused
an assertion failure (e.g., EPA: 12277 filter IDs vs 2660 actual EPA AI docs). Fixed by
applying the agency filter when building `doc_ids_filter` during the estimate step.

**Parameters**: `--doc-types rule proposed_rule`, `--dedup`, `--bootstrap-n 1000`, `--workers 4`

---

## Summary Table

| Agency | Pre-ChatGPT alpha | Post-ChatGPT alpha | Ratio | Pre docs | Post docs | Clean Pattern? |
|--------|-------------------|---------------------|-------|----------|-----------|----------------|
| EPA    | 0.42% [0.28%, 0.57%] | 0.69% [0.48%, 0.91%] | 1.65x | 6,727 | 1,951 | MODERATE |
| FAA    | 0.26% [0.13%, 0.41%] | 1.14% [0.87%, 1.43%] | 4.44x | 7,481 | 2,819 | YES - STRONG |
| CMS    | 1.23% [0.98%, 1.51%] | 3.95% [3.27%, 4.70%] | 3.20x | 363 | 106 | NO - high baseline |
| SEC    | 0.41% [0.09%, 0.98%] | 1.62% [0.52%, 2.95%] | 3.90x | 215 | 114 | YES - but noisy |

ChatGPT cutoff: 2023Q1 (first full quarter after Nov 2022 release)

---

## EPA Detailed Analysis

### Pre/Post Summary
- **Pre-ChatGPT**: 0.42% [0.28%, 0.57%] (28 quarters, 6,727 docs, 729,710 sentences)
- **Post-ChatGPT**: 0.69% [0.48%, 0.91%] (13 quarters, 1,951 docs, 247,116 sentences)
- **Ratio**: 1.65x

### Year-Level Trend
| Year | Alpha | 95% CI | Docs | Pattern |
|------|-------|--------|------|---------|
| 2016 | 0.57% | [0.43%, 0.72%] | 1,134 | Slightly elevated |
| 2017 | 0.38% | [0.23%, 0.53%] | 1,044 | Baseline |
| 2018 | 0.25% | [0.14%, 0.39%] | 1,048 | Low baseline |
| 2019 | 0.25% | [0.14%, 0.37%] | 982 | Low baseline |
| 2020 | 0.38% | [0.25%, 0.52%] | 931 | Baseline |
| 2021 | 0.57% | [0.40%, 0.75%] | 804 | Slightly elevated |
| 2022 | 0.51% | [0.35%, 0.68%] | 784 | Pre-ChatGPT |
| 2023* | 0.60% | [0.40%, 0.80%] | 621 | Mild increase |
| 2024* | 0.52% | [0.35%, 0.70%] | 645 | No clear increase |
| 2025* | 1.00% | [0.74%, 1.26%] | 602 | Clear increase |
| 2026* | 0.75% | [0.45%, 1.07%] | 83 | Elevated (partial) |

### EPA Assessment
- Pre-ChatGPT alpha is NOT near 0% -- it fluctuates 0.2-0.6% with some noisy quarters
- Post-ChatGPT shows only a modest increase; the real jump appears in 2025, not 2023
- 2021 is oddly elevated (0.57%) at the same level as 2023
- The increase starts clearly in **2025Q2** (1.11%) driven by rules (2.02%)
- Does NOT show the clean v1 pattern -- pre-ChatGPT baseline is too variable
- EPA's pre-ChatGPT noise (0.2-0.8% range) makes the post signal hard to distinguish

### EPA Quarterly Time Series (Combined rule + proposed_rule)
```
  2016Q1  0.78%    2016Q2  0.56%    2016Q3  0.57%    2016Q4  0.39%
  2017Q1  0.52%    2017Q2  0.33%    2017Q3  0.28%    2017Q4  0.43%
  2018Q1  0.22%    2018Q2  0.23%    2018Q3  0.22%    2018Q4  0.36%
  2019Q1  0.30%    2019Q2  0.22%    2019Q3  0.24%    2019Q4  0.24%
  2020Q1  0.33%    2020Q2  0.37%    2020Q3  0.48%    2020Q4  0.33%
  2021Q1  0.62%    2021Q2  0.39%    2021Q3  0.77%    2021Q4  0.46%
  2022Q1  0.53%    2022Q2  0.61%    2022Q3  0.46%    2022Q4  0.45%
  ------- ChatGPT Release (Nov 2022) -------
  2023Q1  0.50%    2023Q2  0.83%    2023Q3  0.47%    2023Q4  0.62%
  2024Q1  0.65%    2024Q2  0.41%    2024Q3  0.58%    2024Q4  0.46%
  2025Q1  0.49%    2025Q2  1.11%    2025Q3  1.26%    2025Q4  0.86%
  2026Q1  0.75%
```

---

## FAA Detailed Analysis

### Pre/Post Summary
- **Pre-ChatGPT**: 0.26% [0.13%, 0.41%] (28 quarters, 7,481 docs)
- **Post-ChatGPT**: 1.14% [0.87%, 1.43%] (13 quarters, 2,819 docs)
- **Ratio**: 4.44x

### Year-Level Trend
| Year | Alpha | 95% CI | Docs | Pattern |
|------|-------|--------|------|---------|
| 2016 | 0.22% | [0.11%, 0.37%] | 1,071 | Low baseline |
| 2017 | 0.27% | [0.13%, 0.43%] | 992 | Low baseline |
| 2018 | 0.25% | [0.11%, 0.40%] | 1,037 | Low baseline |
| 2019 | 0.64% | [0.40%, 0.90%] | 779 | Anomalous spike |
| 2020 | 0.20% | [0.08%, 0.34%] | 1,096 | Low baseline |
| 2021 | 0.15% | [0.06%, 0.25%] | 1,250 | Very low |
| 2022 | 0.20% | [0.09%, 0.33%] | 1,256 | Low baseline |
| 2023* | 1.29% | [1.01%, 1.60%] | 1,003 | **Sharp increase** |
| 2024* | 1.14% | [0.86%, 1.44%] | 870 | Sustained high |
| 2025* | 0.78% | [0.58%, 1.02%] | 779 | Moderately high |
| 2026* | 1.73% | [1.36%, 2.12%] | 167 | High (partial) |

### FAA Assessment
- **BEST RESULTS** -- closest to clean v1 pattern
- Pre-ChatGPT baseline is very low (0.15-0.27%) for most years
- 2019 is anomalous (0.64%) driven by a single Q2 spike (1.35%)
- Post-ChatGPT jump is sharp and immediate: 2022Q4=0.47% -> 2023Q2=1.33%
- The increase begins in **2023Q2** and sustains through all subsequent quarters
- CIs are well-separated: pre [0.13%, 0.41%] vs post [0.87%, 1.43%] -- no overlap
- Pre-ChatGPT alpha is near 0% (0.26% with CI touching 0.13%)
- Large sample sizes (7,481 pre docs, 2,819 post docs) give tight estimates

### FAA Quarterly Time Series
```
  2016Q1  0.15%    2016Q2  0.20%    2016Q3  0.22%    2016Q4  0.33%
  2017Q1  0.39%    2017Q2  0.06%    2017Q3  0.23%    2017Q4  0.54%
  2018Q1  0.18%    2018Q2  0.35%    2018Q3  0.23%    2018Q4  0.21%
  2019Q1  0.58%    2019Q2  1.35%    2019Q3  0.56%    2019Q4  0.12%
  2020Q1  0.08%    2020Q2  0.28%    2020Q3  0.15%    2020Q4  0.27%
  2021Q1  0.11%    2021Q2  0.07%    2021Q3  0.14%    2021Q4  0.24%
  2022Q1  0.04%    2022Q2  0.06%    2022Q3  0.23%    2022Q4  0.47%
  ------- ChatGPT Release (Nov 2022) -------
  2023Q1  0.40%    2023Q2  1.33%    2023Q3  1.14%    2023Q4  2.15%   <<< JUMP
  2024Q1  1.56%    2024Q2  0.90%    2024Q3  0.99%    2024Q4  1.08%
  2025Q1  0.68%    2025Q2  1.10%    2025Q3  0.38%    2025Q4  1.00%
  2026Q1  1.73%
```

---

## CMS Detailed Analysis

### Pre/Post Summary
- **Pre-ChatGPT**: 1.23% [0.98%, 1.51%] (28 quarters, 363 docs)
- **Post-ChatGPT**: 3.95% [3.27%, 4.70%] (13 quarters, 106 docs)
- **Ratio**: 3.20x

### CMS Assessment
- **DOES NOT show clean pattern** -- pre-ChatGPT baseline is already high (1.23%)
- Enormous variance: individual quarters range from 0.12% to 13.88% pre-ChatGPT
- Very small sample sizes (5-29 docs per quarter) lead to unreliable estimates
- 2021 is 6.36% -- far higher than any post-ChatGPT year except 2024
- CMS writes extremely long, complex rules (some quarters with 100k+ sentences from ~25 docs)
- The method's word-frequency approach may not work well for CMS's specialized vocabulary
- Pre-ChatGPT has wild swings (e.g., 2018Q2=10.80%, 2022Q1=13.88%) that dwarf any post effect

---

## SEC Detailed Analysis

### Pre/Post Summary
- **Pre-ChatGPT**: 0.41% [0.09%, 0.98%] (27 quarters, 215 docs)
- **Post-ChatGPT**: 1.62% [0.52%, 2.95%] (13 quarters, 114 docs)
- **Ratio**: 3.90x

### SEC Assessment
- Shows a clear increase but with very wide confidence intervals
- Pre-ChatGPT baseline is low (0.41%) but with a wide CI touching near 0%
- Post-ChatGPT CI [0.52%, 2.95%] barely separates from pre [0.09%, 0.98%]
- Very small sample sizes (3-16 docs per quarter) make individual quarters unreliable
- 2021Q3 has 11.13% alpha but only 4 docs (180 sentences) -- clearly noise
- 2022Q4 has 6.85% alpha with only 5 docs (208 sentences)
- The 3.90x ratio is encouraging but driven heavily by a few high-variance quarters
- SEC has too few regulatory documents for reliable per-quarter estimation

---

## Cross-Agency Comparison

### Which agencies show the clean v1 pattern?

1. **FAA: YES** -- Low pre baseline (~0.2%), sharp jump in 2023Q2, sustained elevation.
   Best candidate for demonstrating the method. 4.44x ratio with non-overlapping CIs.

2. **EPA: MODERATE** -- Pre baseline is variable (0.2-0.8%). Post increase is real but
   modest (1.65x). The increase concentrates in 2025, not immediately after ChatGPT.
   Does NOT cleanly replicate v1 because pre-ChatGPT baseline is too noisy.

3. **SEC: SUGGESTIVE** -- 3.90x ratio but sample sizes are too small for reliable
   quarterly estimation. The direction is right but individual quarters are too noisy.

4. **CMS: NO** -- High and wildly variable baseline. Very small sample sizes.
   Not suitable for this analysis.

### When does the increase begin?

| Agency | First sustained elevation | Nature of increase |
|--------|--------------------------|-------------------|
| EPA    | 2025Q2 | Delayed; 2023-2024 look like pre-ChatGPT |
| FAA    | 2023Q2 | Immediate and sharp; matches ChatGPT timeline |
| CMS    | N/A | Too noisy to determine |
| SEC    | 2023Q1 | Suggestive but high variance |

### Is pre-ChatGPT alpha near 0%?

| Agency | Pre-alpha | Near 0%? |
|--------|-----------|----------|
| EPA    | 0.42% | No -- consistently 0.2-0.6% with noisy spikes |
| FAA    | 0.26% | Approximately yes -- CI [0.13%, 0.41%] |
| CMS    | 1.23% | No -- far from 0 |
| SEC    | 0.41% | Borderline -- CI [0.09%, 0.98%] touches near 0 |

---

## Key Findings

1. **FAA is the cleanest single-agency result**: low pre-ChatGPT baseline, sharp post
   increase, large sample sizes, non-overlapping CIs. This is the agency that best
   demonstrates the method's validity.

2. **EPA does NOT cleanly replicate v1**: The pre-ChatGPT baseline (0.42%) is meaningfully
   above 0, and 2021/2022 already show similar levels to 2023/2024. The real EPA increase
   appears to start in 2025, suggesting a delayed adoption pattern.

3. **Agency-specific distributions help**: By building P and Q from EPA-only documents,
   we remove cross-agency vocabulary contamination. But EPA's own regulatory language
   appears to have some inherent overlap with AI-generated text.

4. **Sample size matters enormously**: CMS and SEC have too few documents per quarter for
   reliable estimation. FAA and EPA have ~200-300+ docs per quarter, producing tight CIs.

5. **The 2019 FAA anomaly** (Q2 at 1.35%) warrants investigation -- it could be a
   particular rulemaking with unusual language, or a data artifact.

---

## Technical Notes

- Bug fix in `corpus_ai_usage.py` line 640-643: Added agency filtering when building
  `doc_ids_filter` during matched-mode estimate. Without this fix, the assertion at
  line 791 fails because the filter includes doc IDs from all agencies but the AI
  corpus is filtered to one agency.
- EPA: 2,660 rule docs + 2,036 proposed_rule docs in AI corpus
- FAA: Large corpus with consistent quarterly coverage
- CMS: Very few docs but individual docs can be enormous (100k+ sentences)
- SEC: Small agency with sparse regulatory output
- All runs used `--dedup` but dedup mapper files were not found (warning logged)
- EPA agency+quarter results are identical to quarter-only results (confirmed zero diff)
