# Experiment: Word Length >= 4 + Min 250 Words Original Text

**Date:** 2026-03-21

**Goal:** Combine two filtering strategies to reduce noise from short/function words and
short documents: (1) remove words shorter than 4 characters from distributions, and
(2) exclude AI corpus documents whose original text is under 250 words.

## Pipeline

1. **Filter AI corpus** (min 250 words original text):
   - rule: 12,343 -> 10,261 docs (16.9% removed)
   - proposed_rule: 7,291 -> 5,502 docs (24.5% removed)
2. **Estimate** distributions with `--no-matched` (unmatched corpus, all human docs).
   - rule: 15,561 human docs (1,539,220 sents), 10,261 AI docs (292,166 sents), vocab=20,347
   - proposed_rule: 11,297 human docs (1,325,670 sents), 5,502 AI docs (180,235 sents), vocab=17,386
   - Optimal kappa (rule): human=708.5, AI=472.8
   - Optimal kappa (proposed_rule): human=621.2, AI=397.8
3. **Filter distributions** to words >= 4 chars:
   - rule: 20,347 -> 17,837 words (12.3% removed)
   - proposed_rule: 17,386 -> 15,332 words (11.8% removed)
4. **Infer** with `--stratify-by agency quarter --dedup --bootstrap-n 100 --workers 4`.

## Key Results

### Overall
| doc_type       | mean alpha | median alpha | strata | total docs | total sentences |
|----------------|-----------|-------------|--------|-----------|----------------|
| rule           | 0.0059    | 0.0000      | 1,067  | 18,962    | 2,103,158      |
| proposed_rule  | 0.0094    | 0.0000      | 842    | 13,029    | 1,816,633      |

### Pre vs Post ChatGPT (2023-Q1 cutoff)

**Unweighted (mean of strata):**
| doc_type       | pre-ChatGPT | post-ChatGPT | diff    |
|----------------|------------|-------------|---------|
| rule           | 0.0071     | 0.0032      | -0.0039 |
| proposed_rule  | 0.0092     | 0.0101      | +0.0009 |

**Sentence-weighted:**
| doc_type       | pre-ChatGPT | post-ChatGPT | diff    |
|----------------|------------|-------------|---------|
| rule           | 0.0032     | 0.0025      | -0.0007 |
| proposed_rule  | 0.0069     | 0.0073      | +0.0004 |

### Alpha Distribution (percentiles)
| percentile | rule   | proposed_rule |
|-----------|--------|--------------|
| p10       | 0.0000 | 0.0000       |
| p25       | 0.0000 | 0.0000       |
| p50       | 0.0000 | 0.0000       |
| p75       | 0.0014 | 0.0054       |
| p90       | 0.0113 | 0.0304       |
| p95       | 0.0302 | 0.0549       |
| p99       | 0.1360 | 0.1198       |

### No negative or >0.5 alpha values
- Zero strata with negative alpha (0/1,909).
- Zero strata with alpha > 0.5 (0/1,909).
- Clipped and raw means are identical -- all estimates are in [0, 1].

### Top Agencies by Alpha
**Rules:** FEMA (0.082), COE (0.024), USCBP (0.022), DHS (0.012), FSIS (0.010)
**Proposed rules:** FAA (0.042), COE (0.035), MSHA (0.033), FSIS (0.025), USPS (0.024)

## Comparison with Prior Experiments

| Experiment                    | rule mean | rule pre/post diff | PR mean | PR pre/post diff |
|------------------------------|----------|-------------------|---------|-----------------|
| exp-stratify (quarter only)  | 0.0012   | +0.0006           | 0.0037  | +0.0031         |
| exp-aggressive-a             | 0.0054   | -0.0032           | 0.0098  | +0.0023         |
| exp-lor-filter (LOR 0.3)    | 0.0043   | -0.0013           | 0.0078  | +0.0015         |
| **exp-wl4-min250 (THIS)**   | **0.0059** | **-0.0039**     | **0.0094** | **+0.0009** |

## Observations

1. **Combined filtering produces the largest pre-post divergence for rules.**
   The rule pre-post diff is -0.0039, the most negative across all experiments.
   This means estimated AI usage *decreases* after ChatGPT for rules, which is
   counterintuitive and suggests either (a) the method is measuring stylistic
   noise rather than actual AI usage, or (b) rules genuinely did not adopt
   AI-generated text.

2. **Proposed rules show a small positive post-ChatGPT signal (+0.0009 unweighted,
   +0.0004 sentence-weighted).** This is the smallest positive signal among all
   experiments, much smaller than the +0.0031 in the quarter-only baseline or the
   +0.0023 in aggressive-a.

3. **Overall estimates are higher than the quarter-only baseline** (rule 0.0059 vs
   0.0012; PR 0.0094 vs 0.0037). The agency-quarter stratification inflates per-
   stratum means because small strata with high noise get equal weight. Sentence-
   weighted means are lower (rule 0.0032 pre, 0.0025 post).

4. **No degenerate estimates.** All alpha values are in [0, 1] -- no negative or
   >0.5 outliers. The clipping of short function words and short documents appears
   to eliminate the extreme estimates seen in some other configurations.

5. **Distributions are sparse.** Median alpha is 0.0000 for both doc types, with
   75th percentile at only 0.0014 (rule) and 0.0054 (PR). Over 50% of strata show
   essentially zero AI usage signal.

6. **The min-250-word filter removed ~17-25% of AI corpus documents.** This is a
   meaningful fraction. The removed documents were likely too short to produce
   reliable word-frequency signals, contributing noise.

7. **Word length filtering removed ~12% of vocabulary** (words under 4 chars).
   These are primarily function words (the, a, of, in, etc.) that carry little
   discriminative signal between human and AI text.

8. **FEMA dominates rules** with alpha=0.082, far above the next highest (COE at
   0.024). For proposed rules, FAA leads at 0.042. These agencies should be
   investigated to determine if this reflects genuine AI patterns or corpus
   artifacts.

## Files
- AI corpus: `/lfs/skampere3/0/alexspan/regulations-demo/data/ai-usage-generations/exp-wl4-min250/ai_corpus_{rule,proposed_rule}.parquet`
- Distributions: `.../exp-wl4-min250/distributions/` (full vocab) and `.../distributions_wl4/` (words >= 4 chars)
- Results: `.../exp-wl4-min250/results_wl4.csv.gz`
