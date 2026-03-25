# False Positive Alpha in Pre-ChatGPT Regulatory Data

## Summary

Pre-ChatGPT strata (before 2023Q1) should have zero AI usage, so any nonzero alpha
estimates represent false positives. The overall weighted pre-ChatGPT alpha is **0.28%**
(0.15% for rules, 0.42% for proposed rules). While this is small in absolute terms, it is
nonzero and certain agencies show dramatically higher spurious rates.

## Top 5 Highest-Alpha Pre-ChatGPT Strata

| Rank | Agency | Quarter | Doc Type | Alpha | CI | N_docs | N_sent |
|------|--------|---------|----------|-------|----|--------|--------|
| 1 | NHTSA | 2018Q3 | proposed_rule | 0.164 | [0.072, 0.272] | 2 | 91 |
| 2 | FEMA | 2017Q1 | rule | 0.163 | [0.117, 0.215] | 11 | 337 |
| 3 | FEMA | 2019Q2 | rule | 0.161 | [0.096, 0.229] | 9 | 211 |
| 4 | DOJ | 2018Q3 | rule | 0.138 | [0.015, 0.273] | 2 | 55 |
| 5 | ED | 2017Q2 | proposed_rule | 0.137 | [0.037, 0.256] | 2 | 76 |

### Key observations about these strata:

- **3 of 5 are small strata** (<100 sentences), which inflates noise. Among the top 20
  highest-alpha pre-ChatGPT strata, 12/20 have <100 sentences and 13/20 have <5 documents.

- **FEMA is the standout exception**: It appears twice in the top 5 with moderate-to-large
  strata (337 and 211 sentences, 11 and 9 documents). FEMA has 11 of its 34 pre-ChatGPT
  strata with CI lower bounds above 0.001 -- a 32.4% "significance" rate. Its weighted
  pre-ChatGPT alpha is 2.2%, nearly 8x the overall average.

- **The CI for FEMA 2017Q1 excludes zero** ([0.117, 0.215]), meaning the model is
  confidently wrong that 12-21% of FEMA text is AI-generated in 2017.

## Statistical Properties Driving False Positives

### Stratum size effect
- Spearman correlation between n_sentences and alpha: r=0.063 (p=0.021) -- weak but significant
- Mean alpha by stratum size:
  - <100 sentences: 0.0152
  - 100-500 sentences: 0.0076
  - 500+ sentences: 0.0024
- Small strata are noisier, but the effect is not the whole story.

### Expected log-LR analysis

Under the bag-of-words model, for a human sentence drawn from the global distribution:
- **Expected log-LR = -5.25** (correctly negative, meaning human text is identified as human on average)
- **Std dev = 2.58**
- **P(log-LR > 0 | human) = 2.1%** (under normal approximation)

This means about 2% of human sentences will have a positive log-likelihood ratio
favoring the AI distribution, simply due to variance.

### No words have net positive contribution

Every word in the vocabulary has a negative expected contribution to the log-LR under
human data. The false positive rate comes entirely from **variance**: some sentences
happen to use the "right" combination of words that, taken together, look more like
AI text.

### Top variance-contributing words (Rules)

The words contributing most to log-LR variance are common function/regulatory words
that appear at different rates in human vs. AI text:

| Word | Var Contribution | Human P | AI P |
|------|-----------------|---------|------|
| and | 0.168 | 0.296 | 0.508 |
| this | 0.128 | 0.108 | 0.278 |
| the | 0.105 | 0.661 | 0.794 |
| mips | 0.104 | 0.004 | 0.000 |
| is | 0.060 | 0.134 | 0.241 |
| would | 0.059 | 0.031 | 0.008 |
| federal | 0.057 | 0.018 | 0.099 |

These are words where human and AI frequencies differ notably. The top 30 words
account for 22% of total variance.

## Agency-Level False Positive Analysis

### Mechanism: agency vocabulary mismatch with pooled distribution

The human distribution P is estimated by pooling text across all agencies. When a specific
agency's text differs from this pooled distribution -- particularly on words with large
delta (AI-likeness shift when present) -- the expected log-LR for that agency's text
shifts upward (toward AI), increasing false positive rates.

### Agencies ranked by P(false positive per sentence)

| Agency | Doc Type | P(FP) | Weighted Alpha | N Sentences |
|--------|----------|-------|----------------|-------------|
| TSA | proposed_rule | 9.1% | 0.010 | 402 |
| FAA | proposed_rule | 8.9% | 0.018 | 167,822 |
| USCIS | proposed_rule | 8.5% | 0.020 | 110 |
| FSIS | proposed_rule | 8.2% | 0.025 | 1,459 |
| NIH | rule | 8.0% | -- | 147 |
| MSHA | proposed_rule | 7.6% | 0.015 | 1,848 |
| FEMA | rule | 7.6% | 0.026 | 9,489 |

**Spearman correlation between P(FP) and weighted alpha: r=0.568, p<0.0001**

This strong correlation confirms that agencies whose text deviates from the pooled
human distribution in AI-like directions show higher spurious alpha estimates.

### FEMA deep dive: what makes it look "AI-like"?

FEMA's expected log-LR is -3.85 (vs. -5.25 global), a shift of +1.40 toward AI.
The top words driving this shift are **domain-specific regulatory vocabulary**:

| Word | P(FEMA) | P(global) | Excess Contribution |
|------|---------|-----------|-------------------|
| flood | 0.087 | 0.001 | +0.135 |
| suspension | 0.058 | 0.001 | +0.089 |
| insurance | 0.059 | 0.001 | +0.089 |
| community | 0.061 | 0.002 | +0.067 |
| regulatory | 0.033 | 0.008 | +0.057 |
| date | 0.083 | 0.016 | +0.054 |

These are NFIP (National Flood Insurance Program) terms. They happen to have high
delta values because the AI training (Llama generating regulatory text) produces
these kinds of bureaucratic/regulatory words at elevated rates.

### FAA: similar pattern, much larger scale

FAA has 254,524 sentences and P(FP) of 8.9%. Its distinguishing words are:
- "this" (P_FAA=0.229 vs P_global=0.108)
- "faa" (0.135 vs 0.022)
- "airspace" (0.035 vs 0.006)
- "aviation" (0.017 vs 0.003)
- "safety" (0.027 vs 0.009)

The word "this" is the single largest contributor -- FAA uses "this" at 2x the
global rate, and "this" is also 2.6x more common in AI text than human text.

### NOAA and EPA: similar domain-vocabulary effects

- **NOAA**: fishery/fishing/fisheries/catch terminology
- **EPA**: action/rule/air/requirements terminology

## Implications for the Estimation Framework

1. **The overall false positive rate is ~0.28% weighted** -- small enough that broad trends
   are reliable, but individual agency-quarter estimates can be significantly biased.

2. **FEMA estimates are unreliable** -- pre-ChatGPT alpha of 2.2% means any post-ChatGPT
   FEMA estimates below ~3-4% should not be interpreted as AI usage.

3. **The root cause is distributional mismatch**: the bag-of-words model uses a single
   pooled P distribution. Agencies with distinctive vocabularies (FEMA/NFIP terms,
   FAA/aviation terms, NOAA/fisheries terms) have text that is closer to the AI
   distribution Q simply because Q also overproduces formal regulatory vocabulary.

4. **Potential fix**: Agency-specific or at least agency-adjusted human distributions
   would reduce this bias. Alternatively, an agency-level random effect on the
   log-LR baseline could correct for this.

5. **Small strata amplify the problem**: with few sentences, random variation can push
   the MLE alpha estimate to spuriously high values even when the expected log-LR
   is negative.

## Plots

- `06_agency_vs_spurious_alpha.png` -- 3-panel: P(FP) vs alpha, E[logLR] vs alpha, alpha vs size
- `06_agency_time_trends.png` -- Time series for FEMA, FAA, NOAA, EPA, FSIS, DOT
- `06_false_positive_comprehensive.png` -- 4-panel comprehensive figure
