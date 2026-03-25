# Experiment 03: Per-Agency P/Q Distributions

**Date:** 2026-03-21
**Server:** sk3
**Distribution:** llama-v4-generations, per-agency P/Q (separate human word distribution for each agency)
**Bootstrap:** 100 resamples, dedup enabled, 4 workers
**Doc types:** rule, proposed_rule
**Min docs per agency:** 50

## Setup

Instead of pooling all agencies into a single human word distribution (P), this experiment
builds a separate P distribution for each agency. The AI distribution (Q) is also built
per-agency from the AI corpus. This controls for agency-specific writing style, which
may otherwise create spurious signal.

Stratification remains (agency, quarter) -- same as baseline -- but now the P/Q distributions
are agency-specific rather than pooled.

## Agency Coverage

| Doc type       | Total agencies | Agencies with >= 50 docs | Agencies skipped |
|----------------|---------------|--------------------------|------------------|
| rule           | 44            | 26                       | 18               |
| proposed_rule  | 44            | 20                       | 24               |

**27 unique agencies** across both doc types (26 for rules, 20 for proposed rules, with
substantial overlap).

## Results: Sentence-Weighted Alpha Pre/Post ChatGPT

| Doc type       | Pre-ChatGPT alpha | Post-ChatGPT alpha | Post/Pre ratio | Pre sents     | Post sents   |
|----------------|-------------------|---------------------|----------------|---------------|--------------|
| proposed_rule  | 0.009727          | 0.016078            | 1.65x          | 1,285,619     | 453,716      |
| rule           | 0.005402          | 0.007665            | 1.42x          | 1,522,386     | 517,602      |

### Comparison with pooled (non-per-agency) baseline (from experiment 10, quarter strat)

| Doc type       | Pooled pre-alpha | Per-agency pre-alpha | Pooled post-alpha | Per-agency post-alpha |
|----------------|------------------|----------------------|-------------------|-----------------------|
| proposed_rule  | 0.002606         | 0.009727             | 0.005553          | 0.016078              |
| rule           | 0.001006         | 0.005402             | 0.001464          | 0.007665              |

Per-agency distributions produce **~4-5x higher alpha** than the pooled baseline, both pre
and post ChatGPT. This is a significant finding -- it means per-agency P/Q distributions
do NOT reduce spurious signal; they increase it substantially.

## Per-Agency Pre-ChatGPT Alpha (Spurious Signal)

### proposed_rule -- Top 5 Highest Spurious Alpha

| Agency | Pre-ChatGPT alpha | Post-ChatGPT alpha | Pre sentences |
|--------|-------------------|---------------------|---------------|
| IRS    | 0.031856          | 0.035466            | 44,158        |
| USPS   | 0.028594          | 0.037908            | 3,770         |
| USCBP  | 0.028155          | 0.014713            | 3,395         |
| FWS    | 0.022334          | 0.019361            | 126,764       |
| AMS    | 0.019988          | 0.015495            | 7,058         |

### proposed_rule -- Top 5 Lowest Spurious Alpha

| Agency | Pre-ChatGPT alpha | Post-ChatGPT alpha | Pre sentences |
|--------|-------------------|---------------------|---------------|
| SEC    | 0.003003          | 0.008137            | 22,270        |
| CMS    | 0.003185          | 0.008231            | 188,000       |
| FAA    | 0.004952          | 0.028615            | 171,747       |
| FDA    | 0.005326          | 0.004294            | 29,171        |
| HHS    | 0.005789          | 0.004081            | 11,604        |

### rule -- Top 5 Highest Spurious Alpha

| Agency | Pre-ChatGPT alpha | Post-ChatGPT alpha | Pre sentences |
|--------|-------------------|---------------------|---------------|
| NHTSA  | 0.026874          | 0.061219            | 33,663        |
| FEMA   | 0.023715          | 0.006177            | 9,471         |
| USCIS  | 0.019924          | 0.041659            | 1,705         |
| USPS   | 0.019843          | 0.011752            | 7,001         |
| DOL    | 0.019558          | 0.058379            | 6,190         |

### rule -- Top 5 Lowest Spurious Alpha

| Agency | Pre-ChatGPT alpha | Post-ChatGPT alpha | Pre sentences |
|--------|-------------------|---------------------|---------------|
| FAA    | 0.001677          | 0.001767            | 259,895       |
| NOAA   | 0.002614          | 0.005743            | 176,872       |
| SEC    | 0.002917          | 0.013867            | 29,679        |
| EPA    | 0.002927          | 0.005680            | 363,241       |
| FDIC   | 0.002944          | 0.003197            | 18,004        |

## Key Observations

### 1. Per-agency P/Q distributions INCREASE spurious signal, not decrease it

This is the central finding. Pre-ChatGPT alpha is ~0.97% for proposed_rules and ~0.54% for
rules -- roughly 4-5x higher than the pooled baseline (~0.26% and ~0.10% respectively).

The likely explanation: with per-agency distributions, the AI corpus is split into smaller
per-agency subsets. A smaller AI corpus means the Q distribution is noisier and less
representative, creating more spurious divergence from P. The pooled approach benefits
from having a much larger, more stable Q distribution.

### 2. The post/pre ratio is lower than the pooled baseline

- **proposed_rule:** per-agency 1.65x vs pooled ~2.1x
- **rule:** per-agency 1.42x vs pooled ~1.5x

The higher spurious floor compresses the ratio. If the true signal is similar in absolute
terms, a higher baseline makes the relative increase smaller.

### 3. Large variation in spurious alpha across agencies

Pre-ChatGPT alpha ranges from 0.3% (SEC, CMS) to 3.2% (IRS) for proposed_rules, and from
0.2% (FAA) to 2.7% (NHTSA) for rules. This 10-15x variation suggests that some agency
writing styles are inherently harder to distinguish from AI-generated text, or (more likely)
that the per-agency AI corpus is too small for reliable estimation in some cases.

### 4. Agencies with lowest spurious alpha tend to be the largest

FAA (260K pre-sents for rules), EPA (363K), NOAA (177K), CMS (188K) all have low spurious
alpha. Their large document volumes produce more stable P and Q distributions. Conversely,
high-spurious agencies (USCIS: 1.7K sents, USPS: 3.8K-7K sents, DOL: 6.2K sents) have
small corpora where noise dominates.

### 5. Some agencies show interesting post-ChatGPT jumps

Even with the elevated baseline, some agencies show notable post-ChatGPT increases:
- **FAA proposed_rules:** 0.50% -> 2.86% (5.8x) -- largest relative jump, and FAA has
  low spurious alpha, making this more credible
- **NHTSA rules:** 2.69% -> 6.12% (2.3x) -- but high baseline makes interpretation harder
- **DOL rules:** 1.96% -> 5.84% (3.0x) -- small sample though
- **APHIS proposed_rules:** 1.45% -> 10.07% (6.9x) -- very small sample (1,110 post sents)
- **SEC rules:** 0.29% -> 1.39% (4.8x)

### 6. Some agencies show DECREASING alpha post-ChatGPT

- **FEMA rules:** 2.37% -> 0.62% (drops to 1/4) -- but tiny post sample (628 sents)
- **USCBP proposed_rules:** 2.82% -> 1.47%
- **FWS proposed_rules:** 2.23% -> 1.94%
- **FCC proposed_rules:** 1.39% -> 0.73%

These decreases likely reflect noise from small samples rather than genuine trends.

## Recommendation

**Per-agency P/Q distributions are not recommended.** The pooled approach is superior because:

1. It produces much lower spurious (pre-ChatGPT) alpha, indicating less noise
2. The per-agency AI corpus subsets are too small for stable Q estimation
3. The post/pre ratio (the key signal) is attenuated by the high baseline
4. Agency-specific writing style differences are better handled by stratification
   (which already groups by agency) than by separate P/Q distributions

The pooled P/Q with quarter or year stratification (experiment 10) remains the preferred
approach for trend analysis.

## Output Files

- Results: `/lfs/skampere3/0/alexspan/regulations-demo/data/ai-usage-generations/exp-per-agency/results.csv.gz` (1,469 rows)
- Metadata: `/lfs/skampere3/0/alexspan/regulations-demo/data/ai-usage-generations/exp-per-agency/metadata.json`
- Analysis script: `analyze_per_agency.py`
