# Experiment: Filter Short Words (< 4 characters) from Vocabulary

**Date**: 2026-03-21
**Goal**: Test whether removing short words (< 4 characters) from the distributional AI detection vocabulary reduces spurious alpha (pre-ChatGPT false positives) and improves the signal.

## Setup

- **Estimate**: Unmatched, default thresholds (min_human_count=5, min_ai_count=3), llama-v4-generations corpus
- **Infer**: Stratified by agency + quarter, bootstrap_n=100, pooled distributions
- **Filter**: Removed all words with `len(word) < 4` from the distribution parquet after estimation
- **Control**: Same estimate/infer pipeline but without the word-length filter

## Words Removed by Filter

| Doc Type       | Vocab Before | Vocab After | Words Removed | % Removed |
|----------------|-------------|-------------|---------------|-----------|
| rule           | 20,880      | 18,321      | 2,559         | 12.3%     |
| proposed_rule  | 18,069      | 15,946      | 2,123         | 11.7%     |

Sample removed words: `gpo`, `pl`, `sh`, `cor`, `nw`, `aim`, `und`, `ke`, `cho`, `gl`, `su`, `exp`, `cos`, `sk`, `frl`, `les`, `pat`, `mit`, `rad`, `mer`, etc.

## Results: Pre/Post ChatGPT Alpha (Weighted Average)

### proposed_rule

| Condition       | Pre-ChatGPT Alpha | Post-ChatGPT Alpha | Delta (post-pre) | Ratio |
|-----------------|-------------------|---------------------|-------------------|-------|
| **Filtered (>=4 chars)** | 0.001674 | 0.002484 | **0.000810** | 1.48x |
| Unfiltered (control)     | 0.004236 | 0.008459 | **0.004223** | 2.00x |

### rule

| Condition       | Pre-ChatGPT Alpha | Post-ChatGPT Alpha | Delta (post-pre) | Ratio |
|-----------------|-------------------|---------------------|-------------------|-------|
| **Filtered (>=4 chars)** | 0.001145 | 0.001260 | **0.000115** | 1.10x |
| Unfiltered (control)     | 0.001502 | 0.001870 | **0.000368** | 1.25x |

## Observations

1. **Short-word filtering substantially reduces spurious (pre-ChatGPT) alpha.** The pre-ChatGPT alpha dropped from 0.42% to 0.17% for proposed_rule (60% reduction) and from 0.15% to 0.11% for rule (24% reduction). This suggests short words (abbreviations, codes, two-letter tokens) contribute noise to the distributional signal.

2. **The post-pre delta also decreases, meaning the "signal" shrinks too.** For proposed_rule, the delta dropped from 0.42pp to 0.08pp; for rule, from 0.04pp to 0.01pp. This is concerning: the filter may be removing both noise AND real signal.

3. **The post/pre ratio is weaker with filtering.** Proposed_rule went from 2.00x to 1.48x. Rule went from 1.25x to 1.10x. The relative discrimination between pre and post periods is reduced.

4. **For rule, the filtered result is essentially flat** (1.10x ratio, delta of only 0.01pp), suggesting the word-length filter may over-correct for this doc type.

5. **Pre-ChatGPT alpha is still non-zero even after filtering** (0.17% for proposed_rule, 0.11% for rule), indicating that short words are not the only source of spurious signal.

## Verdict

**The word-length filter did NOT help the overall detection.** While it successfully reduced the spurious pre-ChatGPT alpha (good), it also disproportionately reduced the post-ChatGPT signal (bad). The net effect is a weaker and less discriminative detector. Short words likely carry some real distributional information about AI vs. human writing patterns, and removing them throws the baby out with the bathwater.

Better approaches to explore:
- Word-frequency-based filtering (removing very rare words)
- Log-odds ratio thresholding (keeping only words with strong discriminative power)
- Minimum document-count filtering for the corpus
