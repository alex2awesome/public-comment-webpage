# Experiment 08: Log-Odds Ratio Vocabulary Filtering

**Date**: 2026-03-21
**Hypothesis**: Filtering the vocabulary to only keep words with strong discriminative signal (high |log_odds_ratio|) should improve pre/post ChatGPT separation by removing noise words that do not distinguish AI from human text.

## Setup

- Ran `estimate` with `--no-matched` for rule and proposed_rule against llama-v4-generations
- Created filtered distribution variants at LOR thresholds: 0.3, 0.5, 1.0, 1.5
- Ran `infer` with `--stratify-by agency quarter --bootstrap-n 100` for each variant plus the unfiltered baseline

## Vocabulary Retention

| Threshold | rule vocab | % retained | proposed_rule vocab | % retained |
|-----------|-----------|------------|---------------------|------------|
| baseline  | 20,880    | 100.0%     | 18,069              | 100.0%     |
| LOR>=0.3  | 17,810    | 85.3%      | 15,397              | 85.2%      |
| LOR>=0.5  | 15,856    | 75.9%      | 13,667              | 75.6%      |
| LOR>=1.0  | 10,934    | 52.4%      | 9,470               | 52.4%      |
| LOR>=1.5  | 6,692     | 32.0%      | 5,914               | 32.7%      |

## Key Results

### Pre/Post ChatGPT Separation (weighted by n_sentences)

| Threshold | Pre alpha | Post alpha | Diff    | Ratio  | Pre CI          | Post CI         |
|-----------|-----------|------------|---------|--------|-----------------|-----------------|
| baseline  | 0.0027    | 0.0048     | +0.0021 | 1.75x  | [0.0015,0.0045] | [0.0032,0.0070] |
| LOR>=0.3  | 0.0027    | 0.0047     | +0.0020 | 1.75x  | [0.0015,0.0045] | [0.0031,0.0070] |
| LOR>=0.5  | 0.0032    | 0.0052     | +0.0020 | 1.64x  | [0.0017,0.0052] | [0.0035,0.0077] |
| LOR>=1.0  | 0.0080    | 0.0116     | +0.0036 | 1.45x  | [0.0052,0.0115] | [0.0080,0.0161] |
| LOR>=1.5  | 0.0303    | 0.0439     | +0.0136 | 1.45x  | [0.0207,0.0418] | [0.0314,0.0589] |

### Signal-to-Noise Ratio

| Threshold | SNR   | Signal  | Noise (std) |
|-----------|-------|---------|-------------|
| baseline  | 0.313 | 0.0021  | 0.0066      |
| LOR>=0.3  | 0.314 | 0.0020  | 0.0065      |
| LOR>=0.5  | 0.286 | 0.0020  | 0.0071      |
| LOR>=1.0  | 0.264 | 0.0036  | 0.0136      |
| LOR>=1.5  | 0.351 | 0.0136  | 0.0389      |

### False Positive Analysis (pre-ChatGPT strata with alpha > 0.05)

| Threshold | Mean   | Std    | P95    | Max    | FP rate   |
|-----------|--------|--------|--------|--------|-----------|
| baseline  | 0.0027 | 0.0066 | 0.0276 | 0.1642 | 38/1306 = 2.9% |
| LOR>=0.3  | 0.0027 | 0.0065 | 0.0276 | 0.1699 | 37/1306 = 2.8% |
| LOR>=0.5  | 0.0032 | 0.0071 | 0.0347 | 0.1925 | 44/1306 = 3.4% |
| LOR>=1.0  | 0.0080 | 0.0136 | 0.0684 | 0.2952 | 102/1306 = 7.8% |
| LOR>=1.5  | 0.0303 | 0.0389 | 0.1947 | 0.4733 | 452/1306 = 34.6% |

### By Document Type

**rule** -- very little signal at any threshold:

| Threshold | Pre    | Post   | Diff    | Ratio |
|-----------|--------|--------|---------|-------|
| baseline  | 0.0015 | 0.0018 | +0.0003 | 1.23x |
| LOR>=0.3  | 0.0015 | 0.0019 | +0.0003 | 1.21x |
| LOR>=0.5  | 0.0022 | 0.0024 | +0.0002 | 1.08x |
| LOR>=1.0  | 0.0076 | 0.0084 | +0.0009 | 1.11x |
| LOR>=1.5  | 0.0411 | 0.0561 | +0.0150 | 1.37x |

**proposed_rule** -- stronger signal, best ratio at low thresholds:

| Threshold | Pre    | Post   | Diff    | Ratio |
|-----------|--------|--------|---------|-------|
| baseline  | 0.0042 | 0.0082 | +0.0040 | 1.96x |
| LOR>=0.3  | 0.0041 | 0.0081 | +0.0040 | 1.98x |
| LOR>=0.5  | 0.0043 | 0.0084 | +0.0042 | 1.98x |
| LOR>=1.0  | 0.0085 | 0.0151 | +0.0066 | 1.78x |
| LOR>=1.5  | 0.0204 | 0.0323 | +0.0119 | 1.59x |

## Observations

1. **LOR filtering does NOT improve pre/post separation.** The baseline (no filtering) already achieves the best or tied-best pre/post ratio (1.75x overall). LOR>=0.3 is essentially identical to baseline; higher thresholds degrade results.

2. **Higher LOR thresholds catastrophically inflate false positives.** At LOR>=1.5, pre-ChatGPT alpha rises to 0.030 (from 0.003 baseline) and 34.6% of pre-ChatGPT strata exceed alpha=0.05. This means the model falsely claims ~3% AI usage in text that predates ChatGPT. The reason: when you keep only extreme-LOR words, you throw away the large vocabulary of neutral words that anchor the estimate, causing the model to overfit to a small, noisy subset.

3. **SNR is roughly flat or worse.** Despite larger absolute differences at high thresholds, the noise grows faster than the signal. LOR>=1.5 has the highest SNR (0.351) but this is only marginally above baseline (0.313), and comes at the cost of massive false positives. The slight SNR improvement is a mirage: the "signal" is mostly bias from the distorted vocabulary, not genuine detection of AI text.

4. **proposed_rule shows a genuine ~2x pre/post ratio regardless of threshold.** The best ratio (1.98x) appears at LOR>=0.3 and LOR>=0.5, but these are within noise of the baseline (1.96x). For rules, the ratio is always near 1.1-1.4x, suggesting minimal AI adoption in final rules.

5. **CI widths widen with aggressive filtering.** At LOR>=1.5, CI widths are 7x the baseline (0.021 vs 0.003 pre-ChatGPT). This is the expected consequence of reducing vocabulary: fewer words means less information per sentence, wider uncertainty.

6. **Annual trends show a secular upward drift even pre-ChatGPT at high LOR thresholds.** At LOR>=1.5, alpha rises from 0.022 (2016) to 0.038 (2020) before ChatGPT existed. This is a red flag: the vocabulary is picking up on writing style drift over time, not AI usage.

## Conclusion

**LOR filtering is counterproductive.** The full vocabulary baseline already achieves the cleanest pre/post separation with the lowest false positive rate. Filtering to high-LOR words removes stabilizing vocabulary, inflates alpha estimates, increases noise, and introduces temporal confounds. The approach should not be pursued further.

The best configuration remains the unfiltered baseline vocabulary (20,880 words for rule, 18,069 for proposed_rule).
