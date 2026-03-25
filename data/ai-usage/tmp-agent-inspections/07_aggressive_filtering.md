# Aggressive Vocabulary Filtering Experiments

Date: 2026-03-21

## Goal

Test whether extreme vocabulary filtering parameters produce clean pre/post
ChatGPT separation (i.e., near-zero pre-ChatGPT alpha with a visible post
increase). All experiments use `--no-matched` (unmatched mode), Llama v4
generations, `--stratify-by agency quarter`, and `--bootstrap-n 100`.

---

## Summary Table

| Exp | Setting | Vocab (rule / proposed_rule) | rule pre | rule post | PR pre | PR post |
|-----|---------|------------------------------|----------|-----------|--------|---------|
| A | min_human=100, min_ai=50 | 5837 / 4622 | 0.0020 | 0.0018 | 0.0066 | 0.0115 |
| B | min_human_frac=0.5%, min_ai_frac=0.5% | 262 / 247 | 0.0191 | 0.0291 | 0.0285 | 0.0280 |
| C | max_vocab=1000 | 1000 / 1000 | 0.0049 | 0.0055 | 0.0084 | 0.0122 |
| D | max_vocab=500 | 500 / 500 | 0.0080 | 0.0112 | 0.0116 | 0.0155 |

All values are sentence-weighted averages of `alpha_estimate`. "Pre" is before
2022-12-01, "Post" is from 2022-12-01 onward.

---

## Experiment A: Absolute Count Filtering (min_human=100, min_ai=50)

- **Vocab:** rule=5837, proposed_rule=4622
- **Kappa (rule):** 433.8; **kappa_q:** 293.2
- **Kappa (proposed_rule):** 357.7; **kappa_q:** 244.4

### Results
```
proposed_rule: pre=0.0066 (std=0.0195) post=0.0115
rule:          pre=0.0020 (std=0.0233) post=0.0018
```

**Interpretation:** This is the strictest absolute-count filter. The rule
pre-alpha is extremely low (0.20%) and essentially indistinguishable from zero.
However, the post-alpha is equally low (0.18%), so there is no detectable
signal of AI usage for rules. For proposed rules, there is a small but visible
increase from 0.66% to 1.15%. The large vocabulary (5837 words) suggests many
noisy terms are included despite the high minimum counts.

---

## Experiment B: Fraction-Based Filtering (0.5% threshold)

- **Vocab:** rule=262, proposed_rule=247
- **Kappa (rule):** 104.1; **kappa_q:** 64.6
- **Kappa (proposed_rule):** 89.0; **kappa_q:** 55.5

### Results
```
proposed_rule: pre=0.0285 (std=0.0581) post=0.0280
rule:          pre=0.0191 (std=0.0490) post=0.0291
```

**Interpretation:** This is the most aggressive filter -- only 262 words for
rules and 247 for proposed rules (words that appear in at least 0.5% of all
sentences in both corpora). The pre-alpha is substantially elevated: 1.91% for
rules, 2.85% for proposed rules. The pre/post difference is negligible for
proposed rules and only +1pp for rules. The very high pre-alpha and standard
deviation (4.9-5.8%) suggest significant spurious signal. This filter is
**too aggressive** -- by keeping only extremely common words, it loses
discriminative power and generates noisy/biased estimates.

---

## Experiment C: Top 1000 Most Discriminative Words

- **Vocab:** rule=1000, proposed_rule=1000
- **Kappa (rule):** 191.9; **kappa_q:** 90.8
- **Kappa (proposed_rule):** 188.7; **kappa_q:** 89.6

### Results
```
proposed_rule: pre=0.0084 (std=0.0189) post=0.0122
rule:          pre=0.0049 (std=0.0200) post=0.0055
```

**Interpretation:** With 1000 words (selected by discriminative power between
human/AI), the pre-alpha is low: 0.49% for rules, 0.84% for proposed rules.
Post-alpha shows a modest increase for proposed rules (+0.38pp) but essentially
no change for rules (+0.06pp). Standard deviations are moderate (~2%). This
represents a reasonable balance: low spurious signal with some post-ChatGPT
signal for proposed rules.

---

## Experiment D: Top 500 Most Discriminative Words

- **Vocab:** rule=500, proposed_rule=500
- **Kappa (rule):** 148.4; **kappa_q:** 65.8
- **Kappa (proposed_rule):** 145.9; **kappa_q:** 62.3

### Results
```
proposed_rule: pre=0.0116 (std=0.0272) post=0.0155
rule:          pre=0.0080 (std=0.0258) post=0.0112
```

**Interpretation:** With 500 words, the pre-alpha rises compared to 1000 words
(0.80% for rules, 1.16% for proposed rules). Post-alpha increases show a small
signal for both doc types (+0.32pp rules, +0.39pp proposed rules). The trend
continues: fewer words means more spurious pre-alpha, not less.

---

## Key Findings

1. **More aggressive filtering does NOT produce cleaner separation.** Contrary
   to expectations, restricting the vocabulary more aggressively generally
   *increases* spurious pre-ChatGPT alpha rather than reducing it.

2. **The relationship between vocab size and pre-alpha is non-monotonic:**
   - 5837 words (Exp A): pre=0.0020 (rule), 0.0066 (PR)
   - 1000 words (Exp C): pre=0.0049 (rule), 0.0084 (PR)
   - 500 words (Exp D):  pre=0.0080 (rule), 0.0116 (PR)
   - 262 words (Exp B):  pre=0.0191 (rule), 0.0285 (PR)

   Smaller vocabularies yield higher spurious alpha. This makes sense: with
   fewer words, the MLE has less information and is more affected by noise in
   the word frequency distributions.

3. **Experiment A (strict absolute counts) produces the lowest pre-alpha** but
   also flattens the post-alpha signal. The most conservative approach gives
   near-zero estimates everywhere, suggesting it over-smooths real signal.

4. **No experiment produces a dramatic pre/post separation.** The largest
   relative increase is in Experiment A proposed rules (0.66% -> 1.15%, a 74%
   relative increase), but in absolute terms this is still small.

5. **Kappa values decrease with vocab size**, indicating the hierarchical
   Bayesian agency model has less information to work with when the vocabulary
   is smaller. This is consistent with the elevated spurious alpha.

6. **Fraction-based filtering (Exp B) is distinctly worse** than all other
   approaches. Keeping only very common words produces high noise and no
   discriminative signal.

## Recommendation

The best balance is achieved with the largest vocabulary that still maintains
reasonable minimum count thresholds (Experiment A). The standard defaults or
Experiment C (max_vocab=1000) represent reasonable alternatives. More aggressive
filtering should be avoided as it degrades rather than improves signal quality.
