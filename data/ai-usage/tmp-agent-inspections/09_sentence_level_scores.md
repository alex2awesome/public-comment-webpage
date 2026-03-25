# 09: Sentence-Level P/Q Score Inspection

## Goal
Understand why the distributional method (Liang et al. 2025) does not discriminate
well between human-written and AI-rewritten regulatory text at the sentence level.
Compare against the Liang PR Newswire benchmark to quantify the gap.

---

## 1. Setup

**Rules corpus**: 10,261 rule documents from `ai_corpus_rule.parquet` (>= 250 words).
Built a matched P/Q distribution from the `original_text` (human, P) and `ai_text`
(AI-rewritten, Q).  Vocabulary size after filtering (min_human_count=5,
min_ai_count=3): **18,625 words**.

**PR Newswire corpus**: 4,236 paired human/AI documents from the Liang et al. training
data.  Built matched P/Q distribution with the same thresholds: **15,137 words**.

---

## 2. Sentence-Level Separation: Rules vs PR Newswire

| Metric | Rules | PR Newswire |
|--------|-------|-------------|
| Sentence-level AUC | **0.877** | **0.961** |
| Cohen's d (raw log-ratio) | **0.96** | **2.24** |
| Jensen-Shannon divergence | **0.275** | **0.444** |
| Human mean log-ratio | -4.44 | -6.43 |
| Human std | 2.83 | 5.83 |
| AI mean log-ratio | 7.73 | 6.89 |
| AI std | 17.66 | 6.06 |
| Frac human > AI median | 0.3% | 0.3% |
| Frac AI < human median | 10.0% | 1.1% |

**Key finding**: The rules AUC (0.877) is dramatically worse than PR Newswire (0.961).
Cohen's d is 0.96 vs 2.24 -- the PR Newswire signal is **2.3x stronger** in
standardized-mean terms. The JS divergence is 61% larger for PR Newswire. This
confirms the distributional method has substantially less discriminating power on
regulatory text.

---

## 3. Root Cause 1: Massive Sentence-Length Asymmetry

This is the single biggest problem.

| | Rules Human | Rules AI | PR Newswire Human | PR Newswire AI |
|--|-------------|----------|-------------------|----------------|
| Mean sentence length | **12.0** words | **23.0** words | ~22 words | ~23 words |
| Avg unique vocab words/sentence | **11.2** | **20.2** | ~22 | ~23 |

Rules human sentences average **12 words** while AI sentences average **23 words** --
nearly **2x longer**. PR Newswire sentences are roughly equal length (~22-23 words).

This matters critically because the Bernoulli model computes:

```
log P(s) = baseline_p + sum_{w present} delta_p[w]
```

Each additional word present in a sentence adds one more `delta` term. When AI
sentences have ~20 unique vocab words vs human's ~11, the AI sentences accumulate
nearly twice as many delta terms. Since 82.3% of vocabulary words have LOR < 0
(meaning AI uses them more), the extra words systematically push AI sentences toward
higher logQ - logP, while human sentences don't accumulate enough signal.

The correlation between sentence length and log-ratio confirms this:
- **Human**: corr(length, log-ratio) = -0.04 (essentially no correlation)
- **AI**: corr(length, log-ratio) = **0.67** (strong positive correlation)

---

## 4. Root Cause 2: Vocabulary LOR Distribution is Massively Skewed

| Metric | Rules | PR Newswire |
|--------|-------|-------------|
| Mean LOR | **-0.90** | **+0.08** |
| LOR std | 1.02 | 0.76 |
| % vocab LOR < 0 (AI-heavy) | **82.3%** | ~50% |
| Words with LOR > 2 (human-heavy) | **74** | ~220 |
| Words with LOR < -2 (AI-heavy) | **2,534** | ~220 |

The rules LOR distribution is shifted dramatically negative: the mean is -0.90 and
82.3% of vocabulary words favor AI. This is a **34:1 asymmetry** in strongly
discriminating words (2,534 AI-heavy vs 74 human-heavy at |LOR| > 2).

PR Newswire has roughly symmetric LOR distribution (mean +0.08, balanced counts).

**Why?** The AI rewriter (Llama) uses a much broader vocabulary than the original
rule text. Function words tell the story clearly:

| Word | P(w) human | Q(w) AI | Q/P ratio | LOR |
|------|-----------|---------|-----------|-----|
| the | 0.660 | 0.788 | 1.19 | -0.65 |
| and | 0.298 | 0.484 | 1.62 | -0.79 |
| for | 0.171 | 0.282 | 1.65 | -0.65 |
| is | 0.136 | 0.255 | 1.88 | -0.78 |
| this | 0.111 | 0.282 | 2.55 | -1.15 |
| with | 0.069 | 0.125 | 1.80 | -0.65 |

Even basic function words like "the", "is", "this" appear in a higher fraction of AI
sentences, because AI sentences are longer and thus contain more function words.
"this" appears in 28.2% of AI sentences vs 11.1% of human sentences (2.55x ratio).

---

## 5. Root Cause 3: Garbage/Degenerate AI Text

About **1.0%** of AI sentences have extremely high scores (>50), with some reaching
scores of 200-549. These are degenerate outputs from the Llama rewriter -- long
strings of semi-random words:

> Score=549: "model aircraft accessible trom finite owners inspir bac slight
> breaking string mph too builders job curious accurate sponsor saving fin..."
> (811 words, 0.970 unique fraction)

> Score=366: "auxiliary dish queues dep grains sprawling comb surged distribute
> num overview consist grabbing wars losses assaults blamed adequate abandon bab..."

These are clearly **model failures** (degenerate sampling or hallucinated text), not
genuine rewrites. They inflate the AI mean and variance enormously:

| Threshold | % kept | AI mean | AI std |
|-----------|--------|---------|--------|
| All | 100% | 7.73 | 17.66 |
| <= 100 | 99.6% | 7.01 | 11.28 |
| <= 50 | 99.0% | 6.64 | 10.21 |
| <= 20 | 87.9% | 4.04 | 7.22 |
| <= 10 | 67.6% | 0.97 | 4.94 |

Even at threshold <= 10 (removing 32.4% of AI sentences!), the AI mean is only 0.97.
This means a huge fraction of "legitimate" AI sentences actually score near or below
zero.

---

## 6. Root Cause 4: The 74 Human-Heavy Words Are Domain Jargon

The only words that strongly favor human text (LOR > 2) are extremely narrow
domain-specific acronyms concentrated in healthcare/CMS rulemaking:

| Word | Human freq | AI freq | Ratio |
|------|-----------|---------|-------|
| mips | 0.6% | 0.003% | 187x |
| clinician | 0.2% | 0.003% | 72x |
| ffs | 0.07% | 0.001% | 67x |
| pccm | 0.07% | 0.001% | 67x |
| mcos | 0.07% | 0.001% | 64x |
| ami | 0.2% | 0.004% | 52x |
| cjr | 0.2% | 0.004% | 45x |
| ehr | 0.08% | 0.002% | 39x |

These are acronyms like MIPS (Merit-based Incentive Payment System), CJR
(Comprehensive Joint Replacement), APM (Alternative Payment Model), etc. They are
human-heavy because the Llama rewriter expands or rephrases them into full words.

The problem: these 74 words only help classify sentences from CMS/healthcare rules.
For sentences from other agencies (EPA, FCC, SEC, etc.), there are essentially **no
human-heavy discriminating words**.

---

## 7. Root Cause 5: Suspicious AI-Heavy Words

Some of the most AI-heavy words suggest incomplete cleaning of AI artifacts:

| Word | Human freq | AI freq | Ratio | Concern |
|------|-----------|---------|-------|---------|
| rewritten | 0.0005% | 0.13% | 270x | Model self-reference (cleaning failure) |
| erulemaking | 0.001% | 0.22% | 172x | Likely model artifact |
| boulevard | 0.0009% | 0.24% | 273x | Model hallucinating addresses |
| relations | 0.001% | 0.26% | 200x | Model adding boilerplate |
| necessitated | 0.001% | 0.20% | 200x | Classic "AI style" word |
| rectify | 0.001% | 0.25% | 182x | Classic "AI style" word |
| aims | 0.004% | 0.54% | 129x | Classic "AI style" word |

Some are genuine stylistic markers (necessitated, rectify, aims -- words LLMs
over-use). Others are artifacts of imperfect cleaning (rewritten, erulemaking).

---

## 8. Document-Level Performance

Despite weak sentence-level discrimination, document-level alpha estimation still
separates reasonably well:

| | Human docs | AI docs |
|--|-----------|---------|
| Mean alpha | 0.051 | 0.647 |
| Median alpha | 0.000 | 0.920 |
| [5th, 95th pct] | [0.000, 0.281] | [0.000, 1.000] |
| Document-level AUC | **0.864** | |

The MLE aggregation over many sentences rescues some discrimination. But 0.864 is
still marginal for a method being used to estimate population-level AI usage rates.
Some AI documents get alpha = 0 (false negatives), and some human documents get
alpha > 0.25 (false positives).

---

## 9. Sentence-Length Interaction by Bucket

| Length bucket | Human mean LR | AI mean LR | Difference |
|--------------|---------------|------------|------------|
| 11-15 words | -4.46 | -0.88 | 3.58 |
| 16-20 words | -3.54 | 6.42 | 9.96 |
| 21-30 words | -1.49 | 11.82 | 13.31 |

Discrimination improves dramatically for longer sentences. For the shortest sentences
(11-15 words), the difference is only 3.58, while for 21-30 word sentences it is
13.31. The problem is that most human sentences (the vast majority of the 1M human
sentences) are in the 11-15 word bucket, where discrimination is weakest.

---

## 10. Summary of Diagnosis

The distributional method struggles on regulatory text for five reinforcing reasons:

1. **Sentence length asymmetry (primary)**: Human rule sentences average 12 words vs
   AI's 23 words. The Bernoulli model accumulates per-word deltas, so longer sentences
   get systematically more extreme scores. Since 82% of words favor AI, longer AI
   sentences push positive while short human sentences barely move. PR Newswire does
   not have this problem (~22 vs ~23 words).

2. **Vocabulary LOR is massively skewed**: 82.3% of vocab words have LOR < 0, with a
   34:1 asymmetry in strongly discriminating words. The method relies on words that
   differentially appear in human vs AI text, but almost all such words favor AI. Only
   74 words (mostly CMS acronyms) favor human.

3. **Degenerate AI text inflates scores**: ~1% of AI sentences are garbage (degenerate
   model outputs) with scores 50-549, massively inflating the AI mean and variance.

4. **Human-heavy signal is agency-specific**: The few human-heavy words are healthcare
   acronyms that only help for CMS rules, providing no discrimination for other agencies.

5. **Imperfect AI corpus cleaning**: Words like "rewritten" and "erulemaking" appear
   270x more in AI text, suggesting incomplete removal of model self-references and
   artifacts.

**Recommendation**: Before using the distributional method on regulatory text, the
sentence-length confound must be addressed. Possible approaches:
- Normalize log-ratios by number of vocab words in each sentence
- Re-tokenize to match sentence lengths (e.g., use fixed-length windows instead of
  natural sentences)
- Use only words with |LOR| > threshold to reduce the noise from near-zero LOR words
- Use a length-stratified scoring approach
- Clean the AI corpus more aggressively to remove degenerate outputs
