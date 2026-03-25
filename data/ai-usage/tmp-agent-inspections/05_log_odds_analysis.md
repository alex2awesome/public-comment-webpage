# Log-Odds Ratio Distribution Analysis

**Date:** 2026-03-21
**Source:** `data/ai-usage-generations/ai_usage_distributions/distribution_{rule,proposed_rule,notice,public_submission}.parquet`
**Comparison baseline:** Liang et al. PR Newswire distribution (built from `notebooks/LLM-widespread-adoption-impact/data/training_data/prnewswire/`)

---

## 1. Summary Table: Distribution Width Comparison

| Dataset | Vocab | Mean LOR | Median LOR | Std | IQR | P01 | P99 | Range | Skew | %\|LOR\|>1 | %\|LOR\|>2 | %LOR<0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Rule | 18,140 | -0.515 | -0.502 | 1.082 | 1.351 | -3.19 | +1.94 | 14.64 | -0.36 | 37.7% | 8.9% | 68.8% |
| Proposed Rule | 17,857 | -0.924 | -0.912 | 1.241 | 1.663 | -3.70 | +1.94 | 15.53 | -0.14 | 53.0% | 19.9% | 77.1% |
| Notice | 17,508 | -1.030 | -0.926 | 1.086 | 1.228 | -4.03 | +1.16 | 15.07 | -0.94 | 48.3% | 15.7% | 85.5% |
| Public Submission | 17,480 | -0.729 | -0.632 | 1.440 | 1.849 | -4.60 | +2.30 | 14.06 | -0.42 | 50.1% | 19.4% | 68.1% |
| **PR Newswire (Liang)** | **15,383** | **-0.036** | **+0.009** | **0.764** | **0.724** | **-2.51** | **+1.96** | **10.35** | **-0.56** | **13.6%** | **3.0%** | **49.6%** |

---

## 2. Key Finding: Regulatory Distributions Are Much Wider and More Asymmetric Than PR Newswire

### Width
- **Regulatory std is 1.4-1.9x larger** than PR Newswire (0.764). Rule std = 1.082, Proposed Rule = 1.241, Public Submission = 1.440.
- **IQR is 1.7-2.6x larger**: regulatory IQR ranges from 1.228 (Notice) to 1.849 (Public Submission) vs. 0.724 for PR Newswire.
- **|LOR|>1 fraction**: 37-53% of vocab in regulatory text vs. only 13.6% in PR Newswire. That means 3-4x as many words have strongly discriminative LOR values in the regulatory distributions.
- **|LOR|>2 fraction**: 9-20% in regulatory vs. only 3.0% in PR Newswire.

### Asymmetry (Negative Skew)
- All regulatory distributions are **strongly shifted negative** (AI-favoring). Median LOR ranges from -0.50 (Rule) to -0.93 (Notice). PR Newswire median is nearly zero (+0.009).
- **68-86% of words lean AI-distinctive** (LOR < 0) across regulatory doc types, compared to a balanced 49.6% in PR Newswire.
- This systematic negative shift means the AI reference corpus uses a broader, more distinctive vocabulary than the human regulatory corpus, inflating the apparent separability.

---

## 3. Shape Analysis: Histograms

### Rule Distribution
- Peak at [-1.0, -0.5) with 3,573 words. Approximately bell-shaped but with a long negative tail extending to -10.0.
- The distribution is unimodal, centered around -0.5, with a gentle negative skew (-0.36).

### Proposed Rule Distribution
- Peak at [-1.0, -0.5) with 2,787 words. Similar bell shape but shifted further left (median -0.91).
- More symmetric than Rule (skew -0.14) but wider (std 1.24).

### PR Newswire Distribution
- Sharp peak at [0.0, 0.5) with 4,942 words. Much more concentrated. Centered very close to 0. The classic "most words are uninformative" pattern expected from well-matched P/Q distributions.

**Interpretation:** In PR Newswire, most words have similar frequency in human vs. AI text (LOR near 0), with only tail words being discriminative. In regulatory text, the _entire distribution is shifted_, meaning the generative model systematically over-represents many common words relative to human regulatory writing.

---

## 4. Top 30 Human-Distinctive Words (Highest LOR)

### Rule
| Word | LOR | Human Count | AI Count |
|---|---|---|---|
| gpo | +4.63 | 15,732 | 9 |
| chinook | +4.05 | 2,946 | 3 |
| ami | +3.98 | 2,744 | 3 |
| pct | +3.62 | 1,914 | 3 |
| whale | +3.59 | 1,867 | 3 |
| hd | +3.48 | 3,885 | 7 |
| gillnet | +3.44 | 1,612 | 3 |
| cir | +3.12 | 2,720 | 7 |
| acls | +3.03 | 1,417 | 4 |
| bsai | +2.96 | 4,974 | 15 |
| (+ 20 more...) | | | |

**Pattern:** Almost entirely **domain-specific abbreviations and jargon**: GPO (Government Publishing Office), BSAI (Bering Sea/Aleutian Islands fishery management), MIPS (Medicare payment codes), DRGs (diagnosis-related groups), HCPCS (healthcare procedure codes), chinook/whale/gillnet (NOAA fisheries), loael (toxicology), etc.

These are terms that the LLM rewriter simply does not reproduce because they are highly specialized regulatory vocabulary. The model generates paraphrases that use more generic language.

### Proposed Rule
Same pattern: `mips` (+5.02), `clinicians` (+4.06), `nqf` (+3.92), `rvu` (+3.85), `ltch` (+3.79) -- overwhelmingly CMS/Medicare terminology plus some names from Federal Register metadata (khammond, lotter, daltland, sradovich -- these are FR contact persons).

---

## 5. Top 30 AI-Distinctive Words (Lowest LOR)

### Rule
| Word | LOR | Human Count | AI Count |
|---|---|---|---|
| rewritten | -10.01 | 26 | 30,896 |
| hesitate | -8.07 | 14 | 2,579 |
| happy | -7.86 | 11 | 1,650 |
| rewriting | -7.83 | 9 | 1,309 |
| let | -7.78 | 180 | 23,515 |
| phrasing | -7.57 | 36 | 4,020 |
| rewrite | -6.84 | 25 | 1,356 |
| condensed | -6.60 | 29 | 1,235 |
| concise | -6.54 | 138 | 5,454 |
| writer | -6.52 | 10 | 394 |
| (+ 20 more...) | | | |

### Categorization of Top-50 AI-Distinctive Words (Rule)

**Rewrite/editing meta-language (14 words):** rewritten, rewriting, phrasing, rewrite, condensed, concise, readability, reorganized, brevity, readable, condense, coherence, rewording, aiming

**Conversational/informal (14 words):** hesitate, happy, let, writer, everything, hope, ve, faithful, tried, know, anything, don, helps, aims

**Style/formatting (4 words):** tone, indentation, mentioning, expresses

**Other (18 words):** cotp, critiques, drawbridge, jargon, paraphrased, else, formatting, authoritative, disclaims, nuances, original, phrases, spectators, neh, didn, reorganizing, reformatted, tsob

### Critical Observation: Prompt Leakage

The AI-distinctive words are **not AI writing style markers** (like "moreover", "furthermore", "delve" etc. that one sees in the PR Newswire AI distribution). Instead, they are **prompt leakage** -- the LLM's rewrite meta-commentary bleeding into the output text.

Words like `rewritten`, `rewriting`, `phrasing`, `concise`, `readability`, `hesitate`, `happy`, `let` (as in "let me know"), `hope` (as in "I hope this helps") are the LLM talking about the rewriting task rather than producing rewritten regulatory text.

This is confirmed by the AI counts: `rewritten` appears in 30,896 AI sentences but only 26 human sentences. The model is saying things like "I've rewritten this to be more concise" rather than just producing the rewrite.

### Contrast with PR Newswire AI-Distinctive Words

PR Newswire AI words are genuine style markers: `realm`, `underscoring`, `solidifying`, `esteemed`, `emphasizing`, `underscores`, `endeavors`, `clientele`, `multifaceted`, `captivating`, `unwavering`, `fostering`. These are the "purple prose" markers characteristic of LLM-generated text.

**Almost no overlap:** Only 1 word (`nuances`) appears in both the top-100 AI-distinctive lists for Rule and PR Newswire.

---

## 6. Cross-Distribution Overlap

| Comparison | Top-100 AI-Distinctive Overlap |
|---|---|
| Rule vs. PR Newswire | 1 word |
| Proposed Rule vs. PR Newswire | 1 word |
| Rule vs. Proposed Rule | 56 words |

The Rule and Proposed Rule distributions share 56 of their top-100 AI-distinctive words, indicating the same prompt-leakage pattern in both. But neither overlaps meaningfully with PR Newswire, confirming these are distribution-specific artifacts rather than universal AI writing markers.

---

## 7. Impact of Removing Prompt-Leakage Words

We tested removing 26 obvious prompt-leakage words (rewritten, rewriting, rewrite, phrasing, condensed, concise, readability, readable, reorganized, condense, rewording, coherence, brevity, rephrase, rephrasing, reworded, restructured, simplifying, hesitate, happy, let, hope, faithful, tone, formality, conversational):

| Metric | Rule (full) | Rule (no leak) | Proposed (full) | Proposed (no leak) |
|---|---|---|---|---|
| Mean LOR | -0.515 | -0.508 | -0.924 | -0.917 |
| Std LOR | 1.082 | 1.063 | 1.241 | 1.224 |
| Median LOR | -0.502 | -0.501 | -0.912 | -0.910 |

Removing these 23-25 extreme prompt-leakage words barely affects the bulk statistics (mean shifts by ~0.007) because they are a tiny fraction of the 18K vocabulary. However, they dominate the tails: the mean LOR of these leak words is **-6.03** (Rule) and **-6.14** (Proposed Rule), so they are 5-6 standard deviations from the mean.

---

## 8. Implications

1. **The regulatory P/Q distributions are contaminated by prompt leakage.** The most discriminative AI words are meta-commentary about the rewriting task, not features of AI writing style. This means the distribution is partially detecting "text that talks about rewriting" rather than "text that was AI-rewritten."

2. **The distributions are systematically wider and negatively-shifted** compared to PR Newswire. The IQR is 1.7-2.6x larger, and 68-86% of vocabulary words lean AI-distinctive. This suggests the LLM rewrites of regulatory text diverge more from the originals than LLM rewrites of PR Newswire -- possibly because regulatory language is more specialized and the LLM defaults to generic paraphrasing.

3. **Human-distinctive words are almost entirely domain jargon and abbreviations** that the LLM simply doesn't reproduce (GPO, BSAI, DRGs, MIPS, etc.). This creates a separate detection signal: documents with lots of regulatory abbreviations will score as more "human" regardless of actual authorship.

4. **The lack of overlap with PR Newswire AI markers** (only 1 word in common) means these distributions are measuring something quite different from what the Liang method was designed to detect. The PR Newswire distribution captures genuine stylistic differences; the regulatory distribution captures vocabulary mismatch and prompt contamination.

5. **Recommendation:** The generation pipeline should be audited for prompt leakage. Post-processing should strip LLM meta-commentary (sentences containing "rewritten", "I've", "here is", etc.) before building the Q distribution. The v4 generation pipeline with llama may have addressed this, but the distributions analyzed here still contain the contamination.
