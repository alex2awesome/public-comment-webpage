# Inspection 04: Llama-v4 AI Rewrite Quality Analysis

**Date:** 2026-03-21
**Corpus:** `/Users/spangher/Projects/stanford-research/rfi-research/regulations-demo/data/ai-usage-generations/llama-v4-generations/ai_corpus_rule.parquet`
**Model:** `meta-llama/Llama-3.3-70B-Instruct`
**Corpus size:** 12,343 documents (all `doc_type=rule`)

---

## 1. Long Documents (original > 500 words): Side-by-Side Comparison

**Population:** 9,730 documents have original_text > 500 words. 10 were randomly sampled.

### Vocabulary Similarity

| Document | Agency | Orig Words | AI Words | AI/Orig Ratio | Jaccard |
|----------|--------|-----------|----------|---------------|---------|
| IRS-2015-0031-0019 | IRS | 6,313 | 1,537 | 0.243 | 0.291 |
| FAA-2021-0196-0004 | FAA | 2,356 | 1,233 | 0.523 | 0.394 |
| FAA-2014-0370-0016 | FAA | 6,471 | 506 | 0.078 | 0.162 |
| TREAS-DO-2018-0021-0001 | TREAS | 5,983 | 1,067 | 0.178 | 0.126 |
| FDA-2016-N-1111-1838 | FDA | 1,061 | 1,723 | 1.624 | 0.226 |
| NOAA-NMFS-2020-0141-0003 | NOAA | 875 | 1,836 | 2.098 | 0.036 |
| FDIC-2021-0060-0001 | FDIC | 6,622 | 1,777 | 0.268 | 0.142 |
| EPA-R06-OAR-2019-0211-0007 | EPA | 1,400 | 1,311 | 0.936 | 0.331 |
| EPA-R04-OAR-2016-0421-0003 | EPA | 1,771 | 1,262 | 0.713 | 0.429 |
| FCC-2017-0218-0001 | FCC | 11,854 | 1,309 | 0.110 | 0.181 |

### Sentence Structure Observations

- **AI sentences are longer and fewer.** Original texts have many short sentences (mean 11-22 words, reflecting the Federal Register's line-broken formatting). AI rewrites produce fewer, longer sentences (mean 14-71 words).
- Originals contain boilerplate Federal Register headers (`<<COMMENT 1>>`, `<<COMMENT 2>>`, FR volume/issue citations, GPO URLs). AI rewrites sometimes keep these headers, sometimes strip them, sometimes translate them.
- AI rewrites tend to produce coherent prose paragraphs rather than the structured, section-numbered format of the original.

### Regulatory Jargon Preservation

For the 10 sampled long documents:
- **Mean jargon terms in original:** 20.7
- **Mean jargon terms in AI:** 10.2
- **Mean preservation rate:** ~55%

Commonly **preserved** jargon: `agency`, `federal`, `final`, `proposed`, `provisions`, `comments` -- these are generic administrative terms that appear frequently in any government text.

Commonly **dropped** jargon: `cfr`, `codified`, `notwithstanding`, `herein`, `thereof`, `promulgated`, `rulemaking`, `preamble`, `subpart` -- these are the highly specific regulatory terms that distinguish authentic Federal Register text from generic bureaucratic prose.

This is a significant finding: **the AI preserves the shallow regulatory vocabulary but drops the deep domain-specific terminology**, creating a detectable distributional shift.

### Quality Issues Found in Long Document Samples

- **TREAS-DO-2018-0021-0001**: AI output is garbled gibberish. First 300 chars: `"bicy pas<m 87 agencies specify Certain as expend states perform Banks Acquisition Tradition Foreign planner enriched risks..."` -- complete nonsense.
- **NOAA-NMFS-2020-0141-0003**: AI output contains garbled multilingual characters: `"здійснення령 Entities..."` followed by gibberish tokens.
- **FDA-2016-N-1111-1838**: AI output is 62% longer than the original despite the original being a real regulatory document -- the AI padded it with plausible-sounding but fabricated regulatory content.

---

## 2. Stub Documents (original < 100 words)

**Population:** 1,622 documents (13.1% of corpus) have original_text < 100 words.

These "stub" documents contain primarily Federal Register header metadata:
- FR volume/issue numbers
- Date stamps
- FR Doc numbers
- Page ranges
- Agency and CFR part identifiers

They contain essentially **no substantive regulatory text** in the original.

### What does the AI rewrite look like?

**Length explosion:** Mean AI word count for stubs is 1,147 words -- a **20x expansion** from originals averaging ~58 words. The AI is generating substantial content from almost nothing.

The hallucinated content falls into several categories:

1. **Plausible-but-fabricated regulatory text** (most common): The AI generates realistic-sounding regulations including section numbers, definitions, and procedural requirements. Example: FDIC-2019-0081-0001 produces fake CFR sections (370.1, 370.2, 370.3) with definitions and recordkeeping requirements that sound authentic but are entirely invented.

2. **Verbose rephrasing of metadata** (DOD-2015-HA-0109-0188): The AI rewrites the FR header by spelling out numbers and dates in words: "volume eighty-one, number one hundred seventy-one" instead of "Volume 81, Number 171". This is absurd padding.

3. **Garbled/gibberish output** (FEMA_FRDOC_0001-5237): AI generates fake regulations about "recreational vehicles" and "amphibious vehicles" with HTML-like `<u>` tags, completely unrelated to the original FEMA document.

4. **Recursive self-commentary** (AMS-SC-19-0011-0002): AI text begins with "became" and then produces multiple sequential "revised to" rewrites of the header, each rephrasing it slightly differently, as if the model interpreted the task as an iterative translation exercise.

5. **Overwrought bureaucratese** (SEC-2021-0053-0001): AI produces inflated language like "A new rendition of the directive is forthcoming" and "under the auspices of the federal governing apparatus" -- detectable by its conspicuously formal style.

### Assessment

The stub documents are a **serious data quality problem**. A classifier trained on this corpus would learn that short AI texts are impossible (because even short originals get expanded to 500-1500 words), which is an artifact of the generation process, not a genuine feature of AI-written text.

---

## 3. Overall Corpus Statistics

### Jaccard Similarity Distribution (word sets, all 12,343 docs)

| Statistic | Value |
|-----------|-------|
| Mean | 0.3253 |
| Median | 0.3225 |
| Std Dev | 0.1820 |
| 25th percentile | 0.1819 |
| 75th percentile | 0.4536 |
| % with Jaccard > 0.5 | 17.9% |
| % with Jaccard > 0.3 | 54.5% |
| % with Jaccard < 0.1 | 12.3% |

### Jaccard Histogram

```
[0.00, 0.05):   836 ( 6.8%) ######
[0.05, 0.10):   688 ( 5.6%) #####
[0.10, 0.15):   963 ( 7.8%) #######
[0.15, 0.20):   956 ( 7.7%) #######
[0.20, 0.25):  1074 ( 8.7%) ########
[0.25, 0.30):  1094 ( 8.9%) ########
[0.30, 0.35):  1219 ( 9.9%) #########
[0.35, 0.40):  1236 (10.0%) ##########
[0.40, 0.45):  1133 ( 9.2%) #########
[0.45, 0.50):   925 ( 7.5%) #######
[0.50, 0.60):  1280 (10.4%) ##########
[0.60, 0.70):   683 ( 5.5%) #####
[0.70, 0.80):   208 ( 1.7%) #
[0.80, 0.90):    44 ( 0.4%)
[0.90, 1.00):     4 ( 0.0%)
```

The distribution is roughly normal centered around 0.30-0.35, indicating that most rewrites share about a third of their vocabulary with the original. The tail above 0.7 (near-parroting) is very small (256 docs, 2.1%).

### Word Overlap Percentage (% of original words found in AI)

| Statistic | Value |
|-----------|-------|
| Mean | 43.0% |
| Median | 41.9% |
| Std Dev | 20.0% |

### Length Ratio (AI / Original)

| Statistic | Value |
|-----------|-------|
| Mean | 3.430 |
| Median | 0.529 |
| Original mean word count | 3,027 |
| AI mean word count | 988 |
| AI max word count | 2,017 |
| % where AI is longer | 28.9% |
| % where AI is shorter | 71.0% |
| % where AI is <50% of original | 47.8% |

**Critical observation:** AI text appears to be hard-capped at ~2,000 words (max = 2,017). This means:
- Documents shorter than ~1,500 words get expanded
- Documents longer than ~2,000 words get compressed
- This creates a length convergence artifact that is trivially detectable

### Length Ratio by Original Document Size

| Bucket | N | Mean Ratio | Median Ratio | Mean Jaccard |
|--------|---|-----------|-------------|-------------|
| <100 words | 1,622 | 19.918 | 18.683 | 0.093 |
| 100-500 words | 987 | 4.790 | 2.940 | 0.330 |
| 500-1k words | 1,366 | 1.276 | 1.153 | 0.447 |
| 1k-3k words | 5,319 | 0.556 | 0.498 | 0.417 |
| 3k-10k words | 2,442 | 0.229 | 0.208 | 0.257 |
| 10k+ words | 607 | 0.067 | 0.064 | 0.136 |

The **sweet spot** for realistic rewrites is 500-1000 words original: the AI produces similar-length output (ratio ~1.2) with the highest Jaccard similarity (0.447). Documents outside this range have increasingly artificial length distortions.

### N-gram Overlap Analysis (500 random docs with >200 words)

| Metric | Mean | Median |
|--------|------|--------|
| Word (unigram) Jaccard | 0.362 | 0.347 |
| Bigram Jaccard | 0.197 | 0.172 |
| Trigram Jaccard | 0.141 | 0.111 |
| Exact sentence overlap | 3.5% | 1.3% |

The steep drop from unigram to trigram Jaccard confirms that **Llama is rephrasing, not copying**. The texts share vocabulary but not phrasing.

### Regulatory Jargon Preservation (Full Corpus)

| Statistic | Value |
|-----------|-------|
| Mean jargon terms in original | 15.6 |
| Mean jargon terms in AI | 9.6 |
| Mean preservation rate | 52.3% |
| Median preservation rate | 52.4% |

### Quality Defects in AI Text

| Defect | Count | % of Corpus |
|--------|-------|-------------|
| Empty AI text | 26 | 0.2% |
| AI text < 10 words | 63 | 0.5% |
| HTML-like tags in AI text | 5,602 | 45.4% |
| Non-ASCII garbling (3+ chars) | 1,017 | 8.2% |
| Cyrillic characters | 739 | 6.0% |
| Chinese characters | 649 | 5.3% |
| Korean characters | 268 | 2.2% |
| Greek characters | 265 | 2.1% |
| Meta-commentary ("revised to") | 1,244 | 10.1% |
| "became" translation artifact | 658 | 5.3% |
| "rewritten" self-reference | 663 | 5.4% |
| "here is" self-reference | 230 | 1.9% |

**Nearly half the corpus** (45.4%) contains HTML-like tags, and a combined ~20% has some form of garbling or meta-commentary. These are trivially detectable artifacts.

### Words Frequently Added by AI (Not in Originals)

Top words that appear in AI text but not in the corresponding originals:
`ensure`, `becomes`, `outlined`, `hereby`, `regarding`, `relevant`, `revised`, `specifically`, `furthermore`, `implementation`, `ensuring`, `aims`, `modifications`, `deemed`, `please`

These are characteristic LLM filler words -- they represent the model's stylistic signature rather than regulatory domain vocabulary.

---

## 4. Key Question: Is Llama Simply Parroting Back the Same Regulatory Vocabulary?

**No, but the differences are often artifactual rather than substantive.**

### Evidence Against Pure Parroting

1. **Jaccard similarity is moderate (mean 0.325), not high.** If the model were parroting, we would expect Jaccard > 0.7 consistently. Only 2.1% of documents reach that level.
2. **Bigram/trigram overlap drops sharply** (0.20 / 0.14 vs 0.36 for unigrams), confirming different sentence construction.
3. **Exact sentence overlap is only 3.5%** -- very little verbatim copying.
4. **The model drops ~48% of domain-specific jargon** and replaces it with generic formal language.

### However, the Differences Are Problematic

The rewrites differ from originals in ways that are **artifacts of the generation process**, not genuine stylistic variation:

1. **Length asymmetry:** AI texts converge to ~1,000-2,000 words regardless of original length. This is a trivially detectable signal that a classifier would exploit.
2. **Garbled output:** ~8% of AI texts contain multilingual character garbling (Cyrillic, Korean, Chinese, Greek), making them instantly identifiable.
3. **Meta-commentary leakage:** ~15% of AI texts contain traces of the model "talking about" the rewriting task ("became", "revised to", "here is the rewritten version").
4. **HTML/XML tag injection:** 45% of AI texts contain `<rule>`, `<u>`, or similar tags not present in originals.
5. **Stub hallucination:** For 13% of the corpus (short docs), the AI fabricates plausible but entirely invented regulatory content.

### Implications for AI Detection

A classifier trained on this corpus would likely achieve high accuracy, but for the **wrong reasons**:
- It would learn to detect length convergence, HTML tags, garbled characters, and meta-commentary.
- These signals would **not generalize** to detecting high-quality AI-generated text from better models or better prompting.
- The ~52% jargon preservation rate means the AI rewrites are in a different lexical distribution from authentic regulatory text, but this gap is confounded by the quality defects above.

### Recommendations

1. **Filter out defective AI generations** before training any classifier: remove docs with empty AI text, garbled characters, meta-commentary, and HTML tags.
2. **Length-match the corpus** by truncating or segmenting originals to similar lengths as AI outputs, or by generating multiple AI segments per long original.
3. **Consider the 500-1000 word originals as the highest-quality subset** for evaluation -- these have the most realistic AI rewrites (length ratio ~1.2, highest Jaccard ~0.45).
4. **Be cautious about claims of AI detection accuracy** trained on this corpus -- much of the signal comes from generation artifacts rather than genuine stylistic differences.
