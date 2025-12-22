# Auto-generated metric file for Evidentiary_Support_and_Citations_Rubric
import dspy
import os
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import GeneratedRefFreeLLMJudgeMetric
from typing import ClassVar

DEFAULT_MODEL = dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv("OPENAI_API_KEY"))

class Evidentiary_Support_and_Citations_Rubric_LLMJudge(GeneratedRefFreeLLMJudgeMetric):
    """---
# Metric Card for Evidentiary_Support_and_Citations_Rubric

**Evidentiary Support and Citations** Use of data, analyses, studies, and references that substantiate claims and quantify effects.

## Metric Details

**Evidentiary_Support_and_Citations_Rubric** is a **reference-free** LLM-as-a-Judge metric that prompts an LLM to rate a system output along a single, run-time-specified evaluation axis.
In this case the axis is `**Evidentiary Support and Citations** Use of data, analyses, studies, and references that substantiate claims and quantify effects.`.

The prompt supplies:

1. **Task description** *d*
2. **Rubric** `**Evidentiary Support and Citations** Use of data, analyses, studies, and references that substantiate claims and quantify effects.`
3. **Input text** *x*
4. **Output text** *y*

Greedy decoding (temperature = 0) yields an integer score $\hat{s}\!\in\!\{1,2,3,4,5\}$; higher = better adherence to the axis.

- **Metric Type:** LLM as a Judge
- **Range:** 1-5 (1 = worst, 5 = best)
- **Higher is Better?:** Yes
- **Reference-Based?:** No
- **Input-Required?:** Yes

### Formal Definition

Let $f _{\\theta}$ be the LLM and
$\pi _{\text{RF}}(d,\{axis\},x,y)$ construct the textual prompt.

$$
\hat{s} \;=\; \operatorname*{arg\,max}\limits_{s \in \{1,\dots,5\}} f _{\theta}\!\bigl(s \,\bigl|\, \pi _{\text{RF}}(d,\{axis\},x,y)\bigr)
$$

The metric value is $\operatorname{LJ}^{\text{RF}}_{\{axis\}}(d,x,y)=\hat{s}$.

### Rubric Details

**Criteria:** **Evidentiary Support and Citations** Use of data, analyses, studies, and references that substantiate claims and quantify effects.

#### Scoring Rubric

| Score | Description |
|-------|-------------|
| 1 | Score 1: No evidentiary support<br/>• Purely opinion/anecdote; no citations or links.<br/>• No quantitative information supporting claims.<br/>• References are irrelevant, unverifiable, or obviously inaccurate.<br/>• “See attached” or name-dropping without incorporating any evidence into the argument.<br/>• Fabricated-sounding or conflicting facts with no sources. |
| 2 | Score 2: Minimal, vague, or weak support<br/>• One or two vague references (e.g., “studies show,” “EPA reports”) without titles, dates, docket IDs, or URLs.<br/>• Mentions numbers but provides no source or traceability.<br/>• Sources are low-credibility (blogs, opinion pieces) or links are broken.<br/>• Evidence is not integrated into reasoning; limited or incorrect interpretation of cited material. |
| 3 | Score 3: Adequate baseline support<br/>• At least one specific, credible citation (e.g., docket ID, Federal Register cite, named report/study with year).<br/>• Includes some quantitative data tied to a source.<br/>• Evidence generally supports key claims, though coverage may be partial or selective.<br/>• Minor gaps in traceability (missing page numbers, incomplete links) or reliance on secondary summaries.<br/>• Limited discussion of methods/limitations; little triangulation. |
| 4 | Score 4: Strong, well-integrated support<br/>• Multiple credible, traceable sources (e.g., peer-reviewed studies, government reports, statutes/caselaw, docket materials) with precise identifiers (titles, dates, page/section numbers, URLs).<br/>• Quantifies effects clearly (units, magnitudes, time frames) and ties numbers to sources.<br/>• Evidence is accurately interpreted and woven into arguments that inform agency action.<br/>• Notes key assumptions/limitations; some triangulation across independent sources.<br/>• Attachments/exhibits referenced in-text; links work; minimal inconsistencies. |
| 5 | Score 5: Exceptional, decision-grade evidence use<br/>• Comprehensive, diverse evidence base (primary studies, official data, legal/technical references) with precise, reproducible citations (page/figure numbers, dataset names, docket IDs).<br/>• Provides original analysis or calculations (methods shown, equations/assumptions stated), data tables/appendices, or replicable methodology.<br/>• Quantifies impacts rigorously (comparative baselines, confidence/uncertainty where applicable) and addresses counter-evidence.<br/>• Clearly links evidence to concrete recommendations/options, showing decision relevance.<br/>• All references are authoritative and accessible; no traceability gaps; consistent and accurate interpretation throughout. |

### Inputs and Outputs
- **Inputs:**
  - **Task description** *d*
  - **Rubric** `**Evidentiary Support and Citations** Use of data, analyses, studies, and references that substantiate claims and quantify effects.`
  - **Input text** *x*
  - **Output text** *y*
- **Outputs:**
  - Scalar score $\hat{s} \in \{1,2,3,4,5\}$

## Intended Use

- **Domain:** Policy feedback triage / Evidence quality evaluation
- **Tasks:** 
  - Rank candidate drafts by strength of evidentiary support for escalation decisions
  - Detect vague, missing, or fabricated citations and flag traceability issues
  - Assess whether quantitative claims are sourced and whether magnitudes/timeframes are provided
  - Extract cited sources (titles, years, docket IDs, URLs) and summarize their relevance
  - Classify submissions into rubric categories (1–5) and provide brief rationale
  - Suggest concrete citation improvements or additional evidence needed for escalation
- **Best Suited For:** 
  - Large volumes of text-based policy feedback where fast, consistent triage is needed (e.g., initial pass to prioritize staff review).
  - Drafts that include explicit citation text (titles, authors, docket IDs, URLs) so the model can evaluate specificity and traceability.
  - Scenarios where the goal is to apply a fixed rubric across many items rather than perform final legal/technical adjudication.
  - Environments where the model’s output will be combined with human review or automated link-checking for verification.
  - When submissions make quantitative claims that can be judged for presence/absence of sources and unit/magnitude clarity.
  - Situations that benefit from consistent, descriptive feedback to contributors on how to improve evidence and citations.
- **Not Recommended For:** 
  - Needs real-time verification of external links, paywalled or subscription-only documents, or live data lookups that the model cannot perform.
  - Final agency decision-making or legal admissibility determinations that require authoritative primary-source validation.
  - Highly technical scientific or engineering claims that require reanalysis of primary datasets or replication of methods.
  - When submissions rely on unpublished/confidential documents or proprietary databases the model cannot access or verify.
  - Languages, formats, or domain-specific citation conventions the model has not been trained on or that are highly specialized.
  - Situations where hallucination risk is unacceptable and every citation must be independently confirmed without human oversight.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**
  - [AutoMetrics LLM as a Judge (reference-free)](https://github.com/XenonMolecule/autometrics/blob/main/autometrics/metrics/generated/GeneratedLLMJudgeMetric.py)

### Computational Complexity

- **Efficiency:**
  - Requires a single LLM call per input-output pair.
  - AutoMetrics does parallel calls on batched inputs.

- **Scalability:**
  - Performance is linear in the number of input-output pairs.
  - Performance depends on the underlying LLM model and the dataset size.  Additional consideration would include whether or not the LLM is a reasoning model.

## Known Limitations

- **Biases:** 
  - Authority bias: favoring government/academic sources and downgrading community, industry, or non‑traditional evidence regardless of relevance.
  - Recency bias: giving more weight to newer sources even when older foundational studies or legal citations are more relevant.
  - Format bias: preferring fully formatted citations and URLs over informal but accurate references or attachments.
  - Language and geographic bias: privileging English‑language or U.S. federal sources and undervaluing non‑English or local/regional evidence.
  - Quantitative bias: favoring submissions that present numbers even when qualitative evidence is more informative for the issue.
  - Traceability bias: equating the presence of a URL or docket ID with credibility without verifying content.
  - Publication bias: overvaluing peer‑reviewed literature and undervaluing credible gray literature (technical reports, agency memos).
  - Confirmation bias: tending to credit evidence that aligns with common assumptions about policy impacts more readily than surprising counterevidence.
  - Availability bias: relying on sources the model has been trained on or can recall easily, disadvantaging obscure but valid sources.
  - Conservatism bias: penalizing novel analyses or original calculations because they lack an established publication trail.
- **Task Misalignment Risks:** 
  - Scoring may prioritize traceable citation form over decision relevance, favoring citations that are easily verifiable rather than those most directly informing agency action.
  - The rubric can encourage citation padding—adding many superficial or low‑quality references to get a higher score—rather than substantive evidence integration.
  - It may penalize concise but high‑quality submissions that summarize evidence without reproducing full citations or links.
  - The judge might undervalue first‑hand community testimony or qualitative harms that are highly relevant for policy but lack traditional citations.
  - Because the model cannot access paywalled or proprietary attachments, it may downgrade otherwise strong, decision‑relevant evidence that isn't public.
  - The rubric assumes public, reproducible citations; submissions based on internal agency data or confidential sources could be misjudged.
  - Local or context‑specific evidence (e.g., tribal, municipal data) may be unfairly scored lower for not matching national‑level citation norms.
  - The axis focuses on evidentiary depth and may miss timeliness or urgency signals that warrant escalation despite limited citations.
- **Failure Cases:** 
  - Hallucinated verification: the model asserts that quoted citations are valid or finds supporting text when it cannot actually access the source.
  - False negatives for shorthand citations: downgrading submissions that use accepted shorthand (e.g., 'EPA Toxics 2020') even though a human reviewer would recognize them.
  - False positives for fabricated citations: awarding higher scores to convincing but nonexistent reports or doctored docket IDs because format looks plausible.
  - Inability to follow links or attachments: incorrectly scoring a submission as low‑evidence because links are to paywalled PDFs or attachments the model cannot retrieve.
  - Overrating quantity over quality: giving high scores to submissions listing many low‑credibility links (blogs, press releases) instead of a few strong sources.
  - Misinterpreting statistical claims: failing to spot misuse of statistics (e.g., incorrect denominators, conflating percentage points with percent change) and overrating evidence quality.
  - Failure to detect cherry‑picking: not recognizing when numbers are selectively presented without acknowledging counterevidence or uncertainty.
  - Inconsistency across similar inputs: scoring similar submissions differently due to subtle wording changes that trigger different internal heuristics.
  - Context blindness: treating a legal citation as authoritative even when jurisdiction or statutory context makes it irrelevant to the agency's authority.
  - Adversarial formatting: being manipulated by maliciously formatted text (fake URLs, embedded citations in images) to inflate a submission's score.
  - Over‑penalizing qualitative evidence: incorrectly assigning the lowest scores to persuasive firsthand accounts because they lack traditional citations.
  - Undervaluing original analysis: penalizing submissions that include novel calculations because the judge cannot reproduce methods or validate datasets.

## Related Metrics

- **Related Metrics:**
  - **LevenshteinDistance:** Levenshtein Distance measures the minimum number of single-character edits—insertions, deletions, or substitutions—required to transform one sequence into another.
  - **BARTScore:** BARTScore is a reference-based evaluation metric for text generation that formulates evaluation as a text generation task.
  - **PseudoPARENT:** **PseudoPARENT** is a *custom adaptation* of the PARENT metric for evaluating text generation from structured inputs.

## Further Reading

- **Papers:**
  - [Autometrics](https://github.com/XenonMolecule/autometrics)
  - [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://openreview.net/pdf?id=uccHPGDlao)

## Citation

```
@software{Ryan_Autometrics_2025,
    author = {Ryan, Michael J. and Zhang, Yanzhe and Salunkhe, Amol and Chu, Yi and Rahman, Emily and Xu, Di and Yang, Diyi},
    license = {MIT},
    title = {{Autometrics}},
    url = {https://github.com/XenonMolecule/autometrics},
    version = {1.0.0},
    year = {2025}
}
```

## Metric Card Authors

- **Authors:** This metric card was automatically generated by gpt-5.
- **Acknowledgement of AI Assistance:** This metric card was entirely automatically generated by gpt-5 using the Autometrics library. No human intervention was involved. User discretion is advised.
- **Contact:** For questions about the autometrics library, please contact [Michael J Ryan](mailto:mryan0@stanford.edu)."""

    description: ClassVar[str] = "**Evidentiary Support and Citations** Use of data, analyses, studies, and references that substantiate claims and quantify effects."

    def __init__(self, model: dspy.LM = DEFAULT_MODEL):
        super().__init__(
            name="Evidentiary_Support_and_Citations_Rubric",
            description="**Evidentiary Support and Citations** Use of data, analyses, studies, and references that substantiate claims and quantify effects.",
            axis="**Evidentiary Support and Citations** Use of data, analyses, studies, and references that substantiate claims and quantify effects.\n\nScoring Guidelines:\nScore 1: Score 1: No evidentiary support\n- Purely opinion/anecdote; no citations or links.\n- No quantitative information supporting claims.\n- References are irrelevant, unverifiable, or obviously inaccurate.\n- \u201cSee attached\u201d or name-dropping without incorporating any evidence into the argument.\n- Fabricated-sounding or conflicting facts with no sources.\nScore 2: Score 2: Minimal, vague, or weak support\n- One or two vague references (e.g., \u201cstudies show,\u201d \u201cEPA reports\u201d) without titles, dates, docket IDs, or URLs.\n- Mentions numbers but provides no source or traceability.\n- Sources are low-credibility (blogs, opinion pieces) or links are broken.\n- Evidence is not integrated into reasoning; limited or incorrect interpretation of cited material.\nScore 3: Score 3: Adequate baseline support\n- At least one specific, credible citation (e.g., docket ID, Federal Register cite, named report/study with year).\n- Includes some quantitative data tied to a source.\n- Evidence generally supports key claims, though coverage may be partial or selective.\n- Minor gaps in traceability (missing page numbers, incomplete links) or reliance on secondary summaries.\n- Limited discussion of methods/limitations; little triangulation.\nScore 4: Score 4: Strong, well-integrated support\n- Multiple credible, traceable sources (e.g., peer-reviewed studies, government reports, statutes/caselaw, docket materials) with precise identifiers (titles, dates, page/section numbers, URLs).\n- Quantifies effects clearly (units, magnitudes, time frames) and ties numbers to sources.\n- Evidence is accurately interpreted and woven into arguments that inform agency action.\n- Notes key assumptions/limitations; some triangulation across independent sources.\n- Attachments/exhibits referenced in-text; links work; minimal inconsistencies.\nScore 5: Score 5: Exceptional, decision-grade evidence use\n- Comprehensive, diverse evidence base (primary studies, official data, legal/technical references) with precise, reproducible citations (page/figure numbers, dataset names, docket IDs).\n- Provides original analysis or calculations (methods shown, equations/assumptions stated), data tables/appendices, or replicable methodology.\n- Quantifies impacts rigorously (comparative baselines, confidence/uncertainty where applicable) and addresses counter-evidence.\n- Clearly links evidence to concrete recommendations/options, showing decision relevance.\n- All references are authoritative and accessible; no traceability gaps; consistent and accurate interpretation throughout.\n",
            model=model,
            task_description="Rank candidate policy feedback drafts for follow-up: given a citizen\u2019s submission, \n    determine which responses merit escalation to agency officials.",
            metric_card="provided",
            max_workers=32,
        )

    def __repr__(self):
        return f"Evidentiary_Support_and_Citations_Rubric_LLMJudge(model=dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv(\"OPENAI_API_KEY\")))"

