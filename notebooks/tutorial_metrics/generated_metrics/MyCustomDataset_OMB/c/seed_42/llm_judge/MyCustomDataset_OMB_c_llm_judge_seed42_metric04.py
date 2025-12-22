# Auto-generated metric file for Policy_Citation_and_Evidence_Grounding_gpt-5-mini
import dspy
import os
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import GeneratedRefFreeLLMJudgeMetric
from typing import ClassVar

DEFAULT_MODEL = dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv("OPENAI_API_KEY"))

class Policy_Citation_and_Evidence_Grounding_gpt_5_mini_LLMJudge(GeneratedRefFreeLLMJudgeMetric):
    """---
# Metric Card for Policy_Citation_and_Evidence_Grounding_gpt-5-mini

**Policy Citation and Evidence Grounding** The draft accurately cites relevant OMB guidance memos dates and sections or docket identifiers to anchor the request.

## Metric Details

**Policy_Citation_and_Evidence_Grounding_gpt-5-mini** is a **reference-free** LLM-as-a-Judge metric that prompts an LLM to rate a system output along a single, run-time-specified evaluation axis.
In this case the axis is `**Policy Citation and Evidence Grounding** The draft accurately cites relevant OMB guidance memos dates and sections or docket identifiers to anchor the request.`.

The prompt supplies:

1. **Task description** *d*
2. **Axis rubric** `**Policy Citation and Evidence Grounding** The draft accurately cites relevant OMB guidance memos dates and sections or docket identifiers to anchor the request.`
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

### Inputs and Outputs
- **Inputs:**
  - **Task description** *d*
  - **Axis rubric** `**Policy Citation and Evidence Grounding** The draft accurately cites relevant OMB guidance memos dates and sections or docket identifiers to anchor the request.`
  - **Input text** *x*
  - **Output text** *y*
- **Outputs:**
  - Scalar score $\hat{s} \in \{1,2,3,4,5\}$

## Intended Use

- **Domain:** Policy evaluation / Document triage (Citation verification & evidence grounding)
- **Tasks:** 
  - Rank candidate policy feedback drafts by citation completeness and specificity
  - Flag drafts that cite OMB memos with missing or malformed dates/section identifiers
  - Detect discrepancies between asserted policy claims and the citations provided
  - Triage which citizen submissions merit escalation to agency officials based on citation strength
  - Classify drafts as adequately grounded, insufficiently grounded, or unverifiable (given provided text)
  - Generate structured summaries of cited guidance (when source excerpts provided) for reviewers
  - Identify ambiguous or vague references (e.g., 'recent OMB guidance') that need precise docket/memo identifiers
- **Best Suited For:** 
  - High-volume triage where drafts include explicit citation strings (title, memo date, docket ID) that can be pattern-matched.
  - Workflows that supply the model with the full text or authoritative excerpts of the referenced OMB memos for cross-checking.
  - Situations where the goal is to flag potential issues (missing/malformed citations, unsupported claims) to prioritize human review.
  - Automated pre-screening to reduce reviewer load by filtering clearly well-grounded vs. clearly deficient drafts.
  - Quality-control checks for internal consistency (e.g., whether the date in the citation matches the memo title provided).
  - Integration into a human-in-the-loop pipeline where final escalations and legal interpretations are made by agency officials.
- **Not Recommended For:** 
  - Cases demanding authoritative, up-to-date verification against live federal dockets or OMB repositories when those documents are not provided in the prompt.
  - Situations that require legal interpretation of how a cited OMB memo applies to complex policy facts or regulatory obligations.
  - High-stakes final decisions (e.g., formal agency responses or legal filings) without human expert review and source verification.
  - Inputs with ambiguous, abbreviated, or OCR-corrupted citations that require external research to resolve.
  - Tasks that expect the model to invent or recall precise section contents or docket histories beyond its training cutoff or provided sources.
  - Contexts that require tracking real-time changes to guidance (rescissions, superseding memos) where model cannot access live updates.

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
  - Recency bias: preferring guidance it 'remembers' and penalizing newer or updated guidance beyond the model's knowledge cutoff.
  - Authority/source bias: favoring OMB memos over other legitimate grounding sources (statutes, regs, agency-specific dockets).
  - Exact-match bias: rewarding verbatim citation formatting while penalizing paraphrased or partial but correct references.
  - Jurisdictional bias: assuming OMB guidance applies uniformly across contexts instead of checking topical/applicability differences.
  - Language bias: worse performance on citations in languages or formats not well represented in training data.
  - Confirmation bias: more likely to accept citations that fit expected patterns and to distrust atypical but correct references.
  - Format bias: privileging citations in a specific citation style (dates/section numbers) over other valid identifiers (docket names, RINs).
  - Availability bias: over-relying on well-known/high-profile memos and under-weighting obscure but relevant guidance.
- **Task Misalignment Risks:** 
  - Overfocusing on citation form rather than substantive policy relevance, causing escalations for formally-cited but irrelevant drafts.
  - Penalizing valid drafts that ground requests in non-OMB authorities (statutes, CFR sections, agency dockets) because the axis privileges OMB citations.
  - Treating absence of exact date/section numbers as fatal even when the referenced guidance is clearly identifiable by title or subject.
  - Failing to account for superseded, amended, or revoked guidance and thus mis-evaluating whether a citation is current and applicable.
  - Prioritizing easily-verifiable docket identifiers while ignoring context that affects applicability (e.g., rulemaking stage).
  - Applying the same strictness across all submissions regardless of intended escalation level (informational vs high-priority), leading to mismatched triage.
  - Relying on model-internal knowledge rather than external verification, causing confident but unsupported judgments about citation existence.
  - Encouraging writers to insert boilerplate citations to pass the axis instead of producing accurate, substantive grounding.
- **Failure Cases:** 
  - Hallucinated citations: the judge accepts or invents OMB memo titles/dates/sections that do not exist.
  - False negatives: rejecting correct but paraphrased or abbreviated citations because they do not match expected strings exactly.
  - Outdated-evidence error: marking a citation as valid when it has been superseded or revoked (or marking a current citation as invalid due to model cutoff).
  - Incorrect attribution: assigning a memo to OMB when it originated from another agency or interagency source.
  - Docket misidentification: accepting an incorrect docket identifier that is plausibly formatted but points to a different proceeding.
  - Context-mismatch: flagging a technically accurate citation that is not actually relevant to the citizen's request or jurisdiction.
  - Overconfidence: producing definitive verification statements without external lookup, leading to misleading escalation decisions.
  - Partial-match acceptance: treating a partial or similarly numbered section as fully correct when it materially changes meaning.
  - Failure to detect confidentiality or restricted dockets and recommending escalation where inappropriate.
  - Bias-amplified errors: systematically downgrading submissions from certain demographic or stylistic groups due to language/form differences in citations.
  - Format-only scoring: rewarding drafts that include citations formatted correctly but that misuse or misunderstand the cited guidance content.
  - Chain-of-trust error: assuming a cited memo's guidance applies wholesale without checking whether later guidance or agency interpretations modify its applicability.

## Related Metrics

- **Related Metrics:**
  - **GRMRewardModel:** The GRMRewardModel is a general-purpose reward model designed to evaluate the quality and safety of LLM-generated outputs.
  - **MAUVE:** MAUVE (Measuring the Alignment of Unconditional VErsions) quantifies the similarity between two text distributions (e.
  - **SelfBLEU:** Self-BLEU is a reference-free diversity metric used in text generation tasks.

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

    description: ClassVar[str] = "**Policy Citation and Evidence Grounding** The draft accurately cites relevant OMB guidance memos dates and sections or docket identifiers to anchor the request."

    def __init__(self, model: dspy.LM = DEFAULT_MODEL):
        super().__init__(
            name="Policy_Citation_and_Evidence_Grounding_gpt-5-mini",
            description="**Policy Citation and Evidence Grounding** The draft accurately cites relevant OMB guidance memos dates and sections or docket identifiers to anchor the request.",
            axis="**Policy Citation and Evidence Grounding** The draft accurately cites relevant OMB guidance memos dates and sections or docket identifiers to anchor the request.",
            model=model,
            task_description="Rank candidate policy feedback drafts for follow-up: given a citizen\u2019s submission, \n    determine which responses merit escalation to agency officials.",
            metric_card="provided",
            max_workers=32,
        )

    def __repr__(self):
        return f"Policy_Citation_and_Evidence_Grounding_gpt_5_mini_LLMJudge(model=dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv(\"OPENAI_API_KEY\")))"

