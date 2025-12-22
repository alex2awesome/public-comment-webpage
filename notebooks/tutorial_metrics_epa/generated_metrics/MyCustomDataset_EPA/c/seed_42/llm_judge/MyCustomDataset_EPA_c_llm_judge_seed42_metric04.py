# Auto-generated metric file for Legal_and_Policy_Grounding_gpt-5-mini
import dspy
import os
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import GeneratedRefFreeLLMJudgeMetric
from typing import ClassVar

DEFAULT_MODEL = dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv("OPENAI_API_KEY"))

class Legal_and_Policy_Grounding_gpt_5_mini_LLMJudge(GeneratedRefFreeLLMJudgeMetric):
    """---
# Metric Card for Legal_and_Policy_Grounding_gpt-5-mini

**Legal and Policy Grounding** Accurate use of statutory authority, case law, guidance, and regulatory procedures to frame and justify the request.

## Metric Details

**Legal_and_Policy_Grounding_gpt-5-mini** is a **reference-free** LLM-as-a-Judge metric that prompts an LLM to rate a system output along a single, run-time-specified evaluation axis.
In this case the axis is `**Legal and Policy Grounding** Accurate use of statutory authority, case law, guidance, and regulatory procedures to frame and justify the request.`.

The prompt supplies:

1. **Task description** *d*
2. **Axis rubric** `**Legal and Policy Grounding** Accurate use of statutory authority, case law, guidance, and regulatory procedures to frame and justify the request.`
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
  - **Axis rubric** `**Legal and Policy Grounding** Accurate use of statutory authority, case law, guidance, and regulatory procedures to frame and justify the request.`
  - **Input text** *x*
  - **Output text** *y*
- **Outputs:**
  - Scalar score $\hat{s} \in \{1,2,3,4,5\}$

## Intended Use

- **Domain:** Legal & Policy Review
- **Tasks:** 
  - Rank candidate policy-feedback drafts by adequacy of legal grounding
  - Triage citizen submissions for escalation to agency officials based on statutory/regulatory support
  - Identify missing or incorrect citations to statutes, regulations, or guidance in drafts
  - Highlight procedural requests that depart from agency rules or lack necessary authority
  - Summarize the legal basis cited in each draft for quick human review
  - Flag drafts that require expert legal review due to complexity or potential liability
- **Best Suited For:** 
  - When drafts include explicit citations or quoted statutory/regulatory text that the model can check for consistency
  - When evaluation criteria are well-defined (e.g., must cite statute X or follow procedure Y) so the model can apply deterministic filters
  - For high-volume triage where human review capacity is limited and prioritization is needed
  - When used as a first-pass filter to surface likely misstatements of law or missing procedural steps
  - When source materials (relevant statutes, regulations, agency guidance) are provided alongside submissions for cross-checking
  - When the goal is to produce concise summaries of the legal grounding for human reviewers rather than definitive legal advice
- **Not Recommended For:** 
  - When an authoritative, up-to-date legal determination is required (e.g., before litigation or formal enforcement actions)
  - For interpreting recent changes in law or new case law issued after the model’s knowledge cutoff or without access to current legal databases
  - When jurisdiction-specific nuances or interplay between statutes require expert legal judgment
  - When the submissions involve complex factual disputes that materially affect legal interpretation
  - For handling highly sensitive confidential information without secure, audited infrastructure
  - When outputs will be relied upon as final legal advice rather than as guidance to be vetted by qualified legal staff

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
  - Overconfidence bias: the Judge may present uncertain or hallucinated legal conclusions with high certainty, treating them as authoritative.
  - Authority-weighting bias: the model may overvalue recent or frequently seen sources in its training data regardless of actual bindingness in the relevant jurisdiction.
  - Jurisdictional bias: tendency to default to federal U.S. law or other high-frequency jurisdictions seen in training data, undervaluing local or specialized regulatory regimes.
  - Form-over-substance bias: preferring responses that include formal citations or legal phrasing even when plain-language reasoning is legally sound.
  - Conservatism bias: favoring cautious, procedural recommendations over creative but legally defensible approaches, leading to under-escalation.
  - Recency and dataset bias: failing to recognize changes to law or regulation after the model's cutoff date, or over-relying on outdated guidance.
  - Source familiarity bias: preferring famous cases or statutes and undervaluing niche but controlling authorities, leading to mis-weighted assessments.
  - Language and sociolect bias: misunderstanding submissions that use nonstandard legal language or community-specific framing and devaluing them relative to formally worded drafts.
- **Task Misalignment Risks:** 
  - Jurisdiction mismatch: the axis requires legal grounding per relevant jurisdiction, but the Judge may not detect or correctly apply the specific jurisdictional context of the citizen submission.
  - Overemphasis on citation presence: the Judge might equate lack of explicit citation with poor legal grounding, misaligning with the task's need to assess substantive legal merit.
  - Procedural vs substantive confusion: the Judge may conflate procedural sufficiency (e.g., filing requirements) with substantive legal soundness, misclassifying escalation-worthiness.
  - Scope creep into policy advocacy: the Judge could elevate normative policy preferences rather than strictly assessing legal justification, misaligning with an objective legal-grounding axis.
  - Temporal misalignment: assessing current legal grounding when the task requires evaluating the draft's usefulness despite rapidly changing law or pending rulemakings.
  - Weight-of-authority misassignment: treating persuasive authorities as binding or vice versa, producing misaligned escalation recommendations relative to true legal risk.
  - Over-reliance on training artifacts: ranking based on phrasings or patterns common in training data rather than on actual legal correctness relevant to the citizen's issue.
  - Black-box scoring: producing scalar rankings without transparent legal reasoning, which misaligns with the need for defensible, actionable escalation rationale.
- **Failure Cases:** 
  - False positive escalation: the Judge recommends escalation because a draft includes formal-looking citations that are misapplied or irrelevant to the issue.
  - False negative (missed escalation): the Judge fails to escalate a draft that correctly identifies a compelling statutory or regulatory violation because it lacks formal citations or uses plain language.
  - Jurisdictional error: the Judge evaluates applicability under the wrong jurisdiction (e.g., federal instead of state) and misranks responses accordingly.
  - Hallucinated authority: the Judge cites cases or statutes that do not exist or misstates holdings, then uses those hallucinations to justify its ranking.
  - Misweighted authority: the Judge treats a persuasive agency guidance as if it were binding law, inflating a draft's apparent legal grounding.
  - Temporal staleness: the Judge endorses a draft grounded in an older precedent overturned or superseded after the model's cutoff, leading to inappropriate escalation.
  - Procedural mis-evaluation: the Judge flags a draft as legally insufficient because it doesn't follow a specific filing format, even though the substance is actionable and should be escalated.
  - Over-penalizing lay submissions: the Judge downgrades citizen-authored drafts for lacking citation form even when the substantive legal claim is strong and actionable.
  - Under-recognition of narrow controlling authority: the Judge overlooks a specialized controlling statute or regulation embedded in the submission and ranks the draft low.
  - Over-reliance on surface cues: the Judge privileges drafts that reuse language from well-known legal templates in training data, not because they are better grounded but because they look familiar.
  - Misinterpretation of burdens or standards: the Judge incorrectly assesses the applicable legal standard (e.g., standard of review), leading to wrong conclusions about merit.
  - Lack of explainability: the Judge outputs a ranking with insufficient legal rationale or cites irrelevant authorities, preventing reviewers from validating the evaluation.

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

    description: ClassVar[str] = "**Legal and Policy Grounding** Accurate use of statutory authority, case law, guidance, and regulatory procedures to frame and justify the request."

    def __init__(self, model: dspy.LM = DEFAULT_MODEL):
        super().__init__(
            name="Legal_and_Policy_Grounding_gpt-5-mini",
            description="**Legal and Policy Grounding** Accurate use of statutory authority, case law, guidance, and regulatory procedures to frame and justify the request.",
            axis="**Legal and Policy Grounding** Accurate use of statutory authority, case law, guidance, and regulatory procedures to frame and justify the request.",
            model=model,
            task_description="Rank candidate policy feedback drafts for follow-up: given a citizen\u2019s submission, \n    determine which responses merit escalation to agency officials.",
            metric_card="provided",
            max_workers=32,
        )

    def __repr__(self):
        return f"Legal_and_Policy_Grounding_gpt_5_mini_LLMJudge(model=dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv(\"OPENAI_API_KEY\")))"

