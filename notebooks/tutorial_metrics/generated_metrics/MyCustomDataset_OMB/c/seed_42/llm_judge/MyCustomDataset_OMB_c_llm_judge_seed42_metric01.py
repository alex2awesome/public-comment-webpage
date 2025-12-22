# Auto-generated metric file for Agency_Targeting_and_Recipient_Fit_gpt-5-mini
import dspy
import os
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import GeneratedRefFreeLLMJudgeMetric
from typing import ClassVar

DEFAULT_MODEL = dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv("OPENAI_API_KEY"))

class Agency_Targeting_and_Recipient_Fit_gpt_5_mini_LLMJudge(GeneratedRefFreeLLMJudgeMetric):
    """---
# Metric Card for Agency_Targeting_and_Recipient_Fit_gpt-5-mini

**Agency Targeting and Recipient Fit** Does the draft address the correct agency office and named official who can act on the issue at hand.

## Metric Details

**Agency_Targeting_and_Recipient_Fit_gpt-5-mini** is a **reference-free** LLM-as-a-Judge metric that prompts an LLM to rate a system output along a single, run-time-specified evaluation axis.
In this case the axis is `**Agency Targeting and Recipient Fit** Does the draft address the correct agency office and named official who can act on the issue at hand.`.

The prompt supplies:

1. **Task description** *d*
2. **Axis rubric** `**Agency Targeting and Recipient Fit** Does the draft address the correct agency office and named official who can act on the issue at hand.`
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
  - **Axis rubric** `**Agency Targeting and Recipient Fit** Does the draft address the correct agency office and named official who can act on the issue at hand.`
  - **Input text** *x*
  - **Output text** *y*
- **Outputs:**
  - Scalar score $\hat{s} \in \{1,2,3,4,5\}$

## Intended Use

- **Domain:** Policy evaluation / Triage (text evaluation for government affairs)
- **Tasks:** 
  - Policy feedback triage (rank drafts for escalation)
  - Recipient and office identification (map issues to likely agency office)
  - Named official extraction and consistency checking
  - Scoring recipient fit and escalation priority
  - Suggested redirection or rewrite recommendations to improve targeting
- **Best Suited For:** 
  - The citizen submission and candidate drafts explicitly name agencies, offices, or roles, allowing direct string-match and role-based reasoning.
  - The task is triage-focused (deciding which drafts merit escalation) rather than final authoritative determinations about jurisdiction or law.
  - There is a need to scale rapid, consistent judgments about textual fit and recipient appropriateness across many submissions.
  - When an up-to-date mapping of agency responsibilities or a curated list of relevant offices is provided to the model as context or prompt.
  - When human reviewers will perform a final verification step (e.g., confirming current officials or authority before outreach).
- **Not Recommended For:** 
  - When the decision requires authoritative, real-time verification of current officeholders, contact details, or internal delegation not publicly documented.
  - When escalating could have high legal, safety, or ethical consequences that require certified knowledge from subject-matter experts.
  - When inputs are highly ambiguous, contain sensitive classified information, or include extensive PII that must not be processed by an LLM.
  - When the mapping between issue and agency depends on local laws, nuanced regulatory boundaries, or recent jurisdictional changes after the model’s knowledge cutoff.
  - When multilingual submissions use rare languages or domain-specific jargon the model may not reliably interpret without additional domain data.

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
  - Agency familiarity bias: preferring well-known federal agencies and officials and overlooking lesser-known state/local offices or program-specific units.
  - Recency and training-data bias: relying on outdated personnel or organizational structures present in the model's training data.
  - Name-recognition bias: upwardly favoring drafts that mention prominent officials or high-level titles regardless of operational relevance.
  - Geographic bias: assuming federal jurisdiction for problems that are state, local, or tribal in nature, or vice versa.
  - Confirmation bias: interpreting vague or partial information to fit a plausible agency match the model prefers.
  - Title-over-function bias: equating job titles with actual authority and ignoring functional responsibility or delegated powers.
  - Language and cultural bias: misreading phrasing from non-native speakers or community-specific terminology, leading to incorrect recipient matches.
  - Over-reliance on heuristics: using surface heuristics (words like 'enforcement', 'policy', 'compliance') rather than reasoning about actual mandate or authority.
- **Task Misalignment Risks:** 
  - Focusing too narrowly on exact named officials instead of the correct office or team that actually handles the issue, leading to inappropriate escalation recommendations.
  - Prioritizing formal title matches over operational fit, so the model recommends escalation to someone who looks right on paper but cannot act.
  - Misinterpreting jurisdictional boundaries (federal/state/local/tribal) and recommending escalation to the wrong government level.
  - Assuming a single responsible agency when the issue requires multi-agency coordination, causing under-escalation or misdirection.
  - Failing to account for confidentiality, legal, or safety constraints that make escalation inappropriate despite an apparent recipient match.
  - Valuing named officials even when processes require submitting via a public intake channel or FOIA route, leading to procedural mismatches.
  - Treating an absence of a named official as a non-escalation trigger instead of flagging drafts that need a correct recipient to be identified.
  - Over-weighting the presence of any official name (e.g., a cabinet member) instead of validating whether that person has direct influence over the issue.
- **Failure Cases:** 
  - Recommending escalation to a high-level official (e.g., Secretary) who lacks direct authority over the operational complaint, delaying resolution.
  - Missing that a problem is under state or local jurisdiction and escalating to a federal agency that cannot act, causing wasted effort.
  - Failing to identify that the correct recipient is a program manager or regional office, and marking a draft as non-escalation-worthy.
  - Suggesting escalation to a named individual who has left office or changed roles, due to outdated model knowledge.
  - Flagging routine service requests or informational inquiries as needing escalation to agency officials, overwhelming officials with non-actionable items.
  - Not recognizing multi-agency issues and recommending escalation to only one agency, leading to incomplete follow-up.
  - Recommending escalation when privacy, security, or whistleblower protections mean the submission should not be forwarded to an outside official.
  - Misclassifying politically-targeted or advocacy language as a reason to escalate to elected officials when operational staff are the appropriate recipients.
  - Failing to request clarifying details (location, program identifiers) before deciding, resulting in incorrect recipient matches.
  - Confusing similarly named agencies or offices (e.g., different 'Department of Health' at city vs state vs federal levels) and escalating to the wrong entity.

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

    description: ClassVar[str] = "**Agency Targeting and Recipient Fit** Does the draft address the correct agency office and named official who can act on the issue at hand."

    def __init__(self, model: dspy.LM = DEFAULT_MODEL):
        super().__init__(
            name="Agency_Targeting_and_Recipient_Fit_gpt-5-mini",
            description="**Agency Targeting and Recipient Fit** Does the draft address the correct agency office and named official who can act on the issue at hand.",
            axis="**Agency Targeting and Recipient Fit** Does the draft address the correct agency office and named official who can act on the issue at hand.",
            model=model,
            task_description="Rank candidate policy feedback drafts for follow-up: given a citizen\u2019s submission, \n    determine which responses merit escalation to agency officials.",
            metric_card="provided",
            max_workers=32,
        )

    def __repr__(self):
        return f"Agency_Targeting_and_Recipient_Fit_gpt_5_mini_LLMJudge(model=dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv(\"OPENAI_API_KEY\")))"

