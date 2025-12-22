# Auto-generated metric file for Actionable_recommendations_gpt-5-mini
import dspy
import os
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import GeneratedRefFreeLLMJudgeMetric
from typing import ClassVar

DEFAULT_MODEL = dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv("OPENAI_API_KEY"))

class Actionable_recommendations_gpt_5_mini_LLMJudge(GeneratedRefFreeLLMJudgeMetric):
    """---
# Metric Card for Actionable_recommendations_gpt-5-mini

**Actionable recommendations** Presence of clear asks, proposed changes, or concrete guidance that an official could act upon.

## Metric Details

**Actionable_recommendations_gpt-5-mini** is a **reference-free** LLM-as-a-Judge metric that prompts an LLM to rate a system output along a single, run-time-specified evaluation axis.
In this case the axis is `**Actionable recommendations** Presence of clear asks, proposed changes, or concrete guidance that an official could act upon.`.

The prompt supplies:

1. **Task description** *d*
2. **Axis rubric** `**Actionable recommendations** Presence of clear asks, proposed changes, or concrete guidance that an official could act upon.`
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
  - **Axis rubric** `**Actionable recommendations** Presence of clear asks, proposed changes, or concrete guidance that an official could act upon.`
  - **Input text** *x*
  - **Output text** *y*
- **Outputs:**
  - Scalar score $\hat{s} \in \{1,2,3,4,5\}$

## Intended Use

- **Domain:** Policy feedback evaluation / Text classification
- **Tasks:** 
  - Rank candidate feedback drafts by actionability for escalation to officials
  - Triage citizen submissions to label as 'actionable', 'needs clarification', or 'not actionable'
  - Extract and summarize concrete asks (actor, action, timeline, metrics) from drafts
  - Flag missing information needed for implementation and generate targeted clarification questions
  - Prioritize actionable items by estimated feasibility and clarity
  - Generate short justification notes for why a draft should or should not be escalated
- **Best Suited For:** 
  - When drafts contain explicit language asking for changes or specific actions (clear verbs, timelines, actors).
  - When the evaluation criteria (what counts as actionable) are well-defined and can be encoded as heuristics or examples.
  - When processing moderate to large volumes of citizen feedback for initial triage and prioritization. 
  - When agency procedures and responsibilities are general (not highly technical) so actionability can be judged from plain-language guidance.
  - When rapid, consistent, and reproducible scoring is needed to route items to human reviewers.
- **Not Recommended For:** 
  - When recommendations require deep, domain-specific technical or legal expertise to judge feasibility or compliance.
  - When actionability depends on internal, non-public agency processes or jurisdiction-specific regulations that the model cannot access reliably.
  - For high-stakes or safety-critical escalation decisions where human accountability and formal legal review are required.
  - When drafts are highly implicit, rhetorical, or metaphorical such that asks are not explicitly stated and require heavy inference.
  - When inputs include classified, confidential, or privacy-sensitive information that must not be processed by a general-purpose LLM.

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
  - Favoring submissions that use explicit imperatives or formal structure over narrative or testimonial forms, leading to under-recognition of actionables expressed as stories or grievances.
  - Preference for technical or domain-specific phrasing the model recognizes from training data, disadvantaging lay or community-language descriptions.
  - Length/verbosity bias: longer, more detailed messages may be judged more actionable even if they contain no concrete asks.
  - Recency/training-data bias toward policy ideas and formats common in the model’s training corpus, which may not match local agency practices.
  - Cultural and language bias against non-native phrasing or colloquialisms that convey actionable content indirectly.
  - Status-quo bias that treats novel or unconventional requests as less credible or actionable.
  - Safety/guardrail bias that downranks or refuses to flag items that hint at controversial but legally permissible actions due to caution.
  - Format bias that rewards numbered lists, 'please'/'request' phrasing, or explicit verbs while penalizing subtler formulations.
  - Confirmation bias where the model favors actionables that align with common policy templates it has seen rather than citizens’ bespoke proposals.
- **Task Misalignment Risks:** 
  - Conflating explicit specificity with feasibility—marking as actionable anything that is detailed, regardless of whether an agency can or should act on it.
  - Focusing only on explicit asks and missing implicit requests that require domain inference (e.g., systemic failures described without direct asks).
  - Equating 'actionable' with 'urgent' or 'important' so that high-impact but high-level policy critiques are deprioritized.
  - Failing to account for legal, ethical, or resource constraints that make an otherwise concrete ask infeasible for an agency.
  - Prioritizing requests that fit a template of bureaucratic action (permits, repairs) over governance or policy changes that need deliberation.
  - Scoring based on model’s confidence rather than calibrated probability leading to over/under-escalation.
  - Treating presence of proposed solutions as automatically positive even when they are harmful, illegal, or technically unsound.
  - Optimizing for recall (catching any possible actionables) could produce too much noise for officials, while optimizing for precision could miss subtle but important asks.
- **Failure Cases:** 
  - Marking an emotional report of discriminatory policing as non-actionable because it lacks an explicit ask, thereby failing to escalate a legitimate oversight issue.
  - Flagging a citizen’s rhetorical/opinionated post that proposes illegal measures as actionable and recommending escalation without legality checks.
  - Missing a technical but terse suggestion from a subject-matter expert because of jargon unfamiliar to the model, resulting in under-escalation.
  - Escalating duplicate or spam submissions that contain templated actionable wording, creating unnecessary work for officials.
  - Hallucinating a specific recommended change or implementation step not present in the citizen’s text and including it as justification to escalate.
  - Downgrading community-led proposals expressed in storytelling or first-person narrative because they lack formal request phrasing.
  - Elevating petty or low-impact operational asks (e.g., personal dispute resolution) as actionable over more systemic but less specific policy issues.
  - Failing to detect geographic or temporal specificity errors (e.g., wrong address, ambiguous timeframe) and treating them as actionable without flagging uncertainty.
  - Ignoring marginalized linguistic styles (code-switching, dialect) and thereby missing actionables communicated indirectly.
  - Prioritizing lengthy submissions with many details over concise but precisely actionable one-line requests.
  - Treating feasibility as binary and either fully escalating or rejecting a suggestion, rather than flagging for expert review when unclear.
  - Overseeing safety-sensitive content (threats, calls for violence) as 'action items' instead of routing to appropriate safety/legal workflows.
  - Misclassifying policy critique that implies an ask (e.g., 'This law causes harm, change it') as non-actionable because it lacks specific statutory language or amendment text.
  - Failing to surface resource- or jurisdiction-related constraints (e.g., request outside agency remit) leading to inappropriate escalation.

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

    description: ClassVar[str] = "**Actionable recommendations** Presence of clear asks, proposed changes, or concrete guidance that an official could act upon."

    def __init__(self, model: dspy.LM = DEFAULT_MODEL):
        super().__init__(
            name="Actionable_recommendations_gpt-5-mini",
            description="**Actionable recommendations** Presence of clear asks, proposed changes, or concrete guidance that an official could act upon.",
            axis="**Actionable recommendations** Presence of clear asks, proposed changes, or concrete guidance that an official could act upon.",
            model=model,
            task_description="Rank candidate policy feedback drafts for follow-up: given a citizen\u2019s submission, \n    determine which responses merit escalation to agency officials.",
            metric_card="provided",
            max_workers=32,
        )

    def __repr__(self):
        return f"Actionable_recommendations_gpt_5_mini_LLMJudge(model=dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv(\"OPENAI_API_KEY\")))"

