# Auto-generated metric file for Feasibility_and_Implementation_Detail_gpt-5-mini
import dspy
import os
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import GeneratedRefFreeLLMJudgeMetric
from typing import ClassVar

DEFAULT_MODEL = dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv("OPENAI_API_KEY"))

class Feasibility_and_Implementation_Detail_gpt_5_mini_LLMJudge(GeneratedRefFreeLLMJudgeMetric):
    """---
# Metric Card for Feasibility_and_Implementation_Detail_gpt-5-mini

**Feasibility and Implementation Detail** Practicality of the proposal and inclusion of implementation considerations, tradeoffs, and operational details.

## Metric Details

**Feasibility_and_Implementation_Detail_gpt-5-mini** is a **reference-free** LLM-as-a-Judge metric that prompts an LLM to rate a system output along a single, run-time-specified evaluation axis.
In this case the axis is `**Feasibility and Implementation Detail** Practicality of the proposal and inclusion of implementation considerations, tradeoffs, and operational details.`.

The prompt supplies:

1. **Task description** *d*
2. **Axis rubric** `**Feasibility and Implementation Detail** Practicality of the proposal and inclusion of implementation considerations, tradeoffs, and operational details.`
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
  - **Axis rubric** `**Feasibility and Implementation Detail** Practicality of the proposal and inclusion of implementation considerations, tradeoffs, and operational details.`
  - **Input text** *x*
  - **Output text** *y*
- **Outputs:**
  - Scalar score $\hat{s} \in \{1,2,3,4,5\}$

## Intended Use

- **Domain:** Policy triage and text evaluation
- **Tasks:** 
  - Rank policy feedback drafts by implementation-readiness
  - Identify and extract concrete implementation elements (steps, timelines, responsible parties, resource estimates)
  - Flag proposals that merit escalation due to operational detail or potential impact
  - Summarize tradeoffs, assumptions, and missing operational details
  - Classify drafts by feasibility bands (e.g., near-term implementable, requires analysis, unrealistic)
  - Generate concise justification notes for escalation decisions
- **Best Suited For:** 
  - Drafts are in clear natural language and contain explicit operational elements (timelines, roles, costs, dependencies).
  - Evaluations need consistent, repeatable triage criteria across many submissions.
  - Agency-level processes are general (e.g., budgeting, procurement, staffing) rather than highly specialized technical domains.
  - Decision-makers want rapid pre-screening to surface items that deserve human expert review or escalation.
  - There is ground truth or additional structured context (e.g., agency constraints, approximate budgets) available to supplement the model’s output when needed.
- **Not Recommended For:** 
  - Proposals require detailed domain expertise (e.g., nuclear engineering, specialized medical device regulation) that the model cannot reliably provide.
  - Evaluation requires up-to-date, confidential, or jurisdiction-specific legal/regulatory compliance checks the model cannot access.
  - Inputs are extremely vague, non-textual (e.g., images, audio-only), or too short to infer implementation detail.
  - Decisions depend on sensitive classified information or real-time operational data not supplied to the model.
  - High-stakes legal or binding policy decisions where human expert verification and formal risk assessments are mandatory.

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
  - Recency bias: overvaluing implementation approaches aligned with recent high-profile policy models the model has seen in training.
  - Technical-expertise bias: favoring proposals that reference technical terms or frameworks even if superficially used.
  - Clarity/verbosity bias: equating longer, more detailed prose with higher feasibility regardless of substance.
  - Resource-rich bias: assuming agencies have more budgetary, staffing, or technical resources than they actually do.
  - Status-quo bias: preferring incremental changes with obvious implementation paths over novel, disruptive solutions.
  - Urban/Western context bias: assuming access to infrastructure, institutions, or regulatory environments common in high-income urban settings.
  - Risk-aversion bias: penalizing proposals that involve politically or legally risky but potentially high-impact tradeoffs.
  - Agency-alignment bias: favoring proposals that fit common agency workflows or language, disadvantaging outsider framing.
  - Quantification bias: over-emphasizing numeric estimates and timelines even when unavailable or inappropriate for the idea.
  - Legal/regulatory overconfidence bias: incorrectly assuming or dismissing legal constraints without jurisdiction-specific context.
- **Task Misalignment Risks:** 
  - Prioritizing detailed feasibility over representativeness, causing escalation only for submissions from knowledgeable or professionalized respondents and sidelining lived-experience voices.
  - Misclassifying high-level but strategically important signals (e.g., whistleblower or urgent ethical concerns) as non-actionable because they lack implementation detail.
  - Conflating 'feasible' with 'politically acceptable', leading to non-escalation of technically feasible but politically sensitive matters that require agency attention.
  - Overfitting the axis to technical implementation and ignoring other escalation-worthy dimensions like legality, urgency, or public safety.
  - Encouraging gamed responses where citizens pad drafts with superficial implementation language to trigger escalation, undermining signal quality.
  - Applying a one-size-fits-all feasibility standard across diverse policy domains and local contexts, producing inconsistent escalation decisions.
- **Failure Cases:** 
  - False positive: escalating a draft that reads detailed and plausible but omits critical domain constraints (e.g., regulatory prohibitions).
  - False negative: failing to escalate a brief submission from a subject-matter expert that identifies a high-risk operational flaw because it lacks step-by-step implementation detail.
  - Mistaken timeline assessment: accepting unrealistic timelines or resource estimates included in the draft as credible without cross-checking feasibility.
  - Missing dependencies: endorsing plans that ignore essential stakeholder approvals, procurement processes, or cross-agency coordination needs.
  - Context mismatch: rating a proposal feasible based on assumptions appropriate for one jurisdiction but infeasible in the submitter’s actual locale.
  - Overlooking equity impacts: promoting technically feasible proposals that would exacerbate disparities because the model did not evaluate distributive effects.
  - Terminology trap: downgrading practical community-sourced ideas due to non-standard language or lack of formal planning jargon.
  - Regulatory misassumption: failing to flag proposals that would violate statutes or require legislative changes, leading to unnecessary escalation.
  - Overconfidence in cost estimates: accepting ballpark financials in drafts and escalating without noting high uncertainty or missing cost categories.
  - Red-teaming blind spot: missing adversarially crafted drafts that purposely include plausible-sounding implementation detail to manipulate escalation.

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

    description: ClassVar[str] = "**Feasibility and Implementation Detail** Practicality of the proposal and inclusion of implementation considerations, tradeoffs, and operational details."

    def __init__(self, model: dspy.LM = DEFAULT_MODEL):
        super().__init__(
            name="Feasibility_and_Implementation_Detail_gpt-5-mini",
            description="**Feasibility and Implementation Detail** Practicality of the proposal and inclusion of implementation considerations, tradeoffs, and operational details.",
            axis="**Feasibility and Implementation Detail** Practicality of the proposal and inclusion of implementation considerations, tradeoffs, and operational details.",
            model=model,
            task_description="Rank candidate policy feedback drafts for follow-up: given a citizen\u2019s submission, \n    determine which responses merit escalation to agency officials.",
            metric_card="provided",
            max_workers=32,
        )

    def __repr__(self):
        return f"Feasibility_and_Implementation_Detail_gpt_5_mini_LLMJudge(model=dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv(\"OPENAI_API_KEY\")))"

