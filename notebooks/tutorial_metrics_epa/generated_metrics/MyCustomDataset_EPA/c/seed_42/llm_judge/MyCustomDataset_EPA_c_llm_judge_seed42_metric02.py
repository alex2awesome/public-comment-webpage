# Auto-generated metric file for Specificity_of_Ask_and_Actionability_gpt-5-mini
import dspy
import os
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import GeneratedRefFreeLLMJudgeMetric
from typing import ClassVar

DEFAULT_MODEL = dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv("OPENAI_API_KEY"))

class Specificity_of_Ask_and_Actionability_gpt_5_mini_LLMJudge(GeneratedRefFreeLLMJudgeMetric):
    """---
# Metric Card for Specificity_of_Ask_and_Actionability_gpt-5-mini

**Specificity of Ask and Actionability** Clarity of concrete requests or decisions sought from the agency, with steps the agency can realistically take.

## Metric Details

**Specificity_of_Ask_and_Actionability_gpt-5-mini** is a **reference-free** LLM-as-a-Judge metric that prompts an LLM to rate a system output along a single, run-time-specified evaluation axis.
In this case the axis is `**Specificity of Ask and Actionability** Clarity of concrete requests or decisions sought from the agency, with steps the agency can realistically take.`.

The prompt supplies:

1. **Task description** *d*
2. **Axis rubric** `**Specificity of Ask and Actionability** Clarity of concrete requests or decisions sought from the agency, with steps the agency can realistically take.`
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
  - **Axis rubric** `**Specificity of Ask and Actionability** Clarity of concrete requests or decisions sought from the agency, with steps the agency can realistically take.`
  - **Input text** *x*
  - **Output text** *y*
- **Outputs:**
  - Scalar score $\hat{s} \in \{1,2,3,4,5\}$

## Intended Use

- **Domain:** Policy analysis / text evaluation (public feedback triage)
- **Tasks:** 
  - Triage citizen submissions to decide which should be escalated to agency officials
  - Score or rank drafts by presence of explicit requests and clear decision points
  - Extract concrete action items, requested timelines, and named responsible parties from submissions
  - Generate short rationales explaining why a draft is or is not actionable
  - Suggest minimal clarifying questions to turn an ambiguous submission into an actionable one
  - Batch-filter submissions for follow-up based on preset criteria (e.g., asks requiring policy change)
  - Prioritize responses when limited agency bandwidth is available
  - Create structured metadata (e.g., ask type, specificity level, feasibility indicators) for downstream processing
- **Best Suited For:** 
  - Submissions are written in clear, contemporary language with explicit sentences describing asks or requests.
  - There are well-defined criteria or templates for what constitutes an actionable ask (so the judge can map text to those features).
  - Volume is large and human reviewers need fast triage to find the highest-priority, most actionable items.
  - The task focuses on linguistic/structural signals (presence of steps, timelines, named actors) rather than deep factual verification.
  - Agency follow-up decisions are initially heuristic and will be validated by a human decision-maker later.
  - Inputs are in a language and cultural context the model was trained on and not highly domain- or jurisdiction-specific.
- **Not Recommended For:** 
  - Submissions require legal interpretation, regulatory compliance checks, or answers with binding legal consequences.
  - Actionability depends on confidential, internal agency constraints (budgets, staffing, ongoing negotiations) the model cannot access.
  - Content is highly technical (e.g., engineering specs, complex scientific data) where feasibility requires expert validation.
  - Inputs are vague, heavily implied, or culturally coded such that explicit asks are intentionally indirect.
  - High risk or safety-critical decisions hinge on the evaluation (the model should not be the sole decision-maker).
  - Data contains sensitive personal information or proprietary details where automated processing would create privacy or legal risks.

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
  - Preference for explicit, formal phrasing — the judge favors clear, declarative asks and may underrate implicit or narrative appeals.
  - Technical/domain knowledge bias — the judge may overvalue asks that use technical terminology familiar to it and underrate legitimate community-language requests.
  - Format bias — numbered steps, bullets, or headings are treated as more actionable regardless of substantive feasibility.
  - Agency-remit assumption bias — the judge assumes common agency responsibilities and may mis-evaluate asks that depend on unusual interagency arrangements.
  - Recency/data bias — examples or phrasing similar to recent training data are judged more confidently than novel formulations.
  - Urgency/impact conflation — requests framed as urgent or severe may be scored as more actionable even if they lack concrete steps.
  - Language and cultural bias — idiomatic or culturally specific ways of asking may be misinterpreted as vague or non-actionable.
- **Task Misalignment Risks:** 
  - Over-emphasizing specificity/actionability may deprioritize important high-level or systemic feedback that requires strategic assessment rather than immediate operational steps.
  - Focusing exclusively on what an agency can do risks missing equity or representational concerns embedded in submissions (whose voice is being elevated).
  - The axis may encourage users to rewrite submissions to 'game' escalation by adding superficial steps or actors without real feasibility.
  - Narrow actionability assessment can ignore legal, ethical, or privacy constraints that make a specific ask inappropriate for escalation.
  - Prioritizing concrete asks might undervalue requests that are intentionally exploratory, signaling a need for stakeholder engagement rather than direct agency action.
  - Agency-centric evaluation may miss cross-jurisdictional solutions requiring coordination with other levels of government or external partners.
- **Failure Cases:** 
  - Flagging a passionate narrative that implies a clear request as non-actionable because the ask is not explicitly worded.
  - Scoring as actionable a draft that includes specific steps which are outside the agency's legal authority.
  - Overrating detailed but infeasible plans (e.g., requiring funding or legislation not available) as highly actionable.
  - Failing to detect that a submission's named "actor" is a private organization rather than an agency division, leading to incorrect escalation.
  - Penalizing concise strategic asks (e.g., asking for a review or study) as low-actionability despite being appropriate next steps.
  - Hallucinating plausible implementation steps or actors that the original submission did not request, creating false actionability signals.
  - Treating formatting cues (bullets, headings) as a proxy for quality and elevating poorly substantiated asks.
  - Missing timeline or resource constraints embedded implicitly (e.g., seasonal needs) and thus misjudging feasibility.
  - Underestimating culturally specific or community-framed requests because they lack bureaucratic language, causing inequitable downgrading.
  - Failing to surface potential legal/ethical blocks (privacy, safety) that render an otherwise specific ask inappropriate to escalate.

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

    description: ClassVar[str] = "**Specificity of Ask and Actionability** Clarity of concrete requests or decisions sought from the agency, with steps the agency can realistically take."

    def __init__(self, model: dspy.LM = DEFAULT_MODEL):
        super().__init__(
            name="Specificity_of_Ask_and_Actionability_gpt-5-mini",
            description="**Specificity of Ask and Actionability** Clarity of concrete requests or decisions sought from the agency, with steps the agency can realistically take.",
            axis="**Specificity of Ask and Actionability** Clarity of concrete requests or decisions sought from the agency, with steps the agency can realistically take.",
            model=model,
            task_description="Rank candidate policy feedback drafts for follow-up: given a citizen\u2019s submission, \n    determine which responses merit escalation to agency officials.",
            metric_card="provided",
            max_workers=32,
        )

    def __repr__(self):
        return f"Specificity_of_Ask_and_Actionability_gpt_5_mini_LLMJudge(model=dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv(\"OPENAI_API_KEY\")))"

