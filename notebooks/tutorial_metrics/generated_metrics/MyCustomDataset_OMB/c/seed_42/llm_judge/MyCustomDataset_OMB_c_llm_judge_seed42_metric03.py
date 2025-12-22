# Auto-generated metric file for Channel_and_Procedural_Appropriateness_gpt-5-mini
import dspy
import os
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import GeneratedRefFreeLLMJudgeMetric
from typing import ClassVar

DEFAULT_MODEL = dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv("OPENAI_API_KEY"))

class Channel_and_Procedural_Appropriateness_gpt_5_mini_LLMJudge(GeneratedRefFreeLLMJudgeMetric):
    """---
# Metric Card for Channel_and_Procedural_Appropriateness_gpt-5-mini

**Channel and Procedural Appropriateness** The draft uses the right mode for escalation rather than generic public comment venues and follows expected protocols.

## Metric Details

**Channel_and_Procedural_Appropriateness_gpt-5-mini** is a **reference-free** LLM-as-a-Judge metric that prompts an LLM to rate a system output along a single, run-time-specified evaluation axis.
In this case the axis is `**Channel and Procedural Appropriateness** The draft uses the right mode for escalation rather than generic public comment venues and follows expected protocols.`.

The prompt supplies:

1. **Task description** *d*
2. **Axis rubric** `**Channel and Procedural Appropriateness** The draft uses the right mode for escalation rather than generic public comment venues and follows expected protocols.`
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
  - **Axis rubric** `**Channel and Procedural Appropriateness** The draft uses the right mode for escalation rather than generic public comment venues and follows expected protocols.`
  - **Input text** *x*
  - **Output text** *y*
- **Outputs:**
  - Scalar score $\hat{s} \in \{1,2,3,4,5\}$

## Intended Use

- **Domain:** Policy evaluation / Administrative triage / Text classification
- **Tasks:** 
  - Triage candidate policy feedback drafts to decide escalation vs. public comment routing
  - Rank drafts by procedural correctness for escalation (recipient, channel, urgency, attachments)
  - Detect when a draft omits required procedural elements for escalation (e.g., legal clearance, confidentiality flags)
  - Validate that recommended escalation channels match an explicitly provided escalation matrix or checklist
  - Suggest minimally invasive edits to a draft to make it procedurally appropriate for escalation
- **Best Suited For:** 
  - When the agency provides a clear, written escalation matrix or checklist that the judge can use as criteria.
  - When candidate drafts are in the same language and reasonably well-structured (short to medium length).
  - When the evaluation primarily requires checking for presence/absence of standard procedural elements (recipient, justification, urgency, attachments).
  - When the information concerned is non-classified, non-sensitive personal data, and low to medium risk.
  - When the goal is to pre-screen and prioritize submissions to reduce human reviewer load, not to replace human decision-makers.
- **Not Recommended For:** 
  - When escalation decisions depend on internal, proprietary, or evolving agency SOPs that are not provided to the model.
  - For high-stakes legal or safety-critical escalations where regulatory, statutory, or liability interpretation is required.
  - When submissions contain sensitive personal data, classified information, or require confidentiality protections—use human review and secure systems instead.
  - Where multi-jurisdictional or highly specialized legal/regulatory knowledge determines the appropriate channel.
  - In adversarial contexts (e.g., actors deliberately trying to game escalation rules) without additional audit and human oversight.

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
  - Institutional-formality bias: favoring formal, official channels over informal or community-based routes even when the latter are more appropriate.
  - Availability/recency bias: recommending channels that the model or judge is most familiar with or recently encountered rather than the correct, possibly obscure, pathway.
  - Jurisdictional bias: assuming procedures that apply to one agency or country apply broadly to others.
  - Authority-deference bias: preferring escalation to higher-level officials or legal routes instead of mediation or internal agency mechanisms.
  - Technical/terminology bias: interpreting procedural appropriateness primarily through the presence of formal terms (e.g., 'FOIA', 'ombudsman') which may not indicate true fit.
  - Accessibility bias: undervaluing community accessibility and the citizen’s capacity to follow complex procedural steps, thus recommending impractical channels.
  - Risk-averse bias: defaulting to the most conservative route (legal counsel, inspectors) to avoid under-escalation even when lower-risk options suffice.
  - Privacy/over-protection bias: over-prioritizing confidentiality in ways that prevent necessary public transparency or vice versa.
- **Task Misalignment Risks:** 
  - Equating procedural correctness with overall value: the judge may favor drafts that use correct protocol language even when the core claim is weak or irrelevant.
  - Channel-only focus: the evaluation could ignore substantive factors (evidence strength, legal merits) and escalate solely because the channel seems right.
  - One-size-fits-all procedures: the judge might recommend a standardized protocol that clashes with agency-specific or issue-specific requirements.
  - Neglecting citizen preferences and constraints: recommending escalation paths the citizen cannot reasonably pursue (costly legal routes, time-consuming appeals).
  - Over-escalation as default: aligning with a conservative rubric that grades higher for escalation-minded drafts even when less formal responses are better policy.
  - Misinterpreting urgency: treating emotionally framed comments as requiring immediate formal escalation.
  - Confusing appropriate audience with correct channel: suggesting contact with high-level officials rather than the specialized program office that actually handles the issue.
  - Lack of legal/regulatory nuance: failing to recognize when escalation could violate whistleblower protections, confidentiality rules, or privacy laws.
- **Failure Cases:** 
  - False positive escalation: marking a draft as escalation-worthy because it uses formal language, leading to unnecessary burden on agency staff.
  - False negative omission: failing to flag a subtle but urgent safety or legal compliance issue because it lacks dramatic wording.
  - Wrong-channel recommendation: advising public comment or social media rather than a confidential inspector or hotline needed for safety/abuse reports.
  - Nonexistent or outdated procedure: instructing steps or contacts that don’t exist for the target agency or have changed.
  - Privacy violation suggestion: recommending escalation that would expose sensitive personal data or harm a whistleblower’s protections.
  - Procedural checklist without evidence: producing an escalation template that lacks the specific facts or documents required to make the referral actionable.
  - Burdening the citizen: suggesting expensive or complex legal/administrative avenues the citizen cannot realistically follow.
  - Overloading officials: recommending escalation for high-volume, low-merit submissions, contributing to triage failure at the agency.

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

    description: ClassVar[str] = "**Channel and Procedural Appropriateness** The draft uses the right mode for escalation rather than generic public comment venues and follows expected protocols."

    def __init__(self, model: dspy.LM = DEFAULT_MODEL):
        super().__init__(
            name="Channel_and_Procedural_Appropriateness_gpt-5-mini",
            description="**Channel and Procedural Appropriateness** The draft uses the right mode for escalation rather than generic public comment venues and follows expected protocols.",
            axis="**Channel and Procedural Appropriateness** The draft uses the right mode for escalation rather than generic public comment venues and follows expected protocols.",
            model=model,
            task_description="Rank candidate policy feedback drafts for follow-up: given a citizen\u2019s submission, \n    determine which responses merit escalation to agency officials.",
            metric_card="provided",
            max_workers=32,
        )

    def __repr__(self):
        return f"Channel_and_Procedural_Appropriateness_gpt_5_mini_LLMJudge(model=dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv(\"OPENAI_API_KEY\")))"

