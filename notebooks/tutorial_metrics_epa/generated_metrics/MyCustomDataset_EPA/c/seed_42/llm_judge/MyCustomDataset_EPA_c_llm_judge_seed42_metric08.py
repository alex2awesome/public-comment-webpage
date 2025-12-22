# Auto-generated metric file for Timing_and_Urgency_gpt-5-mini
import dspy
import os
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import GeneratedRefFreeLLMJudgeMetric
from typing import ClassVar

DEFAULT_MODEL = dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv("OPENAI_API_KEY"))

class Timing_and_Urgency_gpt_5_mini_LLMJudge(GeneratedRefFreeLLMJudgeMetric):
    """---
# Metric Card for Timing_and_Urgency_gpt-5-mini

**Timing and Urgency** Presence of deadlines, imminent risks, or time-sensitive windows that increase the need for rapid escalation.

## Metric Details

**Timing_and_Urgency_gpt-5-mini** is a **reference-free** LLM-as-a-Judge metric that prompts an LLM to rate a system output along a single, run-time-specified evaluation axis.
In this case the axis is `**Timing and Urgency** Presence of deadlines, imminent risks, or time-sensitive windows that increase the need for rapid escalation.`.

The prompt supplies:

1. **Task description** *d*
2. **Axis rubric** `**Timing and Urgency** Presence of deadlines, imminent risks, or time-sensitive windows that increase the need for rapid escalation.`
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
  - **Axis rubric** `**Timing and Urgency** Presence of deadlines, imminent risks, or time-sensitive windows that increase the need for rapid escalation.`
  - **Input text** *x*
  - **Output text** *y*
- **Outputs:**
  - Scalar score $\hat{s} \in \{1,2,3,4,5\}$

## Intended Use

- **Domain:** Triage / Decision Support (Policy feedback prioritization and escalation)
- **Tasks:** 
  - Policy feedback prioritization for agency escalation
  - Triage of citizen submissions by urgency
  - Flagging imminent public-safety or regulatory risks
  - Prioritizing time-sensitive policy follow-ups and deadlines
  - Routing submissions to appropriate officials based on urgency
  - Annotating submissions with suggested escalation timelines
- **Best Suited For:** 
  - Submissions include explicit temporal markers (exact dates, deadlines, or event windows) that the model can parse.
  - Messages contain clear urgency language or explicit statements of imminent harm or impact.
  - Structured inputs include metadata (submission timestamp, sender location, affected population) that disambiguate timing.
  - Use cases where the model's output will be a preliminary triage recommendation reviewed by humans.
  - Environments with well-defined escalation rules or thresholds that the model can be calibrated to follow.
  - High-volume intake systems where automated prioritization reduces human workload for clearly time-sensitive items.
- **Not Recommended For:** 
  - Situations requiring legally binding determinations about deadlines or obligations without human legal review.
  - Submissions with implicit or culturally nuanced urgency that require domain-expert interpretation.
  - High-stakes emergencies (medical, criminal, national security) where immediate human intervention or verified sensor data is required.
  - Inputs missing crucial context or metadata (no timestamps, unclear affected parties), increasing risk of misclassification.
  - Adversarial or low-quality texts that use hyperbolic language, which can cause false positives for urgency.
  - Use as a sole decision-maker for escalation in contexts demanding auditability or chain-of-custody evidence without human oversight.

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
  - Recency bias: favoring submissions that reference recent events or dates over equally urgent but less time-stamped issues.
  - Availability bias: over-weighting vivid or emotionally framed submissions that are easier to recall or that mirror training examples.
  - Anchoring on explicit dates: under-recognizing urgency when no explicit deadline is stated but a time-sensitive outcome is implied.
  - Severity amplification: equating forceful language or alarming metaphors with genuine imminent risk.
  - Conservatism bias: defaulting to non-escalation when uncertainty exists about timing to avoid false positives.
  - Domain familiarity bias: better handling of timing cues in well-represented domains (e.g., elections, public health) and worse in niche regulatory areas.
  - Jurisdictional bias: misinterpreting or ignoring local/timezone-specific deadlines and legal timelines.
  - Optimism bias: assuming delays are acceptable or that stakeholders will adapt, reducing escalation likelihood.
  - Temporal anchoring to model knowledge cutoff: discounting contemporary time-sensitive contexts that emerged after training.
  - Formality bias: privileging formally-worded submissions (which often include explicit deadlines) over grassroots or colloquial reports that may be urgent.
- **Task Misalignment Risks:** 
  - Over-prioritizing explicit deadlines: escalating only when a literal date/time appears and missing implied windows that nevertheless require rapid action.
  - Conflating urgency with importance: treating short-term issues as high priority even when their overall policy impact is low, and vice versa.
  - Neglecting resource constraints: recommending escalation without assessing whether agency can act in time or whether escalation would be actionable.
  - Misreading domain-specific timing: failing to differentiate between legal filing deadlines, operational lead times, and policy feedback schedules.
  - Favoring speed over verification: pushing rapid escalation at the expense of verifying whether the timeline claim is accurate or malicious.
  - Single-axis tunnel vision: optimizing only for timing/urgency and ignoring other critical axes (feasibility, provenance, harm) that influence whether escalation is appropriate.
  - Static thresholding: applying a one-size-fits-all urgency threshold across varied policy areas and stakeholder types rather than adapting to context.
  - Over-escalation risk: generating many escalations for borderline timing concerns, creating noise and reviewer fatigue among officials.
- **Failure Cases:** 
  - False negative — implied imminent risk missed: a submission describes symptoms of an imminent public safety hazard (e.g., contaminants appearing before a festival) without an explicit deadline, and the model fails to escalate.
  - False positive — rhetorical urgency escalated: a complaint uses hyperbolic language ('this will destroy us tomorrow') but refers to long-term policy concerns and the model escalates unnecessarily.
  - Missed legal deadline: the model misinterprets jurisdictional filing rules and fails to escalate a submission that mentions an imminent statutory deadline.
  - Timezone/timeframe error: the model ignores or miscalculates timezones and flags a window as passed or not imminent when it is, or vice versa.
  - Ambiguous temporal language failure: phrases like 'soon' or 'in a few weeks' are interpreted inconsistently, producing incorrect prioritization.
  - Adversarial manipulation: someone crafts feed text with fake-sounding deadlines to game the system and cause needless escalations.
  - Delayed escalation due to low confidence: the model hesitates or returns 'needs human review' for clearly urgent items, causing harmful delay.
  - Overload cascade: excessive false positives overwhelm agency reviewers, causing genuine urgent items to be missed in the noise.
  - Domain transfer failure: model handles election-related deadlines well but fails for environmental permitting timelines that require specific lead times.
  - Reliance on metadata: the judge escalates based on submission timestamps or metadata anomalies rather than content, leading to errors if metadata is missing or spoofed.
  - Failure to recognize multi-step timing: missing chained deadlines (e.g., public comment period followed by a short decision window) and not escalating when cumulative timing makes action urgent.
  - Hallucinated deadlines: the model invents specific dates or windows that are not present in the submission, causing inappropriate escalation.

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

    description: ClassVar[str] = "**Timing and Urgency** Presence of deadlines, imminent risks, or time-sensitive windows that increase the need for rapid escalation."

    def __init__(self, model: dspy.LM = DEFAULT_MODEL):
        super().__init__(
            name="Timing_and_Urgency_gpt-5-mini",
            description="**Timing and Urgency** Presence of deadlines, imminent risks, or time-sensitive windows that increase the need for rapid escalation.",
            axis="**Timing and Urgency** Presence of deadlines, imminent risks, or time-sensitive windows that increase the need for rapid escalation.",
            model=model,
            task_description="Rank candidate policy feedback drafts for follow-up: given a citizen\u2019s submission, \n    determine which responses merit escalation to agency officials.",
            metric_card="provided",
            max_workers=32,
        )

    def __repr__(self):
        return f"Timing_and_Urgency_gpt_5_mini_LLMJudge(model=dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv(\"OPENAI_API_KEY\")))"

