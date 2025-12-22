# Auto-generated metric file for Stakeholder_Impact_Magnitude_gpt-5-mini
import dspy
import os
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import GeneratedRefFreeLLMJudgeMetric
from typing import ClassVar

DEFAULT_MODEL = dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv("OPENAI_API_KEY"))

class Stakeholder_Impact_Magnitude_gpt_5_mini_LLMJudge(GeneratedRefFreeLLMJudgeMetric):
    """---
# Metric Card for Stakeholder_Impact_Magnitude_gpt-5-mini

**Stakeholder Impact Magnitude** Scope and severity of anticipated impacts on public health, environment, or regulated entities, including who is affected and how much.

## Metric Details

**Stakeholder_Impact_Magnitude_gpt-5-mini** is a **reference-free** LLM-as-a-Judge metric that prompts an LLM to rate a system output along a single, run-time-specified evaluation axis.
In this case the axis is `**Stakeholder Impact Magnitude** Scope and severity of anticipated impacts on public health, environment, or regulated entities, including who is affected and how much.`.

The prompt supplies:

1. **Task description** *d*
2. **Axis rubric** `**Stakeholder Impact Magnitude** Scope and severity of anticipated impacts on public health, environment, or regulated entities, including who is affected and how much.`
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
  - **Axis rubric** `**Stakeholder Impact Magnitude** Scope and severity of anticipated impacts on public health, environment, or regulated entities, including who is affected and how much.`
  - **Input text** *x*
  - **Output text** *y*
- **Outputs:**
  - Scalar score $\hat{s} \in \{1,2,3,4,5\}$

## Intended Use

- **Domain:** Policy analysis / Regulatory triage
- **Tasks:** 
  - Triage citizen submissions and candidate feedback drafts by potential public impact
  - Prioritize which responses should be escalated to agency officials based on anticipated stakeholder impact magnitude
  - Summarize affected stakeholder groups and likely severity for each candidate draft
  - Produce concise escalation rationales and recommended next actions (e.g., immediate review, monitoring, desk review)
  - Flag mentions of vulnerable populations, regulatory noncompliance, or irreversible harms
  - Apply consistent heuristics to enable reproducible ranking across many items
- **Best Suited For:** 
  - When submissions include concrete descriptions of harms, exposures, or affected populations that can be extracted and compared
  - When the evaluation goal is scalable triage across many items rather than definitive expert adjudication
  - When agency staff will perform final review and need prioritized shortlists with clear rationales
  - When consistent, documented heuristics or scoring rubrics are provided to guide the model’s judgments
  - When time-sensitive screening is required to surface likely high-impact issues for rapid human attention
- **Not Recommended For:** 
  - When the determination requires technical laboratory, epidemiological, or site-specific data the model cannot access or validate
  - When legal or compliance decisions with binding consequences are being made without human expert review
  - When submissions are highly ambiguous, contradictory, or lack factual detail needed to assess impact magnitude
  - When inputs include confidential, classified, or highly sensitive datasets that must not be processed by external models
  - When quantitative risk modeling or probabilistic exposure assessment is required rather than qualitative triage

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
  - Availability bias: over-weighting recent, dramatic, or well-documented incidents seen in training data versus less visible but high-impact harms.
  - Urban/visibility bias: prioritizing impacts in urban or high-profile communities while underestimating harms in rural or marginalized areas.
  - Socioeconomic bias: underestimating impacts on low-income or politically marginalized populations because their harms are less documented.
  - Confirmation bias: preferring interpretations that fit typical narratives of harm (e.g., pollution always equals acute illness) and overlooking atypical pathways.
  - Data-source bias: over-reliance on formal reports and underweighting community testimony or nontraditional evidence.
  - Regulatory bias: assuming the existence or adequacy of regulatory controls and thereby underestimating impact where enforcement is weak.
  - Language bias: poorer performance on submissions in less-represented languages or using nonstandard terminology, leading to misestimation of impact.
  - Severity-over-probability bias: overweighting high-severity but low-likelihood outcomes relative to more probable moderate harms.
  - Anchoring bias: early cues in the submission setting a premature magnitude estimate that is insufficiently updated with new evidence.
  - Expertise bias: failing to recognize technical claims outside the model’s domain knowledge or misinterpreting specialist data.
  - Geographic bias: favoring regions that appear in the model's training data and underprioritizing others.
  - Stakeholder salience bias: giving disproportionate weight to organized stakeholder groups with louder voices compared to diffuse or less-organized affected populations.
- **Task Misalignment Risks:** 
  - Conflating urgency with magnitude: treating immediately visible or time-sensitive issues as automatically high-magnitude even if affected population is small.
  - Equating volume of complaints with impact: assuming many submissions imply larger magnitude rather than reflecting mobilization or awareness.
  - Prioritizing plausibility over distributive effects: focusing on whether harm is plausible instead of who bears disproportionate burdens.
  - Substituting policy preference for impact assessment: letting normative views about regulation shape magnitude judgments instead of impartial impact metrics.
  - Overemphasis on evidence formality: failing to escalate credible community-reported high-magnitude harms because they lack formal measurements.
  - Ignoring jurisdictional context: recommending escalation to agency officials without checking if the agency has authority or capacity to act.
  - Reducing multidimensional impacts to a single scalar: losing nuance by compressing public health, economic, and environmental harms into one score.
  - Neglecting cumulative or indirect impacts: missing large magnitude from many small interacting harms because the axis focuses on immediate direct impacts.
  - Treating stakeholder identity too broadly: failing to distinguish within-group heterogeneity and thereby misjudging who is most affected.
  - Undervaluing uncertainty: either deferring escalation because of uncertainty (false negatives) or escalating reflexively because of precaution (false positives).
- **Failure Cases:** 
  - False negative: missing a widespread contamination event because community reports lack formal lab results and the model underweights anecdotal evidence.
  - False positive: escalating a localized complaint of nuisance odor as high-magnitude despite limited exposure and no health outcomes, causing unnecessary resource diversion.
  - Misidentifying affected stakeholders: labeling an economic impact on a small industry as public-health magnitude while ignoring significant downstream community health effects.
  - Underestimating cumulative harm: failing to recognize that multiple small emissions across facilities collectively create large ecosystem damage.
  - Overestimating severity from ambiguous language: interpreting phrases like 'everyone is getting sick' literally and flagging as high magnitude without corroboration.
  - Failure to detect vulnerable populations: not recognizing that a facility near a school disproportionately exposes children, thus under-ranking magnitude.
  - Being gamed: manipulated or coordinated submissions that exaggerate scope lead the judge to escalate inappropriately.
  - Technical misinterpretation: misreading exposure metrics or units (e.g., µg/m3 vs mg/m3) and thus misestimating magnitude.
  - Jurisdictional error: recommending escalation to the wrong agency because the judge failed to map the regulatory authority, delaying appropriate response.
  - Equity blindspot: systematically underprioritizing harms that primarily affect marginalized groups due to sparse documentation in training data.
  - Temporal failure: failing to account for delayed or long-latency harms (e.g., chronic disease), resulting in under-escalation.
  - Overconfidence in low-quality evidence: producing definitive escalation recommendations despite high uncertainty in the underlying claims.

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

    description: ClassVar[str] = "**Stakeholder Impact Magnitude** Scope and severity of anticipated impacts on public health, environment, or regulated entities, including who is affected and how much."

    def __init__(self, model: dspy.LM = DEFAULT_MODEL):
        super().__init__(
            name="Stakeholder_Impact_Magnitude_gpt-5-mini",
            description="**Stakeholder Impact Magnitude** Scope and severity of anticipated impacts on public health, environment, or regulated entities, including who is affected and how much.",
            axis="**Stakeholder Impact Magnitude** Scope and severity of anticipated impacts on public health, environment, or regulated entities, including who is affected and how much.",
            model=model,
            task_description="Rank candidate policy feedback drafts for follow-up: given a citizen\u2019s submission, \n    determine which responses merit escalation to agency officials.",
            metric_card="provided",
            max_workers=32,
        )

    def __repr__(self):
        return f"Stakeholder_Impact_Magnitude_gpt_5_mini_LLMJudge(model=dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv(\"OPENAI_API_KEY\")))"

