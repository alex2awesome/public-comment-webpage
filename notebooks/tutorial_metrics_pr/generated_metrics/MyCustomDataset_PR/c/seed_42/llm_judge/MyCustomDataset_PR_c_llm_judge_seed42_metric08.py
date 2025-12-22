# Auto-generated metric file for Novelty_and_materiality_of_issues_gpt-5-mini
import dspy
import os
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import GeneratedRefFreeLLMJudgeMetric
from typing import ClassVar

DEFAULT_MODEL = dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv("OPENAI_API_KEY"))

class Novelty_and_materiality_of_issues_gpt_5_mini_LLMJudge(GeneratedRefFreeLLMJudgeMetric):
    """---
# Metric Card for Novelty_and_materiality_of_issues_gpt-5-mini

**Novelty and materiality of issues** Introduction of new information, risks, impacts, or perspectives that could meaningfully affect deliberations.

## Metric Details

**Novelty_and_materiality_of_issues_gpt-5-mini** is a **reference-free** LLM-as-a-Judge metric that prompts an LLM to rate a system output along a single, run-time-specified evaluation axis.
In this case the axis is `**Novelty and materiality of issues** Introduction of new information, risks, impacts, or perspectives that could meaningfully affect deliberations.`.

The prompt supplies:

1. **Task description** *d*
2. **Axis rubric** `**Novelty and materiality of issues** Introduction of new information, risks, impacts, or perspectives that could meaningfully affect deliberations.`
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
  - **Axis rubric** `**Novelty and materiality of issues** Introduction of new information, risks, impacts, or perspectives that could meaningfully affect deliberations.`
  - **Input text** *x*
  - **Output text** *y*
- **Outputs:**
  - Scalar score $\hat{s} \in \{1,2,3,4,5\}$

## Intended Use

- **Domain:** Policy triage and evaluation / Text classification
- **Tasks:** 
  - Triage citizen feedback for escalation to agency officials
  - Rank candidate policy response drafts by novelty and materiality
  - Flag submissions that introduce new factual evidence or risks
  - Summarize why a draft is materially novel for inclusion in briefing notes
  - Prioritize comments for subject-matter expert review based on potential policy impact
- **Best Suited For:** 
  - When background materials (current draft policy, prior public comments, regulatory text) are available for comparison so the model can gauge novelty.
  - When the goal is to filter or rank many free-text submissions quickly to surface a manageable set for human review.
  - When novelty can be inferred from language patterns, described impacts, or explicit new evidence rather than from technical experimental validation.
  - When a clear, operational rubric for “novelty” and “materiality” is supplied and the model’s scoring is used as an initial triage step rather than the final decision.
  - When human-in-the-loop verification is available for items the model marks as high-priority or borderline.
- **Not Recommended For:** 
  - When assessing highly technical, scientific, or legal claims that require primary-source validation, domain-specific measurement, or expert adjudication.
  - When submissions contain sensitive, classified, or legally privileged information that must not be processed by third-party models.
  - When there is no access to up-to-date policy context or prior feedback, making novelty judgments unreliable.
  - When the decision is high-stakes and must be defensible solely on expert evidence rather than model-inferred relevance.
  - When adversarial actors deliberately craft language to game novelty detection—without an adversarial-robust pipeline, the model may be misled.

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
  - Novelty bias: overvaluing phrasing or new terminology even when the underlying point is not substantively different from prior submissions.
  - Recency bias: giving disproportionate weight to information that appears current or framed as 'new' without verifying whether it’s already on the record.
  - Confirmation bias toward prominent framing: favoring issues that align with commonly known risks or the model’s training distribution.
  - Length and specificity bias: treating longer or more technical-sounding submissions as more likely to be material regardless of actual relevance.
  - Source-credibility bias: down-weighting submissions without explicit citations even if they convey novel, actionable observations.
  - Domain expertise bias: misjudging materiality in specialized technical or legal areas due to lack of deep domain knowledge.
  - Linguistic bias: penalizing submissions from non-native speakers or casual writers that present important but tersely stated new information.
  - Political/content bias: unintentionally favoring or disfavoring claims that align with politically salient narratives present in training data.
  - Format bias: preferring structured or formally presented items (tables, numbers) as more material than narrative descriptions.
  - Overcorrection bias: to avoid missing important items, the model may escalate borderline novel claims, increasing false positives.
- **Task Misalignment Risks:** 
  - Prioritizing novelty over practical relevance, causing escalation of technically new but decision-irrelevant details.
  - Failing to recognize cumulative materiality from many similar, non-novel submissions because each individually lacks novelty.
  - Treating phrasing novelty (new wording) as substantive novelty, leading to misaligned escalations.
  - Neglecting agency-specific criteria or statutory/legal thresholds for materiality because the model lacks access to the full docket or internal priorities.
  - Overemphasizing evidentiary form (citations, numbers) rather than plausible real-world impact, disadvantaging credible anecdotal harms.
  - Potentially flagging speculative or hypothetical risks as material when agencies require demonstrated or probable impacts.
  - Misaligning with risk tolerance: escalating items that are novel but low-probability/high-uncertainty contrary to agency preference to focus on higher-likelihood issues.
  - Missing cross-cutting or indirect impacts (e.g., equity implications) because the axis focuses narrowly on explicit new facts.
  - Failing to account for representativeness — promoting unusual but idiosyncratic reports while ignoring systemic but familiar issues.
  - Confusing novelty of language or rhetorical framing with novelty of underlying policy-relevant information.
- **Failure Cases:** 
  - False positive escalation: uniquely worded but irrelevant claims are flagged and routed to officials, wasting review time.
  - False negative omission: familiar wording hides a genuinely new factual claim or evidence, and the judge fails to escalate it.
  - Hallucinated novelty: the model invents or overstates that a submission contains 'new data' or a novel mechanism when none exists.
  - Over-aggregation failure: the judge cannot detect that many non-novel submissions together indicate a material trend and therefore fails to escalate.
  - Source-misinterpretation: downranking material submissions that are poorly cited or anecdotal despite high plausibility of real harms.
  - Adversarial phrasing exploitation: actors game the system by adding novel-sounding but irrelevant phrasing to trigger escalations.
  - Domain-mismatch misses: technical legal or scientific nuance leads to misclassification (e.g., treating a minor procedural change as material or missing a subtle statutory hook).
  - Urgency-misread: the judge fails to flag urgent safety concerns because they are phrased neutrally or lack dramatic language.
  - Equity blind spot: novel impacts on marginalized groups are under-flagged because the language is indirect or the judge lacks contextual knowledge of disproportionate effects.
  - Over-escalation burden: consistent over-flagging of marginally novel items creates review fatigue among officials, degrading the signal-to-noise ratio.

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

    description: ClassVar[str] = "**Novelty and materiality of issues** Introduction of new information, risks, impacts, or perspectives that could meaningfully affect deliberations."

    def __init__(self, model: dspy.LM = DEFAULT_MODEL):
        super().__init__(
            name="Novelty_and_materiality_of_issues_gpt-5-mini",
            description="**Novelty and materiality of issues** Introduction of new information, risks, impacts, or perspectives that could meaningfully affect deliberations.",
            axis="**Novelty and materiality of issues** Introduction of new information, risks, impacts, or perspectives that could meaningfully affect deliberations.",
            model=model,
            task_description="Rank candidate policy feedback drafts for follow-up: given a citizen\u2019s submission, \n    determine which responses merit escalation to agency officials.",
            metric_card="provided",
            max_workers=32,
        )

    def __repr__(self):
        return f"Novelty_and_materiality_of_issues_gpt_5_mini_LLMJudge(model=dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv(\"OPENAI_API_KEY\")))"

