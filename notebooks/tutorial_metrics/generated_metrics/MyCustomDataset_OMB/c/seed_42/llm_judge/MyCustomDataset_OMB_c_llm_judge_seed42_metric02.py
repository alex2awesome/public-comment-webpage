# Auto-generated metric file for Clear_Ask_and_Escalation_Rationale_gpt-5-mini
import dspy
import os
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import GeneratedRefFreeLLMJudgeMetric
from typing import ClassVar

DEFAULT_MODEL = dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv("OPENAI_API_KEY"))

class Clear_Ask_and_Escalation_Rationale_gpt_5_mini_LLMJudge(GeneratedRefFreeLLMJudgeMetric):
    """---
# Metric Card for Clear_Ask_and_Escalation_Rationale_gpt-5-mini

**Clear Ask and Escalation Rationale** The draft states specific requests or decisions sought and explains why escalation is warranted now.

## Metric Details

**Clear_Ask_and_Escalation_Rationale_gpt-5-mini** is a **reference-free** LLM-as-a-Judge metric that prompts an LLM to rate a system output along a single, run-time-specified evaluation axis.
In this case the axis is `**Clear Ask and Escalation Rationale** The draft states specific requests or decisions sought and explains why escalation is warranted now.`.

The prompt supplies:

1. **Task description** *d*
2. **Axis rubric** `**Clear Ask and Escalation Rationale** The draft states specific requests or decisions sought and explains why escalation is warranted now.`
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
  - **Axis rubric** `**Clear Ask and Escalation Rationale** The draft states specific requests or decisions sought and explains why escalation is warranted now.`
  - **Input text** *x*
  - **Output text** *y*
- **Outputs:**
  - Scalar score $\hat{s} \in \{1,2,3,4,5\}$

## Intended Use

- **Domain:** Policy triage / Text evaluation / Administrative decision support
- **Tasks:** 
  - Escalation triage for citizen submissions
  - Ranking candidate feedback drafts by clarity of ask
  - Identifying missing escalation rationale or timing justifications
  - Summarizing the specific requests and why escalation is needed
  - Flagging drafts that require urgent human review
  - Generating concise commentary to help editors improve clarity
- **Best Suited For:** 
  - Inputs are written in clear, grammatical language where asks and rationales are present or clearly absent.
  - A small-to-moderate set of candidate drafts must be ranked quickly using a consistent rubric.
  - Escalation criteria are well defined and do not rely on institutional knowledge or confidential context.
  - High-volume, low-to-medium-stakes workflows where automated triage speeds human review.
  - When the goal is to detect and report missing explicit asks or weak timing justifications rather than to make final policy decisions.
- **Not Recommended For:** 
  - Situations requiring verification of factual claims, legal interpretation, or confidential case details that the model cannot access or validate.
  - High-stakes escalation decisions involving political sensitivity, safety, legal liability, or ethics where human judgment and institutional accountability are required.
  - Inputs that are ambiguous, heavily implied, culturally or politically nuanced, or adversarial, where subtle reading and context are essential.
  - Workflows requiring fully auditable, provenance-backed decisions for regulatory compliance without human review.
  - Very long threads or multi-document contexts where essential context is distributed and must be reconciled before judging escalation rationale.

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
  - Preference for explicit linguistic markers (e.g., 'please escalate', 'we request') which causes implicit or nuanced asks to be underrated.
  - Tendency to equate specificity with quality, thereby favoring verbose, checklist-style drafts over concise but clear requests.
  - Recency or saliency bias toward urgent language (e.g., 'now', 'immediately') that may overweight dramatic but non-critical submissions.
  - Domain bias where the evaluator better recognizes escalation rationale in high-profile areas (e.g., safety/security) than in technical, regulatory, or bureaucratic topics.
  - Cultural and linguistic bias against indirect or polite forms of asking common in some languages and cultures, leading to underestimation of clarity.
  - Conservatism or risk-averse bias prompting under-escalation when uncertainty exists, due to training on safe/responsible responses.
  - Overreliance on surface features (deadlines, named officials) which can penalize legitimate strategic asks that intentionally omit such details.
  - Bias toward drafting norms typical of formal institutions, disadvantaging grassroots or informal citizen voices that still merit escalation.
- **Task Misalignment Risks:** 
  - Treating explicit ask-and-rationale as the sole criterion and ignoring other escalation-relevant factors such as potential harm, legal requirements, or contextual urgency.
  - Conflating clarity with merit: promoting drafts that ask clearly but for inappropriate or low-priority actions, while suppressing ambiguous drafts that point to serious systemic issues.
  - Overemphasizing immediate temporal language and thereby escalating issues that are time-stamped but low-impact while de-prioritizing high-impact long-term matters.
  - Failing to account for confidentiality or safety reasons why a citizen might omit explicit requests or details, and penalizing such cautious wording.
  - Applying a generic checklist across policy domains without adjusting for domain-specific escalation thresholds or required evidence.
  - Prioritizing linguistic form over stakeholder alignment, leading to escalation of well-written but misaligned requests and ignoring poorly worded but substantively important concerns.
  - Using escalation as a binary outcome rather than graded, which can force false all-or-nothing decisions instead of recommending further information-gathering.
  - Relying on model training priors (e.g., professional memo styles) that mismatch common citizen writing and thus mis-evaluate civic submissions.
- **Failure Cases:** 
  - False negative: The model fails to escalate a draft that implies a clear ask by context (e.g., reporting imminent public harm) because the citizen used indirect language.
  - False positive: The model recommends escalation for a draft that contains urgent-sounding language but is a rhetorical complaint without an actionable ask.
  - Missed domain signal: The model ignores technical indicators (e.g., regulatory citations, incident codes) that make an escalation rationale compelling because it lacks domain-specific knowledge.
  - Context loss: The model evaluates an excerpt without context and downgrades a draft that in full would include the necessary escalation rationale or authority contact.
  - Overfitting to format: The model elevates drafts that include template phrases like 'we request' even when the requested action is vague or impractical.
  - Ambiguity handling failure: The model cannot recommend a graded next step (e.g., request more info) and instead either escalates prematurely or not at all.
  - Adversarial input: A submitter uses sensationalist wording to game the model into escalation, resulting in resource misallocation.
  - Privacy/safety oversight: The model suggests escalation that would expose sensitive personal data or jeopardize whistleblower safety because it doesn't flag confidentiality concerns.
  - Hallucination: The model invents a rationale or urgency (e.g., claiming a deadline exists) when none is present and bases an escalation recommendation on that hallucination.
  - Cultural misread: The model undervalues directness-missing but culturally normative phrasing and therefore fails to escalate legitimate requests from certain populations.

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

    description: ClassVar[str] = "**Clear Ask and Escalation Rationale** The draft states specific requests or decisions sought and explains why escalation is warranted now."

    def __init__(self, model: dspy.LM = DEFAULT_MODEL):
        super().__init__(
            name="Clear_Ask_and_Escalation_Rationale_gpt-5-mini",
            description="**Clear Ask and Escalation Rationale** The draft states specific requests or decisions sought and explains why escalation is warranted now.",
            axis="**Clear Ask and Escalation Rationale** The draft states specific requests or decisions sought and explains why escalation is warranted now.",
            model=model,
            task_description="Rank candidate policy feedback drafts for follow-up: given a citizen\u2019s submission, \n    determine which responses merit escalation to agency officials.",
            metric_card="provided",
            max_workers=32,
        )

    def __repr__(self):
        return f"Clear_Ask_and_Escalation_Rationale_gpt_5_mini_LLMJudge(model=dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv(\"OPENAI_API_KEY\")))"

