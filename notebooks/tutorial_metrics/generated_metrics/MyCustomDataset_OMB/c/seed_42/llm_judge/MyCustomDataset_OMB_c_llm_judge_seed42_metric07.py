# Auto-generated metric file for Structure_and_Completeness_of_Correspondence_gpt-5-mini
import dspy
import os
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import GeneratedRefFreeLLMJudgeMetric
from typing import ClassVar

DEFAULT_MODEL = dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv("OPENAI_API_KEY"))

class Structure_and_Completeness_of_Correspondence_gpt_5_mini_LLMJudge(GeneratedRefFreeLLMJudgeMetric):
    """---
# Metric Card for Structure_and_Completeness_of_Correspondence_gpt-5-mini

**Structure and Completeness of Correspondence** The draft includes standard elements such as date subject greeting body closing and signature for clarity.

## Metric Details

**Structure_and_Completeness_of_Correspondence_gpt-5-mini** is a **reference-free** LLM-as-a-Judge metric that prompts an LLM to rate a system output along a single, run-time-specified evaluation axis.
In this case the axis is `**Structure and Completeness of Correspondence** The draft includes standard elements such as date subject greeting body closing and signature for clarity.`.

The prompt supplies:

1. **Task description** *d*
2. **Axis rubric** `**Structure and Completeness of Correspondence** The draft includes standard elements such as date subject greeting body closing and signature for clarity.`
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
  - **Axis rubric** `**Structure and Completeness of Correspondence** The draft includes standard elements such as date subject greeting body closing and signature for clarity.`
  - **Input text** *x*
  - **Output text** *y*
- **Outputs:**
  - Scalar score $\hat{s} \in \{1,2,3,4,5\}$

## Intended Use

- **Domain:** Text Evaluation / Correspondence Triage
- **Tasks:** 
  - Triage citizen policy feedback drafts for escalation readiness
  - Checklist validation of correspondence elements (date, subject, greeting, body, closing, signature)
  - Ranking multiple draft responses by structural completeness and clarity
  - Generating recommendations to complete or standardize missing correspondence elements
  - Quality assurance for templates and staff responses before agency review
- **Best Suited For:** 
  - When evaluating English-language drafts that follow conventional formal correspondence norms
  - When clear checklist criteria for structure and completeness are provided to the judge
  - When large batches of candidate drafts must be filtered quickly for basic readiness
  - When the goal is to surface drafts that are missing standard elements rather than to judge policy content
  - When consistent, repeatable evaluation is needed to reduce reviewer workload
- **Not Recommended For:** 
  - When escalation decisions require legal interpretation, policy analysis, or domain expertise about substantive merits
  - When drafts contain highly sensitive personal data or classified information that require human-only handling
  - When correspondence is in languages or cultural formats the model was not trained on or where norms differ markedly
  - When nuanced prioritization depends on context not present in the text (e.g., ongoing case history, external timelines)
  - When final high-stakes decisions are required without human review or accountability

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
  - Formal-structure bias: preferring drafts that look 'complete' over those with substantive but informal content.
  - Template bias: overvaluing boilerplate or templated replies because they contain the expected fields.
  - Length bias: equating longer, fully formatted drafts with higher quality or importance.
  - Cultural/linguistic bias: penalizing correspondence that follows different cultural salutations, date formats, or signature norms.
  - Channel bias: favoring email-like formats and underweighting social media, text, or attachment-based submissions.
  - Training-data bias: leaning on patterns from formal written corpora and undervaluing authentic citizen language.
  - Accessibility bias: failing to recognize alternative indicators of completeness (e.g., attached form or scanned signature) and penalizing them.
  - Politeness/formality bias: equating polite, formal wording with legitimacy or urgency.
- **Task Misalignment Risks:** 
  - Axis narrowness: the axis ignores substantive correctness, urgency, legal risk, or policy relevance, so it can rank structurally perfect but irrelevant drafts higher.
  - False security: treating presence of all fields as a signal the draft is safe to escalate when it may contain harmful or confidential content.
  - Context loss: not accounting for agency-specific formatting conventions and escalation thresholds, leading to inconsistent prioritization across agencies.
  - Overfitting to format: encouraging reviewers to favour edits that add missing elements rather than addressing content that merits escalation.
  - Multimodal misalignment: the axis assumes text fields are present and readable and therefore misaligns for inputs with attachments, images, or non-text evidence.
  - Language mismatch: evaluating correspondence completeness in one language or convention can misclassify valid submissions in other languages or dialects.
- **Failure Cases:** 
  - High-priority terse report missed: a short emergency report lacking formal greeting but containing urgent policy-relevant info is ranked low and not escalated.
  - Boilerplate escalation: a generic, well-structured template is flagged as high priority and escalated despite no substantive issue.
  - Attachment blindness: a draft that relies on an attached form or scanned letter is downgraded because the model doesn't detect completeness within attachments.
  - Cultural format false negative: a submission using non-Western date formats or salutations is marked incomplete and deprioritized.
  - Signature misread: electronically signed or single-word signatures are flagged as missing/signature invalid and the draft is under-ranked.
  - Multilingual failure: a valid, fully complete draft in a less-represented language is judged incomplete due to unfamiliar salutations or structure.
  - Format spoofing: malicious actors craft superficially complete drafts to game the axis and push low-quality or harmful content through escalation.
  - Incorrect element inference: the model hallucinates the presence or absence of elements (e.g., claiming a subject line exists when it does not) and misranks drafts.

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

    description: ClassVar[str] = "**Structure and Completeness of Correspondence** The draft includes standard elements such as date subject greeting body closing and signature for clarity."

    def __init__(self, model: dspy.LM = DEFAULT_MODEL):
        super().__init__(
            name="Structure_and_Completeness_of_Correspondence_gpt-5-mini",
            description="**Structure and Completeness of Correspondence** The draft includes standard elements such as date subject greeting body closing and signature for clarity.",
            axis="**Structure and Completeness of Correspondence** The draft includes standard elements such as date subject greeting body closing and signature for clarity.",
            model=model,
            task_description="Rank candidate policy feedback drafts for follow-up: given a citizen\u2019s submission, \n    determine which responses merit escalation to agency officials.",
            metric_card="provided",
            max_workers=32,
        )

    def __repr__(self):
        return f"Structure_and_Completeness_of_Correspondence_gpt_5_mini_LLMJudge(model=dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv(\"OPENAI_API_KEY\")))"

