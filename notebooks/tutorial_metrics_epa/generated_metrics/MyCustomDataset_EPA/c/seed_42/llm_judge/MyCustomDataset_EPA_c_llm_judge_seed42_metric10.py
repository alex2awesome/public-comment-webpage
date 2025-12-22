# Auto-generated metric file for Clarity_Structure_and_Procedural_Completeness_gpt-5-mini
import dspy
import os
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import GeneratedRefFreeLLMJudgeMetric
from typing import ClassVar

DEFAULT_MODEL = dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv("OPENAI_API_KEY"))

class Clarity_Structure_and_Procedural_Completeness_gpt_5_mini_LLMJudge(GeneratedRefFreeLLMJudgeMetric):
    """---
# Metric Card for Clarity_Structure_and_Procedural_Completeness_gpt-5-mini

**Clarity Structure and Procedural Completeness** Organization, readability, civility, correct docket identifiers, accessible attachments, and contact details that facilitate agency action.

## Metric Details

**Clarity_Structure_and_Procedural_Completeness_gpt-5-mini** is a **reference-free** LLM-as-a-Judge metric that prompts an LLM to rate a system output along a single, run-time-specified evaluation axis.
In this case the axis is `**Clarity Structure and Procedural Completeness** Organization, readability, civility, correct docket identifiers, accessible attachments, and contact details that facilitate agency action.`.

The prompt supplies:

1. **Task description** *d*
2. **Axis rubric** `**Clarity Structure and Procedural Completeness** Organization, readability, civility, correct docket identifiers, accessible attachments, and contact details that facilitate agency action.`
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
  - **Axis rubric** `**Clarity Structure and Procedural Completeness** Organization, readability, civility, correct docket identifiers, accessible attachments, and contact details that facilitate agency action.`
  - **Input text** *x*
  - **Output text** *y*
- **Outputs:**
  - Scalar score $\hat{s} \in \{1,2,3,4,5\}$

## Intended Use

- **Domain:** Administrative triage / Text evaluation for policy feedback
- **Tasks:** 
  - Triage citizen policy feedback for escalation
  - Detect missing or malformed docket identifiers and metadata
  - Assess organization and readability (headings, clear ask, logical flow)
  - Evaluate civility and tone and flag abusive or non-constructive language
  - Check presence and basic usability of contact details (email, phone, mailing address)
  - Identify references to attachments and flag accessibility/description problems
  - Recommend next procedural steps (escalate, request clarification, reject, or redact)
- **Best Suited For:** 
  - Submissions provided as plain text or structured fields (title, body, attachments) where metadata can be parsed automatically
  - Feedback that includes explicit docket numbers, dates, or contact fields that can be pattern-checked
  - High-volume triage needs where rapid, consistent prioritization of clearly actionable items is required
  - Situations where the goal is to identify procedural omissions or formatting problems rather than adjudicate technical correctness
  - When non-sensitive content is evaluated (no need to handle verified PII or authentication)
- **Not Recommended For:** 
  - Cases that require validating external resources (confirming that a link resolves or that a docket exists) because the model has no external access
  - High-stakes legal or policy determinations requiring subject-matter experts or legal counsel
  - Situations requiring authentication of submitter identity or verification of attachments' provenance
  - Extremely technical policy feedback where correctness depends on domain expertise (e.g., detailed engineering specifications, clinical protocols)
  - Poorly formatted, ambiguous, or heavily redacted submissions lacking sufficient context for reliable automated judgment

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
  - Formality bias: prefers submissions that follow formal grammar and organization, disadvantaging non-native speakers or informal community styles.
  - Civility-as-quality bias: equates polite tone with suitability for escalation and may downrank substantive but impassioned feedback.
  - Format-pattern bias: overweights familiar docket/contact formats seen in training data and penalizes unusual but valid formats.
  - English-language bias: performs better on standard English and may mis-evaluate submissions in other languages or dialects.
  - Socioeconomic/tech access bias: favors submissions that include attachments or structured metadata, penalizing those who cannot attach files or use certain technologies.
  - Training-data recency bias: follows formatting conventions common in its training set and may miss newer procedural requirements or docket identifier formats.
  - Confirmation bias in verification: when uncertain, the model may infer that identifiers or attachments are correct based on partial matches rather than flagging uncertainty.
- **Task Misalignment Risks:** 
  - Checklist over-optimization: the model may prioritize meeting the axis checklist (format, attachments, contact) over substantive policy relevance, leading to escalation of procedurally complete but irrelevant drafts.
  - Escalation vs triage mismatch: the model could conflate 'procedurally complete' with 'requires immediate agency official attention', misdirecting limited staff resources.
  - Privacy and PII risk: focusing on contact completeness could encourage escalation of sensitive personal data that should instead be redacted or handled differently.
  - Inability to externally validate: because it cannot verify docket numbers or attachment accessibility, the model may mark items as complete or correct when they are not.
  - Civility gatekeeping: using civility as a primary filter could exclude critical perspectives that use forceful language, misaligning with public comment inclusion goals.
  - Over-reliance on surface signals: the axis emphasizes readability and structure, which may cause the model to miss high-value but poorly formatted submissions that merit escalation.
- **Failure Cases:** 
  - False positive escalation: a grammatically perfect but substantively irrelevant submission is escalated because it meets all checklist items.
  - False negative omission: a substantive, time-sensitive comment written in a conversational or nonstandard style is not escalated due to poor formatting or tone.
  - Hallucinated attachments: the model asserts that attachments are accessible or adequate based on textual mention, when attachments are missing or corrupted.
  - Docket identifier mis-evaluation: the model accepts or rejects docket numbers based on format heuristics, failing to detect legitimate new formats or subtle errors.
  - Missed contact info in nonstandard form: contact details written in phonewords, images, or obfuscated formats are overlooked and the submission is marked procedurally incomplete.
  - Tonal misclassification: passionate advocacy is labeled 'uncivil' and deprioritized even when it contains actionable procedural information.
  - Inconsistent scoring: identical issues are scored differently across similar drafts because the model's heuristics are sensitive to superficial phrasing differences.
  - Failure to flag sensitive PII: the model escalates submissions containing unnecessary personal data without recognizing privacy concerns or required redaction steps.
  - Language and OCR errors: submissions with OCR artifacts or multilingual content are misread, producing incorrect assessments of attachments and docket references.
  - Overconfidence in uncertain judgments: the model provides definite judgments about procedural completeness despite low internal confidence or lack of external verification.

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

    description: ClassVar[str] = "**Clarity Structure and Procedural Completeness** Organization, readability, civility, correct docket identifiers, accessible attachments, and contact details that facilitate agency action."

    def __init__(self, model: dspy.LM = DEFAULT_MODEL):
        super().__init__(
            name="Clarity_Structure_and_Procedural_Completeness_gpt-5-mini",
            description="**Clarity Structure and Procedural Completeness** Organization, readability, civility, correct docket identifiers, accessible attachments, and contact details that facilitate agency action.",
            axis="**Clarity Structure and Procedural Completeness** Organization, readability, civility, correct docket identifiers, accessible attachments, and contact details that facilitate agency action.",
            model=model,
            task_description="Rank candidate policy feedback drafts for follow-up: given a citizen\u2019s submission, \n    determine which responses merit escalation to agency officials.",
            metric_card="provided",
            max_workers=32,
        )

    def __repr__(self):
        return f"Clarity_Structure_and_Procedural_Completeness_gpt_5_mini_LLMJudge(model=dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv(\"OPENAI_API_KEY\")))"

