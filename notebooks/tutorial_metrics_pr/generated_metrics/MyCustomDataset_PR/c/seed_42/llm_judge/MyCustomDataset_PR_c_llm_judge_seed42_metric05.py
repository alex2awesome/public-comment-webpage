# Auto-generated metric file for Document_genre_suitability_gpt-5-mini
import dspy
import os
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import GeneratedRefFreeLLMJudgeMetric
from typing import ClassVar

DEFAULT_MODEL = dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv("OPENAI_API_KEY"))

class Document_genre_suitability_gpt_5_mini_LLMJudge(GeneratedRefFreeLLMJudgeMetric):
    """---
# Metric Card for Document_genre_suitability_gpt-5-mini

**Document genre suitability** Whether the item is a formal comment, analysis, or summary versus email routing, headers, contact lists, or boilerplate.

## Metric Details

**Document_genre_suitability_gpt-5-mini** is a **reference-free** LLM-as-a-Judge metric that prompts an LLM to rate a system output along a single, run-time-specified evaluation axis.
In this case the axis is `**Document genre suitability** Whether the item is a formal comment, analysis, or summary versus email routing, headers, contact lists, or boilerplate.`.

The prompt supplies:

1. **Task description** *d*
2. **Axis rubric** `**Document genre suitability** Whether the item is a formal comment, analysis, or summary versus email routing, headers, contact lists, or boilerplate.`
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
  - **Axis rubric** `**Document genre suitability** Whether the item is a formal comment, analysis, or summary versus email routing, headers, contact lists, or boilerplate.`
  - **Input text** *x*
  - **Output text** *y*
- **Outputs:**
  - Scalar score $\hat{s} \in \{1,2,3,4,5\}$

## Intended Use

- **Domain:** Text Classification / Document Genre Identification (applied to policy feedback triage)
- **Tasks:** 
  - Triage policy feedback drafts for escalation
  - Classify documents into genre labels: formal comment, analysis, summary, email header, contact list, boilerplate
  - Detect presence of substantive policy claims or recommendations
  - Filter out routing/administrative metadata from textual content
  - Produce brief justifications for genre labels to support human reviewers
  - Prioritize items for human review based on genre + content signals
- **Best Suited For:** 
  - Input texts are digital, machine-readable, and primarily prose (no scanned images or embedded attachments).
  - Documents contain clear discourse cues (arguments, recommendations, citations) and are of short-to-moderate length (a few paragraphs to a few pages).
  - There is a need for high-throughput initial triage to reduce manual review workload. 
  - Labeled examples or a small rule set are available to align the judge’s genre definitions with agency conventions.
  - The goal is to separate substantive feedback from administrative/boilerplate material, not to make final legal or policy determinations.
  - Outputs will be used to surface candidates for human escalation rather than to fully automate consequential decisions.
- **Not Recommended For:** 
  - Inputs are scanned documents, images, or attachments requiring OCR that may lose layout/context. 
  - Critical decisions require provenance verification (authorship, representativeness), legal interpretation, or regulatory compliance judgments beyond textual genre. 
  - Documents are extremely short, fragmentary, or contain only terse headers where genre cues are ambiguous. 
  - Text relies heavily on domain-specific legal jargon or novel formats not represented in the judge’s training data without adaptation. 
  - There is a requirement for complete explainability/auditability at a fine-grained legal standard (the model’s probabilistic judgments may be insufficient). 
  - Content includes sensitive PII or classified material for which automated triage is not permitted without specialized safeguards.

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
  - Length bias: preferring longer documents as substantive and shorter ones as non-substantive, even when short items are high-value summaries or urgent points.
  - Format bias: over-weighting visual/format cues (headings, bullet lists, salutations) and misclassifying plain-text substantive arguments as boilerplate.
  - Template-matching bias: labeling anything matching common boilerplate or form-language as non-substantive regardless of whether it contains unique policy points.
  - Keyword bias: relying on presence/absence of policy keywords or legal citations and missing novel phrasing of substantive points.
  - Modality bias: treating attachments or embedded content (e.g., images, tables, PDFs) as less likely to contain substantive feedback if not parsed.
  - Source bias: expecting certain agency-specific structures and penalizing submissions that use different conventions (cross-agency or international variations).
  - Language/cultural bias: underperforming on non-standard dialects, non-English inputs, or culturally different document conventions.
  - Recency/training bias: favoring genres seen frequently in training data and failing on rarer or emerging formats.
  - Conservatism bias: defaulting to classifying ambiguous content as non-escalation to avoid false positives, potentially missing important items.
  - Overconfidence bias: producing overly certain genre judgments without surfacing uncertainty or needing human review for borderline cases.
- **Task Misalignment Risks:** 
  - Axis ignores urgency/impact: documents that are non-standard in genre but contain urgent or high-impact policy input may be deprioritized by a pure genre filter.
  - Content vs. form mismatch: the axis focuses on form rather than substance, so substantive comments embedded in headers, footers, or email threads can be missed.
  - Loss of nuance about partial relevance: multi-part submissions (routing metadata + substantive comment) may be forced into a single genre label, losing escalation-relevant signal.
  - Misaligns with stakeholder identity: the axis doesn’t capture who submitted the comment (e.g., subject-matter expert vs general public), which affects escalation decisions.
  - Agency-process misfit: different agencies require different routing policies; a single genre rubric may not match each agency’s escalation rules.
  - Attachment handling mismatch: classifying the primary message but ignoring attachments with substantive analysis creates misalignment with the true escalation need.
  - Contextual dependency ignored: prior correspondence, docket numbers, or linked documents that change genre relevance are not considered by a narrow genre classifier.
  - Evaluation threshold mismatch: the axis does not define a threshold for escalation (e.g., minimal substantive content required), so operational decisions may diverge from the rubric.
- **Failure Cases:** 
  - A short, high-value executive summary is labeled as 'non-substantive' because of length-based heuristics and not escalated.
  - A long routing email that contains a pasted formal comment is labeled 'email/routing' and the embedded comment is ignored.
  - A submission includes a mix of boilerplate and a single paragraph of policy recommendations; the model discards the whole item as boilerplate.
  - An attached PDF with detailed analysis is not parsed and the plain-text email is classified as non-escalation.
  - Templates with customized insertions are classified as boilerplate due to partial template-match, losing the unique substantive insertions.
  - Non-English or code-switched comments are misclassified as headers/metadata because the judge lacks robust multilingual pattern recognition.
  - Adversarial formatting (e.g., inserting substantive text inside a signature block) fools the classifier into skipping escalation.
  - The model overreports confidence and provides no 'uncertain' flag for borderline documents that require human review.
  - Inconsistent decisions across similar inputs because of sensitivity to punctuation/whitespace or minor formatting differences.
  - Agency-specific header conventions (e.g., docket IDs placed in the body) are misinterpreted as non-substantive and dropped.
  - The judge prioritizes keyword presence and escalates documents with policy terms used in an unrelated boilerplate context, causing false positives.
  - A comment containing legal citations but no recommendations is escalated despite being informational only, increasing reviewer burden.

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

    description: ClassVar[str] = "**Document genre suitability** Whether the item is a formal comment, analysis, or summary versus email routing, headers, contact lists, or boilerplate."

    def __init__(self, model: dspy.LM = DEFAULT_MODEL):
        super().__init__(
            name="Document_genre_suitability_gpt-5-mini",
            description="**Document genre suitability** Whether the item is a formal comment, analysis, or summary versus email routing, headers, contact lists, or boilerplate.",
            axis="**Document genre suitability** Whether the item is a formal comment, analysis, or summary versus email routing, headers, contact lists, or boilerplate.",
            model=model,
            task_description="Rank candidate policy feedback drafts for follow-up: given a citizen\u2019s submission, \n    determine which responses merit escalation to agency officials.",
            metric_card="provided",
            max_workers=32,
        )

    def __repr__(self):
        return f"Document_genre_suitability_gpt_5_mini_LLMJudge(model=dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv(\"OPENAI_API_KEY\")))"

