# Auto-generated metric file for Structure_and_organization_gpt-5-mini
import dspy
import os
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import GeneratedRefFreeLLMJudgeMetric
from typing import ClassVar

DEFAULT_MODEL = dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv("OPENAI_API_KEY"))

class Structure_and_organization_gpt_5_mini_LLMJudge(GeneratedRefFreeLLMJudgeMetric):
    """---
# Metric Card for Structure_and_organization_gpt-5-mini

**Structure and organization** Presence of headings, numbering, and well organized arguments that aid rapid review by officials.

## Metric Details

**Structure_and_organization_gpt-5-mini** is a **reference-free** LLM-as-a-Judge metric that prompts an LLM to rate a system output along a single, run-time-specified evaluation axis.
In this case the axis is `**Structure and organization** Presence of headings, numbering, and well organized arguments that aid rapid review by officials.`.

The prompt supplies:

1. **Task description** *d*
2. **Axis rubric** `**Structure and organization** Presence of headings, numbering, and well organized arguments that aid rapid review by officials.`
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
  - **Axis rubric** `**Structure and organization** Presence of headings, numbering, and well organized arguments that aid rapid review by officials.`
  - **Input text** *x*
  - **Output text** *y*
- **Outputs:**
  - Scalar score $\hat{s} \in \{1,2,3,4,5\}$

## Intended Use

- **Domain:** Document Triage / Text Evaluation
- **Tasks:** 
  - Policy feedback triage (escalation prioritization based on organization)
  - Evaluating presence and quality of headings, numbering, and sections
  - Scoring drafts against an organization/formatting rubric
  - Flagging submissions that need reformatting before review
  - Sorting submissions by readability and scannability for officials
  - Generating structured summaries and outlines of submissions
  - Quality control of public comments for adherence to submission templates
  - Assisting template and guideline design based on common structural errors
- **Best Suited For:** 
  - When documents are in a single common language (e.g., English) and use conventional document structure.
  - High-volume intake where consistent, automated triage of organizational quality is needed to save reviewer time.
  - When a clear, written rubric for structure/organization exists and can be applied automatically.
  - When surface features (headings, numbering, short executive summaries) are reliable proxies for faster human review.
  - When submissions are electronic and preserve formatting (not plain-text pasted without markup).
- **Not Recommended For:** 
  - When the primary evaluation requires substantive policy judgment, legal interpretation, or factual verification rather than organizational assessment.
  - When documents use domain-specific or nonstandard organizational conventions that the judge has not been primed on.
  - When submissions contain sensitive, classified, or legally privileged content requiring human oversight.
  - When the input is noisy or formatting is lost (e.g., scanned images or OCR with errors), making structure detection unreliable.
  - When the goal is to detect author intent, rhetorical strategy, or nuanced tone rather than explicit structural elements.

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
  - Surface-cue bias: over-reliance on explicit headings/numbering as proxies for quality or urgency.
  - Template bias: preference for formats seen frequently in training data, penalizing nonstandard but valid structures.
  - Language proficiency bias: penalizing well-reasoned submissions from non-native speakers that use less conventional organization.
  - Cultural/document-convention bias: favoring Western/English document structures (e.g., 'Executive Summary', numbered lists).
  - Length bias: equating longer, more sectioned responses with higher importance regardless of content.
  - Formatting-gaming bias: susceptible to being fooled by submissions that mimic structure without substantive content.
  - Training-data recency bias: preferring contemporary online formatting norms over legacy or domain-specific formats.
  - Punctuation/tokenization bias: misinterpreting or missing structure due to unusual punctuation or encoding artifacts.
- **Task Misalignment Risks:** 
  - Escalation misprioritization: promoting well-structured but low-impact items while downgrading brief but urgent submissions.
  - Content-neglect risk: failing to surface policy-significant content that lacks formal headings or numbering.
  - Overfitting to format: optimizing for checklistable features rather than officials' actual review needs.
  - Accessibility misalignment: preferring visual layout over accessibility-friendly formats (e.g., plain text or assistive-technology-friendly phrasing).
  - Agency-format mismatch: enforcing structures not used or valued by the target agency, causing false negatives/positives.
  - Equity misalignment: systematically disadvantaging communities or submitters who use different rhetorical conventions.
- **Failure Cases:** 
  - False positive: a perfectly structured draft with headings and numbered recommendations that contains inaccurate or irrelevant policy content is escalated.
  - False negative: a terse, poorly sectioned citizen report contains urgent legal violations but is assigned low priority because of weak structure.
  - Missed subtext: nuanced or coded complaints (e.g., brief anecdotal reports) lack explicit headings and are overlooked.
  - Formatting noise: OCR errors, pasted email threads, or embedded images break heading detection and produce low structure scores.
  - Gaming the evaluator: submitters add superficial headings and numbered lists to manipulate escalation outcomes.
  - Inconsistent scoring: model rates similar structures differently across contexts due to sensitivity to wording or tokenization.
  - Multilingual failure: non-English submissions use valid organizational cues unfamiliar to the model and are mis-rated.
  - Attachment blindness: important content conveyed in attachments, screenshots, or linked documents is ignored because the judge evaluates only the visible structure.
  - Overconfidence/hallucination: model invents headings or reorganizes content in its internal representation and rates structure inaccurately.
  - Edge-format failure: proposals using domain-specific formats (e.g., regulatory templates, legal citations) are misinterpreted as unstructured.

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

    description: ClassVar[str] = "**Structure and organization** Presence of headings, numbering, and well organized arguments that aid rapid review by officials."

    def __init__(self, model: dspy.LM = DEFAULT_MODEL):
        super().__init__(
            name="Structure_and_organization_gpt-5-mini",
            description="**Structure and organization** Presence of headings, numbering, and well organized arguments that aid rapid review by officials.",
            axis="**Structure and organization** Presence of headings, numbering, and well organized arguments that aid rapid review by officials.",
            model=model,
            task_description="Rank candidate policy feedback drafts for follow-up: given a citizen\u2019s submission, \n    determine which responses merit escalation to agency officials.",
            metric_card="provided",
            max_workers=32,
        )

    def __repr__(self):
        return f"Structure_and_organization_gpt_5_mini_LLMJudge(model=dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv(\"OPENAI_API_KEY\")))"

