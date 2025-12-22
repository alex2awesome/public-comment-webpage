# Auto-generated metric file for Clarity_Structure_and_Procedural_Completeness_Rubric
import dspy
import os
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import GeneratedRefFreeLLMJudgeMetric
from typing import ClassVar

DEFAULT_MODEL = dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv("OPENAI_API_KEY"))

class Clarity_Structure_and_Procedural_Completeness_Rubric_LLMJudge(GeneratedRefFreeLLMJudgeMetric):
    """---
# Metric Card for Clarity_Structure_and_Procedural_Completeness_Rubric

**Clarity Structure and Procedural Completeness** Organization, readability, civility, correct docket identifiers, accessible attachments, and contact details that facilitate agency action.

## Metric Details

**Clarity_Structure_and_Procedural_Completeness_Rubric** is a **reference-free** LLM-as-a-Judge metric that prompts an LLM to rate a system output along a single, run-time-specified evaluation axis.
In this case the axis is `**Clarity Structure and Procedural Completeness** Organization, readability, civility, correct docket identifiers, accessible attachments, and contact details that facilitate agency action.`.

The prompt supplies:

1. **Task description** *d*
2. **Rubric** `**Clarity Structure and Procedural Completeness** Organization, readability, civility, correct docket identifiers, accessible attachments, and contact details that facilitate agency action.`
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

### Rubric Details

**Criteria:** **Clarity Structure and Procedural Completeness** Organization, readability, civility, correct docket identifiers, accessible attachments, and contact details that facilitate agency action.

#### Scoring Rubric

| Score | Description |
|-------|-------------|
| 1 | Score 1 — Deficient<br/>• Disorganized or largely unintelligible; no logical flow.<br/>• Lacks civility or uses hostile/abusive language.<br/>• No docket ID or rule identifier; or clearly wrong action referenced.<br/>• Mentions attachments but none provided or they are inaccessible (e.g., image-only scans without text, broken links).<br/>• No contact information (name/affiliation/email/phone) for follow-up.<br/>• Formatting severely impedes readability (garbled text, excessive errors). |
| 2 | Score 2 — Poor<br/>• Minimal structure; difficult to follow; arguments scattered.<br/>• Tone marginally civil or unnecessarily inflammatory.<br/>• Docket/rule reference is missing or ambiguous (e.g., incorrect number, only a vague topic).<br/>• Attachments referenced but incomplete, unlabeled, or not readily accessible/searchable.<br/>• Contact info is incomplete (e.g., name only, no way to reach submitter) or buried.<br/>• Formatting issues (missing headings, no pagination) hinder quick review. |
| 3 | Score 3 — Adequate<br/>• Generally readable with a basic structure (intro/body/conclusion), but limited headings or summary.<br/>• Civil tone.<br/>• Includes either the correct docket ID or a clear rule/title reference; minor inconsistencies possible.<br/>• Attachments provided and mostly accessible, though with some formatting/searchability issues or sparse labeling.<br/>• Provides at least one reliable contact method (email or phone) and a name/affiliation.<br/>• Minor formatting gaps (e.g., no page numbers, inconsistent citation style) but content is usable. |
| 4 | Score 4 — Strong<br/>• Well-organized: clear headings, logical flow, succinct paragraphs; possibly a brief summary of key points.<br/>• Consistently civil and professional.<br/>• Correct and specific docket ID and rule/title included in header or opening.<br/>• Attachments are clearly labeled, searchable (text-based), and referenced in-text; links work.<br/>• Complete contact block (name, affiliation, email, phone) and sign-off.<br/>• Clean formatting with pagination, dated letterhead or similar metadata; only minor, non-impeding omissions. |
| 5 | Score 5 — Exemplary<br/>• Exemplary structure and readability: executive summary, numbered sections, clear asks/recommendations.<br/>• Unambiguously civil, respectful, and audience-appropriate.<br/>• Precise procedural details: correct docket ID(s), rule title, date, and any relevant agency office; consistent throughout.<br/>• Attachments and exhibits are fully accessible (searchable PDFs), clearly titled and cited in-text; hyperlinks and citations resolve; page and exhibit numbers provided.<br/>• Complete, prominent contact details (name, role/affiliation, email, phone, address) and a clear signature/date.<br/>• Accessibility and usability best practices: plain language, clear formatting, consistent citation style, alt text or descriptions where applicable, making it immediately actionable for agency officials. |

### Inputs and Outputs
- **Inputs:**
  - **Task description** *d*
  - **Rubric** `**Clarity Structure and Procedural Completeness** Organization, readability, civility, correct docket identifiers, accessible attachments, and contact details that facilitate agency action.`
  - **Input text** *x*
  - **Output text** *y*
- **Outputs:**
  - Scalar score $\hat{s} \in \{1,2,3,4,5\}$

## Intended Use

- **Domain:** Document Triage / Text Evaluation
- **Tasks:** 
  - Triage public comments and citizen submissions for escalation to agency officials
  - Score and label drafts for clarity, organization, and civility against the provided rubric
  - Extract and validate presence/format of procedural identifiers (docket IDs, rule titles) from text
  - Detect and flag missing or incomplete contact information and produce a follow-up checklist
  - Assess attachments for accessibility when text-based content or searchable OCR is included
  - Generate short summary or executive-notes highlighting deficiencies that block agency action
  - Pre-screen submissions for potential procedural red flags (hostile tone, ambiguous docket references, inaccessible exhibits)
- **Best Suited For:** 
  - Input documents are provided as machine-readable text or searchable OCR (not image-only scans).
  - The rubric and expected docket ID formats are well-specified or follow consistent patterns the model can learn to recognize.
  - The goal is rapid triage and prioritization rather than final legal judgment.
  - Attachments are included inline or as clearly labeled text excerpts, enabling accessibility checks without external link fetching.
  - Human reviewers are available to act on items the model flags as high-priority or ambiguous.
- **Not Recommended For:** 
  - Attachments or evidence are only available via external links that the model cannot fetch or verify (broken links or remote files).
  - Submissions are image-only scans without OCR or have low-quality scans that impede text extraction.
  - Tasks require authoritative verification of docket numbers, filing status, or other facts that need access to agency databases or the internet.
  - High-stakes legal interpretation, official determinations, identity/authenticity verification, or decisions with regulatory liability are required without human oversight.
  - Multilingual inputs or domain-specific jargon without provided glossaries that might lead to tone and intent misclassification.

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
  - Formality bias: preferring professional, well-formatted language and downranking grassroots or informal but actionable submissions.
  - Surface-cue bias: overweighting explicit metadata (docket numbers, headers) over implicit or paraphrased procedural references.
  - Native-language bias: misjudging clarity or civility for non-native or dialectal phrasing as less readable or less civil.
  - Civility detection errors: failing to detect sarcasm, coded hostility, or polite but hostile framing and thus mis-scoring tone.
  - Attachment accessibility bias: penalizing attachments that are accessible to humans but not machine-readable (e.g., scanned PDFs without OCR).
  - Conservatism/leniency calibration bias: systematic tendency to score more harshly or leniently depending on training examples or default thresholds.
  - Template familiarity bias: rewarding submissions that follow common templates and penalizing novel but valid organizational formats.
  - Confirmation/anchoring bias: initial cues (e.g., a clear docket ID) anchoring the score and causing the model to overlook other deficiencies.
- **Task Misalignment Risks:** 
  - Over-prioritizing formatting over substance, causing highly substantive but informally presented comments to be under-escalated.
  - Penalizing legitimate privacy-conscious submissions that intentionally omit full contact details, thereby missing actionable policy signals.
  - Treating presence of a docket ID as sufficient for escalation even when the substantive content references the wrong proceeding or misunderstanding.
  - Equating machine-readable attachments with real-world accessibility and ignoring context where a human reviewer could readily access materials.
  - Reducing complex procedural nuance (e.g., comments intended for different agency office or multiple dockets) to a single score, losing important distinctions for escalation.
  - Favoring submissions that match bureaucratic expectations and marginalizing nontraditional stakeholders (community groups, oral testimony transcripts).
  - Applying rigid civility standards that could suppress escalation of urgent but emotionally charged citizen feedback.
  - Failing to account for redacted or anonymized submissions that are nevertheless delegable to officials for policy significance.
- **Failure Cases:** 
  - False negative: a clearly actionable submission is scored low because it uses informal structure and lacks a formal header.
  - False positive: a well-formatted template letter with incorrect docket ID is scored highly and escalated despite pointing to the wrong rule.
  - OCR/attachment failure: attachments are present as scanned images and are treatable as inaccessible, causing downgrading despite being readable to human reviewers.
  - Link resolution failure: the model marks hyperlinks as broken because it cannot follow or validate them, even though they work in a browser.
  - Civility misclassification: polite but critical language is flagged as hostile due to sarcasm or idiomatic expressions, lowering the civility score.
  - Contact detection error: the model misses contact information that is present but embedded in a signature image or unconventional location.
  - Docket parsing error: transposed digits or slight formatting differences cause the model to treat a correct docket reference as absent/incorrect.
  - Inconsistent scoring: nearly identical submissions receive different scores across runs due to nondeterministic model outputs or subtle prompt sensitivity.
  - Over-reliance on explicit headings: submissions with clear logical flow but without explicit headings are underrated relative to wordier submissions with headings.
  - Accessibility overreach: the model downgrades submissions for not including alt text or accessibility metadata even when the agency does not require it, causing unnecessary non-escalation.
  - Privacy-preserving submission missed: anonymous or redacted submissions with high policy relevance are rejected because contact details are missing.
  - Hallucinated content: the model asserts the presence of attachments or contact details that do not exist, creating incorrect recommendations for escalation.

## Related Metrics

- **Related Metrics:**
  - **LevenshteinDistance:** Levenshtein Distance measures the minimum number of single-character edits—insertions, deletions, or substitutions—required to transform one sequence into another.
  - **BARTScore:** BARTScore is a reference-based evaluation metric for text generation that formulates evaluation as a text generation task.
  - **PseudoPARENT:** **PseudoPARENT** is a *custom adaptation* of the PARENT metric for evaluating text generation from structured inputs.

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
            name="Clarity_Structure_and_Procedural_Completeness_Rubric",
            description="**Clarity Structure and Procedural Completeness** Organization, readability, civility, correct docket identifiers, accessible attachments, and contact details that facilitate agency action.",
            axis="**Clarity Structure and Procedural Completeness** Organization, readability, civility, correct docket identifiers, accessible attachments, and contact details that facilitate agency action.\n\nScoring Guidelines:\nScore 1: Score 1 \u2014 Deficient\n- Disorganized or largely unintelligible; no logical flow.\n- Lacks civility or uses hostile/abusive language.\n- No docket ID or rule identifier; or clearly wrong action referenced.\n- Mentions attachments but none provided or they are inaccessible (e.g., image-only scans without text, broken links).\n- No contact information (name/affiliation/email/phone) for follow-up.\n- Formatting severely impedes readability (garbled text, excessive errors).\nScore 2: Score 2 \u2014 Poor\n- Minimal structure; difficult to follow; arguments scattered.\n- Tone marginally civil or unnecessarily inflammatory.\n- Docket/rule reference is missing or ambiguous (e.g., incorrect number, only a vague topic).\n- Attachments referenced but incomplete, unlabeled, or not readily accessible/searchable.\n- Contact info is incomplete (e.g., name only, no way to reach submitter) or buried.\n- Formatting issues (missing headings, no pagination) hinder quick review.\nScore 3: Score 3 \u2014 Adequate\n- Generally readable with a basic structure (intro/body/conclusion), but limited headings or summary.\n- Civil tone.\n- Includes either the correct docket ID or a clear rule/title reference; minor inconsistencies possible.\n- Attachments provided and mostly accessible, though with some formatting/searchability issues or sparse labeling.\n- Provides at least one reliable contact method (email or phone) and a name/affiliation.\n- Minor formatting gaps (e.g., no page numbers, inconsistent citation style) but content is usable.\nScore 4: Score 4 \u2014 Strong\n- Well-organized: clear headings, logical flow, succinct paragraphs; possibly a brief summary of key points.\n- Consistently civil and professional.\n- Correct and specific docket ID and rule/title included in header or opening.\n- Attachments are clearly labeled, searchable (text-based), and referenced in-text; links work.\n- Complete contact block (name, affiliation, email, phone) and sign-off.\n- Clean formatting with pagination, dated letterhead or similar metadata; only minor, non-impeding omissions.\nScore 5: Score 5 \u2014 Exemplary\n- Exemplary structure and readability: executive summary, numbered sections, clear asks/recommendations.\n- Unambiguously civil, respectful, and audience-appropriate.\n- Precise procedural details: correct docket ID(s), rule title, date, and any relevant agency office; consistent throughout.\n- Attachments and exhibits are fully accessible (searchable PDFs), clearly titled and cited in-text; hyperlinks and citations resolve; page and exhibit numbers provided.\n- Complete, prominent contact details (name, role/affiliation, email, phone, address) and a clear signature/date.\n- Accessibility and usability best practices: plain language, clear formatting, consistent citation style, alt text or descriptions where applicable, making it immediately actionable for agency officials.\n",
            model=model,
            task_description="Rank candidate policy feedback drafts for follow-up: given a citizen\u2019s submission, \n    determine which responses merit escalation to agency officials.",
            metric_card="provided",
            max_workers=32,
        )

    def __repr__(self):
        return f"Clarity_Structure_and_Procedural_Completeness_Rubric_LLMJudge(model=dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv(\"OPENAI_API_KEY\")))"

