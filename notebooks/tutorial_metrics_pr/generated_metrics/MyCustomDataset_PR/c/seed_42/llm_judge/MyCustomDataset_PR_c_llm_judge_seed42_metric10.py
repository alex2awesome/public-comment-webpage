# Auto-generated metric file for Signal_to_noise_ratio_gpt-5-mini
import dspy
import os
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import GeneratedRefFreeLLMJudgeMetric
from typing import ClassVar

DEFAULT_MODEL = dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv("OPENAI_API_KEY"))

class Signal_to_noise_ratio_gpt_5_mini_LLMJudge(GeneratedRefFreeLLMJudgeMetric):
    """---
# Metric Card for Signal_to_noise_ratio_gpt-5-mini

**Signal to noise ratio** Proportion of substantive content relative to extraneous headers, signatures, duplication, or formatting artifacts.

## Metric Details

**Signal_to_noise_ratio_gpt-5-mini** is a **reference-free** LLM-as-a-Judge metric that prompts an LLM to rate a system output along a single, run-time-specified evaluation axis.
In this case the axis is `**Signal to noise ratio** Proportion of substantive content relative to extraneous headers, signatures, duplication, or formatting artifacts.`.

The prompt supplies:

1. **Task description** *d*
2. **Axis rubric** `**Signal to noise ratio** Proportion of substantive content relative to extraneous headers, signatures, duplication, or formatting artifacts.`
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
  - **Axis rubric** `**Signal to noise ratio** Proportion of substantive content relative to extraneous headers, signatures, duplication, or formatting artifacts.`
  - **Input text** *x*
  - **Output text** *y*
- **Outputs:**
  - Scalar score $\hat{s} \in \{1,2,3,4,5\}$

## Intended Use

- **Domain:** Text classification / Content triage
- **Tasks:** 
  - Triage citizen feedback for escalation
  - Filter and label substantive vs. extraneous content
  - Extract candidate policy-relevant excerpts for reviewer attention
  - Score or rank drafts by signal-to-noise ratio
  - Preprocess submissions for human review or downstream summarization
- **Best Suited For:** 
  - Input submissions are primarily in English and reasonably well-formed (few OCR errors or corrupted encoding).
  - There is a clear, operational definition of what counts as ‘substantive’ (examples or annotation guidelines available).
  - Submissions are moderate to long in length where boilerplate, signatures, and quoted threads are present and detectable.
  - Large-scale automated triage is needed to prioritize a subset of items for human review.
  - You want consistent, repeatable SNR scoring and automatic extraction of likely policy-relevant passages.
- **Not Recommended For:** 
  - Inputs contain heavy OCR artifacts, malformed encodings, or numerous non-textual artifacts that impede reliable parsing.
  - Material is in languages or dialects the model handles poorly or mixes many languages without clear markers.
  - Determining policy relevance requires deep domain expertise or legal judgement that goes beyond signal/noise identification.
  - High-stakes escalation decisions that cannot tolerate false negatives without human oversight.
  - Substantive content is extremely implicit or embedded in metaphors or coded language where surface cues are unreliable.

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
  - Format bias: favoring responses with clean structure, headings, or bullet lists even if those contain little substantive content.
  - Length bias: equating longer text with higher signal, disadvantaging concise but important submissions.
  - Template bias: discounting or ignoring content that resembles standard templates or legal boilerplate even when those templates contain critical legal claims.
  - Language/style bias: penalizing nonstandard grammar, dialects, or translations that appear noisy though content is substantive.
  - Header/signature bias: treating salutations, signatures, and address blocks as pure noise and discarding substantive information contained there.
  - Duplication bias: automatically down-weighting repeated phrases or quoted threads that may nonetheless indicate emphasis or chronology important to escalation.
  - OCR/artifact bias: misclassifying OCR errors, line breaks, or special characters as noise, losing true signal especially from scanned attachments.
  - Cultural formatting bias: preferring Western-centric document conventions (subject lines, concise abstracts) over other valid formats.
  - Citations-as-noise bias: counting dense legal citations or references as low signal because they disrupt prose or appear mechanical.
  - Length-of-paragraph bias: treating long paragraphs with embedded nuance as noisy because they lack visual formatting cues.
- **Task Misalignment Risks:** 
  - Overemphasis on signal-to-noise may deprioritize highly relevant but brief submissions (e.g., urgent tip or legal complaint).
  - Focusing on visible noise metrics can miss contextual signals needed for escalation, such as sender credibility, attachments, or prior correspondence.
  - Ranking by SNR alone may escalate polished but low-impact feedback while ignoring messy but high-impact reports.
  - The judge may fail to consider agency-specific thresholds (legal thresholds, statutory relevance) and thus mis-rank policy-critical content.
  - Signal heuristics can be gamed (e.g., adversaries adding boilerplate to appear substantive or inserting noise to avoid detection), undermining triage integrity.
  - Cross-lingual and translation issues may cause systemic underestimation of signal for non-primary languages, reducing equitable escalation.
  - Treating repeated quoted threads as noise may break chronological context necessary to assess escalation urgency.
  - Narrow axis use could encourage human reviewers to rely solely on automated SNR scores instead of reading for real-world relevance.
- **Failure Cases:** 
  - Substantive content hidden in an email thread's header (e.g., a one-line clarification) is removed as 'noise' and the draft is not escalated.
  - A concise whistleblower tip (three sentences) is ranked very low because of short length despite high policy relevance.
  - Scanned PDF with OCR errors yields garbled tokens and the judge marks it as low-signal even though the original contained detailed allegations.
  - Legal complaint using boilerplate statutory language is marked as 'template noise' and deprioritized despite constituting a valid legal claim.
  - Quotations and repeated content from multiple respondents are collapsed as duplication, losing evidence of widespread concern that should trigger escalation.
  - Submissions in less-common languages or with translated structure are mis-scored due to formatting or punctuation differences, reducing their priority.
  - Deliberately obfuscated malicious input (lots of headers, fake signatures) is scored as high-signal because of polished formatting, leading to false positives.
  - Embedded attachments or links with substantive reports are ignored because the judge only evaluates visible inline text.
  - Important numbered evidence embedded inside a signature block is discarded as non-substantive.
  - Threshold miscalibration causes a cluster of mid-signal but important messages to be consistently ignored, creating blind spots in escalation coverage.

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

    description: ClassVar[str] = "**Signal to noise ratio** Proportion of substantive content relative to extraneous headers, signatures, duplication, or formatting artifacts."

    def __init__(self, model: dspy.LM = DEFAULT_MODEL):
        super().__init__(
            name="Signal_to_noise_ratio_gpt-5-mini",
            description="**Signal to noise ratio** Proportion of substantive content relative to extraneous headers, signatures, duplication, or formatting artifacts.",
            axis="**Signal to noise ratio** Proportion of substantive content relative to extraneous headers, signatures, duplication, or formatting artifacts.",
            model=model,
            task_description="Rank candidate policy feedback drafts for follow-up: given a citizen\u2019s submission, \n    determine which responses merit escalation to agency officials.",
            metric_card="provided",
            max_workers=32,
        )

    def __repr__(self):
        return f"Signal_to_noise_ratio_gpt_5_mini_LLMJudge(model=dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv(\"OPENAI_API_KEY\")))"

