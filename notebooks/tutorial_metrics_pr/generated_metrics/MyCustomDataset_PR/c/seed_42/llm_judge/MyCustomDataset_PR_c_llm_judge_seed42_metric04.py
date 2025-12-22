# Auto-generated metric file for Clarity_and_coherence_gpt-5-mini
import dspy
import os
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import GeneratedRefFreeLLMJudgeMetric
from typing import ClassVar

DEFAULT_MODEL = dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv("OPENAI_API_KEY"))

class Clarity_and_coherence_gpt_5_mini_LLMJudge(GeneratedRefFreeLLMJudgeMetric):
    """---
# Metric Card for Clarity_and_coherence_gpt-5-mini

**Clarity and coherence** Readability and logical flow of prose versus garbled OCR, placeholders, or fragmented content.

## Metric Details

**Clarity_and_coherence_gpt-5-mini** is a **reference-free** LLM-as-a-Judge metric that prompts an LLM to rate a system output along a single, run-time-specified evaluation axis.
In this case the axis is `**Clarity and coherence** Readability and logical flow of prose versus garbled OCR, placeholders, or fragmented content.`.

The prompt supplies:

1. **Task description** *d*
2. **Axis rubric** `**Clarity and coherence** Readability and logical flow of prose versus garbled OCR, placeholders, or fragmented content.`
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
  - **Axis rubric** `**Clarity and coherence** Readability and logical flow of prose versus garbled OCR, placeholders, or fragmented content.`
  - **Input text** *x*
  - **Output text** *y*
- **Outputs:**
  - Scalar score $\hat{s} \in \{1,2,3,4,5\}$

## Intended Use

- **Domain:** Text Evaluation / Document Triage
- **Tasks:** 
  - Policy feedback triage and escalation based on clarity
  - Ranking candidate citizen feedback drafts by readability and logical flow
  - Filtering submissions for OCR/artifact corruption
  - Prioritizing messages for human review in high-volume intake systems
  - Quality-control checks for auto-generated draft responses
  - Identifying items needing clarification before escalation
  - Labeling drafts for follow-up vs discard based on prose coherence
- **Best Suited For:** 
  - High-volume intake of short-to-medium citizen submissions where readability predicts actionability.
  - When candidate drafts vary primarily in surface quality (grammar, completeness, structure) rather than domain-specific correctness.
  - English (or well-supported language) inputs with consistent formatting and limited domain jargon.
  - Workflows that use a secondary human reviewer for factual validation after clarity-based triage.
  - When a clear scoring rubric or examples of 'coherent' vs 'garbled' text are provided to the judge.
- **Not Recommended For:** 
  - Long, highly technical, legal, or policy-dense documents where correctness and domain expertise—not just clarity—dictate escalation.
  - Multilingual inputs, heavy dialect/slang, or badly OCRed texts that require language-specific restoration or manual reading.
  - High-stakes decisions where factual accuracy, legal interpretation, or sensitive content must be adjudicated without expert human oversight.
  - Cases requiring inference of intent, credibility, or hidden context beyond what is present in the prose.
  - Adversarial, obfuscated, or intentionally misleading submissions where surface clarity may mask malicious content.

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
  - Preference for Standardized Grammar: favors submissions with standard English grammar and penalizes nonstandard dialects or non-native constructions.
  - Length/verbosity bias: shorter, concise sentences may be scored as clearer even when they omit essential context, while verbose but precise drafts may be penalized.
  - Format and punctuation bias: messages with correct punctuation and paragraphing are preferred over those lacking formatting (e.g., line breaks, bullets).
  - Polished-surface bias: well-edited wording can be treated as inherently higher quality, masking lack of relevance or factual accuracy.
  - Domain-language bias: use of specialized jargon or legal/technical shorthand may be labeled unclear when it is perfectly clear to agency experts.
  - OCR/scan artifact sensitivity: the judge may over-react to OCR noise (weird tokens, broken words) and downgrade otherwise meaningful content.
  - Bias against placeholders/fragments: presence of placeholders like [NAME] or fragmented input can cause harsh clarity penalties even if context is recoverable.
  - Cultural/conversational style bias: informal or culturally-specific rhetorical styles (e.g., storytelling, indirect appeals) are more likely to be judged as incoherent.
  - Length-as-quality bias: longer rewrites or clarifying additions may be favored over terse, precise citizen feedback.
  - Multilingual bias: submissions mixing languages or using non-Latin scripts may be scored as unclear instead of being recognized as multilingual content.
- **Task Misalignment Risks:** 
  - Clarity != Importance: focusing on clarity can miss the primary objective of escalation, which is to surface urgent or high-impact content regardless of prose quality.
  - False security from polish: clear, coherent drafts that are irrelevant or harmful could be escalated unnecessarily if clarity is treated as a proxy for importance.
  - Over-penalizing noisy but critical input: noisy submissions (OCR scans, SMS logs, shorthand notes) that contain urgent policy signals may be filtered out.
  - Encourages editing over escalation: the judge may favor recommending editorial fixes rather than escalation, delaying attention to time-sensitive issues.
  - Mismatch with domain expectations: clarity metrics that don't incorporate domain expertise may reject well-formed technical notes that use domain shorthand.
  - Equity harms: de-prioritizing submissions from non-native speakers or marginalized communities reduces representativeness of escalated items.
  - Context-stripping: focusing on sentence-level coherence can ignore cross-document cues (attachments, metadata) needed to assess escalation need.
  - Adversarial exploitation: actors can craft superficially clear messages to manipulate escalation pipelines if clarity is over-weighted.
- **Failure Cases:** 
  - False negative: a typed photo of a damaged infrastructure report with OCR glitches is judged incoherent and not escalated despite being urgent.
  - False positive: a rewritten, grammatically perfect complaint about a trivial administrative issue is escalated while more substantive but messy submissions are ignored.
  - Dialect penalty: feedback written in a regional dialect or nonstandard grammar is marked low-clearance and excluded from escalation despite relevance.
  - Jargon misread as garble: a technical submission using agency-specific acronyms is classified as incoherent instead of being recognized as expert shorthand.
  - Placeholder confusion: a draft containing placeholders like [REDACTED] or [FIGURE] is flagged as fragmented and deprioritized even though placeholders point to attached evidence.
  - Multilingual drop: a bilingual submission with important content in a non-English segment is scored poorly and not forwarded to multilingual reviewers.
  - OCR artifact cascade: a single misrecognized character (e.g., ‘0’ vs ‘O’) leads to overall judgment that the document is unreadable and drops escalation.
  - Adversarial clarity: malicious actors submit polished, coherent but harmful disinformation to trigger unnecessary escalations.
  - Granularity failure: judge treats a long, well-structured policy suggestion as low priority because it lacks crisp focal sentences, losing nuance required by officials.
  - Context-loss failure: judge evaluates an excerpt in isolation and deems it incoherent when the full submission (attachments/previous messages) would make it clear.
  - Overtrimming: the judge rewrites messy but substantive input into a polished form internally and then uses that polish to escalate, masking the original user's voice and consent issues.
  - Latency failure: flagging many near-unclear items for human review (due to strict clarity thresholds) overwhelms triage staff and delays responses.

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

    description: ClassVar[str] = "**Clarity and coherence** Readability and logical flow of prose versus garbled OCR, placeholders, or fragmented content."

    def __init__(self, model: dspy.LM = DEFAULT_MODEL):
        super().__init__(
            name="Clarity_and_coherence_gpt-5-mini",
            description="**Clarity and coherence** Readability and logical flow of prose versus garbled OCR, placeholders, or fragmented content.",
            axis="**Clarity and coherence** Readability and logical flow of prose versus garbled OCR, placeholders, or fragmented content.",
            model=model,
            task_description="Rank candidate policy feedback drafts for follow-up: given a citizen\u2019s submission, \n    determine which responses merit escalation to agency officials.",
            metric_card="provided",
            max_workers=32,
        )

    def __repr__(self):
        return f"Clarity_and_coherence_gpt_5_mini_LLMJudge(model=dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv(\"OPENAI_API_KEY\")))"

