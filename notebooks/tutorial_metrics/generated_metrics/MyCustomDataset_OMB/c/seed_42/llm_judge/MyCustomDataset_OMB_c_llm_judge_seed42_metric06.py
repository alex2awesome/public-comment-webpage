# Auto-generated metric file for Tone_and_Professionalism_gpt-5-mini
import dspy
import os
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import GeneratedRefFreeLLMJudgeMetric
from typing import ClassVar

DEFAULT_MODEL = dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv("OPENAI_API_KEY"))

class Tone_and_Professionalism_gpt_5_mini_LLMJudge(GeneratedRefFreeLLMJudgeMetric):
    """---
# Metric Card for Tone_and_Professionalism_gpt-5-mini

**Tone and Professionalism** The language is respectful objective and non partisan avoiding rants slogans or accusatory rhetoric.

## Metric Details

**Tone_and_Professionalism_gpt-5-mini** is a **reference-free** LLM-as-a-Judge metric that prompts an LLM to rate a system output along a single, run-time-specified evaluation axis.
In this case the axis is `**Tone and Professionalism** The language is respectful objective and non partisan avoiding rants slogans or accusatory rhetoric.`.

The prompt supplies:

1. **Task description** *d*
2. **Axis rubric** `**Tone and Professionalism** The language is respectful objective and non partisan avoiding rants slogans or accusatory rhetoric.`
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
  - **Axis rubric** `**Tone and Professionalism** The language is respectful objective and non partisan avoiding rants slogans or accusatory rhetoric.`
  - **Input text** *x*
  - **Output text** *y*
- **Outputs:**
  - Scalar score $\hat{s} \in \{1,2,3,4,5\}$

## Intended Use

- **Domain:** Text Evaluation / Policy Triage / Content Moderation
- **Tasks:** 
  - Ranking citizen feedback drafts by tone and professionalism
  - Flagging responses that contain accusatory, partisan, or inflammatory language
  - Prioritizing messages for escalation to agency officials
  - Suggesting neutral rewrites or phrasing improvements
  - Quality assurance of staff replies for respectful/objective language
  - Batch pre-screening to reduce human review load
- **Best Suited For:** 
  - High-volume streams of citizen feedback where consistent, repeatable pre-screening is needed to surface the most problematic items.
  - Materials written in standard English with clear lexical cues of tone (insults, threats, profanity, rhetorical slogans).
  - When you need explainable, point-by-point highlights of problematic wording and suggested neutral alternatives.
  - Integration as an automated triage step that forwards flagged items to human reviewers for final decision.
  - Applying a well-defined rubric for tone and professionalism across many short-to-medium length drafts.
- **Not Recommended For:** 
  - Short, highly ambiguous, or context-dependent messages where intent cannot be inferred without additional metadata or conversation history.
  - Cases that require legal, regulatory, or subject-matter expertise to determine whether language merits escalation beyond tone (e.g., threats that imply legal risk).
  - Cross-cultural or idiomatic expressions and code-switching where cultural norms determine whether phrasing is unprofessional or acceptable.
  - When messages are adversarially crafted to hide hostility via coded language, dog-whistles, or sarcasm that the model may miss.
  - Making final high-stakes escalation decisions (e.g., law enforcement referrals) without human oversight and verification.

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
  - Cultural and linguistic bias: preference for standard forms of English and Western norms of politeness may penalize dialects, translated text, or non-Western rhetorical styles.
  - Politeness bias: the model may overvalue neutral or understated language and devalue emotionally charged but legitimate advocacy.
  - Status-quo / institutional bias: tendency to favor language that aligns with bureaucratic/professional norms, which can disadvantage critical or reformist voices.
  - Political bias: training data or alignment choices could tilt judgments against language associated with particular political movements or ideologies.
  - Non-native speaker penalty: grammatical errors, indirect phrasing, or atypical syntax from non-native writers can be labeled unprofessional even when content is valid.
  - Overfiltering bias: the model may conflate passionate or urgent language with incivility and erroneously suppress it.
  - Form over substance bias: placing excessive weight on tone can lead to ignoring factual accuracy, risk, or legal content that merits escalation.
  - Recency and domain-training bias: model judgments may reflect the norms present in its training corpus rather than the specific agency’s or community’s standards.
- **Task Misalignment Risks:** 
  - Narrow axis drift: focusing on tone alone can miss content that should be escalated for factual, safety, or legal reasons despite professional wording.
  - Under-escalation risk: polite, neutral phrasing used to hide threats, misinformation, or policy-relevant allegations may be overlooked.
  - Over-escalation risk: strongly worded but legitimate citizen advocacy could be escalated unnecessarily because it violates a strict 'professional' standard.
  - Equity risk: marginalized communities using different rhetorical norms may be systematically deprioritized for escalation.
  - Context ignorance: failing to consider prior submissions, known complainant credibility, or topic sensitivity can produce misaligned escalation decisions.
  - Single-axis optimization: optimizing the judge for tone can produce brittle behavior when reused for different axes (e.g., accuracy, safety) without re-calibration.
  - Operational mismatch: agency protocols may require escalation based on content categories or legal thresholds that tone-based judgments do not capture.
  - Explainability gap: if the judge only outputs a binary escalation label without justification tied to both tone and substantive criteria, human reviewers may be unable to correct errors.
- **Failure Cases:** 
  - False negative: a submission containing credible threats or doxxing information is worded politely and is not escalated because it passes the tone check.
  - False positive: a passionate complaint with angry language but clear evidence of harm is flagged as unprofessional and unnecessarily escalated to legal teams.
  - Sarcasm/misinterpretation: sarcastic or ironic phrasing is interpreted as polite or neutral, masking abusive intent.
  - Dogwhistle evasion: coded or euphemistic hostile language bypasses the tone filter and goes unflagged.
  - Non-native phrasing misclassification: an earnest, relevant submission from a non-native speaker is marked unprofessional due to atypical syntax.
  - Adversarial paraphrase: actors deliberately rewrite toxic content in polite register to avoid escalation, exploiting the judge’s focus on tone.
  - Inconsistent thresholds: the model applies different standards across similar cases, producing unreliable escalation decisions.
  - Lack of justification: the judge returns a decision without clear reasoning tied to the axis, preventing effective human review or appeal.

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

    description: ClassVar[str] = "**Tone and Professionalism** The language is respectful objective and non partisan avoiding rants slogans or accusatory rhetoric."

    def __init__(self, model: dspy.LM = DEFAULT_MODEL):
        super().__init__(
            name="Tone_and_Professionalism_gpt-5-mini",
            description="**Tone and Professionalism** The language is respectful objective and non partisan avoiding rants slogans or accusatory rhetoric.",
            axis="**Tone and Professionalism** The language is respectful objective and non partisan avoiding rants slogans or accusatory rhetoric.",
            model=model,
            task_description="Rank candidate policy feedback drafts for follow-up: given a citizen\u2019s submission, \n    determine which responses merit escalation to agency officials.",
            metric_card="provided",
            max_workers=32,
        )

    def __repr__(self):
        return f"Tone_and_Professionalism_gpt_5_mini_LLMJudge(model=dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv(\"OPENAI_API_KEY\")))"

