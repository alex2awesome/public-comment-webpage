# Auto-generated metric file for Factual_Accuracy_of_Names_Titles_and_Terms_gpt-5-mini
import dspy
import os
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import GeneratedRefFreeLLMJudgeMetric
from typing import ClassVar

DEFAULT_MODEL = dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv("OPENAI_API_KEY"))

class Factual_Accuracy_of_Names_Titles_and_Terms_gpt_5_mini_LLMJudge(GeneratedRefFreeLLMJudgeMetric):
    """---
# Metric Card for Factual_Accuracy_of_Names_Titles_and_Terms_gpt-5-mini

**Factual Accuracy of Names Titles and Terms** Officials names offices dates and terminology such as OMB are correct and consistently used.

## Metric Details

**Factual_Accuracy_of_Names_Titles_and_Terms_gpt-5-mini** is a **reference-free** LLM-as-a-Judge metric that prompts an LLM to rate a system output along a single, run-time-specified evaluation axis.
In this case the axis is `**Factual Accuracy of Names Titles and Terms** Officials names offices dates and terminology such as OMB are correct and consistently used.`.

The prompt supplies:

1. **Task description** *d*
2. **Axis rubric** `**Factual Accuracy of Names Titles and Terms** Officials names offices dates and terminology such as OMB are correct and consistently used.`
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
  - **Axis rubric** `**Factual Accuracy of Names Titles and Terms** Officials names offices dates and terminology such as OMB are correct and consistently used.`
  - **Input text** *x*
  - **Output text** *y*
- **Outputs:**
  - Scalar score $\hat{s} \in \{1,2,3,4,5\}$

## Intended Use

- **Domain:** Text evaluation / policy feedback triage
- **Tasks:** 
  - Rank candidate policy feedback drafts by factual accuracy of names, titles, dates, and terminology
  - Detect and flag inconsistent uses of official names, acronyms, and office titles within a draft
  - Normalize different variants of agency and office names to a single canonical form
  - Identify likely incorrect or outdated names/titles/dates for human verification
  - Assign confidence indicators to factual name/term checks to prioritize escalation
  - Produce short justification notes citing the specific mismatch or uncertainty
- **Best Suited For:** 
  - Drafts that reference well-established, widely documented entities (e.g., Cabinet departments, OMB, EPA) where canonical names are stable
  - Short to medium-length feedback drafts with clearly delimited named entities and dates
  - Workflows that include an authoritative reference list (official names/titles) supplied to the model for cross-checking
  - Triage settings where the goal is to prioritize human review rather than make final determinations
  - Environments that accept model-generated confidence flags and require human verification for escalated items
  - When internal consistency (same spelling/abbreviation used throughout) is the primary criterion
- **Not Recommended For:** 
  - Cases requiring up-to-the-minute verification of personnel, recent appointments, or newly created offices that may postdate the model's knowledge cutoff
  - High-stakes legal or compliance decisions where authoritative documentation and human sign-off are mandatory
  - Drafts containing obscure, local, or informal office names not in common training data without an authoritative reference provided
  - Inputs with heavy OCR / transcription errors, nonstandard encodings, or poor formatting that impede reliable entity extraction
  - Tasks that require external live-data lookup (current URLs, live agency rosters) unless those sources are supplied to the model
  - Situations demanding legal interpretation of titles, authority, or statutory effect rather than surface factual naming consistency

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
  - Recency bias: the judge may miss recent appointments or reorganizations that occurred after its training cutoff and therefore mark current names or titles as incorrect.
  - Prominence bias: well-known officials and large federal agencies are more likely to be recognized and validated than low-profile local officials or small offices.
  - Anglocentric / U.S.-centric bias: the judge is more reliable for U.S. federal terminology and may perform worse on state, local, or non‑U.S. institutional names and titles.
  - Formalization bias: preference for canonical/formal names and titles causes the judge to penalize colloquial, shorthand, or community-specific references that are nevertheless meaningful.
  - Abbreviation expansion bias: the model may over- or under-expand acronyms (e.g., assuming an acronym maps to the most common expansion in training data rather than the jurisdictionally correct one).
  - Spelling-normalization bias: the judge may autocorrect unusual but correct spellings or transliterations, treating them as errors.
  - Confirmation/context bias: the judge can infer a likely name/title from surrounding context and then treat deviations as errors even if the original was correct.
  - Source prominence bias: the judge favors information forms/phrases common in its training sources (press releases, official web pages) over citizen phrasing.
- **Task Misalignment Risks:** 
  - Overweighting surface factuality: prioritizing perfect names/titles may push the judge to de-escalate otherwise important substantive feedback that uses informal language.
  - Penalizing lay language: citizens often use shorthand or common parlance; treating these as disqualifying can reduce escalation of high-value submissions.
  - Jurisdiction misclassification: focusing on exact agency naming without verifying jurisdiction may cause wrong escalations or missed escalations across federal/state/local boundaries.
  - Encouraging conservative edits: the axis incentivizes changing a draft to canonical labels rather than preserving intent or nuance, potentially altering the author's message.
  - Insufficient context sensitivity: the judge may require perfectly specified addressees where a single relevant office would have sufficed for escalation.
  - False reliance on terminology correctness: equating correct terminology with overall suitability for escalation can ignore tone, confidentiality, or procedural appropriateness.
- **Failure Cases:** 
  - False negative due to recency: a correct current official is flagged as incorrect because the judge's knowledge cutoff predates their appointment.
  - Alias confusion: a draft uses a common alias (e.g., 'the Secretary') and the judge incorrectly treats the reference as ambiguous and non-actionable.
  - Jurisdiction collision: the judge mistakes a state agency acronym for a federal one (or vice versa) and recommends the wrong escalation target.
  - Transliteration/spelling mismatch: non‑standard Romanization or an uncommon spelling of a foreign official's name is marked wrong despite being correct.
  - Acronym mis-expansion: the judge expands an acronym to a more common organization in training data rather than the specific office intended by the citizen.
  - Overcorrection of informal phrasing: a citizen's plain-language reference (e.g., 'the finance people at OMB') is flagged as unacceptable even though intent and recipient are clear.
  - OCR/typo sensitivity: small typos or OCR artifacts in uploaded text cause the judge to flag otherwise correct names or dates.
  - Title drift: the agency recently renamed an office and the judge marks older but still-used titles as incorrect, or vice versa.
  - Person/office conflation: the judge treats a phrase intended as an office (e.g., 'Office of the Chief Medical Officer') as a person name and mis-evaluates factuality.
  - Ambiguous date handling: the judge treats approximate or partial dates provided by a citizen as factual errors rather than contextually acceptable references.

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

    description: ClassVar[str] = "**Factual Accuracy of Names Titles and Terms** Officials names offices dates and terminology such as OMB are correct and consistently used."

    def __init__(self, model: dspy.LM = DEFAULT_MODEL):
        super().__init__(
            name="Factual_Accuracy_of_Names_Titles_and_Terms_gpt-5-mini",
            description="**Factual Accuracy of Names Titles and Terms** Officials names offices dates and terminology such as OMB are correct and consistently used.",
            axis="**Factual Accuracy of Names Titles and Terms** Officials names offices dates and terminology such as OMB are correct and consistently used.",
            model=model,
            task_description="Rank candidate policy feedback drafts for follow-up: given a citizen\u2019s submission, \n    determine which responses merit escalation to agency officials.",
            metric_card="provided",
            max_workers=32,
        )

    def __repr__(self):
        return f"Factual_Accuracy_of_Names_Titles_and_Terms_gpt_5_mini_LLMJudge(model=dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv(\"OPENAI_API_KEY\")))"

