# Auto-generated metric file for Novelty_and_Added_Value_gpt-5-mini
import dspy
import os
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import GeneratedRefFreeLLMJudgeMetric
from typing import ClassVar

DEFAULT_MODEL = dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv("OPENAI_API_KEY"))

class Novelty_and_Added_Value_gpt_5_mini_LLMJudge(GeneratedRefFreeLLMJudgeMetric):
    """---
# Metric Card for Novelty_and_Added_Value_gpt-5-mini

**Novelty and Added Value** Extent to which the submission introduces new information, perspectives, or solutions not already prominent in the record.

## Metric Details

**Novelty_and_Added_Value_gpt-5-mini** is a **reference-free** LLM-as-a-Judge metric that prompts an LLM to rate a system output along a single, run-time-specified evaluation axis.
In this case the axis is `**Novelty and Added Value** Extent to which the submission introduces new information, perspectives, or solutions not already prominent in the record.`.

The prompt supplies:

1. **Task description** *d*
2. **Axis rubric** `**Novelty and Added Value** Extent to which the submission introduces new information, perspectives, or solutions not already prominent in the record.`
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
  - **Axis rubric** `**Novelty and Added Value** Extent to which the submission introduces new information, perspectives, or solutions not already prominent in the record.`
  - **Input text** *x*
  - **Output text** *y*
- **Outputs:**
  - Scalar score $\hat{s} \in \{1,2,3,4,5\}$

## Intended Use

- **Domain:** Policy triage / Text classification / Public comment analysis
- **Tasks:** 
  - Triage citizen submissions for escalation to agency officials based on novelty and added value
  - Ranking public comments by how much new information or perspectives they introduce
  - Identifying submissions that propose new policy options, implementation mechanisms, or evidence not present in the record
  - Labeling comments as paraphrase/redundant vs substantively novel
  - Prioritizing stakeholder feedback for human review in rulemaking or consultation processes
  - Generating short rationales that explain why a submission was flagged as novel
- **Best Suited For:** 
  - When the judge has access to the full prior record or a reliable summary against which to compare candidate submissions.
  - When novelty can be detected through language-level signals (new facts, alternative arguments, or distinct solution proposals) rather than deep technical validation.
  - For high-throughput triage where the goal is to surface likely-novel items for human experts rather than to make final determinations.
  - When comments are in clear, well-structured prose and include explicit claims, data, or recommendations.
  - When agency reviewers want explainable, reproducible heuristics (e.g., semantic similarity thresholds, presence of novel evidence) to guide escalation.
- **Not Recommended For:** 
  - When evaluation requires domain-expert validation of technical, legal, or scientific novelty (e.g., novel clinical trial claims or complex regulatory legal arguments) without expert oversight.
  - If the judge does not have reliable access to the canonical record or if that record is incomplete or highly fragmented.
  - For final, high-stakes decisions about policy change or legal interpretation where false positives/negatives could cause significant harm.
  - When submissions deliberately obscure meaning (e.g., coded language, satire, or highly rhetorical political messaging) that requires contextual or human cultural interpretation.
  - In situations demanding provenance verification of factual claims (the model can flag novelty but not independently verify new data sources).

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
  - Lexical novelty bias: the judge favors submissions that use different wording or synonyms even when they repeat existing points.
  - Length/verbosity bias: longer drafts are treated as more novel or valuable than concise ones.
  - Citation and formality bias: feedback that includes citations, technical jargon, or formal structure is scored higher regardless of substantive novelty.
  - Conservatism bias: suggestions aligned with prevailing agency views or mainstream sources are more likely to be flagged as valuable.
  - Representation bias: perspectives from marginalized communities or uncommon framings are underrecognized as novel because they are underrepresented in the model's training data.
  - Recency bias: the judge overweights recently discussed facts or events and underrates longstanding but still-novel angles.
  - Domain familiarity bias: topics similar to the model's dominant training-domain content are judged more accurately, while niche technical/legal points are underappreciated.
  - Sensationalism bias: striking or emotionally charged language is mistaken for substantive novelty.
  - Language/translation bias: submissions in non-dominant languages or with nonstandard phrasing are penalized even when substantive.
- **Task Misalignment Risks:** 
  - Conflating lexical novelty with substantive novelty — the evaluation may reward novel wording rather than new facts, arguments, or solutions.
  - Prioritizing novelty at the expense of relevance or feasibility — novel ideas that are impractical or irrelevant may be escalated over actionable but incremental input.
  - Failing to account for corroborative value — duplicate or confirmatory submissions that strengthen a record may be undervalued despite their importance for escalation.
  - Assuming a complete record — the judge may label contributions as non-novel because it incorrectly believes the record already contains similar points.
  - Overlooking legal or procedural constraints — novel proposals that conflict with statutes or agency constraints might be flagged as valuable without recognizing invalidity.
  - Treating novelty as the sole escalation criterion — other escalation-worthy dimensions (urgency, harm, representativeness) may be ignored.
  - Inconsistent weighting across domains — the same type of novelty may be judged differently depending on topic due to uneven training exposure.
  - Relying on surface cues (format, citations) rather than substantive evidence of added value.
- **Failure Cases:** 
  - False positive novelty: a submission that paraphrases existing comments in new language is escalated as 'novel' when it adds no new content.
  - False negative novelty: a succinct technical correction or narrowly scoped novel legal argument is missed because it is brief or jargon-heavy.
  - Hallucinated novelty: the judge asserts a submission is novel relative to the record because it hallucinates what is or isn’t in the record.
  - Over-escalation of impractical ideas: creative but infeasible policy suggestions are flagged for escalation, wasting official attention.
  - Under-escalation of corroborative signals: many similar submissions that collectively indicate a trend are individually discounted and thus the pattern is missed.
  - Language access failure: non-native phrasing or multilingual submissions are systematically scored lower, causing exclusion of important perspectives.
  - Prompt-sensitivity inconsistency: small changes in prompt wording or example audits produce different novelty rankings for the same submission.
  - Domain blindspot: expert technical or legal novelties in niche fields are misinterpreted or ignored because the judge lacks domain competence.
  - Format-driven error: submissions without citations or polished structure are deprioritized even when they contain original, high-value insights.

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

    description: ClassVar[str] = "**Novelty and Added Value** Extent to which the submission introduces new information, perspectives, or solutions not already prominent in the record."

    def __init__(self, model: dspy.LM = DEFAULT_MODEL):
        super().__init__(
            name="Novelty_and_Added_Value_gpt-5-mini",
            description="**Novelty and Added Value** Extent to which the submission introduces new information, perspectives, or solutions not already prominent in the record.",
            axis="**Novelty and Added Value** Extent to which the submission introduces new information, perspectives, or solutions not already prominent in the record.",
            model=model,
            task_description="Rank candidate policy feedback drafts for follow-up: given a citizen\u2019s submission, \n    determine which responses merit escalation to agency officials.",
            metric_card="provided",
            max_workers=32,
        )

    def __repr__(self):
        return f"Novelty_and_Added_Value_gpt_5_mini_LLMJudge(model=dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv(\"OPENAI_API_KEY\")))"

