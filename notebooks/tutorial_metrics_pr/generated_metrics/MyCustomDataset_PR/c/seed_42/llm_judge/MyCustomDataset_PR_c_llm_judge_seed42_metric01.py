# Auto-generated metric file for Substantive_policy_relevance_gpt-5-mini
import dspy
import os
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import GeneratedRefFreeLLMJudgeMetric
from typing import ClassVar

DEFAULT_MODEL = dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv("OPENAI_API_KEY"))

class Substantive_policy_relevance_gpt_5_mini_LLMJudge(GeneratedRefFreeLLMJudgeMetric):
    """---
# Metric Card for Substantive_policy_relevance_gpt-5-mini

**Substantive policy relevance** Degree to which the text addresses the regulation or policy issues and their implications rather than administrative metadata or unrelated content.

## Metric Details

**Substantive_policy_relevance_gpt-5-mini** is a **reference-free** LLM-as-a-Judge metric that prompts an LLM to rate a system output along a single, run-time-specified evaluation axis.
In this case the axis is `**Substantive policy relevance** Degree to which the text addresses the regulation or policy issues and their implications rather than administrative metadata or unrelated content.`.

The prompt supplies:

1. **Task description** *d*
2. **Axis rubric** `**Substantive policy relevance** Degree to which the text addresses the regulation or policy issues and their implications rather than administrative metadata or unrelated content.`
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
  - **Axis rubric** `**Substantive policy relevance** Degree to which the text addresses the regulation or policy issues and their implications rather than administrative metadata or unrelated content.`
  - **Input text** *x*
  - **Output text** *y*
- **Outputs:**
  - Scalar score $\hat{s} \in \{1,2,3,4,5\}$

## Intended Use

- **Domain:** Policy triage / Text classification / Policy analysis
- **Tasks:** 
  - Triage incoming public comments by substantive relevance
  - Rank feedback drafts for escalation to agency officials
  - Identify and flag comments that cite specific regulatory impacts or statutory provisions
  - Detect whether a submission raises new policy issues versus administrative/meta points
  - Prioritize stakeholder-submitted evidence or concrete implementation concerns
  - Generate short rationales explaining why a comment merits escalation
- **Best Suited For:** 
  - Submissions written in clear, well-formed language that explicitly mention policy issues, statutes, or stakeholder impacts.
  - A defined rubric or examples are provided so the judge can align evaluations to agency escalation criteria.
  - Moderate to high volume of comments where preliminary automated filtering is needed before expert review.
  - Domains with stable, well-documented terminology (e.g., environmental regulation, telecom policy) that the model can learn to recognize.
  - Use as a first-pass prioritization tool to surface likely-substantive feedback for human experts to review.
  - When speed and consistency of initial relevance judgments are more important than definitive legal interpretation.
- **Not Recommended For:** 
  - Inputs that are highly ambiguous, fragmented, or very short such that substantive intent cannot be inferred reliably.
  - Situations requiring definitive legal judgments, binding interpretations, or authoritative policy decisions.
  - Comments containing technical facts that must be empirically verified or that hinge on up-to-date jurisdictional law.
  - High-stakes escalation decisions where false positives/negatives carry major legal, safety, or reputational risk without human oversight.
  - Adversarial, manipulative, or phishing-style submissions designed to deceive or game automated triage.
  - Multilingual or dialect-heavy submissions when the model lacks robust language support or domain adaptation.
  - Cases where stakeholder identity, provenance, or trustworthiness must be established before escalation (authentication required).

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
  - Keyword bias: over-weighting presence of specific policy words or citations and underrating substantive discussion that uses different phrasing.
  - Length/verbosity bias: equating longer responses with greater substantive relevance and discounting concise but targeted submissions.
  - Domain expertise bias: favoring content that uses familiar terminology while penalizing technically correct but specialized language the model finds unfamiliar.
  - Recency/context bias: misjudging relevance because the model lacks knowledge of the particular regulatory stage, jurisdiction, or recent developments.
  - Political or ideological bias: treating language from certain political perspectives as more or less policy-relevant due to learned correlations in training data.
  - Civility/tonality bias: penalizing emotional or rhetorical submissions even when they contain substantive policy proposals or evidence.
  - Confirmation bias: prioritizing content that aligns with preconceptions about what constitutes 'policy' (e.g., legalistic language) over lived-experience evidence about implementation impacts.
  - Source/form bias: undervaluing first-person narratives or grassroots testimony relative to academic or industry-sounding submissions.
- **Task Misalignment Risks:** 
  - Conflating 'substantive' with 'technical/legal' — the judge may narrow the axis to formal legal argumentation and miss practical implementation issues that are highly relevant.
  - Treating administrative metadata as irrelevant even when it signals important context (e.g., identifying affected populations or timelines), leading to false negatives.
  - Over-relying on surface cues (keywords, citations) so responses that are substantively relevant but phrased colloquially are deprioritized.
  - Failing to account for cross-cutting relevance: a response may be policy-relevant because it reveals unintended consequences even if it doesn't propose a legal fix.
  - Using a fixed threshold for escalation that doesn't adapt to the agency's priorities or docket (e.g., treating all mentions of cost as equally important).
  - Failing to incorporate stakeholder credibility or representativeness — treating a single detailed anecdote and a formal analysis as equivalent without nuance.
  - Applying a one-size-fits-all rubric across diverse policy domains (environment, finance, health), causing domain-specific signals to be ignored.
  - Prioritizing comprehensiveness over actionability so tentative but novel insights that merit escalation are missed.
- **Failure Cases:** 
  - False negative: A short, concrete report from a regulated-business operator describes a novel compliance workaround that would undermine the rule’s intent, but the judge dismisses it as anecdotal and non-substantive.
  - False positive: A long submission filled with legalistic citations that are irrelevant to the specific rule is escalated despite lacking implications for the agency’s decision.
  - Keyword trap: A comment uses synonyms or plain language to describe a technical implementation failure (e.g., 'safety sensor sometimes fails when it rains') but lacks formal terminology and is scored low.
  - Jargon misinterpretation: Highly technical engineering feedback is misread as noise because the judge lacks domain knowledge and thus fails to escalate an important technical risk.
  - Tone penalty: An angry public comment that documents repeated harms and provides actionable remedies is discounted due to abusive or emotive language.
  - Context omission: A submission references a related statute or recent enforcement action unknown to the model, so the judge underestimates its relevance.
  - Overgeneralization: The judge escalates submissions that make broad policy claims without evidence simply because they mention policy topics repeatedly.
  - Edge-case legal nuance missed: A comment points out a narrow preemption or jurisdictional issue that would invalidate parts of the rule, but the judge misses it and fails to escalate.
  - Representative weighting error: Multiple brief but consistent reports from the same community are individually judged as non-substantive rather than recognized as cumulative evidence.
  - Domain transfer error: The same rubric applied across domains treats economic impact statements in the monetary domain differently from qualitative equity impacts, resulting in inconsistent escalation decisions.

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

    description: ClassVar[str] = "**Substantive policy relevance** Degree to which the text addresses the regulation or policy issues and their implications rather than administrative metadata or unrelated content."

    def __init__(self, model: dspy.LM = DEFAULT_MODEL):
        super().__init__(
            name="Substantive_policy_relevance_gpt-5-mini",
            description="**Substantive policy relevance** Degree to which the text addresses the regulation or policy issues and their implications rather than administrative metadata or unrelated content.",
            axis="**Substantive policy relevance** Degree to which the text addresses the regulation or policy issues and their implications rather than administrative metadata or unrelated content.",
            model=model,
            task_description="Rank candidate policy feedback drafts for follow-up: given a citizen\u2019s submission, \n    determine which responses merit escalation to agency officials.",
            metric_card="provided",
            max_workers=32,
        )

    def __repr__(self):
        return f"Substantive_policy_relevance_gpt_5_mini_LLMJudge(model=dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv(\"OPENAI_API_KEY\")))"

