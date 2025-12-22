# Auto-generated metric file for Topic_Relevance_and_Docket_Alignment_gpt-5-mini
import dspy
import os
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import GeneratedRefFreeLLMJudgeMetric
from typing import ClassVar

DEFAULT_MODEL = dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv("OPENAI_API_KEY"))

class Topic_Relevance_and_Docket_Alignment_gpt_5_mini_LLMJudge(GeneratedRefFreeLLMJudgeMetric):
    """---
# Metric Card for Topic_Relevance_and_Docket_Alignment_gpt-5-mini

**Topic Relevance and Docket Alignment** Degree to which the response directly addresses the citizen issue and the correct rulemaking docket without drifting into unrelated topics.

## Metric Details

**Topic_Relevance_and_Docket_Alignment_gpt-5-mini** is a **reference-free** LLM-as-a-Judge metric that prompts an LLM to rate a system output along a single, run-time-specified evaluation axis.
In this case the axis is `**Topic Relevance and Docket Alignment** Degree to which the response directly addresses the citizen issue and the correct rulemaking docket without drifting into unrelated topics.`.

The prompt supplies:

1. **Task description** *d*
2. **Axis rubric** `**Topic Relevance and Docket Alignment** Degree to which the response directly addresses the citizen issue and the correct rulemaking docket without drifting into unrelated topics.`
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
  - **Axis rubric** `**Topic Relevance and Docket Alignment** Degree to which the response directly addresses the citizen issue and the correct rulemaking docket without drifting into unrelated topics.`
  - **Input text** *x*
  - **Output text** *y*
- **Outputs:**
  - Scalar score $\hat{s} \in \{1,2,3,4,5\}$

## Intended Use

- **Domain:** Policy evaluation / Text classification (Triage & Escalation)
- **Tasks:** 
  - Escalation triage of citizen feedback
  - Docket assignment / matching candidate responses to rulemaking dockets
  - Ranking candidate policy feedback drafts by topical relevance
  - Filtering out responses that drift into unrelated topics
  - Generating brief rationales for why a draft should be escalated
  - Batch triage to surface likely high-priority items for human review
- **Best Suited For:** 
  - The citizen submission, candidate responses, and a clearly defined list of dockets or short docket descriptions/keywords are included in the input.
  - Clear, objective rubric or example labels (aligned / misaligned / borderline) are provided for in-context learning.
  - Triage is low-to-moderate risk and used to prioritize human reviewers rather than to make final legal/policy decisions.
  - Large volumes of submissions where consistent, automated pre-screening can greatly reduce human workload.
  - When quick, explainable short rationales are needed to justify why a response was flagged for escalation.
  - When docket mappings are stable and do not require real-time external data lookups.
- **Not Recommended For:** 
  - Dockets or alignment rules are not provided in the prompt and the model is expected to discover or verify live docket identifiers or up-to-date procedural rules.
  - Decisions are legally binding, high-stakes, or require certified subject-matter expertise (e.g., nuanced regulatory interpretation) without mandatory human sign-off.
  - Citizen submissions are highly technical, use domain-specific terminology, or require deep policy context that the model was not given.
  - Inputs are adversarial, intentionally ambiguous, or crafted to induce hallucinated alignment rationale.
  - Requirements include auditable provenance against agency databases or verification against external systems the model cannot access.
  - Situations needing precise mapping to internal agency priorities or internal workflows that are not encoded into the prompt.

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
  - Lexical matching bias: favors drafts that reuse the same words or docket numbers found in the submission rather than assessing semantic alignment.
  - Recency/data bias: may preferentially match to more recent or well-known dockets the model has seen during training instead of the correct, possibly obscure docket.
  - Agency-name salience bias: overemphasizes matches on agency names or program titles even when the substantive issue relates to a different docket.
  - Length/verbosity bias: interprets longer, more detailed drafts as more relevant or authoritative regardless of topical alignment.
  - Conservatism/neutrality bias: prefers cautious, bland language and may penalize concise or strongly worded but relevant feedback.
  - Formatting bias: gives undue weight to drafts that follow an expected structure (e.g., headers, docket lines) and downgrades nonstandard but relevant submissions.
  - Domain-knowledge bias: under-weights technical regulatory nuance if the model lacks specific statutory/regulatory background.
  - Language and cultural bias: lower-quality performance on submissions using nonstandard English, idioms, or other languages leading to misclassification.
- **Task Misalignment Risks:** 
  - Overemphasis on explicit docket tags: treating the presence/absence of a docket number as the primary criterion rather than overall relevance.
  - Semantic vs procedural misalignment: conflating topical relevance with whether the draft follows agency submission procedures (format vs substance).
  - Single-docket framing: forcing a one-docket decision when the citizen’s issue legitimately spans multiple dockets or broader agency responsibilities.
  - Escalation threshold mismatch: using an internal proxy (e.g., strong language or legal citation) that doesn't match agency rules about what requires escalation.
  - Neglect of urgency or risk: focusing only on topical match and missing safety, legal, or time-sensitive signals that should change escalation priority.
  - Preference for surface plausibility: aligning drafts that sound plausible but are substantively incorrect with the docket, thereby missing misleading but on-topic content.
- **Failure Cases:** 
  - False negative: a highly relevant draft that uses different terminology or implicit references is rated as off-topic and not escalated.
  - False positive: a draft that includes the correct docket number but primarily discusses an unrelated subject is escalated inappropriately.
  - Docket misattribution: the model assigns the response to a similar but wrong docket (e.g., related policy area) and misses the actual responsible rulemaking.
  - Cross-docket failure: fails to recognize that a single submission legitimately pertains to multiple dockets and forces a single-category decision.
  - Overlooking attachments/quotes: ignores relevant information embedded in attachments, quoted emails, or referenced documents and misjudges relevance.
  - Hallucinated alignment: invents or asserts linkage to a docket or policy rationale that isn't present, giving a false justification for escalation.
  - Inconsistent scoring: similar drafts receive different relevance assessments due to minor phrasing changes or formatting noise.
  - Missed urgency: does not escalate content that is time-sensitive or legally risky because topical cues alone appeared low-priority.
  - Mistranslation/idiom failure: misinterprets nonliteral language (sarcasm, idioms) and misclassifies topical relevance.
  - Stylistic confusion: penalizes concise, bullet-style drafts despite perfect topical alignment because they lack full prose context.

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

    description: ClassVar[str] = "**Topic Relevance and Docket Alignment** Degree to which the response directly addresses the citizen issue and the correct rulemaking docket without drifting into unrelated topics."

    def __init__(self, model: dspy.LM = DEFAULT_MODEL):
        super().__init__(
            name="Topic_Relevance_and_Docket_Alignment_gpt-5-mini",
            description="**Topic Relevance and Docket Alignment** Degree to which the response directly addresses the citizen issue and the correct rulemaking docket without drifting into unrelated topics.",
            axis="**Topic Relevance and Docket Alignment** Degree to which the response directly addresses the citizen issue and the correct rulemaking docket without drifting into unrelated topics.",
            model=model,
            task_description="Rank candidate policy feedback drafts for follow-up: given a citizen\u2019s submission, \n    determine which responses merit escalation to agency officials.",
            metric_card="provided",
            max_workers=32,
        )

    def __repr__(self):
        return f"Topic_Relevance_and_Docket_Alignment_gpt_5_mini_LLMJudge(model=dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv(\"OPENAI_API_KEY\")))"

