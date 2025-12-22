# Auto-generated metric file for Evidence_and_specificity_gpt-5-mini
import dspy
import os
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import GeneratedRefFreeLLMJudgeMetric
from typing import ClassVar

DEFAULT_MODEL = dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv("OPENAI_API_KEY"))

class Evidence_and_specificity_gpt_5_mini_LLMJudge(GeneratedRefFreeLLMJudgeMetric):
    """---
# Metric Card for Evidence_and_specificity_gpt-5-mini

**Evidence and specificity** Use of citations to statutes, sections, data, or reasoned analysis supporting claims or recommendations.

## Metric Details

**Evidence_and_specificity_gpt-5-mini** is a **reference-free** LLM-as-a-Judge metric that prompts an LLM to rate a system output along a single, run-time-specified evaluation axis.
In this case the axis is `**Evidence and specificity** Use of citations to statutes, sections, data, or reasoned analysis supporting claims or recommendations.`.

The prompt supplies:

1. **Task description** *d*
2. **Axis rubric** `**Evidence and specificity** Use of citations to statutes, sections, data, or reasoned analysis supporting claims or recommendations.`
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
  - **Axis rubric** `**Evidence and specificity** Use of citations to statutes, sections, data, or reasoned analysis supporting claims or recommendations.`
  - **Input text** *x*
  - **Output text** *y*
- **Outputs:**
  - Scalar score $\hat{s} \in \{1,2,3,4,5\}$

## Intended Use

- **Domain:** Text Evaluation / Policy Feedback Review
- **Tasks:** 
  - Rank candidate policy feedback drafts by strength of evidence and specificity
  - Detect and classify explicit citations (statutes, sections, case law, datasets) and identify missing citations
  - Assess whether claims are supported by reasoned analysis versus unsupported assertions
  - Flag drafts that warrant escalation to agency officials for legal or technical review
  - Produce concise summaries of evidentiary strengths and weaknesses for each draft
  - Suggest concrete edits to increase citation specificity or strengthen reasoning
- **Best Suited For:** 
  - Triage at scale where many drafts must be prioritized quickly for human review
  - When drafts include explicit citations or quoted statutory language that the model can parse for specificity
  - When the goal is to identify missing or vague evidence (e.g., 'relevant statute' with no section) rather than to perform definitive legal interpretation
  - When producing structured summaries of evidentiary strengths/weaknesses to guide expert reviewers
  - When used as a first-pass filter combined with downstream human validation and source verification
- **Not Recommended For:** 
  - When definitive, binding legal judgments or formal legal advice are required (should be done by qualified counsel)
  - When assessment depends on statutes, regulations, or case law enacted or changed after the model's knowledge cutoff or that require live database access
  - When high-stakes escalation decisions depend on primary-source verification (the model cannot access or certify external documents)
  - When nuanced statutory interpretation, agency precedent, or complex technical evidence requires domain experts
  - When the dataset contains jurisdiction-specific or obscure citations the model is unlikely to have seen and may hallucinate

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
  - Preference for documents that include formal, standard citations (e.g., statute numbers, section references) over first‑hand testimony or qualitative evidence, disadvantaging lived-experience feedback that lacks formal references.
  - Jurisdictional bias toward sources and statutes common in the model's training data (e.g., U.S.-centric), causing lower scores for correct citations from less-represented jurisdictions or international law.
  - Source-quality bias favoring well-known or frequently cited outlets (e.g., major journals, federal statutes) even when less-known sources are more relevant or higher quality for the specific policy area.
  - Recency bias or staleness: tendency to treat older-cited statutes or data as equally authoritative even when the model lacks awareness of more recent changes.
  - Format bias: penalizing credible evidence presented in nonstandard citation formats (URLs, inline descriptions, community reports) because it lacks recognizable citation tokens.
  - Confirmation bias where the judge rates evidence that aligns with its learned priors more favorably than equally supported, but countervailing, evidence.
  - Precision bias that equates the number of citations with strength of evidence rather than assessing relevance or methodological rigor.
  - Language bias against drafts citing non-English sources or community documents because the model's training data contains fewer such examples.
  - Overreliance on surface lexical signals (e.g., presence of statute numbers, 'study found') rather than evaluating the substance or validity of the claims.
  - Conservatism bias that favors conservative/defensive language with many caveats and citations over bold but potentially important claims lacking dense referencing.
- **Task Misalignment Risks:** 
  - Prioritizing citation density or formal legal references can cause the judge to escalate drafts that are technically well-cited but irrelevant to agencies' practical decision needs.
  - The axis encourages emphasis on explicit citations, which may deprioritize urgent firsthand reports or systemic problems that are evidenced primarily by patterns of narrative rather than named statutes.
  - If the judge treats any citation token as valid evidence without verification, escalation decisions can be misaligned with actual evidentiary quality and agency burden of proof.
  - Focusing on evidence specificity may ignore other escalation criteria (e.g., immediacy, potential harm, confidentiality, political sensitivity), leading to poor operational decisions.
  - The judge may downgrade drafts aimed at prompting exploratory agency inquiry (high-level policy questions) because they lack detailed statutory citations, even though escalation is appropriate.
  - Relying on the model's internal knowledge to assess citations risks misalignment when statutes have been amended, repealed, or interpreted differently since the model's training cutoff.
  - Encouraging citation-style conformity may disadvantage community-sourced feedback formats that agencies nevertheless consider valuable, misaligning with real-world intake processes.
  - The axis could incentivize writers to include superficial or misleading citations (gaming) — the judge may escalate these while missing substantive, uncited concerns.
- **Failure Cases:** 
  - Hallucinated citations in a candidate draft (fabricated statute numbers, fake studies) pass the judge's superficial checks and receive high escalation scores.
  - Legitimate but nonstandard evidence (community survey, interview transcripts) is scored very low because it lacks formal citations, causing failure to escalate high‑priority issues.
  - Citations to the correct legal provision in a less common jurisdiction are ignored or misidentified as irrelevant, leading to false negatives for escalation.
  - A draft lists many irrelevant or marginally related citations to appear evidence-rich and is escalated despite the citations not supporting its core claims.
  - The judge mistakes mere citation formatting (e.g., 'Smith et al. 2020') for strong causal evidence and escalates drafts with low-quality or correlational studies.
  - Timely drafts referencing recent regulatory changes are downgraded because the model lacks knowledge of those changes and treats the citations as unsupported.
  - Adversarial submissions include plausible-looking but incorrect URLs or DOIs and get ranked highly because the judge does not verify source contents.
  - Drafts containing sensitive personal information that should be redacted are escalated without flagging privacy concerns because the axis focuses on citations over safety.
  - Inconsistent scoring where two substantively similar drafts receive divergent evaluations because one uses formal citation tokens and the other paraphrases the same evidence.
  - False positives: minor procedural feedback with dense statutory quotes is escalated while substantive systemic complaints lacking citations are ignored.
  - False negatives: high-quality analytical drafts that synthesize multiple data types but rely on implicit inference (not line-by-line citations) are rated poorly and not escalated.
  - Failure to distinguish between primary sources (statutes, datasets) and secondary commentary (op-eds, blog posts), causing misranking of evidence strength.

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

    description: ClassVar[str] = "**Evidence and specificity** Use of citations to statutes, sections, data, or reasoned analysis supporting claims or recommendations."

    def __init__(self, model: dspy.LM = DEFAULT_MODEL):
        super().__init__(
            name="Evidence_and_specificity_gpt-5-mini",
            description="**Evidence and specificity** Use of citations to statutes, sections, data, or reasoned analysis supporting claims or recommendations.",
            axis="**Evidence and specificity** Use of citations to statutes, sections, data, or reasoned analysis supporting claims or recommendations.",
            model=model,
            task_description="Rank candidate policy feedback drafts for follow-up: given a citizen\u2019s submission, \n    determine which responses merit escalation to agency officials.",
            metric_card="provided",
            max_workers=32,
        )

    def __repr__(self):
        return f"Evidence_and_specificity_gpt_5_mini_LLMJudge(model=dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv(\"OPENAI_API_KEY\")))"

