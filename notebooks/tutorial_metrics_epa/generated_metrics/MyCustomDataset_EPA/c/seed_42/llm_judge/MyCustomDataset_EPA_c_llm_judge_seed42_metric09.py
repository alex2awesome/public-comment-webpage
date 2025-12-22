# Auto-generated metric file for Source_Credibility_and_Representation_gpt-5-mini
import dspy
import os
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import GeneratedRefFreeLLMJudgeMetric
from typing import ClassVar

DEFAULT_MODEL = dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv("OPENAI_API_KEY"))

class Source_Credibility_and_Representation_gpt_5_mini_LLMJudge(GeneratedRefFreeLLMJudgeMetric):
    """---
# Metric Card for Source_Credibility_and_Representation_gpt-5-mini

**Source Credibility and Representation** Author expertise, organizational standing, coalition support, and whether the comment represents large or affected constituencies.

## Metric Details

**Source_Credibility_and_Representation_gpt-5-mini** is a **reference-free** LLM-as-a-Judge metric that prompts an LLM to rate a system output along a single, run-time-specified evaluation axis.
In this case the axis is `**Source Credibility and Representation** Author expertise, organizational standing, coalition support, and whether the comment represents large or affected constituencies.`.

The prompt supplies:

1. **Task description** *d*
2. **Axis rubric** `**Source Credibility and Representation** Author expertise, organizational standing, coalition support, and whether the comment represents large or affected constituencies.`
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
  - **Axis rubric** `**Source Credibility and Representation** Author expertise, organizational standing, coalition support, and whether the comment represents large or affected constituencies.`
  - **Input text** *x*
  - **Output text** *y*
- **Outputs:**
  - Scalar score $\hat{s} \in \{1,2,3,4,5\}$

## Intended Use

- **Domain:** Text classification / Policy analysis / Stakeholder triage
- **Tasks:** 
  - Prioritizing public comments for escalation to agency officials
  - Identifying and flagging submissions that cite institutional affiliations or credentials
  - Detecting and grouping submissions from named organizations or coalitions
  - Estimating whether a comment likely represents an affected or large constituency
  - Summarizing the credibility-related rationale for escalation decisions
  - Suggesting follow-up verification actions (e.g., request membership lists, corroborating sources)
- **Best Suited For:** 
  - When submissions include clear, explicit affiliation metadata (organization names, job titles, letterhead text).
  - When commenters provide verifiable citations, reports, or domain-specific technical content that indicate subject-matter expertise.
  - When multiple submissions use consistent organizational language or identical sign-on lists suggesting coalition support.
  - For moderate-to-large volumes of text where automated triage is needed to surface likely high-value items for human review.
  - When the goal is prioritization and rationale generation rather than final, legally binding determinations.
- **Not Recommended For:** 
  - When authoritative verification is required (e.g., confirming that an organization exists, validating membership counts, or authenticating signatures) without access to external databases.
  - When submissions are anonymous, extremely brief, or intentionally obfuscated such that textual credibility signals are absent.
  - When subtle local political or coalition dynamics (back-channel relationships, inferred influence) must be judged — these often require human institutional knowledge.
  - When inputs include non-textual evidence (scanned letterheads, attachments, or metadata) that the model cannot interpret without preprocessing.
  - When legal conclusions, formal regulatory decisions, or sensitive determinations about protected groups are at stake and require human oversight.

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
  - Institutional prestige bias: over-weighting well-known organizations, universities, or government-sounding affiliations versus community groups.
  - Credential-recognition bias: favoring conventional academic/professional credentials and failing to appreciate nontraditional or local expertise.
  - Language and formatting bias: penalizing submissions that are informal, poorly formatted, or in non-dominant languages.
  - Quantity-over-quality bias: treating counts (signatures, endorsements) as a straightforward proxy for representativeness or legitimacy.
  - Training-data recency and geographic bias: better recognizing organizations and norms common in the model’s training data (e.g., US/Western institutions).
  - Confirmation bias: leaning toward signals that match typical patterns of credible sources in training data.
  - Visibility/platform bias: favoring submitters who appear on public platforms or media the model 'knows' about.
  - Socioeconomic bias: disadvantaging marginalized communities whose submissions lack formal markers of authority.
- **Task Misalignment Risks:** 
  - Prioritizing formal authority over affectedness: escalating input from experts/organizations even when marginalized individuals present crucial lived-experience information.
  - Equating endorsement volume with representativeness: recommending escalation based on number of signatories rather than demographic or geographic representativeness.
  - Privacy and safety misalignment: escalating submissions that reveal personal data or sensitive details without regard for harm to individuals.
  - Policy/legal misalignment: flagging or escalating comments based on perceived credibility even when agency criteria require other legal/evidentiary thresholds.
  - Overreliance on unverifiable claims: making escalation recommendations despite inability to verify affiliations, potentially propagating false claims.
  - Cultural/context misalignment: misjudging credibility when local norms for demonstrating support differ from training-data norms.
  - Instrumentalization risk: optimizing for signals easy to detect by the model (names, org labels) rather than for the agency’s substantive needs.
- **Failure Cases:** 
  - False positives: escalating polished comments from a small but well-branded organization that misrepresents its support or expertise.
  - False negatives: failing to escalate high-impact grassroots testimony (affected individuals/communities) that lacks formal credentials or polished presentation.
  - Hallucinated affiliations: asserting or relying on organizational ties or credential claims that the submitter did not actually make.
  - Gamed inputs: being misled by manufactured coalitions, purchased endorsements, or coordinated campaigns designed to appear broadly representative.
  - Language and format failures: mis-evaluating credible non-English or informally written submissions as low-credibility.
  - Sparse-metadata failure: inability to judge credibility when the submission provides minimal self-description, leading to incorrect low-priority assignment.
  - Overcounting signatures: treating listed counts or references to large numbers as evidence of representativeness when numbers are vague or unverifiable.
  - Context-mismatch: recommending escalation for expert-sounding input that is irrelevant to the specific policy question or outside the agency’s jurisdiction.
  - Equity-related harm: systematically deprioritizing inputs from marginalized groups, producing skewed escalation that reduces representativeness.

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

    description: ClassVar[str] = "**Source Credibility and Representation** Author expertise, organizational standing, coalition support, and whether the comment represents large or affected constituencies."

    def __init__(self, model: dspy.LM = DEFAULT_MODEL):
        super().__init__(
            name="Source_Credibility_and_Representation_gpt-5-mini",
            description="**Source Credibility and Representation** Author expertise, organizational standing, coalition support, and whether the comment represents large or affected constituencies.",
            axis="**Source Credibility and Representation** Author expertise, organizational standing, coalition support, and whether the comment represents large or affected constituencies.",
            model=model,
            task_description="Rank candidate policy feedback drafts for follow-up: given a citizen\u2019s submission, \n    determine which responses merit escalation to agency officials.",
            metric_card="provided",
            max_workers=32,
        )

    def __repr__(self):
        return f"Source_Credibility_and_Representation_gpt_5_mini_LLMJudge(model=dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv(\"OPENAI_API_KEY\")))"

