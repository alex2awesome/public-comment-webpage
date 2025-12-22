# Auto-generated metric file for Stakeholder_identity_and_relevance_gpt-5-mini
import dspy
import os
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import GeneratedRefFreeLLMJudgeMetric
from typing import ClassVar

DEFAULT_MODEL = dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv("OPENAI_API_KEY"))

class Stakeholder_identity_and_relevance_gpt_5_mini_LLMJudge(GeneratedRefFreeLLMJudgeMetric):
    """---
# Metric Card for Stakeholder_identity_and_relevance_gpt-5-mini

**Stakeholder identity and relevance** Clear identification of the submitting entity and its stake or expertise in the matter without relying on protected attributes.

## Metric Details

**Stakeholder_identity_and_relevance_gpt-5-mini** is a **reference-free** LLM-as-a-Judge metric that prompts an LLM to rate a system output along a single, run-time-specified evaluation axis.
In this case the axis is `**Stakeholder identity and relevance** Clear identification of the submitting entity and its stake or expertise in the matter without relying on protected attributes.`.

The prompt supplies:

1. **Task description** *d*
2. **Axis rubric** `**Stakeholder identity and relevance** Clear identification of the submitting entity and its stake or expertise in the matter without relying on protected attributes.`
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
  - **Axis rubric** `**Stakeholder identity and relevance** Clear identification of the submitting entity and its stake or expertise in the matter without relying on protected attributes.`
  - **Input text** *x*
  - **Output text** *y*
- **Outputs:**
  - Scalar score $\hat{s} \in \{1,2,3,4,5\}$

## Intended Use

- **Domain:** Text evaluation / Policy intake triage
- **Tasks:** 
  - Rank candidate policy feedback drafts for escalation based on stakeholder identity and relevance
  - Extract and summarize declared affiliations, roles, and stated expertise from submissions
  - Assess potential conflicts of interest or vested interests described by the submitter
  - Compare stakeholder claims to policy topic to determine topical relevance
  - Flag ambiguous, unverifiable, or potentially fabricated identity claims for human follow-up
  - Generate short clarifying questions to confirm stake or expertise before escalation
- **Best Suited For:** 
  - Submissions that include explicit, unambiguous declarations of affiliation, role, or credentials that can be judged against policy criteria
  - Contexts where escalation rules and relevance criteria are clearly defined and provided to the judge
  - Policy domains where textual signals of expertise (terminology, citations, prior public positions) reliably correlate with real-world relevance
  - Workflows that treat the judge's output as an advisory triage step followed by human verification for high-stakes cases
  - Environments where privacy constraints permit analysis of the submitter's stated non-sensitive identity details
- **Not Recommended For:** 
  - Cases requiring external identity verification (employment, licensure, or official status) or legal adjudication, which need authoritative checks beyond text analysis
  - When submitters are anonymous, extremely terse, or provide no affiliation or evidence of stake or expertise
  - Situations that demand inference of protected attributes (race, religion, sexual orientation, etc.) or other sensitive attributes for decision-making
  - Adversarial settings where actors may fabricate or spoof identities and where automated heuristics can be gamed without human oversight
  - Jurisdiction-specific or highly technical regulatory determinations that require up-to-date domain-specific knowledge or legal interpretation not present in the judge's training data

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
  - Favoring submissions that explicitly state institutional affiliations (organizational name bias), even when unaffiliated but knowledgeable individuals are equally relevant.
  - Length and detail bias: longer or more technical submissions will be judged as more expert or stakeholder-relevant regardless of actual authority.
  - Jargon and credential bias: use of field-specific terminology or credentials (real or self-claimed) will inflate perceived expertise, disadvantaging credible lay stakeholders.
  - Name/organization recognition bias: familiar or well-known organization names will be elevated over unknown local groups with equal stake.
  - Recency and public-profile bias: submissions from entities with an online presence or recent publicity are more likely to be escalated than anonymous but directly affected citizens.
  - Tone and politeness bias: civil, formal wording may be scored higher than emotionally charged but substantive first-person accounts from affected individuals.
  - Proxy attribute bias: the model may rely on non-protected proxies (address, ZIP code, occupation) that correlate with protected attributes, indirectly reproducing discriminatory patterns.
- **Task Misalignment Risks:** 
  - Overemphasizing identity clarity at the expense of substantive content, causing relevant technical comments from anonymous or vague submissions to be ignored.
  - Treating the axis as requiring full identity verification, which could promote unsafe inference/guessing about submitters' backgrounds or private data.
  - Escalation policy misalignment where the judge equates organizational prestige with relevance, rather than measuring direct stake or unique expertise.
  - Conflating declared affiliation with verified authority, leading to over-escalation of unverified claims and under-escalation of verifiable lived-experience testimony.
  - Applying the axis uniformly across heterogeneous submission types (e.g., petitions, technical comments, personal narratives) without adjusting criteria for different stakeholder roles.
  - Encouraging collection or inference of protected attributes to improve classification accuracy, which conflicts with the axis constraint to avoid protected attributes.
- **Failure Cases:** 
  - False positive escalation: the model escalates a submission from a purported 'expert' that used impressive jargon but lacks verifiable affiliation or stake.
  - False negative omission: the model fails to escalate a short, anonymous first-person account describing direct harm because it lacks formal identity markers.
  - Hallucinated affiliation: the judge invents or confidently asserts an affiliation or credential not present in the text, leading to improper escalation.
  - Over-reliance on external knowledge: the model uses web-knowledge to assume an organization is relevant when the submitter is unrelated, causing misdirected follow-up.
  - Privacy breach: the model attempts to infer or reveal identifying information from minimal clues (e.g., ZIP, uncommon job title), exposing sensitive data.
  - Role confusion: the model treats an advocacy group's general position as expert technical input and escalates it as if it were subject-matter expertise.
  - Proxy discrimination: the judge downgrades submissions from areas or institutions correlated with protected attributes, producing unequal escalation outcomes.

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

    description: ClassVar[str] = "**Stakeholder identity and relevance** Clear identification of the submitting entity and its stake or expertise in the matter without relying on protected attributes."

    def __init__(self, model: dspy.LM = DEFAULT_MODEL):
        super().__init__(
            name="Stakeholder_identity_and_relevance_gpt-5-mini",
            description="**Stakeholder identity and relevance** Clear identification of the submitting entity and its stake or expertise in the matter without relying on protected attributes.",
            axis="**Stakeholder identity and relevance** Clear identification of the submitting entity and its stake or expertise in the matter without relying on protected attributes.",
            model=model,
            task_description="Rank candidate policy feedback drafts for follow-up: given a citizen\u2019s submission, \n    determine which responses merit escalation to agency officials.",
            metric_card="provided",
            max_workers=32,
        )

    def __repr__(self):
        return f"Stakeholder_identity_and_relevance_gpt_5_mini_LLMJudge(model=dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv(\"OPENAI_API_KEY\")))"

