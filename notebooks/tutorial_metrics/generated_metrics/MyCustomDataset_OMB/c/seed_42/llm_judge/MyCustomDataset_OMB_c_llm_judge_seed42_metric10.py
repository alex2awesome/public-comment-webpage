# Auto-generated metric file for Follow_Up_Path_and_Contactability_gpt-5-mini
import dspy
import os
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import GeneratedRefFreeLLMJudgeMetric
from typing import ClassVar

DEFAULT_MODEL = dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv("OPENAI_API_KEY"))

class Follow_Up_Path_and_Contactability_gpt_5_mini_LLMJudge(GeneratedRefFreeLLMJudgeMetric):
    """---
# Metric Card for Follow_Up_Path_and_Contactability_gpt-5-mini

**Follow Up Path and Contactability** The draft offers a concrete path for next steps meeting request timelines and contact information for response.

## Metric Details

**Follow_Up_Path_and_Contactability_gpt-5-mini** is a **reference-free** LLM-as-a-Judge metric that prompts an LLM to rate a system output along a single, run-time-specified evaluation axis.
In this case the axis is `**Follow Up Path and Contactability** The draft offers a concrete path for next steps meeting request timelines and contact information for response.`.

The prompt supplies:

1. **Task description** *d*
2. **Axis rubric** `**Follow Up Path and Contactability** The draft offers a concrete path for next steps meeting request timelines and contact information for response.`
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
  - **Axis rubric** `**Follow Up Path and Contactability** The draft offers a concrete path for next steps meeting request timelines and contact information for response.`
  - **Input text** *x*
  - **Output text** *y*
- **Outputs:**
  - Scalar score $\hat{s} \in \{1,2,3,4,5\}$

## Intended Use

- **Domain:** Text Evaluation / Triage
- **Tasks:** 
  - Policy feedback triage and escalation
  - Draft response ranking for agency follow-up readiness
  - Customer/citizen correspondence screening for contactability
  - Quality control of template responses for next-step clarity
  - Prioritization of messages that require routing to officials
- **Best Suited For:** 
  - Drafts are short to medium length and contain explicit procedural language (e.g., 'please meet', 'we will follow up by', 'contact at')
  - The evaluation goal is binary or scalar (present/absent or degree of concreteness) rather than fact verification
  - Responses follow institutional templates or consistent phrasing that the judge can learn to recognize
  - There is no need to validate contact information against external databases — only to check presence and format
  - High volume of submissions where rapid, consistent triage is required
- **Not Recommended For:** 
  - Situations that require external validation of contact details, calendar availability, or the truthfulness of claims
  - Cases needing nuanced legal, medical, or highly specialized policy interpretation to decide escalation
  - Inputs that are long, ambiguous, or heavily implicit about next steps or contactability
  - Workflows that involve sensitive personal data requiring strict privacy or compliance checks beyond surface text features
  - Multilingual materials or noisy OCR text where format detection and language understanding degrade

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
  - Surface-form bias: preferring drafts that explicitly list contact details and timelines regardless of their appropriateness or accuracy.
  - Authority bias: favoring drafts that invoke official-sounding steps or titles even if those steps are infeasible or irrelevant to the agency.
  - Privacy-neglect bias: underweighting concerns about exposing sensitive contact details or asking for contact where doing so would risk the submitter.
  - English/format bias: treating English-language or Western-styled contact conventions (email/phone) as normative and penalizing other cultural or local conventions.
  - Technical-access bias: assuming recipients have access to email/phone or can attend meetings, disadvantaging drafts that suggest low-bandwidth or asynchronous paths.
  - Recency/verbosity bias: equating more detailed timelines and multiple contact options with higher quality, even when succinct or limited contact is preferable.
  - Policy-ignorance bias: giving equal credit to contact paths that violate agency rules because the judge lacks domain-specific constraints.
  - Template bias: over-recognizing templated phrasings (e.g., “please contact us at”) as sufficient evidence of contactability without verifying completeness.
- **Task Misalignment Risks:** 
  - Focusing solely on contactability may ignore whether escalation is appropriate, ethical, or permitted under agency rules (e.g., privacy, confidentiality, or legal restrictions).
  - The axis conflates presence of contact/timelines with effective escalation, missing crucial dimensions like content accuracy, urgency, or jurisdictional relevance.
  - Automated prioritization based on this axis could systematically escalate low-quality but well-formatted drafts while leaving out urgent reports that lack explicit contact fields.
  - Evaluating drafts by this axis might incentivize authors to include contact info even when doing so endangers the submitter (e.g., sensitive complaints, whistleblowing).
  - The axis may promote a single-mode view of contact (meeting requests, phone, email) and fail to recognize acceptable alternative contacts (advocates, ombudspersons, anonymous reporting portals).
  - If the judge lacks access to agency-specific escalation workflows, it may recommend contact paths that conflict with mandated procedures, causing operational friction.
  - Narrow axis emphasis can produce blind spots for accessibility needs (e.g., accommodations, language access) that affect actual contactability.
- **Failure Cases:** 
  - False positive escalation: the judge flags a draft as actionable because it lists contact info, but the contact is inappropriate (private email, non-agency account) or fabricated.
  - False negative escalation: a high-priority submission lacking explicit contact fields is downgraded despite clear urgency and a need for agency follow-up.
  - Privacy breach risk: the judge recommends adding contact prompts or public meeting requests for submissions that should remain confidential, exposing the submitter to harm.
  - Policy-conflict recommendation: the judge elevates a draft that requests direct meetings contrary to agency protocols (e.g., legal review required before contact).
  - Misinterpreted timelines: ambiguous phrasing like 'as soon as possible' or 'within two weeks' is mis-scored or normalized incorrectly, leading to incorrect urgency assessments.
  - Unverifiable contact acceptance: the judge accepts embedded phone/email without verifying format, leading to escalation attempts that bounce or reach the wrong person.
  - Cultural/format misread: nonstandard contact methods (social-service caseworker, community liaison) are penalized, reducing appropriate follow-up for certain populations.
  - Template gaming: actors craft superficially complete contact/timeline fields to get escalated even when the underlying request is irrelevant or malicious.
  - Accessibility neglect: the judge recommends in-person meetings without noting accommodation needs or alternatives, creating barriers for some submitters.
  - Ambiguity in scope: the judge escalates drafts whose requested follow-up falls outside the agency's jurisdiction, wasting resources and causing delays.

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

    description: ClassVar[str] = "**Follow Up Path and Contactability** The draft offers a concrete path for next steps meeting request timelines and contact information for response."

    def __init__(self, model: dspy.LM = DEFAULT_MODEL):
        super().__init__(
            name="Follow_Up_Path_and_Contactability_gpt-5-mini",
            description="**Follow Up Path and Contactability** The draft offers a concrete path for next steps meeting request timelines and contact information for response.",
            axis="**Follow Up Path and Contactability** The draft offers a concrete path for next steps meeting request timelines and contact information for response.",
            model=model,
            task_description="Rank candidate policy feedback drafts for follow-up: given a citizen\u2019s submission, \n    determine which responses merit escalation to agency officials.",
            metric_card="provided",
            max_workers=32,
        )

    def __repr__(self):
        return f"Follow_Up_Path_and_Contactability_gpt_5_mini_LLMJudge(model=dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv(\"OPENAI_API_KEY\")))"

