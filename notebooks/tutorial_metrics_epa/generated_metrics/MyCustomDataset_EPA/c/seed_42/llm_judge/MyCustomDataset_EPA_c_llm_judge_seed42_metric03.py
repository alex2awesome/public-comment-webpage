# Auto-generated metric file for Evidentiary_Support_and_Citations_gpt-5-mini
import dspy
import os
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import GeneratedRefFreeLLMJudgeMetric
from typing import ClassVar

DEFAULT_MODEL = dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv("OPENAI_API_KEY"))

class Evidentiary_Support_and_Citations_gpt_5_mini_LLMJudge(GeneratedRefFreeLLMJudgeMetric):
    """---
# Metric Card for Evidentiary_Support_and_Citations_gpt-5-mini

**Evidentiary Support and Citations** Use of data, analyses, studies, and references that substantiate claims and quantify effects.

## Metric Details

**Evidentiary_Support_and_Citations_gpt-5-mini** is a **reference-free** LLM-as-a-Judge metric that prompts an LLM to rate a system output along a single, run-time-specified evaluation axis.
In this case the axis is `**Evidentiary Support and Citations** Use of data, analyses, studies, and references that substantiate claims and quantify effects.`.

The prompt supplies:

1. **Task description** *d*
2. **Axis rubric** `**Evidentiary Support and Citations** Use of data, analyses, studies, and references that substantiate claims and quantify effects.`
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
  - **Axis rubric** `**Evidentiary Support and Citations** Use of data, analyses, studies, and references that substantiate claims and quantify effects.`
  - **Input text** *x*
  - **Output text** *y*
- **Outputs:**
  - Scalar score $\hat{s} \in \{1,2,3,4,5\}$

## Intended Use

- **Domain:** Policy feedback evaluation / Text evaluation / Evidence assessment
- **Tasks:** 
  - Triage citizen-submitted feedback drafts for escalation
  - Rank candidate responses by strength of evidentiary support
  - Identify missing or weak citations and request substantiation
  - Flag claims that cite low-credibility or irrelevant sources
  - Recommend whether a draft should be escalated to agency officials based on evidence strength
  - Summarize the supporting evidence and note key gaps or uncertainties
  - Prepare concise evidence-quality notes for human reviewers
- **Best Suited For:** 
  - Drafts include explicit citations or links that are machine-readable (DOIs, URLs, full bibliographic references).
  - The domain relies on publicly available literature or government reports that can be programmatically retrieved for verification.
  - High-volume triage is required to prioritize which drafts need human escalation based on citation quality.
  - The goal is to enforce a consistent, repeatable checklist-style assessment of evidentiary support across many submissions.
  - Non-technical evidentiary judgments (e.g., presence/absence of citation, recency, basic source type) are sufficient for escalation decisions.
- **Not Recommended For:** 
  - Claims depend on proprietary, paywalled, or non-digitized sources that the model cannot access to verify content.
  - Evaluations require deep domain expertise (complex clinical trial methodology, advanced econometrics, or specialized lab methods) where expert peer review is necessary.
  - Inputs are primarily anecdotal, experiential, or normative claims without empirical anchors — where evidentiary criteria are inappropriate.
  - There is a high risk environment where incorrect triage could cause legal/regulatory harm without human expert oversight.
  - Citations are ambiguous, incomplete, or suspected to be fabricated — requiring human-led source verification and forensic checks.

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
  - Preference for formal, peer-reviewed, and English-language sources, disadvantaging lived experience, local knowledge, non-English, or grey literature.
  - Tendency to favor quantitative claims and large-N studies over qualitative or case-based evidence, undervaluing narrative or contextual insights.
  - Recency bias that gives undue weight to newer studies even when older foundational work is more relevant or methodologically stronger.
  - Authority bias favoring well-known institutions (Ivy-league, major journals, large think tanks) over smaller or community-based sources.
  - Publication bias where published positive-result studies are prioritized and null or negative findings are underweighted.
  - Accessibility bias that penalizes evidence behind paywalls or in obscure outlets despite potential quality.
  - Geographic/jurisdictional bias favoring evidence from high-income countries and national contexts over local or low-income contexts.
  - Citation-count or prominence bias where frequently-cited works are assumed to be higher quality without assessing methods.
  - Over-reliance on surface cues (URLs, DOI patterns) which can be gamed or may not reflect true credibility.
  - Model-specific tendency to hallucinate or fabricate citations and details when assessing evidentiary support.
- **Task Misalignment Risks:** 
  - Over-prioritizing formal citations could cause the judge to de-escalate submissions where lived experience or legal/regulatory concerns should trigger agency attention.
  - The axis may push the judge to escalate only when quantitative estimates are present, missing policy risks that require precautionary action without strong quantification.
  - Emphasis on citation presence can incentivize gaming by submitters who add irrelevant or low-quality citations to appear evidence-based.
  - The judge may conflate 'having many citations' with 'strong evidence' and ignore methodological flaws, leading to false escalations.
  - Demanding jurisdictionally specific evidence might exclude important cross-jurisdictional lessons that are nevertheless relevant to agencies.
  - Strict evidentiary thresholds could systematically disadvantage under-resourced communities and skew escalation toward well-funded stakeholders.
  - The axis may not align with urgent/time-sensitive policy contexts where rapid escalation is needed despite incomplete evidence.
  - Focusing on citation form over content relevance can divert attention from policy feasibility, legal constraints, or equity impacts that matter for escalation.
- **Failure Cases:** 
  - False positive escalation: a draft contains many citations from low-quality or irrelevant sources and is escalated despite weak substantive support.
  - False negative de-escalation: a submission based on strong lived experience or legal obligation lacks formal citations and is incorrectly marked as low-evidence.
  - Hallucinated citations: the LLM Judge invents plausible-looking references (authors, titles, DOIs) when assessing support.
  - Misattribution: the judge credits claims to high-quality sources when the cited work actually contradicts or does not support the claim.
  - Overemphasis on quantitative precision: the judge demands precise effect sizes and downranks legitimate uncertainty-aware recommendations.
  - Jurisdictional error: the judge rates foreign or international evidence as irrelevant even when it is applicable to the agency's context.
  - Paywall bias: high-quality evidence behind paywalls is treated as less valuable because the judge cannot verify content.
  - Outdated-evidence reliance: the judge cites older studies as current support despite newer contradictory research existing beyond its knowledge window.
  - Sensitivity to citation formatting: well-supported drafts with informal or nonstandard citations are penalized for form over substance.
  - Gaming vulnerability: malicious submitters attach numerous irrelevant but authoritative-looking citations to trigger escalation.
  - Correlation/causation confusion: the judge accepts correlational studies as proving causal policy effects and escalates inappropriately.
  - Excessive conservatism: the judge suppresses escalation for novel but plausible concerns that lack established literature, reducing agency situational awareness.

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

    description: ClassVar[str] = "**Evidentiary Support and Citations** Use of data, analyses, studies, and references that substantiate claims and quantify effects."

    def __init__(self, model: dspy.LM = DEFAULT_MODEL):
        super().__init__(
            name="Evidentiary_Support_and_Citations_gpt-5-mini",
            description="**Evidentiary Support and Citations** Use of data, analyses, studies, and references that substantiate claims and quantify effects.",
            axis="**Evidentiary Support and Citations** Use of data, analyses, studies, and references that substantiate claims and quantify effects.",
            model=model,
            task_description="Rank candidate policy feedback drafts for follow-up: given a citizen\u2019s submission, \n    determine which responses merit escalation to agency officials.",
            metric_card="provided",
            max_workers=32,
        )

    def __repr__(self):
        return f"Evidentiary_Support_and_Citations_gpt_5_mini_LLMJudge(model=dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv(\"OPENAI_API_KEY\")))"

