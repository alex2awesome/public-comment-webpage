# Auto-generated metric file for Stakeholder_Impact_Magnitude_Rubric
import dspy
import os
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import GeneratedRefFreeLLMJudgeMetric
from typing import ClassVar

DEFAULT_MODEL = dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv("OPENAI_API_KEY"))

class Stakeholder_Impact_Magnitude_Rubric_LLMJudge(GeneratedRefFreeLLMJudgeMetric):
    """---
# Metric Card for Stakeholder_Impact_Magnitude_Rubric

**Stakeholder Impact Magnitude** Scope and severity of anticipated impacts on public health, environment, or regulated entities, including who is affected and how much.

## Metric Details

**Stakeholder_Impact_Magnitude_Rubric** is a **reference-free** LLM-as-a-Judge metric that prompts an LLM to rate a system output along a single, run-time-specified evaluation axis.
In this case the axis is `**Stakeholder Impact Magnitude** Scope and severity of anticipated impacts on public health, environment, or regulated entities, including who is affected and how much.`.

The prompt supplies:

1. **Task description** *d*
2. **Rubric** `**Stakeholder Impact Magnitude** Scope and severity of anticipated impacts on public health, environment, or regulated entities, including who is affected and how much.`
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

### Rubric Details

**Criteria:** **Stakeholder Impact Magnitude** Scope and severity of anticipated impacts on public health, environment, or regulated entities, including who is affected and how much.

#### Scoring Rubric

| Score | Description |
|-------|-------------|
| 1 | Score 1 (Minimal/unclear impact)<br/>• Does not identify specific stakeholders beyond the author, or uses generic phrases (“people,” “businesses”) without detail.<br/>• No quantification of impacts; purely opinion or anecdote with no credible linkage to broader consequences.<br/>• Scope appears trivial or isolated; no geographic context; no mention of severity, duration, or reversibility.<br/>• Lacks references or factual support; speculative or hypothetical harms/benefits. |
| 2 | Score 2 (Low/narrow impact)<br/>• Identifies a stakeholder group but limited to a small, localized set (e.g., a single facility or neighborhood) without showing broader relevance.<br/>• Impacts described qualitatively with minimal or imprecise numbers (e.g., “some costs,” “could affect a few”).<br/>• Severity unclear or appears modest; little discussion of health/environmental endpoints or regulatory/economic consequences.<br/>• Sparse or weak support; limited connection to the specific policy decision. |
| 3 | Score 3 (Moderate/limited breadth)<br/>• Clearly names at least one stakeholder category (e.g., a sector, community, agency) and provides some magnitude indicators (rough counts, ranges, concentrations, or examples).<br/>• Impact scope is regional or sector-specific, or potentially broader but not fully substantiated.<br/>• Describes meaningful but not critical severity (e.g., compliance costs, localized health risks) with partial discussion of duration or reversibility.<br/>• Provides some data or citations, but quantification and linkage to the policy outcome are incomplete. |
| 4 | Score 4 (High/substantial impact)<br/>• Specifies multiple stakeholder groups and/or large constituencies (e.g., statewide/multi-state populations, national industry associations, hundreds of facilities).<br/>• Provides concrete, policy-relevant quantification: numbers of people/facilities/states; pollutant levels (ppm, mg/m3); dollar costs; job or service impacts; compliance or enforcement risks.<br/>• Articulates serious health/environmental or operational/economic consequences, including vulnerable populations and/or equity considerations; discusses timing (urgency), duration, and potential irreversibility.<br/>• Supports claims with credible sources or agency data and shows a clear causal link to the proposed action. |
| 5 | Score 5 (Critical/exceptional impact warranting escalation)<br/>• Demonstrates national or multi-sector consequences with severe public health/environmental harms or major economic/operational disruption; clearly identifies very large affected populations or regulated entities.<br/>• Delivers precise, comprehensive quantification (e.g., estimated cases, exposures, deaths, emissions changes, billions in costs, number of jurisdictions/facilities affected) and compares scenarios to show magnitude.<br/>• Highlights statutory/regulatory ramifications (e.g., sanctions, NAAQS nonattainment, ESA implications), cross-agency/cascading effects, irreversibility, and urgent time windows.<br/>• Elevates equity by detailing disproportionate impacts on vulnerable/overburdened communities.<br/>• Substantiated with multiple credible sources, datasets, or expert analyses; strong docket-specific linkage that makes the high-impact consequences credible and imminent. |

### Inputs and Outputs
- **Inputs:**
  - **Task description** *d*
  - **Rubric** `**Stakeholder Impact Magnitude** Scope and severity of anticipated impacts on public health, environment, or regulated entities, including who is affected and how much.`
  - **Input text** *x*
  - **Output text** *y*
- **Outputs:**
  - Scalar score $\hat{s} \in \{1,2,3,4,5\}$

## Intended Use

- **Domain:** Policy analysis / Regulatory review (Text classification & Triage)
- **Tasks:** 
  - Triage public comments for escalation to agency officials
  - Prioritize docket submissions by potential public-health/environmental impact
  - Extract and categorize named stakeholders and affected populations
  - Detect and quantify claims of harm and compare to rubric thresholds
  - Identify presence and quality of supporting evidence or citations
  - Flag comments citing statutory/regulatory triggers (e.g., NAAQS, ESA) for legal review
  - Summarize magnitude-related arguments for briefings to decision-makers
  - Batch-screen large volumes of feedback to surface high-impact submissions
- **Best Suited For:** 
  - Large volumes of citizen/organizational submissions where rapid, consistent triage is needed
  - Inputs that contain explicit stakeholder names, numeric data, or citations that the model can parse against the rubric
  - Policy areas with clear, known regulatory metrics or thresholds that map well to the scoring guidance
  - When used as preliminary screening to reduce human workload and surface likely high-impact items
  - Situations that require consistent application of a defined rubric across many short textual submissions
  - When docket-specific context and any relevant regulatory references are provided along with the submissions
- **Not Recommended For:** 
  - Submissions that hinge on highly technical, domain-specific evidence (e.g., advanced epidemiology, toxicology, complex engineering failure modes) without supporting documentation or domain expert review
  - Cases requiring verification against live, proprietary, or classified datasets (the model cannot access or confirm external data)
  - Inputs with vague, ambiguous, or rhetorical language lacking concrete claims where the model might overinterpret intent
  - Legal or compliance determinations that must be made by counsel (e.g., formal statutory triggering, enforcement decisions)
  - Assessing lived-experience or equity impacts that require community validation or qualitative field investigation
  - Final escalation decisions for critical/high-impact items without human subject-matter expert confirmation

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
  - Availability/data bias — favors submissions with explicit numeric data or citations and downranks credible-but-uncited lived-experience or local knowledge.
  - Quantification bias — overprefers quantifiable impacts and undervalues qualitative or hard-to-measure harms (e.g., psychosocial impacts).
  - Visibility/stakeholder bias — favors large, well-documented stakeholder groups (industry, statewide populations) and under-recognizes small, marginalized, or informal communities.
  - Recency/publication bias — higher weight given to recent or widely circulated sources versus older but still relevant evidence.
  - Regulatory familiarity bias — judges with stronger knowledge of particular statutes/regulations will rate impacts differently than those lacking that context.
  - Severity-amplification bias — language that uses vivid descriptors may be interpreted as higher impact even without supporting data.
  - Conservatism/precaution bias — tendency to treat ambiguous harms as low-impact to avoid false positives (or conversely to escalate to be safe), depending on model calibration.
  - Citation-credibility bias — treats presence of any citation as strong corroboration even if the source is weak or non-docket-specific.
  - Geographic/urban bias — assumes impacts described in urban or high-profile areas are broader than similar impacts in rural or less-visible areas.
  - Source-type bias — prefers government/academic sources over community groups or gray-literature, potentially missing actionable local evidence.
- **Task Misalignment Risks:** 
  - Overemphasis on numeric thresholds — the axis rewards precise quantification, which may push the judge to flag only submissions that can supply numbers even when qualitative arguments merit escalation.
  - Conflating credibility with impact — powerful rhetoric or polished evidence may be scored as higher impact regardless of actual stakeholder magnitude.
  - Neglecting cumulative or synergistic impacts — the judge might evaluate submissions in isolation rather than considering how they interact with existing exposures or policies.
  - Equity underweighting — although equity is in the rubric, automated evaluation may fail to detect disproportionate effects on marginalized communities without explicit mention.
  - Scope-drift to other axes — the judge could conflate clarity, relevance, or timeliness with impact magnitude (e.g., a clear, timely submission scored high despite small stakeholder size).
  - Legal/regulatory ignorance — failing to recognize statutory triggers or regulatory thresholds that materially change the significance of an impact.
  - Time-window misalignment — not recognizing urgent windows for regulatory response (e.g., comment deadlines, imminent permits) and thus mis-ranking urgency as impact.
  - Over-reliance on docket-specific linkage — penalizing generalizable high-impact concerns that lack immediate docket citations even when escalation is warranted.
- **Failure Cases:** 
  - False negative: a submission describing severe but qualitative harms in an underserved community (no citations) is scored low because it lacks numbers, missing a high-impact escalation need.
  - False positive: a well-cited industry-commissioned analysis with flawed assumptions is scored high and escalated, triggering unnecessary agency attention.
  - Missed cumulative risk: multiple submissions describing related modest harms are each scored low, but together imply substantial regional risk that is not escalated.
  - Misattributed stakeholders: the judge incorrectly infers affected populations (e.g., statewide rather than local) from ambiguous wording, skewing magnitude assessment.
  - Numeric misestimation: when a submission provides statistics but with unclear units or baselines, the judge makes incorrect conversions and mis-scores impact.
  - Ignoring vulnerable groups: impacts on children, elderly, or overburdened communities are missed when not explicitly labeled, undercounting equity concerns.
  - Temporal blindness: urgent, time-limited harms (e.g., imminent permit approval) are not flagged because the model fails to detect or weigh timing windows.
  - Source-gaming vulnerability: submitters include superficial citations or figures to appear high-impact, and the judge lacks nuance to detect weak or misapplied evidence.
  - Inconsistent scoring: similar submissions receive divergent scores due to wording differences or search heuristics, reducing reliability for escalation workflows.
  - Alert fatigue: the judge is calibrated too conservatively and escalates many marginal cases, overwhelming officials and reducing responsiveness to truly critical submissions.

## Related Metrics

- **Related Metrics:**
  - **LevenshteinDistance:** Levenshtein Distance measures the minimum number of single-character edits—insertions, deletions, or substitutions—required to transform one sequence into another.
  - **BARTScore:** BARTScore is a reference-based evaluation metric for text generation that formulates evaluation as a text generation task.
  - **PseudoPARENT:** **PseudoPARENT** is a *custom adaptation* of the PARENT metric for evaluating text generation from structured inputs.

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

    description: ClassVar[str] = "**Stakeholder Impact Magnitude** Scope and severity of anticipated impacts on public health, environment, or regulated entities, including who is affected and how much."

    def __init__(self, model: dspy.LM = DEFAULT_MODEL):
        super().__init__(
            name="Stakeholder_Impact_Magnitude_Rubric",
            description="**Stakeholder Impact Magnitude** Scope and severity of anticipated impacts on public health, environment, or regulated entities, including who is affected and how much.",
            axis="**Stakeholder Impact Magnitude** Scope and severity of anticipated impacts on public health, environment, or regulated entities, including who is affected and how much.\n\nScoring Guidelines:\nScore 1: Score 1 (Minimal/unclear impact)\n- Does not identify specific stakeholders beyond the author, or uses generic phrases (\u201cpeople,\u201d \u201cbusinesses\u201d) without detail.\n- No quantification of impacts; purely opinion or anecdote with no credible linkage to broader consequences.\n- Scope appears trivial or isolated; no geographic context; no mention of severity, duration, or reversibility.\n- Lacks references or factual support; speculative or hypothetical harms/benefits.\nScore 2: Score 2 (Low/narrow impact)\n- Identifies a stakeholder group but limited to a small, localized set (e.g., a single facility or neighborhood) without showing broader relevance.\n- Impacts described qualitatively with minimal or imprecise numbers (e.g., \u201csome costs,\u201d \u201ccould affect a few\u201d).\n- Severity unclear or appears modest; little discussion of health/environmental endpoints or regulatory/economic consequences.\n- Sparse or weak support; limited connection to the specific policy decision.\nScore 3: Score 3 (Moderate/limited breadth)\n- Clearly names at least one stakeholder category (e.g., a sector, community, agency) and provides some magnitude indicators (rough counts, ranges, concentrations, or examples).\n- Impact scope is regional or sector-specific, or potentially broader but not fully substantiated.\n- Describes meaningful but not critical severity (e.g., compliance costs, localized health risks) with partial discussion of duration or reversibility.\n- Provides some data or citations, but quantification and linkage to the policy outcome are incomplete.\nScore 4: Score 4 (High/substantial impact)\n- Specifies multiple stakeholder groups and/or large constituencies (e.g., statewide/multi-state populations, national industry associations, hundreds of facilities).\n- Provides concrete, policy-relevant quantification: numbers of people/facilities/states; pollutant levels (ppm, mg/m3); dollar costs; job or service impacts; compliance or enforcement risks.\n- Articulates serious health/environmental or operational/economic consequences, including vulnerable populations and/or equity considerations; discusses timing (urgency), duration, and potential irreversibility.\n- Supports claims with credible sources or agency data and shows a clear causal link to the proposed action.\nScore 5: Score 5 (Critical/exceptional impact warranting escalation)\n- Demonstrates national or multi-sector consequences with severe public health/environmental harms or major economic/operational disruption; clearly identifies very large affected populations or regulated entities.\n- Delivers precise, comprehensive quantification (e.g., estimated cases, exposures, deaths, emissions changes, billions in costs, number of jurisdictions/facilities affected) and compares scenarios to show magnitude.\n- Highlights statutory/regulatory ramifications (e.g., sanctions, NAAQS nonattainment, ESA implications), cross-agency/cascading effects, irreversibility, and urgent time windows.\n- Elevates equity by detailing disproportionate impacts on vulnerable/overburdened communities.\n- Substantiated with multiple credible sources, datasets, or expert analyses; strong docket-specific linkage that makes the high-impact consequences credible and imminent.\n",
            model=model,
            task_description="Rank candidate policy feedback drafts for follow-up: given a citizen\u2019s submission, \n    determine which responses merit escalation to agency officials.",
            metric_card="provided",
            max_workers=32,
        )

    def __repr__(self):
        return f"Stakeholder_Impact_Magnitude_Rubric_LLMJudge(model=dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv(\"OPENAI_API_KEY\")))"

