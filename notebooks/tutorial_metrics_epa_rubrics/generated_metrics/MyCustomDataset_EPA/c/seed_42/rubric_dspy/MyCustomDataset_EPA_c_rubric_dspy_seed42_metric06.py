# Auto-generated metric file for Feasibility_and_Implementation_Detail_Rubric
import dspy
import os
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import GeneratedRefFreeLLMJudgeMetric
from typing import ClassVar

DEFAULT_MODEL = dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv("OPENAI_API_KEY"))

class Feasibility_and_Implementation_Detail_Rubric_LLMJudge(GeneratedRefFreeLLMJudgeMetric):
    """---
# Metric Card for Feasibility_and_Implementation_Detail_Rubric

**Feasibility and Implementation Detail** Practicality of the proposal and inclusion of implementation considerations, tradeoffs, and operational details.

## Metric Details

**Feasibility_and_Implementation_Detail_Rubric** is a **reference-free** LLM-as-a-Judge metric that prompts an LLM to rate a system output along a single, run-time-specified evaluation axis.
In this case the axis is `**Feasibility and Implementation Detail** Practicality of the proposal and inclusion of implementation considerations, tradeoffs, and operational details.`.

The prompt supplies:

1. **Task description** *d*
2. **Rubric** `**Feasibility and Implementation Detail** Practicality of the proposal and inclusion of implementation considerations, tradeoffs, and operational details.`
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

**Criteria:** **Feasibility and Implementation Detail** Practicality of the proposal and inclusion of implementation considerations, tradeoffs, and operational details.

#### Scoring Rubric

| Score | Description |
|-------|-------------|
| 1 | Score 1: Not feasible; no implementation detail<br/>• Purely aspirational or rhetorical; no concrete actions the agency could take<br/>• Ignores agency authority, legal/process constraints, or proposes unlawful/unfunded actions<br/>• No identification of responsible offices, timelines, or resources<br/>• No discussion of tradeoffs, risks, monitoring, or enforcement<br/>• Misaligned to the docket/problem or requests unrelated actions |
| 2 | Score 2: Minimal feasibility; vague, incomplete<br/>• Mentions a potential step but lacks who/how/when specifics<br/>• Little to no citation of relevant authorities (statute/CFR/docket) or operational levers<br/>• Assumes unrealistic conditions (e.g., immediate compliance, unlimited resources)<br/>• Omits tradeoffs, risks, and metrics; no consideration of stakeholder or interagency needs<br/>• May conflate issues or partially misinterpret agency remit |
| 3 | Score 3: Partially feasible; some operational thinking<br/>• Identifies a plausible action pathway and references relevant programs/dockets or authorities<br/>• Outlines several implementation steps, but lacks depth on timeline, cost, staffing, or ownership<br/>• Acknowledges at least one constraint or risk, but offers limited mitigation<br/>• Touches on tradeoffs qualitatively; limited discussion of monitoring or evaluation<br/>• Leaves major open questions that would require substantial agency development |
| 4 | Score 4: Feasible and well-scoped; high implementation detail<br/>• Specifies concrete regulatory/administrative levers (e.g., proposed CFR parts/sections, label language, guidance revisions), responsible offices, and a realistic sequence of steps<br/>• Provides timeline/milestones and approximate resource needs or funding pathways<br/>• Discusses tradeoffs (e.g., compliance costs vs. health benefits) and provides risk/mitigation strategies<br/>• Addresses enforcement/oversight, data and measurement (baseline, metrics, sources), and process requirements (e.g., OMB A-4, PRA, public comment)<br/>• Notes interagency and stakeholder coordination and legal boundaries<br/>• Actionable with limited additional scoping |
| 5 | Score 5: Implementation-ready; comprehensive and practical<br/>• Supplies near plug-and-play artifacts (e.g., draft rule/guidance text, label language, templates), precise CFR/authority citations, and clear ownership across offices<br/>• Phased rollout with milestones, budget ranges and funding sources, staffing/training plans, and procurement/IT/data needs as applicable<br/>• Robust tradeoff analysis with quantified impacts/benefits and alternatives; equity/EJ and small entity impacts assessed<br/>• Detailed monitoring, reporting, and evaluation framework (metrics, baselines, data governance), plus enforcement strategy<br/>• Identifies risks/dependencies and contingency plans; aligns with legal/process constraints and includes interagency review steps<br/>• Cites precedents/pilots or empirical evidence supporting feasibility |

### Inputs and Outputs
- **Inputs:**
  - **Task description** *d*
  - **Rubric** `**Feasibility and Implementation Detail** Practicality of the proposal and inclusion of implementation considerations, tradeoffs, and operational details.`
  - **Input text** *x*
  - **Output text** *y*
- **Outputs:**
  - Scalar score $\hat{s} \in \{1,2,3,4,5\}$

## Intended Use

- **Domain:** Policy Evaluation / Regulatory Analysis
- **Tasks:** 
  - Prioritizing citizen-submitted policy feedback for escalation based on implementability
  - Assessing draft proposals for presence of regulatory levers and administrative pathways
  - Scoring proposals against the Feasibility and Implementation Detail rubric and explaining score rationale
  - Flagging missing ownership, timelines, resource estimates, and enforcement considerations
  - Suggesting concrete next steps to make a proposal more actionable (e.g., offices to engage, types of analyses needed)
  - Reviewing draft guidance/regulatory text for operational completeness at a high level
  - Comparing proposals to typical agency processes (public comment, PRA/OMB requirements) to identify compliance gaps
  - Generating checklists or template artifacts (milestones, stakeholder matrix) to move proposals toward implementation
- **Best Suited For:** 
  - Citizen submissions that propose administrative, regulatory, or guidance changes within a federal/state agency remit and cite or reference relevant dockets or statutes.
  - Proposals of moderate complexity where mapping to typical agency processes (rulemaking, guidance, labeling changes, enforcement) is sufficient to assess feasibility.
  - Workflows where rapid triage is needed to escalate only those drafts that are operationally specific and actionable.
  - Situations where the judge can access (or is provided) background context such as the relevant docket, agency organization chart, or existing guidance to ground its assessment.
  - When the goal is identifying missing implementation elements (ownership, timeline, resources, metrics) rather than producing legally binding text.
- **Not Recommended For:** 
  - Highly technical engineering, scientific, or economic proposals requiring specialist modeling, empirical data validation, or domain-specific simulation (e.g., detailed cost-benefit quantification).
  - Requests for authoritative, jurisdiction-specific legal interpretation or certified citations to statutes/CFR where post-cutoff or current accuracy is required.
  - High-stakes legal/regulatory decisions where liability or litigation risk makes agency legal counsel and subject-matter experts mandatory.
  - Classified, sensitive, or proprietary program details that cannot be evaluated without secure internal data or agency access.
  - Situations requiring real-time verification of agency capacity, budget appropriations, or internal staffing that only agency records can confirm.

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
  - Regulatory-formality bias: favors responses that use formal regulatory language and citations, penalizing plain-language but practical suggestions.
  - Domain-knowledge bias: rewards proposals that reflect the judge's familiar policy domains while downgrading feasible ideas outside its expertise.
  - Conservatism/status-quo bias: prefers incremental administrative actions over novel or transformational approaches that lack established precedents.
  - Resource-centric bias: overweights explicit budget/staff estimates and underweights low-cost operational fixes or partnerships.
  - Legalistic bias: privileges precisely cited statutes/CFR references and may penalize correct but uncited authorities.
  - Process-procedure bias: favors suggestions that map neatly onto standard administrative processes (e.g., rulemaking) over informal or community-driven implementation routes.
  - Risk-averse bias: discounts ambitious proposals for assuming higher risk even if benefits justify the risk.
  - Interagency familiarity bias: rewards proposals that name familiar federal agencies while penalizing proposals involving less-common actors or non-governmental partners.
  - Language and style bias: scores higher well-structured, detailed prose and penalizes terse, non-native-English, or differently formatted submissions.
  - Overfitting to rubric bias: encourages drafting to hit rubric keywords (timelines, milestones) even when those details would be speculative or premature.
- **Task Misalignment Risks:** 
  - Over-emphasizing implementation detail could filter out high-priority policy issues that legitimately require escalation despite lacking plug-and-play plans.
  - Penalizing novel or systemic proposals that are transformative but intentionally vague about implementation until scoping occurs.
  - Treating the presence of citations or formal language as a proxy for feasibility rather than verifying substantive alignment with agency authority.
  - Conflating operational feasibility with political or budgetary feasibility, leading to rejecting proposals that are administratively possible but politically unlikely (or vice versa).
  - Rewarding exhaustive detail from authors who have resources to craft implementation-ready drafts, thereby biasing against ordinary citizens' submissions.
  - Failing to escalate rapid-response or emergency suggestions because they lack full timelines or staffing plans that are realistic only in crisis contexts.
  - Prioritizing actions that fit conventional agency levers and missing cross-sector or community-led solutions that require new partnerships.
  - Encouraging gaming where commenters add superficial implementation language to obtain escalation, producing noise for officials.
- **Failure Cases:** 
  - False negative: A practical, low-cost community partnership proposal is scored low because it lacks formal citations and staffing estimates.
  - False positive: A long, detailed draft with plausible-sounding citations and cost estimates is scored high despite relying on incorrect or non-applicable legal authority.
  - Hallucinated verification: The judge accepts a cited statute or CFR section at face value without checking accuracy, leading to incorrect feasibility assessments.
  - Domain mismatch: The judge misidentifies the responsible office (e.g., assigns a state-level responsibility to a federal agency) and thus undervalues feasibility.
  - Overconfidence in numeric estimates: Treats an unsupported budget/timeline number as credible and escalates an implausible plan.
  - Under-appreciation of political constraints: Scores a technically feasible regulatory action highly even though it is politically or legally blocked.
  - Misreading pilot proposals: Downgrades proposals designed as pilots because they lack final rollout plans, despite being appropriate next steps.
  - Inconsistency across similar drafts: Scores similar levels of detail differently due to subtle wording or the judge's transient heuristic preferences.
  - Format-sensitivity failure: Penalizes actionable content presented in a narrative or testimonial form rather than structured implementation sections.
  - Equity/EJ blindspot: Fails to recognize that brief, equity-focused recommendations can be highly actionable, scoring them low for lacking operational metrics.
  - Timing/context failure: Uses outdated process knowledge (e.g., old OMB requirements) and mis-evaluates feasibility under current rules.
  - Enforcement ambiguity failure: Scores a plan high for outlining monitoring metrics but misses that the proposed enforcement mechanism would be unlawful or infeasible.

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

    description: ClassVar[str] = "**Feasibility and Implementation Detail** Practicality of the proposal and inclusion of implementation considerations, tradeoffs, and operational details."

    def __init__(self, model: dspy.LM = DEFAULT_MODEL):
        super().__init__(
            name="Feasibility_and_Implementation_Detail_Rubric",
            description="**Feasibility and Implementation Detail** Practicality of the proposal and inclusion of implementation considerations, tradeoffs, and operational details.",
            axis="**Feasibility and Implementation Detail** Practicality of the proposal and inclusion of implementation considerations, tradeoffs, and operational details.\n\nScoring Guidelines:\nScore 1: Score 1: Not feasible; no implementation detail\n- Purely aspirational or rhetorical; no concrete actions the agency could take\n- Ignores agency authority, legal/process constraints, or proposes unlawful/unfunded actions\n- No identification of responsible offices, timelines, or resources\n- No discussion of tradeoffs, risks, monitoring, or enforcement\n- Misaligned to the docket/problem or requests unrelated actions\nScore 2: Score 2: Minimal feasibility; vague, incomplete\n- Mentions a potential step but lacks who/how/when specifics\n- Little to no citation of relevant authorities (statute/CFR/docket) or operational levers\n- Assumes unrealistic conditions (e.g., immediate compliance, unlimited resources)\n- Omits tradeoffs, risks, and metrics; no consideration of stakeholder or interagency needs\n- May conflate issues or partially misinterpret agency remit\nScore 3: Score 3: Partially feasible; some operational thinking\n- Identifies a plausible action pathway and references relevant programs/dockets or authorities\n- Outlines several implementation steps, but lacks depth on timeline, cost, staffing, or ownership\n- Acknowledges at least one constraint or risk, but offers limited mitigation\n- Touches on tradeoffs qualitatively; limited discussion of monitoring or evaluation\n- Leaves major open questions that would require substantial agency development\nScore 4: Score 4: Feasible and well-scoped; high implementation detail\n- Specifies concrete regulatory/administrative levers (e.g., proposed CFR parts/sections, label language, guidance revisions), responsible offices, and a realistic sequence of steps\n- Provides timeline/milestones and approximate resource needs or funding pathways\n- Discusses tradeoffs (e.g., compliance costs vs. health benefits) and provides risk/mitigation strategies\n- Addresses enforcement/oversight, data and measurement (baseline, metrics, sources), and process requirements (e.g., OMB A-4, PRA, public comment)\n- Notes interagency and stakeholder coordination and legal boundaries\n- Actionable with limited additional scoping\nScore 5: Score 5: Implementation-ready; comprehensive and practical\n- Supplies near plug-and-play artifacts (e.g., draft rule/guidance text, label language, templates), precise CFR/authority citations, and clear ownership across offices\n- Phased rollout with milestones, budget ranges and funding sources, staffing/training plans, and procurement/IT/data needs as applicable\n- Robust tradeoff analysis with quantified impacts/benefits and alternatives; equity/EJ and small entity impacts assessed\n- Detailed monitoring, reporting, and evaluation framework (metrics, baselines, data governance), plus enforcement strategy\n- Identifies risks/dependencies and contingency plans; aligns with legal/process constraints and includes interagency review steps\n- Cites precedents/pilots or empirical evidence supporting feasibility\n",
            model=model,
            task_description="Rank candidate policy feedback drafts for follow-up: given a citizen\u2019s submission, \n    determine which responses merit escalation to agency officials.",
            metric_card="provided",
            max_workers=32,
        )

    def __repr__(self):
        return f"Feasibility_and_Implementation_Detail_Rubric_LLMJudge(model=dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv(\"OPENAI_API_KEY\")))"

