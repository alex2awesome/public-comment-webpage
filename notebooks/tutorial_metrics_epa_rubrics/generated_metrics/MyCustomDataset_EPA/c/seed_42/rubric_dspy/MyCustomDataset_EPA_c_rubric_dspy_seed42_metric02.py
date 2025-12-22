# Auto-generated metric file for Specificity_of_Ask_and_Actionability_Rubric
import dspy
import os
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import GeneratedRefFreeLLMJudgeMetric
from typing import ClassVar

DEFAULT_MODEL = dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv("OPENAI_API_KEY"))

class Specificity_of_Ask_and_Actionability_Rubric_LLMJudge(GeneratedRefFreeLLMJudgeMetric):
    """---
# Metric Card for Specificity_of_Ask_and_Actionability_Rubric

**Specificity of Ask and Actionability** Clarity of concrete requests or decisions sought from the agency, with steps the agency can realistically take.

## Metric Details

**Specificity_of_Ask_and_Actionability_Rubric** is a **reference-free** LLM-as-a-Judge metric that prompts an LLM to rate a system output along a single, run-time-specified evaluation axis.
In this case the axis is `**Specificity of Ask and Actionability** Clarity of concrete requests or decisions sought from the agency, with steps the agency can realistically take.`.

The prompt supplies:

1. **Task description** *d*
2. **Rubric** `**Specificity of Ask and Actionability** Clarity of concrete requests or decisions sought from the agency, with steps the agency can realistically take.`
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

**Criteria:** **Specificity of Ask and Actionability** Clarity of concrete requests or decisions sought from the agency, with steps the agency can realistically take.

#### Scoring Rubric

| Score | Description |
|-------|-------------|
| 1 | Score 1: No clear ask or not actionable<br/>• No explicit request or decision sought; purely opinion/rhetoric.<br/>• Requests are unrelated to the docket/topic or outside agency authority.<br/>• Lacks any concrete step, timeline, or responsible entity.<br/>• No references to rule text, docket, or statutory/regulatory levers.<br/>• Demands broad outcomes (e.g., “fix pollution”) without means. |
| 2 | Score 2: Implicit or vague ask; low actionability<br/>• General support/oppose posture with at most a broad, non-specific request.<br/>• Asks framed in vague terms (“consider,” “do better”) without how-to.<br/>• Proposed actions are unrealistic or largely outside agency control.<br/>• No identification of who at the agency should act or when.<br/>• Minimal linkage to docket/sections; little to no evidence or rationale. |
| 3 | Score 3: Clear primary ask but limited operational detail<br/>• A specific decision or direction is stated (e.g., retain/withdraw a proposal).<br/>• Feasible within agency authority, but implementation steps are thin.<br/>• May name the docket/rule, but lacks citations to sections or draft text.<br/>• Little specificity on timelines, metrics, or responsible office.<br/>• Few or no alternatives/contingencies; limited evidentiary support. |
| 4 | Score 4: Concrete, feasible asks with clear implementation guidance<br/>• Multiple specific requests tied to the docket/rule sections or program area.<br/>• Identifies the responsible office/unit and suggests realistic steps (e.g., revise label language; exclude retrospective reviews; include co-benefits in BCA).<br/>• Cites relevant statutes/caselaw/precedents; provides supporting data.<br/>• May propose thresholds, criteria, or process steps; suggests timelines.<br/>• Some ambiguity remains (e.g., no draft text or detailed monitoring plan). |
| 5 | Score 5: Highly specific and fully actionable plan<br/>• Enumerated asks each tied to exact levers (e.g., CFR/docket section edits, guidance updates, label text), with proposed wording or redlines.<br/>• Assigns responsibility (office/role), timeline/milestones, and success metrics/thresholds.<br/>• Grounds requests in statutory authority and constraints; includes feasible alternatives/fallbacks.<br/>• Anticipates implementation (e.g., data needs, reporting, stakeholder engagement, review panels) and proposes verification/monitoring.<br/>• Clearly within agency remit and immediately executable; provides contact and commitment to assist. |

### Inputs and Outputs
- **Inputs:**
  - **Task description** *d*
  - **Rubric** `**Specificity of Ask and Actionability** Clarity of concrete requests or decisions sought from the agency, with steps the agency can realistically take.`
  - **Input text** *x*
  - **Output text** *y*
- **Outputs:**
  - Scalar score $\hat{s} \in \{1,2,3,4,5\}$

## Intended Use

- **Domain:** Policy evaluation / Regulatory comment triage / Text classification
- **Tasks:** 
  - Triage and prioritization of public comments for agency escalation
  - Automated scoring of policy feedback drafts against a rubric (specificity/actionability)
  - Highlighting and extracting explicit asks, timelines, and responsible offices from submissions
  - Generating summary rationales for escalation decisions to assist human reviewers
  - Batch processing and ranking of candidate responses for docket-focused review
- **Best Suited For:** 
  - High-volume review contexts where consistent, rubric-based scoring is required to surface the most actionable submissions for human attention.
  - Submissions in English with reasonably clear prose and standard regulatory vocabulary (docket IDs, section references, statutory citations).
  - When the agency can provide the model with contextual metadata (docket number, relevant rule text, examples of scored comments) to calibrate judgments.
  - Triage workflows that require highlighting exact language supporting a score (e.g., explicit wording of asks, named offices, timelines) rather than final legal determinations.
  - Scenarios where the goal is to prioritize or summarize items for legal/policy staff who will conduct deeper review and decide on escalation.
- **Not Recommended For:** 
  - Cases that require definitive legal interpretation of statutes, delegation-of-authority questions, or binding legal advice without human legal review.
  - Highly technical scientific or engineering feasibility assessments that require domain specialists and empirical validation.
  - Comments in languages other than those the model was trained/tuned on, or submissions with heavy use of nonstandard abbreviations or poor OCR quality.
  - Situations with extremely low tolerance for hallucination (e.g., automatic redlining or issuing directives) where outputs would be acted on without human oversight.
  - Environments lacking representative training examples or contextual docket/rule information, which increases risks of inconsistent or inaccurate scoring.

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
  - Surface-form bias: overvaluing explicit citations, docket numbers, or legalistic phrasing even when underlying requests are impractical.
  - Recency and training-data bias: relying on patterns in training corpora that may not reflect current statutes, agency structures, or recent rulemaking practices.
  - Formality bias: favoring submissions that use technical or formal language and penalizing plain-language citizen input regardless of substance.
  - Jurisdictional bias: assuming common agency responsibilities or structures and misattributing which office/unit can act.
  - Conservatism bias: penalizing novel or unconventional but feasible implementation ideas because they deviate from familiar templates.
  - Confirmation bias from keywords: interpreting presence of certain words (e.g., 'CFR', 'docket', 'redline') as proxies for high actionability.
  - Language and cultural bias: lower scoring for submissions that use different rhetorical styles, idioms, or non-native-English phrasing.
  - Overconfidence bias: producing confident judgments about legal or operational feasibility despite lacking authoritative verification.
- **Task Misalignment Risks:** 
  - Keyword overfitting: optimizing for presence of axis-specific words (docket, CFR, redline) rather than true operational specificity, misaligning evaluation with real-world implementability.
  - Scope mismatch: treating any detailed request as actionable even when it requires funding, interagency coordination, or statutory change outside the agency's authority.
  - Overemphasis on formal legal markers: penalizing valid citizen asks that propose practical, program-level changes without formal citations.
  - Inflexible granularity: applying the same specificity standard to short public comments and to long policy memos, producing unfair comparisons.
  - Neglect of feasibility constraints: scoring high on 'specificity' without assessing whether the agency has the personnel, budget, or statutory authority to execute the plan.
  - Single-axis tunnel vision: ignoring other relevant evaluation axes (e.g., equity, evidence quality, legal risk), causing escalation of technically specific but harmful or legally invalid requests.
  - Misinterpretation of delegated asks: failing to detect when the citizen directs action to another entity (state, private sector), and thus mis-scoring agency-relevance.
  - Temporal mismatch: rewarding immediate-execution specificity when the agency's appropriate response is long-term rulemaking or research, not immediate operational changes.
- **Failure Cases:** 
  - False positive: a comment includes detailed-sounding redlines and office names but proposes actions that violate statute or require Congressional appropriation, yet receives a top score.
  - False negative: a succinct, plain-language submission that names a feasible procedural change (e.g., alter intake workflow) lacks citations and is scored low despite being immediately implementable.
  - Inconsistent calibration: near-duplicate comments with minor wording differences get different scores because the model latches onto differing surface tokens.
  - Gaming: submitters deliberately inject legalistic boilerplate (fake docket numbers, mock citations) to get higher escalation scores which the model cannot reliably verify.
  - Jurisdiction error: the model assigns responsibility to the wrong office/unit (e.g., regional enforcement vs. rulemaking staff) leading to improper escalation routing.
  - Context miss: the model fails to recognize that a requested metric or timeline is unrealistic given known rulemaking timelines and scores it as actionable.
  - Ambiguity failure: when a comment contains multiple asks with mixed levels of specificity, the model either averages out and obscures the actionable element or incorrectly focuses on the least actionable part.
  - Adversarial phrasing: politically charged or rhetorical language masks concrete asks that the model dismisses as opinion, missing otherwise actionable items.
  - Evidence-check failure: the model accepts citations or data claims at face value and rates an ask higher despite the underlying evidence being irrelevant or fabricated.
  - Cross-axis conflict: the model escalates a highly specific but legally risky proposal because it optimizes for this axis alone, ignoring legal risk that would disqualify escalation.

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

    description: ClassVar[str] = "**Specificity of Ask and Actionability** Clarity of concrete requests or decisions sought from the agency, with steps the agency can realistically take."

    def __init__(self, model: dspy.LM = DEFAULT_MODEL):
        super().__init__(
            name="Specificity_of_Ask_and_Actionability_Rubric",
            description="**Specificity of Ask and Actionability** Clarity of concrete requests or decisions sought from the agency, with steps the agency can realistically take.",
            axis="**Specificity of Ask and Actionability** Clarity of concrete requests or decisions sought from the agency, with steps the agency can realistically take.\n\nScoring Guidelines:\nScore 1: Score 1: No clear ask or not actionable\n- No explicit request or decision sought; purely opinion/rhetoric.\n- Requests are unrelated to the docket/topic or outside agency authority.\n- Lacks any concrete step, timeline, or responsible entity.\n- No references to rule text, docket, or statutory/regulatory levers.\n- Demands broad outcomes (e.g., \u201cfix pollution\u201d) without means.\nScore 2: Score 2: Implicit or vague ask; low actionability\n- General support/oppose posture with at most a broad, non-specific request.\n- Asks framed in vague terms (\u201cconsider,\u201d \u201cdo better\u201d) without how-to.\n- Proposed actions are unrealistic or largely outside agency control.\n- No identification of who at the agency should act or when.\n- Minimal linkage to docket/sections; little to no evidence or rationale.\nScore 3: Score 3: Clear primary ask but limited operational detail\n- A specific decision or direction is stated (e.g., retain/withdraw a proposal).\n- Feasible within agency authority, but implementation steps are thin.\n- May name the docket/rule, but lacks citations to sections or draft text.\n- Little specificity on timelines, metrics, or responsible office.\n- Few or no alternatives/contingencies; limited evidentiary support.\nScore 4: Score 4: Concrete, feasible asks with clear implementation guidance\n- Multiple specific requests tied to the docket/rule sections or program area.\n- Identifies the responsible office/unit and suggests realistic steps (e.g., revise label language; exclude retrospective reviews; include co-benefits in BCA).\n- Cites relevant statutes/caselaw/precedents; provides supporting data.\n- May propose thresholds, criteria, or process steps; suggests timelines.\n- Some ambiguity remains (e.g., no draft text or detailed monitoring plan).\nScore 5: Score 5: Highly specific and fully actionable plan\n- Enumerated asks each tied to exact levers (e.g., CFR/docket section edits, guidance updates, label text), with proposed wording or redlines.\n- Assigns responsibility (office/role), timeline/milestones, and success metrics/thresholds.\n- Grounds requests in statutory authority and constraints; includes feasible alternatives/fallbacks.\n- Anticipates implementation (e.g., data needs, reporting, stakeholder engagement, review panels) and proposes verification/monitoring.\n- Clearly within agency remit and immediately executable; provides contact and commitment to assist.\n",
            model=model,
            task_description="Rank candidate policy feedback drafts for follow-up: given a citizen\u2019s submission, \n    determine which responses merit escalation to agency officials.",
            metric_card="provided",
            max_workers=32,
        )

    def __repr__(self):
        return f"Specificity_of_Ask_and_Actionability_Rubric_LLMJudge(model=dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv(\"OPENAI_API_KEY\")))"

