# Auto-generated metric file for Novelty_and_Added_Value_Rubric
import dspy
import os
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import GeneratedRefFreeLLMJudgeMetric
from typing import ClassVar

DEFAULT_MODEL = dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv("OPENAI_API_KEY"))

class Novelty_and_Added_Value_Rubric_LLMJudge(GeneratedRefFreeLLMJudgeMetric):
    """---
# Metric Card for Novelty_and_Added_Value_Rubric

**Novelty and Added Value** Extent to which the submission introduces new information, perspectives, or solutions not already prominent in the record.

## Metric Details

**Novelty_and_Added_Value_Rubric** is a **reference-free** LLM-as-a-Judge metric that prompts an LLM to rate a system output along a single, run-time-specified evaluation axis.
In this case the axis is `**Novelty and Added Value** Extent to which the submission introduces new information, perspectives, or solutions not already prominent in the record.`.

The prompt supplies:

1. **Task description** *d*
2. **Rubric** `**Novelty and Added Value** Extent to which the submission introduces new information, perspectives, or solutions not already prominent in the record.`
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

**Criteria:** **Novelty and Added Value** Extent to which the submission introduces new information, perspectives, or solutions not already prominent in the record.

#### Scoring Rubric

| Score | Description |
|-------|-------------|
| 1 | Score 1: No novelty or value-add<br/>• Purely generic sentiment or off-topic content; no linkage to the specific rule/decision.<br/>• Duplicates widely available talking points or form-letter language; no new facts, citations, or analysis.<br/>• No concrete recommendations or implementation details.<br/>• Contains assertions without evidence and no unique stakeholder perspective.<br/>• Would not inform agency deliberation beyond what is already known. |
| 2 | Score 2: Minimal novelty; marginal value-add<br/>• Minor personalization (e.g., brief anecdote) but no broader implications, evidence, or analysis.<br/>• Reiterates common arguments/sources already pervasive in the docket (e.g., cites standard EPA/OMB guidance) without new interpretation.<br/>• Recommendations are high-level (“do more research,” “protect public health”) with no specificity.<br/>• Limited relevance to key decision levers; lacks citations or verifiability.<br/>• Unlikely to change or enrich the record meaningfully. |
| 3 | Score 3: Some novelty; modest value-add<br/>• Provides specific local/sector examples, preliminary data, or targeted observations that are not widely represented, but limited in scope or rigor.<br/>• Adds a nuance to known arguments (e.g., highlights a particular subpopulation or use case) with partial evidence or plausible logic.<br/>• Offers actionable suggestions with some implementation detail, but lacking depth, feasibility analysis, or legal/policy anchoring.<br/>• Uses citations or basic methods, though transparency or robustness is limited.<br/>• Could help refine agency understanding but unlikely to materially shift analysis on its own. |
| 4 | Score 4: Substantial novelty; strong value-add<br/>• Introduces new evidence (dataset, analysis, field measurements, comparative assessment) or an underrepresented stakeholder perspective that fills a clear gap in the record.<br/>• Proposes specific, feasible, and detailed solutions (e.g., differentiated mitigation by product/use, concrete criteria, timelines), with rationale and consideration of trade-offs.<br/>• Presents a well-supported legal/policy interpretation or procedural insight not already prominent, tied to statutory/regulatory levers.<br/>• Methods and citations are transparent; claims are verifiable; implications for agency decision points are explicit.<br/>• Likely to influence internal deliberations or prompt targeted follow-up. |
| 5 | Score 5: High originality and potential impact; decisive value-add<br/>• Contributes a unique, high-quality dataset or rigorous analysis (e.g., reproducible methods, quantification of impacts, sensitivity/scenario analysis) that materially expands the evidentiary base.<br/>• Surfaces a novel problem framing or cross-cutting perspective (e.g., overlooked pathway, equity impact, cross-agency linkage) absent from the record.<br/>• Delivers a detailed, implementable policy/technical proposal with operational steps, legal authority mapping, risk/benefit quantification, and potential metrics for evaluation.<br/>• Demonstrates exceptional credibility (expert sources, peer-reviewed or methodologically sound work, transparent assumptions) and clear relevance to the decision.<br/>• Has clear potential to change the agency’s assessment, trigger new analysis, or shape final policy design. |

### Inputs and Outputs
- **Inputs:**
  - **Task description** *d*
  - **Rubric** `**Novelty and Added Value** Extent to which the submission introduces new information, perspectives, or solutions not already prominent in the record.`
  - **Input text** *x*
  - **Output text** *y*
- **Outputs:**
  - Scalar score $\hat{s} \in \{1,2,3,4,5\}$

## Intended Use

- **Domain:** Policy feedback triage / Text evaluation
- **Tasks:** 
  - Triage and rank public comments by novelty and added value
  - Extract and summarize proposed policy and implementation details from submissions
  - Detect and flag unique datasets, measurements, or evidence claims for human follow-up
  - Identify underrepresented stakeholder perspectives or novel problem framings
  - Provide rationale tied to the scoring rubric for each ranked comment
  - Filter out generic/form-letter responses and low-value duplicative submissions
  - Prioritize comments likely to change agency deliberation for escalation
  - Generate short, structured briefing notes for agency officials about high-value submissions
- **Best Suited For:** 
  - When the full rule text, docket, and prior submissions are provided so the model can compare new material against the record
  - High-volume comment periods where rapid, consistent triage is needed to surface candidates for human review
  - When submissions are in clear, well-formed English and contain explicit proposals, citations, or data claims
  - As a first-pass prioritization tool to reduce reviewer workload and highlight potentially novel inputs
  - When the agency wants standardized, reproducible application of the specified novelty/value rubric
  - When the model’s outputs will be combined with human validation for verification and final escalation decisions
- **Not Recommended For:** 
  - When the model lacks access to the full docket or contextual materials — it may misjudge novelty without the record
  - For final, unsupervised escalation decisions in high-stakes or legally sensitive cases without expert human oversight
  - When submissions include technical datasets or methods requiring reproducibility checks or external validation the model cannot perform
  - Where regulatory or statutory interpretation hinges on subtle legal analysis beyond general language understanding
  - For multilingual submissions unsupported by the model’s reliable language proficiency or for highly idiomatic local dialects
  - When adversarial, deceptive, or fabricated evidence is suspected and forensic verification is required rather than heuristic detection

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
  - Length bias — the model may equate longer submissions with greater novelty or value, penalizing concise but original inputs.
  - Citation/formality bias — content with formal citations or academic style may be rated higher than credible lived-experience or community submissions lacking references.
  - Expertise/technical bias — technical analyses or jargon-heavy submissions may be favored, disadvantaging legitimate stakeholder perspectives.
  - Recency/training-data bias — the model’s sense of what is ‘already prominent in the record’ may reflect its training corpus rather than the actual docket, skewing novelty judgments.
  - Surface-chunking bias — the model may rely on surface features (keywords, numbers, section headings) as proxies for originality instead of substantive contribution.
  - Verification bias — unverifiable but potentially novel claims (anecdotes, proprietary data) may be downgraded compared with verifiable but low-impact citations.
  - Conservatism bias — preference for familiar policy framings can penalize genuinely novel or radical proposals.
  - Language and cultural bias — non‑native phrasing, atypical narrative structures, or culturally specific frames may be interpreted as low-quality and thus low-novelty.
  - Authority bias — submissions referencing well-known institutions or journals are treated as more credible and novel than those from smaller community groups.
  - Format bias — well-structured, enumerated recommendations are scored higher than freeform prose even when both contain similar substance.
- **Task Misalignment Risks:** 
  - Prioritizing novelty over relevance — highly novel submissions that are irrelevant to key decision levers could be escalated while routine but decision-critical comments are deprioritized.
  - Undervaluing consensus signalling — broadly repeated points that indicate stakeholder support may be low-scored for novelty despite being important for agency deliberation.
  - Favoring provable over persuasive evidence — the rubric can penalize compelling qualitative testimony that lacks citations even though it meaningfully informs policy impact.
  - Overfitting to citation presence — equating citations with added value risks missing original ideas that are pragmatic but uncited.
  - Treating uniqueness as inherently valuable — rare or idiosyncratic claims may be escalated despite being unreliable or irrelevant to statutory/regulatory levers.
  - Misinterpreting local particulars — localized or community-specific information might be marked as low novelty if the model incorrectly judges them as already represented.
  - Neglecting legal/feasibility alignment — emphasizing novelty could elevate ideas that cannot be implemented or are legally infeasible, diverting agency attention.
  - Biasing toward analyzable inputs — the model may systematically favor submissions that are easy to quantify or reproduce, misaligning with the agency’s need for contextual or equity-focused insights.
- **Failure Cases:** 
  - False positives: a long, well-cited comment restates obscure literature but adds no new insight and is incorrectly scored high for novelty.
  - False negatives: succinct, original policy proposals with clear implementation steps are scored low because they lack formal citations or verbose justification.
  - Gaming: submitters add spurious citations, fabricated datasets, or boilerplate analytic-sounding text to inflate novelty scores.
  - Inconsistent scoring: minor paraphrases of the same novel idea receive different scores due to phrasing or formatting differences.
  - Fabricated evidence undetected: the model fails to flag or verify fabricated datasets or fake citations that falsely increase perceived novelty.
  - Overlooking equity/lived experience: community testimony that contextualizes impacts is scored low for lacking generalizable data.
  - Jargon misinterpretation: domain-specific shorthand or acronyms are misread, causing underestimation of a comment’s originality.
  - Overweighting marginal anecdotes: a single unrepresentative anecdote is escalated as novel despite offering no generalizable insight.
  - Duplicate-counting: unique argument components repeated across multiple comments result in multiple escalations even though the information is not additive.
  - Feasibility blind spot: novel proposals lacking legal or operational feasibility analysis are scored high but would be impractical for agency follow-up.
  - Language penalty: non-native English submissions with high substantive novelty are downgraded because of grammatical or stylistic differences.
  - Context loss: the model rates a submission as highly novel without recognizing its dependence on other docket materials, misrepresenting standalone value.

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

    description: ClassVar[str] = "**Novelty and Added Value** Extent to which the submission introduces new information, perspectives, or solutions not already prominent in the record."

    def __init__(self, model: dspy.LM = DEFAULT_MODEL):
        super().__init__(
            name="Novelty_and_Added_Value_Rubric",
            description="**Novelty and Added Value** Extent to which the submission introduces new information, perspectives, or solutions not already prominent in the record.",
            axis="**Novelty and Added Value** Extent to which the submission introduces new information, perspectives, or solutions not already prominent in the record.\n\nScoring Guidelines:\nScore 1: Score 1: No novelty or value-add\n- Purely generic sentiment or off-topic content; no linkage to the specific rule/decision.\n- Duplicates widely available talking points or form-letter language; no new facts, citations, or analysis.\n- No concrete recommendations or implementation details.\n- Contains assertions without evidence and no unique stakeholder perspective.\n- Would not inform agency deliberation beyond what is already known.\nScore 2: Score 2: Minimal novelty; marginal value-add\n- Minor personalization (e.g., brief anecdote) but no broader implications, evidence, or analysis.\n- Reiterates common arguments/sources already pervasive in the docket (e.g., cites standard EPA/OMB guidance) without new interpretation.\n- Recommendations are high-level (\u201cdo more research,\u201d \u201cprotect public health\u201d) with no specificity.\n- Limited relevance to key decision levers; lacks citations or verifiability.\n- Unlikely to change or enrich the record meaningfully.\nScore 3: Score 3: Some novelty; modest value-add\n- Provides specific local/sector examples, preliminary data, or targeted observations that are not widely represented, but limited in scope or rigor.\n- Adds a nuance to known arguments (e.g., highlights a particular subpopulation or use case) with partial evidence or plausible logic.\n- Offers actionable suggestions with some implementation detail, but lacking depth, feasibility analysis, or legal/policy anchoring.\n- Uses citations or basic methods, though transparency or robustness is limited.\n- Could help refine agency understanding but unlikely to materially shift analysis on its own.\nScore 4: Score 4: Substantial novelty; strong value-add\n- Introduces new evidence (dataset, analysis, field measurements, comparative assessment) or an underrepresented stakeholder perspective that fills a clear gap in the record.\n- Proposes specific, feasible, and detailed solutions (e.g., differentiated mitigation by product/use, concrete criteria, timelines), with rationale and consideration of trade-offs.\n- Presents a well-supported legal/policy interpretation or procedural insight not already prominent, tied to statutory/regulatory levers.\n- Methods and citations are transparent; claims are verifiable; implications for agency decision points are explicit.\n- Likely to influence internal deliberations or prompt targeted follow-up.\nScore 5: Score 5: High originality and potential impact; decisive value-add\n- Contributes a unique, high-quality dataset or rigorous analysis (e.g., reproducible methods, quantification of impacts, sensitivity/scenario analysis) that materially expands the evidentiary base.\n- Surfaces a novel problem framing or cross-cutting perspective (e.g., overlooked pathway, equity impact, cross-agency linkage) absent from the record.\n- Delivers a detailed, implementable policy/technical proposal with operational steps, legal authority mapping, risk/benefit quantification, and potential metrics for evaluation.\n- Demonstrates exceptional credibility (expert sources, peer-reviewed or methodologically sound work, transparent assumptions) and clear relevance to the decision.\n- Has clear potential to change the agency\u2019s assessment, trigger new analysis, or shape final policy design.\n",
            model=model,
            task_description="Rank candidate policy feedback drafts for follow-up: given a citizen\u2019s submission, \n    determine which responses merit escalation to agency officials.",
            metric_card="provided",
            max_workers=32,
        )

    def __repr__(self):
        return f"Novelty_and_Added_Value_Rubric_LLMJudge(model=dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv(\"OPENAI_API_KEY\")))"

