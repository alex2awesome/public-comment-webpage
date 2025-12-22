# Auto-generated metric file for Topic_Relevance_and_Docket_Alignment_Rubric
import dspy
import os
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import GeneratedRefFreeLLMJudgeMetric
from typing import ClassVar

DEFAULT_MODEL = dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv("OPENAI_API_KEY"))

class Topic_Relevance_and_Docket_Alignment_Rubric_LLMJudge(GeneratedRefFreeLLMJudgeMetric):
    """---
# Metric Card for Topic_Relevance_and_Docket_Alignment_Rubric

**Topic Relevance and Docket Alignment** Degree to which the response directly addresses the citizen issue and the correct rulemaking docket without drifting into unrelated topics.

## Metric Details

**Topic_Relevance_and_Docket_Alignment_Rubric** is a **reference-free** LLM-as-a-Judge metric that prompts an LLM to rate a system output along a single, run-time-specified evaluation axis.
In this case the axis is `**Topic Relevance and Docket Alignment** Degree to which the response directly addresses the citizen issue and the correct rulemaking docket without drifting into unrelated topics.`.

The prompt supplies:

1. **Task description** *d*
2. **Rubric** `**Topic Relevance and Docket Alignment** Degree to which the response directly addresses the citizen issue and the correct rulemaking docket without drifting into unrelated topics.`
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

**Criteria:** **Topic Relevance and Docket Alignment** Degree to which the response directly addresses the citizen issue and the correct rulemaking docket without drifting into unrelated topics.

#### Scoring Rubric

| Score | Description |
|-------|-------------|
| 1 | • Entirely wrong topic or docket: addresses a different rulemaking, statute, or program (e.g., TSCA, pesticides, NAAQS) than the citizen’s issue.<br/>• No mention or clear indication of the correct docket, rule title, or subject matter.<br/>• Content is generic or boilerplate, or primarily about unrelated policy areas.<br/>• Introduces citations or docket IDs that conflict with the citizen’s issue.<br/>• Would mislead triagers about applicability; merits no escalation. |
| 2 | • Predominantly misaligned: brief superficial overlap in terminology, but the substance targets another docket/rulemaking.<br/>• Confuses statutes or agencies (e.g., mixes CAA procedural rule with FIFRA or TSCA topics).<br/>• Mentions EPA broadly but lacks any concrete linkage to the correct docket or the citizen’s specific concerns.<br/>• Includes incorrect or mismatched docket IDs, or multiple unrelated docket references.<br/>• Not actionable for the citizen’s issue; generally should not be escalated. |
| 3 | • Mixed alignment: acknowledges the correct general topic area or partially references the right rule/docket, but substantial portions drift to other dockets or unrelated issues.<br/>• Addresses peripheral aspects of the citizen’s issue without engaging core concerns; may cite ambiguous or multiple dockets.<br/>• Lacks precise statutory/program context or misstates parts of it.<br/>• Some useful relevance, but noise/unrelated content is significant (roughly 40–60%).<br/>• Escalation only if no better-aligned drafts exist. |
| 4 | • Mostly aligned to the correct docket/rulemaking and the citizen’s concerns, with minor tangents or omissions.<br/>• Uses correct statutory/program framing; no conflicting docket references. May omit the explicit docket ID but clearly matches the rule title/subject.<br/>• Responds to most key points raised by the citizen with minimal drift and no cross-program confusion.<br/>• Actionable and appropriate for follow-up; usually merits escalation. |
| 5 | • Fully aligned: explicitly cites the correct docket ID and rule title, or unambiguously matches them; stays entirely within the correct statutory/program context.<br/>• Directly addresses the citizen’s specific issues point-by-point; no unrelated topics or mixed dockets.<br/>• Accurate, precise references to relevant sections, requirements, and precedents; no factual or jurisdictional drift.<br/>• Maximally actionable for agency officials; clearly merits escalation. |

### Inputs and Outputs
- **Inputs:**
  - **Task description** *d*
  - **Rubric** `**Topic Relevance and Docket Alignment** Degree to which the response directly addresses the citizen issue and the correct rulemaking docket without drifting into unrelated topics.`
  - **Input text** *x*
  - **Output text** *y*
- **Outputs:**
  - Scalar score $\hat{s} \in \{1,2,3,4,5\}$

## Intended Use

- **Domain:** Policy triage / text evaluation
- **Tasks:** 
  - Triage citizen feedback for escalation to agency officials
  - Scoring candidate response drafts for docket/topic relevance
  - Matching drafts to rulemaking dockets or rule titles
  - Flagging drafts with incorrect or conflicting docket IDs
  - Prioritizing drafts for human review based on alignment score
  - Generating short rationales explaining alignment or misalignment
- **Best Suited For:** 
  - Citizen submissions that include explicit docket IDs, rule titles, or clear descriptive keywords, enabling straightforward matching.
  - High-volume triage where consistent, repeatable relevance scoring is needed to prioritize human review.
  - Drafts and submissions written in clear, moderately formal English without extensive domain-specific jargon.
  - When the correct statutory/program context is well-defined and there are established examples of aligned vs. misaligned responses for calibration.
  - Situations where the goal is to filter out clearly wrong-topic drafts before detailed legal review by agency staff.
  - Workflows that combine automated LLM judgments with a human-in-the-loop for edge cases and final escalation decisions.
- **Not Recommended For:** 
  - Cases requiring authoritative legal interpretation, binding determinations of jurisdiction, or formal legal advice — these should be handled by qualified agency counsel.
  - Submissions that are highly ambiguous, extremely short or lacking context (no docket identifiers or subject keywords), where the model cannot disambiguate reliably.
  - When multiple overlapping dockets or evolving rulemakings exist and up-to-date docket mappings are required beyond the model’s knowledge cutoff.
  - Technical or scientific comments needing domain-expert assessment to determine relevance (e.g., complex emissions modeling), rather than surface-level topical matching.
  - Low-resource languages or nonstandard dialects where model performance on semantic matching is unreliable.
  - Adversarial or intentionally misleading drafts crafted to confuse automated triage — these require human scrutiny.

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
  - Surface-citation bias: favors drafts that explicitly include docket IDs or statute names, even when substantive relevance is weak.
  - Terminology bias: rewards drafts that use exact agency jargon or formal phrasing, disadvantaging plain-language but accurate responses.
  - Recency/training bias: leans on docket/rule examples seen during training and may overapply those patterns to new or niche rulemakings.
  - Agency conflation bias: more likely to confuse related environmental programs (e.g., mixing CAA and TSCA) if training data blurred distinctions.
  - Length/verbosity bias: assumes longer, more detailed drafts are more aligned, penalizing concise but precise replies.
  - Conservatism bias: prefers literal matches to the stated docket over reasonable inferences about implied relevance.
  - Formatting bias: favors drafts with formal citation formatting and penalizes drafts that convey correct linkage without standard formatting.
  - Language proficiency bias: rates non-native phrasing or nonstandard English as less aligned despite correct content.
  - Overfitting to examples bias: may rigidly apply narrow criteria learned from annotated examples and ignore allowable variability in real drafts.
  - Confirmation bias: if initial cues suggest a docket, the model may interpret ambiguous content as supporting that docket rather than seeking alternative explanations.
- **Task Misalignment Risks:** 
  - Over-prioritizing explicit docket mentions, causing under-escalation of drafts that correctly address the issue without naming the docket.
  - Treating docket alignment as the sole criterion and ignoring other triage-relevant factors (urgency, legal risk, stakeholder impact).
  - Penalizing drafts that deliberately avoid naming dockets for privacy or procedural reasons, misclassifying them as unrelated.
  - Failing to recognize multi-docket or multi-issue submissions and forcing a single-docket judgment when escalation should be broader.
  - Applying statutory/program context rigidly and misclassifying responses that appropriately interpret ambiguous citizen descriptions.
  - Confusing correct-but-paraphrased references with misalignment, thereby undervaluing accurate but differently-worded explanations.
  - Prioritizing surface similarity to training examples over real-world applicability, leading to inconsistent escalations across similar cases.
- **Failure Cases:** 
  - False negative: a highly relevant draft addresses the citizen’s issue precisely but omits the docket ID and is scored low for lack of explicit citation.
  - False positive: a draft includes the correct docket ID in passing but the substantive content addresses a different issue, yet receives a high score.
  - Docket confusion: model mistakes a similarly named rulemaking for the target docket and escalates an irrelevant draft.
  - Cross-program drift missed: draft mixes elements of two programs (e.g., CAA and FIFRA) and the model fails to detect the confusion, producing a mid/high score.
  - Overpenalize tangents: a useful draft with a short unrelated background paragraph is downgraded disproportionately despite clear core relevance.
  - Under-detect multi-docket relevance: model forces a single-score label and misses that the draft appropriately addresses multiple dockets that require escalation.
  - Misread citizen intent: ambiguous citizen submission requires inference, and the model favors drafts that match a wrong inferred docket.
  - Formatting trap: drafts with nonstandard citation format or plain-language docket references are misclassified as generic/boilerplate.
  - Language artefact failure: non-native English wording that correctly maps to the docket is scored poorly due to syntax differences.
  - Inconsistent scoring: similar drafts receive different scores across runs due to sensitivity to minor wording or prompt noise.
  - Overreliance on training examples: novel or emerging rulemakings are mishandled because model maps them to best-fit historical dockets incorrectly.
  - Malicious or careless insertion: a draft includes an incorrect docket ID (typo or malicious), and the model accepts it as authoritative and escalates incorrectly.

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

    description: ClassVar[str] = "**Topic Relevance and Docket Alignment** Degree to which the response directly addresses the citizen issue and the correct rulemaking docket without drifting into unrelated topics."

    def __init__(self, model: dspy.LM = DEFAULT_MODEL):
        super().__init__(
            name="Topic_Relevance_and_Docket_Alignment_Rubric",
            description="**Topic Relevance and Docket Alignment** Degree to which the response directly addresses the citizen issue and the correct rulemaking docket without drifting into unrelated topics.",
            axis="**Topic Relevance and Docket Alignment** Degree to which the response directly addresses the citizen issue and the correct rulemaking docket without drifting into unrelated topics.\n\nScoring Guidelines:\nScore 1: - Entirely wrong topic or docket: addresses a different rulemaking, statute, or program (e.g., TSCA, pesticides, NAAQS) than the citizen\u2019s issue.\n- No mention or clear indication of the correct docket, rule title, or subject matter.\n- Content is generic or boilerplate, or primarily about unrelated policy areas.\n- Introduces citations or docket IDs that conflict with the citizen\u2019s issue.\n- Would mislead triagers about applicability; merits no escalation.\nScore 2: - Predominantly misaligned: brief superficial overlap in terminology, but the substance targets another docket/rulemaking.\n- Confuses statutes or agencies (e.g., mixes CAA procedural rule with FIFRA or TSCA topics).\n- Mentions EPA broadly but lacks any concrete linkage to the correct docket or the citizen\u2019s specific concerns.\n- Includes incorrect or mismatched docket IDs, or multiple unrelated docket references.\n- Not actionable for the citizen\u2019s issue; generally should not be escalated.\nScore 3: - Mixed alignment: acknowledges the correct general topic area or partially references the right rule/docket, but substantial portions drift to other dockets or unrelated issues.\n- Addresses peripheral aspects of the citizen\u2019s issue without engaging core concerns; may cite ambiguous or multiple dockets.\n- Lacks precise statutory/program context or misstates parts of it.\n- Some useful relevance, but noise/unrelated content is significant (roughly 40\u201360%).\n- Escalation only if no better-aligned drafts exist.\nScore 4: - Mostly aligned to the correct docket/rulemaking and the citizen\u2019s concerns, with minor tangents or omissions.\n- Uses correct statutory/program framing; no conflicting docket references. May omit the explicit docket ID but clearly matches the rule title/subject.\n- Responds to most key points raised by the citizen with minimal drift and no cross-program confusion.\n- Actionable and appropriate for follow-up; usually merits escalation.\nScore 5: - Fully aligned: explicitly cites the correct docket ID and rule title, or unambiguously matches them; stays entirely within the correct statutory/program context.\n- Directly addresses the citizen\u2019s specific issues point-by-point; no unrelated topics or mixed dockets.\n- Accurate, precise references to relevant sections, requirements, and precedents; no factual or jurisdictional drift.\n- Maximally actionable for agency officials; clearly merits escalation.\n",
            model=model,
            task_description="Rank candidate policy feedback drafts for follow-up: given a citizen\u2019s submission, \n    determine which responses merit escalation to agency officials.",
            metric_card="provided",
            max_workers=32,
        )

    def __repr__(self):
        return f"Topic_Relevance_and_Docket_Alignment_Rubric_LLMJudge(model=dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv(\"OPENAI_API_KEY\")))"

