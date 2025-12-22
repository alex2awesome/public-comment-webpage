# Auto-generated metric file for Timing_and_Urgency_Rubric
import dspy
import os
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import GeneratedRefFreeLLMJudgeMetric
from typing import ClassVar

DEFAULT_MODEL = dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv("OPENAI_API_KEY"))

class Timing_and_Urgency_Rubric_LLMJudge(GeneratedRefFreeLLMJudgeMetric):
    """---
# Metric Card for Timing_and_Urgency_Rubric

**Timing and Urgency** Presence of deadlines, imminent risks, or time-sensitive windows that increase the need for rapid escalation.

## Metric Details

**Timing_and_Urgency_Rubric** is a **reference-free** LLM-as-a-Judge metric that prompts an LLM to rate a system output along a single, run-time-specified evaluation axis.
In this case the axis is `**Timing and Urgency** Presence of deadlines, imminent risks, or time-sensitive windows that increase the need for rapid escalation.`.

The prompt supplies:

1. **Task description** *d*
2. **Rubric** `**Timing and Urgency** Presence of deadlines, imminent risks, or time-sensitive windows that increase the need for rapid escalation.`
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

**Criteria:** **Timing and Urgency** Presence of deadlines, imminent risks, or time-sensitive windows that increase the need for rapid escalation.

#### Scoring Rubric

| Score | Description |
|-------|-------------|
| 1 | Score 1 (No meaningful urgency)<br/>• No dates, timelines, or time windows referenced.<br/>• Only generic urgency language (e.g., “urgent,” “ASAP”) without specifics.<br/>• Deadlines mentioned are already past or clearly stale.<br/>• No described consequence for delay.<br/>• No sources or identifiers to verify timing claims. |
| 2 | Score 2 (Weak or implied urgency)<br/>• Hints at timing (“soon,” “before it’s too late”) but no concrete date/time.<br/>• Self-imposed or arbitrary timelines without external driver.<br/>• Low or unclear consequence if delayed.<br/>• Minimal or non-verifiable references (no docket/order/event link).<br/>• Potentially relevant timing exists but is not tied to an action or decision point. |
| 3 | Score 3 (Moderate, non-imminent or under-specified urgency)<br/>• Includes a date/window but it is not imminent (e.g., >2 weeks away) or loosely connected to the request.<br/>• Consequences are mentioned but generic or not clearly tied to the agency’s decision.<br/>• Some supporting detail (e.g., docket ID, meeting month) but incomplete citation or no link.<br/>• Asks for expedited action but lacks specific, time-bound next steps.<br/>• Could justify escalation soon, but not necessarily immediate. |
| 4 | Score 4 (Clear, near-term urgency)<br/>• Specific, externally driven deadline or event within the near term (about 3–10 business days).<br/>• Concrete consequence for missing the window (e.g., statutory close of comment period, loss of funding opportunity, scheduled vote/hearing).<br/>• Verifiable references (docket ID, public notice, court schedule) or named officials/venues.<br/>• Provides time-bound, actionable next steps appropriate for escalation.<br/>• Minor gaps in documentation or consequence detail, but overall credible and pressing. |
| 5 | Score 5 (Compelling, imminent urgency)<br/>• Imminent deadline or risk (e.g., within 24–72 hours or by a specific near date/time) clearly stated.<br/>• High-consequence outcome if missed (legal noncompliance, safety/public health risk, irreversible decision, expiring funds).<br/>• Multiple, verifiable signals (e.g., docket ID + linked notice/order + scheduled agenda) and up-to-date timestamps.<br/>• Explicit, immediate action requested with designated contacts/officials and precise timing.<br/>• Urgency is externally mandated (statute, court, scheduled proceeding) rather than self-imposed. |

### Inputs and Outputs
- **Inputs:**
  - **Task description** *d*
  - **Rubric** `**Timing and Urgency** Presence of deadlines, imminent risks, or time-sensitive windows that increase the need for rapid escalation.`
  - **Input text** *x*
  - **Output text** *y*
- **Outputs:**
  - Scalar score $\hat{s} \in \{1,2,3,4,5\}$

## Intended Use

- **Domain:** Triage / Content Evaluation (Policy Escalation Timing Assessment)
- **Tasks:** 
  - Policy feedback triage to prioritize submissions for agency escalation
  - Automatic detection and extraction of dates, deadlines, docket IDs, and named events from citizen submissions
  - Scoring submissions on timing-and-urgency using the provided rubric
  - Batch prioritization of incoming comments to meet near-term comment periods or hearings
  - Generating short rationale notes (e.g., which evidence supports the urgency score) for human reviewers
  - Flagging submissions that require immediate human verification of external references or legal deadlines
- **Best Suited For:** 
  - Submissions that include concrete dates, deadlines, docket IDs, hearing times, or named public notices that can be parsed from text.
  - High-volume intake workflows where rapid, consistent triage reduces human workload by filtering clearly time-sensitive items.
  - Situations where urgency is externally driven (e.g., statutory deadlines, scheduled votes) and the required signals appear in the submission.
  - When submissions follow semi-structured templates (forms with separate fields for dates, event type, contact, and evidence).
  - Workflows that pair automated scoring with a short human verification step for top-priority items.
- **Not Recommended For:** 
  - Submissions that require real-time verification of external links, docket statuses, or current court schedules without an external data lookup service, since the model cannot confirm live facts.
  - Complex legal interpretation about whether a cited date actually triggers an agency obligation or statutory timeline (requires legal expertise).
  - Highly ambiguous or rhetorical submissions that use vague urgency language but contain no verifiable timing information.
  - Safety-critical or irreversible decision-making where false negatives or positives would cause substantial harm without immediate human oversight.
  - Multilingual or poorly formatted inputs without reliable language detection and normalization, which can lead to missed or mis-parsed time references.

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
  - Explicitness bias: preferring submissions with concrete dates or docket IDs and downranking equally urgent but vaguely worded reports.
  - Verifiability bias: favoring submissions that include links or identifiers even when the underlying urgency is comparable to ones without those artifacts.
  - Conservatism bias: defaulting to lower urgency when evidence is ambiguous to avoid false positives, which can under-escalate real risks.
  - Rhetorical-sensitivity bias: being misled by urgent-sounding language (e.g., “urgent,” “ASAP”) even when no real external deadline exists.
  - Source-format bias: better scoring for submissions that match familiar formats (legal docket style, meeting notices) and worse for informal or community-sourced formats.
  - Recency bias toward timestamps: privileging recent timestamps or explicit 'now' statements over context that implies time-sensitivity without dates.
  - Domain-knowledge bias: depending on generic rules of urgency rather than agency-specific statutory timing requirements, disadvantaging domain-specific judgments.
  - Link-dependence bias: overvaluing the presence of links/IDs and undervaluing credible claims without online artifacts (e.g., phone reports from constituents).
- **Task Misalignment Risks:** 
  - Equating explicitness with true urgency: the evaluation could conflate well-documented formatting with genuine need for escalation, misaligning with the task goal of surfacing real time-sensitive risks.
  - Ignoring implicit time-critical context: the model may miss submissions where urgency is implied by context (e.g., a planned event referenced without date) leading to failure to escalate when needed.
  - Over-prioritizing external verification: requiring verifiable links or docket numbers could block escalation for emergent safety reports that lack public notices.
  - Applying general business-day heuristics that conflict with agency-specific deadlines (e.g., treating a 3-day statutory window as longer/shorter), causing mis-scored urgency.
  - Treating self-imposed timelines as equal to externally mandated ones, thereby misguiding agency officials about the true immediacy of action required.
  - Failing to separate high-consequence but non-imminent issues (need policy attention) from low-consequence imminent events (need operational response), reducing signal quality.
  - Incentivizing terse, date-heavy citizen submissions (to get escalated) rather than accurate reporting, thereby changing user behavior away from the task’s intent.
  - Conflating multiple timelines in one submission (e.g., short-term and long-term asks) and failing to identify which require immediate escalation.
- **Failure Cases:** 
  - Missed imminent risk because the submitter used vague language (e.g., 'before long') without dates, causing a low score despite real urgency.
  - False escalation when a user adds fabricated docket IDs or links, because the model cannot reliably validate external resources.
  - Under-escalation of public-safety reports that lack formal documentation (e.g., reports from on-the-ground staff) due to link/verifiability heuristics.
  - Over-escalation of rhetorical pleas labeled 'urgent' that have no external deadline, wasting agency attention on low-risk items.
  - Incorrect time-window calculation because the model mishandles business days, holidays, or time zones, shifting a near-term event into 'non-imminent.'
  - Failure to detect statutory or process-driven deadlines specific to an agency, leading to misclassification of urgency level.
  - Hallucinated verification: the model invents supporting identifiers or links to justify a higher urgency score, presenting false confidence to reviewers.
  - Ambiguity-handling error where submissions containing multiple timelines get an averaged score that obscures an actually imminent item.
  - Bias against nonstandard formats (social-media screenshots, phone transcripts), producing systematically lower urgency ratings for marginalized voices.
  - Delayed escalation recommendation because the model waits for explicit 'immediate' phrasing when an actionable 48-hour window is implied by context.

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

    description: ClassVar[str] = "**Timing and Urgency** Presence of deadlines, imminent risks, or time-sensitive windows that increase the need for rapid escalation."

    def __init__(self, model: dspy.LM = DEFAULT_MODEL):
        super().__init__(
            name="Timing_and_Urgency_Rubric",
            description="**Timing and Urgency** Presence of deadlines, imminent risks, or time-sensitive windows that increase the need for rapid escalation.",
            axis="**Timing and Urgency** Presence of deadlines, imminent risks, or time-sensitive windows that increase the need for rapid escalation.\n\nScoring Guidelines:\nScore 1: Score 1 (No meaningful urgency)\n- No dates, timelines, or time windows referenced.\n- Only generic urgency language (e.g., \u201curgent,\u201d \u201cASAP\u201d) without specifics.\n- Deadlines mentioned are already past or clearly stale.\n- No described consequence for delay.\n- No sources or identifiers to verify timing claims.\nScore 2: Score 2 (Weak or implied urgency)\n- Hints at timing (\u201csoon,\u201d \u201cbefore it\u2019s too late\u201d) but no concrete date/time.\n- Self-imposed or arbitrary timelines without external driver.\n- Low or unclear consequence if delayed.\n- Minimal or non-verifiable references (no docket/order/event link).\n- Potentially relevant timing exists but is not tied to an action or decision point.\nScore 3: Score 3 (Moderate, non-imminent or under-specified urgency)\n- Includes a date/window but it is not imminent (e.g., >2 weeks away) or loosely connected to the request.\n- Consequences are mentioned but generic or not clearly tied to the agency\u2019s decision.\n- Some supporting detail (e.g., docket ID, meeting month) but incomplete citation or no link.\n- Asks for expedited action but lacks specific, time-bound next steps.\n- Could justify escalation soon, but not necessarily immediate.\nScore 4: Score 4 (Clear, near-term urgency)\n- Specific, externally driven deadline or event within the near term (about 3\u201310 business days).\n- Concrete consequence for missing the window (e.g., statutory close of comment period, loss of funding opportunity, scheduled vote/hearing).\n- Verifiable references (docket ID, public notice, court schedule) or named officials/venues.\n- Provides time-bound, actionable next steps appropriate for escalation.\n- Minor gaps in documentation or consequence detail, but overall credible and pressing.\nScore 5: Score 5 (Compelling, imminent urgency)\n- Imminent deadline or risk (e.g., within 24\u201372 hours or by a specific near date/time) clearly stated.\n- High-consequence outcome if missed (legal noncompliance, safety/public health risk, irreversible decision, expiring funds).\n- Multiple, verifiable signals (e.g., docket ID + linked notice/order + scheduled agenda) and up-to-date timestamps.\n- Explicit, immediate action requested with designated contacts/officials and precise timing.\n- Urgency is externally mandated (statute, court, scheduled proceeding) rather than self-imposed.\n",
            model=model,
            task_description="Rank candidate policy feedback drafts for follow-up: given a citizen\u2019s submission, \n    determine which responses merit escalation to agency officials.",
            metric_card="provided",
            max_workers=32,
        )

    def __repr__(self):
        return f"Timing_and_Urgency_Rubric_LLMJudge(model=dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv(\"OPENAI_API_KEY\")))"

