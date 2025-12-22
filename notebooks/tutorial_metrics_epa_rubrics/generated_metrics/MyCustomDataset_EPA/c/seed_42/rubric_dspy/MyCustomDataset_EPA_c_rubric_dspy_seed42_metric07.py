# Auto-generated metric file for Legal_and_Policy_Grounding_Rubric
import dspy
import os
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import GeneratedRefFreeLLMJudgeMetric
from typing import ClassVar

DEFAULT_MODEL = dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv("OPENAI_API_KEY"))

class Legal_and_Policy_Grounding_Rubric_LLMJudge(GeneratedRefFreeLLMJudgeMetric):
    """---
# Metric Card for Legal_and_Policy_Grounding_Rubric

**Legal and Policy Grounding** Accurate use of statutory authority, case law, guidance, and regulatory procedures to frame and justify the request.

## Metric Details

**Legal_and_Policy_Grounding_Rubric** is a **reference-free** LLM-as-a-Judge metric that prompts an LLM to rate a system output along a single, run-time-specified evaluation axis.
In this case the axis is `**Legal and Policy Grounding** Accurate use of statutory authority, case law, guidance, and regulatory procedures to frame and justify the request.`.

The prompt supplies:

1. **Task description** *d*
2. **Rubric** `**Legal and Policy Grounding** Accurate use of statutory authority, case law, guidance, and regulatory procedures to frame and justify the request.`
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

**Criteria:** **Legal and Policy Grounding** Accurate use of statutory authority, case law, guidance, and regulatory procedures to frame and justify the request.

#### Scoring Rubric

| Score | Description |
|-------|-------------|
| 1 | Score 1: No legal/policy grounding or fundamentally incorrect<br/>• No citations to statutes, regulations, case law, or guidance; or wholly irrelevant references.<br/>• Misstates or contradicts core statutory requirements (e.g., asserts cost consideration where explicitly prohibited; confuses program jurisdiction).<br/>• Off-topic relative to the docket or procedural context; does not reference the administrative record.<br/>• Requests actions clearly outside agency authority or legal process.<br/>• Contains obvious legal inaccuracies that would mislead decision-makers. |
| 2 | Score 2: Minimal and/or error-prone grounding<br/>• Mentions a law or policy area in general terms but without specific, verifiable citations (e.g., “the Clean Air Act says…”).<br/>• One or more notable errors in interpreting authority, procedure, or case law; or mismatched citations.<br/>• Weak linkage to the docket/topic; limited awareness of rulemaking posture (e.g., proposal vs. interim decision).<br/>• Requests are largely non-actionable or partially outside agency authority.<br/>• Relies mainly on opinion; minimal engagement with guidance (EO 12866, OMB A-4, EPA Guidelines) or misapplies them. |
| 3 | Score 3: Adequate but incomplete grounding<br/>• Includes some correct, relevant citations (e.g., specific statute sections or docket ID), with generally accurate interpretations.<br/>• Demonstrates basic understanding of procedural posture and agency authority, but lacks depth (e.g., few or no FR page cites; limited case law).<br/>• Requests are mostly within agency authority but not fully justified by the cited legal framework.<br/>• May omit key constraints (e.g., when costs can/cannot be considered) or rely on generic guidance references without application to the issue.<br/>• Minor inaccuracies or gaps, but not misleading in material ways. |
| 4 | Score 4: Strong, accurate legal/policy framework<br/>• Multiple precise citations (U.S.C., C.F.R., Federal Register pages, docket materials) and correct summaries of their relevance.<br/>• Correctly distinguishes statutory constraints (e.g., when cost-benefit analysis is permissible/required/prohibited) and cites controlling case law where appropriate.<br/>• Appropriately applies guidance (EO 12866, OMB Circular A-4, EPA economic/science guidelines) to the specific rule/action.<br/>• Requests are specific and clearly within agency authority (e.g., adjust analysis scope, clarify presentation of benefits/co-benefits, modify mitigation measures), with a clear legal/procedural rationale.<br/>• Minimal to no errors; arguments are well-scoped to the responsible program office and docket. |
| 5 | Score 5: Exemplary, comprehensive legal and policy grounding<br/>• Exhaustive, accurate use of statutory authority, case law, and guidance tailored to the docket (e.g., cites exact sections, FR pages, prior agency guidance, and relevant precedents like Whitman v. American Trucking; Michigan v. EPA if applicable).<br/>• Demonstrates clear command of procedural posture and regulatory requirements, including limitations on decision criteria (e.g., CAA §109 vs. other CAA provisions), and integrates record evidence (RIAs, risk assessments, SAP/SACC reports).<br/>• Translates legal grounding into precise, actionable, within-authority requests (e.g., specific analytical corrections, record supplementation, reconsideration of mitigation, clarity on benefit categorization, compliance timelines), and anticipates legal risk or alternative lawful pathways.<br/>• Accurately references and applies cross-cutting requirements where relevant (ESA consultations, TSCA best available science, public disclosure constraints) without overreach.<br/>• No inaccuracies; cites sources transparently; acknowledges countervailing constraints and explains why the proposed action remains lawful. |

### Inputs and Outputs
- **Inputs:**
  - **Task description** *d*
  - **Rubric** `**Legal and Policy Grounding** Accurate use of statutory authority, case law, guidance, and regulatory procedures to frame and justify the request.`
  - **Input text** *x*
  - **Output text** *y*
- **Outputs:**
  - Scalar score $\hat{s} \in \{1,2,3,4,5\}$

## Intended Use

- **Domain:** Legal/Policy Evaluation (Administrative Law)
- **Tasks:** 
  - Rank candidate policy feedback drafts by strength of legal/policy grounding
  - Triage public comments or citizen submissions for escalation to agency officials based on legal merit
  - Identify missing or incorrect statutory or regulatory citations in policy feedback
  - Assess whether recommended actions fall within agency authority and procedural posture
  - Flag drafts that misapply or omit relevant guidance (EO 12866, OMB A-4, agency guidance)
  - Provide suggested edits to increase specificity and citation accuracy for escalation-ready drafts
- **Best Suited For:** 
  - When drafts include explicit citations or quoted regulatory language that can be checked for internal consistency.
  - When the task is triage/quality-control across many short drafts where speed and consistent rubric application are needed.
  - When assessing whether a draft’s request is broadly within agency authority and procedural posture (e.g., proposal vs. final rule).
  - When evaluating surface-level legal accuracy and completeness (presence/absence of required citation types, correct mention of controlling constraints).
  - When reviewers need standardized, reproducible scoring against the provided 1–5 rubric for dataset labeling or workflow prioritization.
  - When the docket context and key documents (statutes, regulations, Federal Register notices) are supplied to the model as part of the prompt.
- **Not Recommended For:** 
  - When definitive, up-to-date legal research is required (novel case law, recent decisions, or FR pages beyond the model’s training cutoff).
  - For providing final legal advice or formal attorney-level certification of legal sufficiency.
  - When the underlying administrative record or docket is not provided and claims depend on record-specific evidence or citations.
  - When drafts involve classified, privileged, or confidential materials that require secure human handling.
  - For highly technical subject-matter where specialized agency program knowledge (e.g., complex statutory interplay across agencies) is essential and the model lacks domain-specific data.
  - When small citation errors (exact FR page numbers, current docket statuses) must be verified — model outputs should not be the sole source for bibliographic accuracy.

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
  - Surface-citation bias: overvaluing drafts that include many explicit citations regardless of the citations' relevance or accuracy.
  - Federal-centrism: favoring federal statutes and Supreme Court precedent over state, local, or international legal authorities that may be relevant.
  - Precedent prominence bias: overweighting well-known, frequently cited cases from training data and underweighting niche but controlling authorities.
  - Conservatism bias: preferring cautious, conservative legal framings and penalizing novel but potentially valid legal arguments.
  - Formality bias: preferring formally phrased legal language and penalizing clear, plain-language but legally sound requests.
  - English-language and US-law bias: reduced accuracy evaluating non-English submissions or legal frameworks outside the U.S. context.
  - Recency/data cutoff bias: missing very recent cases, guidance, or docket developments and thus undervaluing otherwise well-grounded feedback.
  - Citation-format bias: equating correct citation format (e.g., U.S.C., C.F.R., FR page) with legal correctness even when substance is weak.
  - Authority-mismatch bias: assuming a cited authority is controlling without checking jurisdiction or doctrinal relevance.
  - Anchoring to prompt cues: scoring heavily based on presence/absence of specific keywords from the rubric rather than holistic legal merit.
- **Task Misalignment Risks:** 
  - Overemphasis on citation completeness: the evaluator may downgrade actionable feedback that lacks exhaustive citations even though escalation is still warranted.
  - Neglecting operational/practical considerations: the rubric centers legal correctness but may ignore feasibility, urgency, or policy importance that justify escalation.
  - Confusing legal formality with material relevance: ranking may prioritize technically precise citations over persuasive, record-grounded issues.
  - Penalizing novel but plausible legal theories: the evaluator could incorrectly mark innovative arguments as weak for lacking prior precedent.
  - Ignoring docket-specific context: the judge might not properly weigh procedural posture (proposal vs. final rule) or docket history if not explicitly cited.
  - Overly narrow scope: the axis focuses on legal grounding and may miss cross-cutting issues (e.g., scientific evidence gaps) that merit escalation for different reasons.
  - Authority-overreach mixup: the model might conflate suggestions that are legally permissible but outside the specific program office’s remit as non-actionable.
  - Binary escalation framing: treating legal grounding as the sole escalation criterion can miss multi-factor judgments where legal plausibility is necessary but not sufficient.
- **Failure Cases:** 
  - Hallucinated citations: the judge invents statutes, CFR sections, FR page numbers, or cases when assessing grounding and awards or deducts points erroneously.
  - False positive escalation: the evaluator scores minimal or flawed legal grounding as strong because of superficial citations or confident wording.
  - False negative escalation: dismissing meritorious feedback that lacks explicit citations though the substance is legally sound and actionable.
  - Jurisdictional errors: the model applies federal authorities where state/local law controls (or vice versa), mis-scoping the request’s authority.
  - Misapplied precedent: citing controlling cases but applying them incorrectly to the docket’s facts or regulatory statute.
  - Citation mismatch: recognizing a citation but failing to verify that the cited text supports the asserted legal claim.
  - Procedural posture mistakes: misreading whether the rule is proposed, interim, or final and penalizing feedback that is appropriate to the actual posture.
  - Overreliance on training priors: defaulting to commonly seen interpretations (e.g., standard cost-benefit rules) even when the docket or statute requires a different approach.
  - Inconsistent scoring across similar drafts: small wording differences lead to large and unjustified score variance due to sensitivity to keywords.
  - Vulnerability to adversarial inputs: submitters can game the evaluator by adding boilerplate citations or legalese to inflate scores.
  - Failure to flag cross-cutting constraints: not recognizing relevant non-legal constraints (ESA, confidentiality, tribal consultation) that limit the requested action.
  - Opaque rationale: producing scores without a verifiable chain of legal reasoning, making it hard for humans to audit or correct errors.

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

    description: ClassVar[str] = "**Legal and Policy Grounding** Accurate use of statutory authority, case law, guidance, and regulatory procedures to frame and justify the request."

    def __init__(self, model: dspy.LM = DEFAULT_MODEL):
        super().__init__(
            name="Legal_and_Policy_Grounding_Rubric",
            description="**Legal and Policy Grounding** Accurate use of statutory authority, case law, guidance, and regulatory procedures to frame and justify the request.",
            axis="**Legal and Policy Grounding** Accurate use of statutory authority, case law, guidance, and regulatory procedures to frame and justify the request.\n\nScoring Guidelines:\nScore 1: Score 1: No legal/policy grounding or fundamentally incorrect\n- No citations to statutes, regulations, case law, or guidance; or wholly irrelevant references.\n- Misstates or contradicts core statutory requirements (e.g., asserts cost consideration where explicitly prohibited; confuses program jurisdiction).\n- Off-topic relative to the docket or procedural context; does not reference the administrative record.\n- Requests actions clearly outside agency authority or legal process.\n- Contains obvious legal inaccuracies that would mislead decision-makers.\nScore 2: Score 2: Minimal and/or error-prone grounding\n- Mentions a law or policy area in general terms but without specific, verifiable citations (e.g., \u201cthe Clean Air Act says\u2026\u201d).\n- One or more notable errors in interpreting authority, procedure, or case law; or mismatched citations.\n- Weak linkage to the docket/topic; limited awareness of rulemaking posture (e.g., proposal vs. interim decision).\n- Requests are largely non-actionable or partially outside agency authority.\n- Relies mainly on opinion; minimal engagement with guidance (EO 12866, OMB A-4, EPA Guidelines) or misapplies them.\nScore 3: Score 3: Adequate but incomplete grounding\n- Includes some correct, relevant citations (e.g., specific statute sections or docket ID), with generally accurate interpretations.\n- Demonstrates basic understanding of procedural posture and agency authority, but lacks depth (e.g., few or no FR page cites; limited case law).\n- Requests are mostly within agency authority but not fully justified by the cited legal framework.\n- May omit key constraints (e.g., when costs can/cannot be considered) or rely on generic guidance references without application to the issue.\n- Minor inaccuracies or gaps, but not misleading in material ways.\nScore 4: Score 4: Strong, accurate legal/policy framework\n- Multiple precise citations (U.S.C., C.F.R., Federal Register pages, docket materials) and correct summaries of their relevance.\n- Correctly distinguishes statutory constraints (e.g., when cost-benefit analysis is permissible/required/prohibited) and cites controlling case law where appropriate.\n- Appropriately applies guidance (EO 12866, OMB Circular A-4, EPA economic/science guidelines) to the specific rule/action.\n- Requests are specific and clearly within agency authority (e.g., adjust analysis scope, clarify presentation of benefits/co-benefits, modify mitigation measures), with a clear legal/procedural rationale.\n- Minimal to no errors; arguments are well-scoped to the responsible program office and docket.\nScore 5: Score 5: Exemplary, comprehensive legal and policy grounding\n- Exhaustive, accurate use of statutory authority, case law, and guidance tailored to the docket (e.g., cites exact sections, FR pages, prior agency guidance, and relevant precedents like Whitman v. American Trucking; Michigan v. EPA if applicable).\n- Demonstrates clear command of procedural posture and regulatory requirements, including limitations on decision criteria (e.g., CAA \u00a7109 vs. other CAA provisions), and integrates record evidence (RIAs, risk assessments, SAP/SACC reports).\n- Translates legal grounding into precise, actionable, within-authority requests (e.g., specific analytical corrections, record supplementation, reconsideration of mitigation, clarity on benefit categorization, compliance timelines), and anticipates legal risk or alternative lawful pathways.\n- Accurately references and applies cross-cutting requirements where relevant (ESA consultations, TSCA best available science, public disclosure constraints) without overreach.\n- No inaccuracies; cites sources transparently; acknowledges countervailing constraints and explains why the proposed action remains lawful.\n",
            model=model,
            task_description="Rank candidate policy feedback drafts for follow-up: given a citizen\u2019s submission, \n    determine which responses merit escalation to agency officials.",
            metric_card="provided",
            max_workers=32,
        )

    def __repr__(self):
        return f"Legal_and_Policy_Grounding_Rubric_LLMJudge(model=dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv(\"OPENAI_API_KEY\")))"

