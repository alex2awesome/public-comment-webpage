# Auto-generated metric file for Source_Credibility_and_Representation_Rubric
import dspy
import os
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import GeneratedRefFreeLLMJudgeMetric
from typing import ClassVar

DEFAULT_MODEL = dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv("OPENAI_API_KEY"))

class Source_Credibility_and_Representation_Rubric_LLMJudge(GeneratedRefFreeLLMJudgeMetric):
    """---
# Metric Card for Source_Credibility_and_Representation_Rubric

**Source Credibility and Representation** Author expertise, organizational standing, coalition support, and whether the comment represents large or affected constituencies.

## Metric Details

**Source_Credibility_and_Representation_Rubric** is a **reference-free** LLM-as-a-Judge metric that prompts an LLM to rate a system output along a single, run-time-specified evaluation axis.
In this case the axis is `**Source Credibility and Representation** Author expertise, organizational standing, coalition support, and whether the comment represents large or affected constituencies.`.

The prompt supplies:

1. **Task description** *d*
2. **Rubric** `**Source Credibility and Representation** Author expertise, organizational standing, coalition support, and whether the comment represents large or affected constituencies.`
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

**Criteria:** **Source Credibility and Representation** Author expertise, organizational standing, coalition support, and whether the comment represents large or affected constituencies.

#### Scoring Rubric

| Score | Description |
|-------|-------------|
| 1 | Score 1 (Very Low credibility/representation):<br/>• Anonymous or unclear author identity; no affiliation or role provided.<br/>• No credentials or evidence of domain expertise.<br/>• No claim of representing others; purely individual opinion with no indication of being affected.<br/>• No contact info, letterhead, or docket-specific context.<br/>• Signals of astroturfing/misrepresentation or generic, mass-produced text without source details. |
| 2 | Score 2 (Low credibility/limited representation):<br/>• Identified individual with minimal context (e.g., resident, voter) and no relevant credentials.<br/>• May claim personal affectedness but provides no specifics or substantiation.<br/>• No organizational affiliation or coalition support.<br/>• Limited transparency (e.g., name but no role/contact), or weak alignment to the docket domain. |
| 3 | Score 3 (Moderate credibility OR representation):<br/>• EITHER an individual with clear, relevant expertise (e.g., professional title, degree) but not representing a broader constituency,<br/>• OR a small/local organization (community group, local nonprofit) with some members/clients explicitly mentioned.<br/>• Some transparency (contact info/letterhead) and clear relevance to the docket.<br/>• No coalition support, or support from only one additional allied group.<br/>• Demonstrates some proximity to impacts (local effects, member experience) but limited scope. |
| 4 | Score 4 (High credibility and/or substantial representation):<br/>• Recognized organization (regional/national association, NGO, trade group, agency, academic center) with stated membership or stakeholder base; or an author with strong, directly relevant credentials (e.g., PhD, executive role) writing on behalf of the organization.<br/>• Provides quantitative or specific representation claims (e.g., number of members, jurisdictions served) and shows direct affectedness (e.g., regulated entities, frontline communities).<br/>• Good transparency: letterhead, full contact info, docket citations, signatory roles.<br/>• May include multiple co-signers or endorsements, though coalition breadth is limited (e.g., a few allied orgs).<br/>• Clear mission and domain alignment to the issue. |
| 5 | Score 5 (Very high credibility AND broad/affected representation with coalition strength):<br/>• Major national or multi-regional organization(s) or coalition letter with numerous and/or diverse signatories (industry, municipalities, NGOs, labor, community-based orgs) clearly listed.<br/>• Explicit, substantial constituency reach (e.g., thousands of members, majority of sector coverage, large populations served) or uniquely high affectedness (e.g., utilities under the rule, heavily burdened communities) documented in the submission.<br/>• Author signatories hold senior roles and/or possess strong domain expertise (e.g., executives, technical directors, PhDs), writing in their official capacities.<br/>• Full transparency (official letterhead, contacts, docket references) and strong domain relevance.<br/>• May reference prior formal engagements (testimony, advisory panels) or provide organization-generated data, indicating established policy participation. |

### Inputs and Outputs
- **Inputs:**
  - **Task description** *d*
  - **Rubric** `**Source Credibility and Representation** Author expertise, organizational standing, coalition support, and whether the comment represents large or affected constituencies.`
  - **Input text** *x*
  - **Output text** *y*
- **Outputs:**
  - Scalar score $\hat{s} \in \{1,2,3,4,5\}$

## Intended Use

- **Domain:** Policy feedback evaluation / Text classification
- **Tasks:** 
  - Triage of public comments for escalation to agency officials
  - Credibility scoring of policy feedback submissions
  - Identifying and enumerating signatories and coalition breadth in letters
  - Flagging potential astroturfing or mass-produced comments
  - Prioritizing high-impact stakeholder inputs for reviewer attention
  - Generating short rationales tied to rubric criteria for each score
  - Filtering submissions that require human verification or follow-up
- **Best Suited For:** 
  - Submissions primarily in English and in standard textual formats (letters, comments, emails) where identity, affiliation, and signatory information are present or omitted explicitly.
  - Large volumes of comments requiring consistent, repeatable triage according to a transparent rubric.
  - Situations where the goal is to prioritize which submissions merit human escalation rather than to conclusively verify external claims.
  - When comments include structured cues (letterhead, contact blocks, explicit membership counts, docket citations, lists of co-signers) that are directly mappable to the scoring guidelines.
  - Contexts that benefit from quick, explainable justifications for scores (e.g., to create human-review queues or summary dashboards).
- **Not Recommended For:** 
  - Cases that require external verification of claims (membership numbers, official roles, prior testimony) or authoritative fact-checking — the model cannot reliably confirm off-document facts.
  - Submissions that are multimedia-heavy (scanned images of letters, audio/video testimony) without accurate OCR/transcription, or texts in languages the model is weak in.
  - High-stakes legal or policy decisions where incorrect credibility assessments could cause material harm and thus require domain-expert adjudication.
  - Highly adversarial or purposely deceptive inputs designed to spoof organizational credentials or fabricate coalitions without corroborating evidence.
  - Narrow, highly technical domains with uncommon institutional codewords or localized organizational structures the model may not recognize without additional context.

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
  - Credentialism bias: privileging formal academic/professional credentials over lived experience or community knowledge.
  - Institutional/status bias: favoring large, well-known organizations and letterhead, disadvantaging small grassroots groups.
  - Format/visual bias: treating presence of letterhead, full contact info, or formal formatting as proxies for credibility.
  - Language and cultural bias: penalizing submissions from non-native English speakers or different rhetorical styles.
  - Name-recognition bias: giving extra weight to familiar institution names or senior-sounding titles.
  - Coalition-size bias: equating larger signatory lists with substantive representation without verifying diversity or authenticity.
  - Anonymity-averse bias: automatically lowering scores for anonymous submissions even when anonymity protects vulnerable individuals.
  - Surface-evidence bias: overrelying on explicit numeric claims (member counts, jurisdictions) that may be unverified or inflated.
  - Confirmation bias: interpreting ambiguous claims in a way that aligns with prior model exposure to similar documents.
  - Recency/training-data bias: preferring organizational types and presentation styles common in the model's training set.
- **Task Misalignment Risks:** 
  - Undervaluing affected individuals: the axis can systematically deprioritize individual but highly affected stakeholders who lack formal affiliation or credentials.
  - Overvaluing self-interested groups: well-resourced industry or advocacy organizations may be scored high despite narrow or biased interests.
  - Penalizing protected/vulnerable voices: anonymity or withheld contact info may reflect safety needs, not low credibility, leading to harmful exclusion.
  - Misreading coalition breadth: counting signatories as representation without assessing whether signers reflect diverse or relevant constituencies.
  - Conflating form with substance: the rubric may prioritize presentation quality (letterhead, citations) over factual accuracy or on-the-ground relevance.
  - Domain mismatch: a single rubric may fail to account for domain-specific credibility signals (e.g., patents in tech vs lived experience in community health).
  - False sense of verification: assigning high credibility based on claims the LLM cannot externally verify (membership numbers, prior testimony).
  - Disincentivizing grassroots testimony: public procedures may favor institutional commenters, skewing agency attention away from frontline impacts.
  - Ambiguity in multi-author submissions: the rubric may not specify how to score mixed-author inputs (expert co-signers plus anonymous community members).
  - Overemphasis on coalition breadth could ignore depth of impact (a small but uniquely affected group may be more relevant than a large but peripheral coalition).
- **Failure Cases:** 
  - Classifying an anonymous whistleblower's detailed, verifiable technical account as 'Very Low' because contact info is absent, thereby failing to escalate a critical issue.
  - Elevating a polished coalition letter with fabricated signatories and inflated membership counts to 'Very High' because of formatting and large numbers.
  - Downgrading grassroots submissions written in non-standard English or with culturally distinct rhetorical styles, missing important affectedness evidence.
  - Failing to detect astroturfing when a lobby group submits many coordinated comments with slight variations, instead scoring them as broad grassroots representation.
  - Scoring a small local nonprofit low because it lacks letterhead or formal membership numbers, despite the organization being the primary affected party.
  - Treating long, data-heavy industry comments with unverifiable proprietary claims as high credibility while ignoring conflicts of interest.
  - Missing domain nuance: giving a high score to an author with a PhD in an adjacent field whose expertise does not apply to the docket's technical area.
  - Misinterpreting co-signer lists: counting numerous minor endorsements as evidence of broad representation when signers are duplicate or irrelevant entities.
  - Penalizing a victim/survivor who omits contact info for privacy, thereby silencing a directly affected constituency in the escalation pipeline.
  - Failing to flag obviously fake organizations with realistic-sounding names because the model overrelies on lexical heuristics rather than verification.

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

    description: ClassVar[str] = "**Source Credibility and Representation** Author expertise, organizational standing, coalition support, and whether the comment represents large or affected constituencies."

    def __init__(self, model: dspy.LM = DEFAULT_MODEL):
        super().__init__(
            name="Source_Credibility_and_Representation_Rubric",
            description="**Source Credibility and Representation** Author expertise, organizational standing, coalition support, and whether the comment represents large or affected constituencies.",
            axis="**Source Credibility and Representation** Author expertise, organizational standing, coalition support, and whether the comment represents large or affected constituencies.\n\nScoring Guidelines:\nScore 1: Score 1 (Very Low credibility/representation):\n- Anonymous or unclear author identity; no affiliation or role provided.\n- No credentials or evidence of domain expertise.\n- No claim of representing others; purely individual opinion with no indication of being affected.\n- No contact info, letterhead, or docket-specific context.\n- Signals of astroturfing/misrepresentation or generic, mass-produced text without source details.\nScore 2: Score 2 (Low credibility/limited representation):\n- Identified individual with minimal context (e.g., resident, voter) and no relevant credentials.\n- May claim personal affectedness but provides no specifics or substantiation.\n- No organizational affiliation or coalition support.\n- Limited transparency (e.g., name but no role/contact), or weak alignment to the docket domain.\nScore 3: Score 3 (Moderate credibility OR representation):\n- EITHER an individual with clear, relevant expertise (e.g., professional title, degree) but not representing a broader constituency,\n- OR a small/local organization (community group, local nonprofit) with some members/clients explicitly mentioned.\n- Some transparency (contact info/letterhead) and clear relevance to the docket.\n- No coalition support, or support from only one additional allied group.\n- Demonstrates some proximity to impacts (local effects, member experience) but limited scope.\nScore 4: Score 4 (High credibility and/or substantial representation):\n- Recognized organization (regional/national association, NGO, trade group, agency, academic center) with stated membership or stakeholder base; or an author with strong, directly relevant credentials (e.g., PhD, executive role) writing on behalf of the organization.\n- Provides quantitative or specific representation claims (e.g., number of members, jurisdictions served) and shows direct affectedness (e.g., regulated entities, frontline communities).\n- Good transparency: letterhead, full contact info, docket citations, signatory roles.\n- May include multiple co-signers or endorsements, though coalition breadth is limited (e.g., a few allied orgs).\n- Clear mission and domain alignment to the issue.\nScore 5: Score 5 (Very high credibility AND broad/affected representation with coalition strength):\n- Major national or multi-regional organization(s) or coalition letter with numerous and/or diverse signatories (industry, municipalities, NGOs, labor, community-based orgs) clearly listed.\n- Explicit, substantial constituency reach (e.g., thousands of members, majority of sector coverage, large populations served) or uniquely high affectedness (e.g., utilities under the rule, heavily burdened communities) documented in the submission.\n- Author signatories hold senior roles and/or possess strong domain expertise (e.g., executives, technical directors, PhDs), writing in their official capacities.\n- Full transparency (official letterhead, contacts, docket references) and strong domain relevance.\n- May reference prior formal engagements (testimony, advisory panels) or provide organization-generated data, indicating established policy participation.\n",
            model=model,
            task_description="Rank candidate policy feedback drafts for follow-up: given a citizen\u2019s submission, \n    determine which responses merit escalation to agency officials.",
            metric_card="provided",
            max_workers=32,
        )

    def __repr__(self):
        return f"Source_Credibility_and_Representation_Rubric_LLMJudge(model=dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv(\"OPENAI_API_KEY\")))"

