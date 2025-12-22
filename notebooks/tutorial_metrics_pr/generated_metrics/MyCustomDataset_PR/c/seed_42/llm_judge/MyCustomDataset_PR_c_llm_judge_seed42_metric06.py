# Auto-generated metric file for Direct_alignment_with_docket_and_section_gpt-5-mini
import dspy
import os
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import GeneratedRefFreeLLMJudgeMetric
from typing import ClassVar

DEFAULT_MODEL = dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv("OPENAI_API_KEY"))

class Direct_alignment_with_docket_and_section_gpt_5_mini_LLMJudge(GeneratedRefFreeLLMJudgeMetric):
    """---
# Metric Card for Direct_alignment_with_docket_and_section_gpt-5-mini

**Direct alignment with docket and section** Explicit references to the relevant agency proceeding, rule sections, or issues under consideration.

## Metric Details

**Direct_alignment_with_docket_and_section_gpt-5-mini** is a **reference-free** LLM-as-a-Judge metric that prompts an LLM to rate a system output along a single, run-time-specified evaluation axis.
In this case the axis is `**Direct alignment with docket and section** Explicit references to the relevant agency proceeding, rule sections, or issues under consideration.`.

The prompt supplies:

1. **Task description** *d*
2. **Axis rubric** `**Direct alignment with docket and section** Explicit references to the relevant agency proceeding, rule sections, or issues under consideration.`
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
  - **Axis rubric** `**Direct alignment with docket and section** Explicit references to the relevant agency proceeding, rule sections, or issues under consideration.`
  - **Input text** *x*
  - **Output text** *y*
- **Outputs:**
  - Scalar score $\hat{s} \in \{1,2,3,4,5\}$

## Intended Use

- **Domain:** Regulatory policy triage / Text classification and document triage
- **Tasks:** 
  - Rank public comment or feedback drafts by how explicitly they reference the agency docket and specific rule sections
  - Detect and extract docket numbers, rule section citations, and quoted regulatory language from candidate drafts
  - Score drafts for direct alignment with a specified proceeding, rule sections, or enumerated issues
  - Flag drafts that lack explicit references or that reference irrelevant dockets/sections
  - Produce short explanations for why a draft merits escalation based on citation alignment
- **Best Suited For:** 
  - When candidate drafts contain explicit, correctly formatted docket numbers or section citations that can be matched by string or pattern matching
  - For high-volume triage where speed is important and initial filtering of clearly aligned vs. not-aligned drafts is needed
  - When there is a clear, machine-readable description of the docket and sections of interest to use as matching targets
  - When labeled examples or heuristics are available to calibrate the model's scoring and ranking behavior
  - When outputs will be reviewed by agency staff who can validate citations and make final escalation decisions
- **Not Recommended For:** 
  - When final escalation decisions depend on legal interpretation, nuanced policy judgment, or factual validation against up-to-date external records
  - If candidate drafts use implicit, vague, or argumentative references without explicit docket/section citations (high ambiguity)
  - For scanned PDFs or images without reliable OCR, or when relevant citations are embedded in attachments the model cannot access
  - When the agency requires provable, auditable verification of citation accuracy without human verification, because the model can hallucinate or misattribute citations
  - When the docket or rule text has changed recently and the model has no access to live updates or authoritative external databases

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
  - Citation-form bias: favors drafts that use formal citations or legalese over lay-language submissions that are still substantively aligned.
  - Formatting bias: rewards specific formatting (section numbers, docket lines) rather than substantive connection to issues.
  - Domain-knowledge bias: advantages writers familiar with regulatory citation conventions and disadvantages the general public.
  - Recency/jurisdiction bias: may overemphasize currently prominent dockets or canonical sections while downweighting peripheral but relevant references.
  - Verification-blindness bias: assumes cited docket/section exists and is relevant without checking accuracy, privileging plausible-looking but incorrect citations.
  - Language bias: disfavors non-English or nonstandard phrasing that nonetheless maps to the docket/sections.
  - Conservatism bias: prefers comments that conform to official framing and may downrank novel critiques that don't directly reference sections.
  - Length/verbosity bias: longer, more detailed citations may be treated as higher-quality even if redundant.
- **Task Misalignment Risks:** 
  - Overnarrow focus: the axis may cause the judge to ignore important qualities like urgency, factual newness, or harm potential that don't require direct section citations.
  - False security: equating explicit references with correctness can encourage accepting low-quality drafts that pad with citations rather than adding substance.
  - Gaming risk: submitters can intentionally add boilerplate or fabricated section references to trigger escalation despite low relevance.
  - Equity misalignment: prioritizing citation-form decreases accessibility and may systematically exclude valuable input from non-expert citizens.
  - Cross-issue neglect: policy concerns that span multiple sections or are upstream/downstream of the docket may be undervalued if not matched to a single section.
  - Verification workload shift: the judge might escalate more items that appear aligned but require manual verification, increasing human burden.
  - Format dependency: reliance on docket/section strings misaligns the judge for informal channels (emails, oral testimony) where alignment is implicit.
- **Failure Cases:** 
  - False positive escalation: a draft contains plausible-looking but fabricated docket numbers or section citations and is escalated despite being irrelevant.
  - False negative rejection: a highly relevant citizen comment paraphrases the rule or cites policy outcomes without formal section numbers and is erroneously downranked.
  - Mis-parsing numerical references: the judge misinterprets ambiguous numbers (e.g., internal policy IDs vs. section numbers) and mis-assesses alignment.
  - Cross-references missed: the draft discusses an issue that affects the docket indirectly (e.g., upstream statute) but lacks a direct section citation and is missed.
  - Multilingual failure: a non-English submission that references the docket in translated or alternate notation is not recognized as aligned.
  - Over-escalation of boilerplate: mass-produced templates that insert docket/section strings are escalated despite lacking unique, substantive points.
  - Under-escalation of novel arguments: creative or systemic critiques that propose new interpretive lenses are rejected because they don't cite existing sections.
  - Formatting edge cases: PDFs, scanned documents, or unusual encodings hide section references causing the judge to miss alignment.
  - Hallucinated verification: the judge asserts that a cited section supports a claim when it does not, leading to incorrect prioritization.
  - Bias-amplified exclusion: aggregate patterns cause systematic under-escalation of submissions from particular demographic groups who use different citation norms.

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

    description: ClassVar[str] = "**Direct alignment with docket and section** Explicit references to the relevant agency proceeding, rule sections, or issues under consideration."

    def __init__(self, model: dspy.LM = DEFAULT_MODEL):
        super().__init__(
            name="Direct_alignment_with_docket_and_section_gpt-5-mini",
            description="**Direct alignment with docket and section** Explicit references to the relevant agency proceeding, rule sections, or issues under consideration.",
            axis="**Direct alignment with docket and section** Explicit references to the relevant agency proceeding, rule sections, or issues under consideration.",
            model=model,
            task_description="Rank candidate policy feedback drafts for follow-up: given a citizen\u2019s submission, \n    determine which responses merit escalation to agency officials.",
            metric_card="provided",
            max_workers=32,
        )

    def __repr__(self):
        return f"Direct_alignment_with_docket_and_section_gpt_5_mini_LLMJudge(model=dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv(\"OPENAI_API_KEY\")))"

