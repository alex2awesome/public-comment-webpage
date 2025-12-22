# Auto-generated metric file for Topical_Relevance_to_the_Submission_gpt-5-mini
import dspy
import os
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import GeneratedRefFreeLLMJudgeMetric
from typing import ClassVar

DEFAULT_MODEL = dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv("OPENAI_API_KEY"))

class Topical_Relevance_to_the_Submission_gpt_5_mini_LLMJudge(GeneratedRefFreeLLMJudgeMetric):
    """---
# Metric Card for Topical_Relevance_to_the_Submission_gpt-5-mini

**Topical Relevance to the Submission** The content tightly maps to the citizens submission and the OMB guidance it invokes avoiding broad unrelated themes.

## Metric Details

**Topical_Relevance_to_the_Submission_gpt-5-mini** is a **reference-free** LLM-as-a-Judge metric that prompts an LLM to rate a system output along a single, run-time-specified evaluation axis.
In this case the axis is `**Topical Relevance to the Submission** The content tightly maps to the citizens submission and the OMB guidance it invokes avoiding broad unrelated themes.`.

The prompt supplies:

1. **Task description** *d*
2. **Axis rubric** `**Topical Relevance to the Submission** The content tightly maps to the citizens submission and the OMB guidance it invokes avoiding broad unrelated themes.`
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
  - **Axis rubric** `**Topical Relevance to the Submission** The content tightly maps to the citizens submission and the OMB guidance it invokes avoiding broad unrelated themes.`
  - **Input text** *x*
  - **Output text** *y*
- **Outputs:**
  - Scalar score $\hat{s} \in \{1,2,3,4,5\}$

## Intended Use

- **Domain:** Text classification / Policy triage / Content relevance evaluation
- **Tasks:** 
  - Triage candidate feedback for escalation based on topical relevance
  - Rank multiple candidate responses by how closely they map to the citizen's submission
  - Flag drafts that reference unrelated themes or invoke guidance incorrectly
  - Annotate which sentences or paragraphs in a draft are on-topic vs off-topic relative to the submission and OMB guidance
  - Provide short rationales explaining why a draft merits or does not merit escalation
- **Best Suited For:** 
  - The citizen submission and the specific OMB guidance excerpt are provided verbatim, allowing direct semantic comparison.
  - Candidate drafts are short to moderate length (a few paragraphs) focused on policy feedback rather than long policy memos.
  - Tasks require identification of topical alignment and surface-level mapping rather than deep legal interpretation or operational decisions.
  - Multiple candidate drafts need consistent, repeatable ranking or flagging according to a single relevance axis.
  - Evaluation must be fast and scalable across many submissions where human reviewers will perform final checks.
- **Not Recommended For:** 
  - Inputs lack the cited OMB guidance text or include only vague/incomplete references, preventing reliable mapping.
  - Materials require authoritative legal interpretation, confidential agency context, or factual verification beyond the provided texts.
  - Candidate drafts are extremely long, technical, or contain specialized jargon that the model hasn't been primed on.
  - Submissions or drafts are in languages other than those the model was trained for or contain heavy code/data tables needing domain-specific parsing.
  - High-stakes escalation decisions where errors could cause legal or safety harm and where automated judgment would be used without human oversight.

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
  - Lexical overlap bias: favoring responses that reuse words or phrases from the submission or OMB text even when semantic alignment is weak.
  - Citation-surface bias: preferring candidates that explicitly mention OMB guidance (or plausible-looking citations) regardless of correct interpretation or relevance.
  - Formality/verbosity bias: ranking longer or more formal-sounding drafts higher because they appear authoritative rather than more concise, directly relevant replies.
  - Recency/training-data bias: privileging interpretations common in the model's training data over niche or novel readings of OMB guidance.
  - Confirmation bias toward expected framing: interpreting ambiguous submissions in ways that confirm common policy framings instead of the citizen's intended angle.
  - Language and cultural bias: downranking submissions or responses that use nonstandard English, idioms, or culturally specific concerns even when topically relevant.
  - Overgeneralization bias: treating mentions of related policy topics as sufficient topical relevance even when the draft diverges from the submission’s specific focus.
  - Neglect-of-implicit-intent bias: ignoring implicitly stated citizen priorities (e.g., equity or local impacts) when they are not explicitly worded.
  - Anchoring bias: initial high-scoring candidates set a baseline that skews comparison, causing subtler but more relevant candidates to be undervalued.
  - Hallucination acceptance bias: insufficiently penalizing drafts that assert incorrect facts about OMB guidance or agency authority if they 'sound' relevant.
- **Task Misalignment Risks:** 
  - Overemphasis on topical relevance alone: escalating drafts that are topically aligned but legally or procedurally inaccurate, harmful, or unsafe.
  - Neglecting escalation criteria beyond topical fit: ignoring urgency, factual accuracy, citizen harm, or policy impact which matter for escalation decisions.
  - Rewarding surface-level matches: producing rankings that favor paraphrase or keyword matches rather than substantive interpretation of what should be escalated.
  - Failure to account for multi-issue submissions: misranking drafts when a citizen submission spans several distinct topics and the judge prioritizes only one.
  - Alignment to OMB phrasing rather than intent: punishing drafts that correctly apply the spirit of guidance but use different terminology than the cited OMB text.
  - Sensitivity to adversarial or strategic wording: candidates that game topical relevance (e.g., stuffing keywords) may be escalated despite low real-world usefulness.
  - Ignoring contextual constraints: not considering agency jurisdiction or procedural feasibility, causing irrelevant but topical drafts to be escalated.
  - Overlooking equity and representational concerns: treating topical fit as sufficient without evaluating whether escalation amplifies marginalized voices appropriately.
- **Failure Cases:** 
  - False positive: a draft repeats OMB language and keywords, ranks highest for topical relevance, but misapplies the guidance and should not be escalated.
  - False negative: a concise draft that reframes the submission with accurate actionable points uses different terminology and is scored low despite being highly relevant.
  - Ambiguity mishandling: the judge fails to detect that the citizen's submission is asking about two separate issues and ranks drafts that address only the less important one.
  - Citation hallucination: a candidate claims support from specific OMB sections that do not exist or are mischaracterized, and the judge still rates it as relevant.
  - Paraphrase penalty: semantically equivalent responses that paraphrase the OMB guidance are undervalued because they lack literal textual overlap.
  - Length bias failure: verbose but tangential drafts outrank concise, focused replies that more tightly map to the submission.
  - Cross-jurisdiction error: a draft is topically relevant but recommends actions an agency cannot take; the judge escalates it because topical fit was the sole criterion.
  - Nonstandard language failure: responses tailored to a submission in informal or nonstandard English are mis-scored as irrelevant despite clear topical alignment.
  - Multi-topic collapse: when the submission touches on policy and operational concerns, the judge consolidates them incorrectly and misses escalation-worthy operational risks.
  - Adversarial robustness failure: an adversary crafts a draft that maximizes lexical similarity to the submission to force escalation despite malicious intent or disinformation.

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

    description: ClassVar[str] = "**Topical Relevance to the Submission** The content tightly maps to the citizens submission and the OMB guidance it invokes avoiding broad unrelated themes."

    def __init__(self, model: dspy.LM = DEFAULT_MODEL):
        super().__init__(
            name="Topical_Relevance_to_the_Submission_gpt-5-mini",
            description="**Topical Relevance to the Submission** The content tightly maps to the citizens submission and the OMB guidance it invokes avoiding broad unrelated themes.",
            axis="**Topical Relevance to the Submission** The content tightly maps to the citizens submission and the OMB guidance it invokes avoiding broad unrelated themes.",
            model=model,
            task_description="Rank candidate policy feedback drafts for follow-up: given a citizen\u2019s submission, \n    determine which responses merit escalation to agency officials.",
            metric_card="provided",
            max_workers=32,
        )

    def __repr__(self):
        return f"Topical_Relevance_to_the_Submission_gpt_5_mini_LLMJudge(model=dspy.LM(model='openai/gpt-5-mini', api_key=os.getenv(\"OPENAI_API_KEY\")))"

