# Auto-generated metric file for Clarity_of_Expression_gpt-4o-mini
import dspy
import os
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import GeneratedRefBasedLLMJudgeMetric
from typing import ClassVar

DEFAULT_MODEL = dspy.LM(model='openai/gpt-4o-mini', api_key=os.getenv("OPENAI_API_KEY"))

class Clarity_of_Expression_gpt_4o_mini_LLMJudge(GeneratedRefBasedLLMJudgeMetric):
    """---
# Metric Card for Clarity_of_Expression_gpt-4o-mini

**Clarity of Expression** Simplified sentences should clearly convey the message without losing the original meaning.

## Metric Details

**Clarity_of_Expression_gpt-4o-mini** is a **reference-based** LLM-as-a-Judge metric that prompts an LLM to rate a system output along a single, run-time-specified evaluation axis.
In this case the axis is `**Clarity of Expression** Simplified sentences should clearly convey the message without losing the original meaning.`.

The prompt supplies:

1. **Task description** *d*
2. **Axis rubric** `**Clarity of Expression** Simplified sentences should clearly convey the message without losing the original meaning.`
3. **Input text** *x*
4. **Reference text** *r*
5. **Output text** *y*

Greedy decoding (temperature = 0) yields an integer score $\hat{s}\!\in\!\{1,2,3,4,5\}$; higher = better adherence to the axis.

- **Metric Type:** LLM as a Judge
- **Range:** 1-5 (1 = worst, 5 = best)
- **Higher is Better?:** Yes
- **Reference-Based?:** Yes
- **Input-Required?:** Yes (plus reference)

### Formal Definition

Let $f _{\\theta}$ be the LLM and
$\pi _{\text{RB}}(d,\{axis\},x,r,y)$ construct the textual prompt.

$$
\hat{s} \;=\; \operatorname*{arg\,max}\limits_{s \in \{1,\dots,5\}} f _{\theta}\!\bigl(s \,\bigl|\, \pi _{\text{RB}}(d,\{axis\},x,r,y)\bigr)
$$

The metric value is $\operatorname{LJ}^{\text{RB}}_{\{axis\}}(d,x,r,y)=\hat{s}$.

### Inputs and Outputs
- **Inputs:**
  - **Task description** *d*
  - **Axis rubric** `**Clarity of Expression** Simplified sentences should clearly convey the message without losing the original meaning.`
  - **Input text** *x*
  - **Reference text** *r*
  - **Output text** *y*
- **Outputs:**
  - Scalar score $\hat{s} \in \{1,2,3,4,5\}$

## Intended Use

- **Domain:** Text Simplification
- **Tasks:** 
  - Sentence Simplification
  - Educational Material Creation
  - Technical Document Simplification
  - Accessibility Content Modification
  - Plain Language Translation
- **Best Suited For:** 
  - The original sentences contain complex language or jargon that may confuse a broader audience.
  - There is a need to make information accessible for individuals with varying reading levels.
  - The audience includes non-experts who require straightforward explanations.
  - Simplified content is intended for educational purposes or public communication.
- **Not Recommended For:** 
  - The original content is already clear and requires no simplification.
  - The audience is highly specialized and requires technical information without simplification.
  - The task requires maintaining a specific linguistic style or complexity for literary purposes.
  - The simplified sentence needs to preserve nuanced meanings that are lost in oversimplification.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**
  - [AutoMetrics LLM as a Judge (reference-based)](https://github.com/XenonMolecule/autometrics/blob/main/autometrics/metrics/generated/GeneratedLLMJudgeMetric.py)

### Computational Complexity

- **Efficiency:**
  - Requires a single LLM call per input-output pair.
  - AutoMetrics does parallel calls on batched inputs.

- **Scalability:**
  - Performance is linear in the number of input-output pairs.
  - Performance depends on the underlying LLM model and the dataset size.  Additional consideration would include whether or not the LLM is a reasoning model.

## Known Limitations

- **Biases:** 
  - The model may favor more simplistic language over contextually appropriate complexity, leading to oversimplification.
  - Cultural biases may influence what is considered clear or understandable, privileging certain linguistic styles.
  - The model might demonstrate confirmation bias, focusing on examples it has seen in training that match its perception of clarity.
- **Task Misalignment Risks:** 
  - Clarity can be subjective; what is clear to one audience may not be to another, leading to misalignment in evaluation.
  - The evaluation may prioritize brevity over clarity, causing important nuances from the original sentence to be lost.
  - Differences in familiarity with specific terminology or concepts could lead to a mismatch between the intended audience's understanding and the simplification provided.
- **Failure Cases:** 
  - The model may produce a simplification that, while concise, overlooks key details necessary for understanding.
  - A simplified sentence might introduce misunderstandings through the use of ambiguous language or phrasing.
  - If the audience's background knowledge doesn't align with the simplification, it could lead to confusion rather than clarity.

## Related Metrics

- **Related Metrics:**
  - **LevenshteinDistance:** Levenshtein Distance measures the minimum number of single-character edits—insertions, deletions, or substitutions—required to transform one sequence into another.
  - **CIDEr:** CIDEr (Consensus-based Image Description Evaluation) measures the similarity between a candidate image caption and a set of human-generated reference captions.
  - **HammingDistance:** Hamming Distance measures the number of positions at which two equal-length sequences differ.

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

- **Authors:** This metric card was automatically generated by gpt-4o-mini.
- **Acknowledgement of AI Assistance:** This metric card was entirely automatically generated by gpt-4o-mini using the Autometrics library. No human intervention was involved. User discretion is advised.
- **Contact:** For questions about the autometrics library, please contact [Michael J Ryan](mailto:mryan0@stanford.edu)."""

    description: ClassVar[str] = "**Clarity of Expression** Simplified sentences should clearly convey the message without losing the original meaning."

    def __init__(self, model: dspy.LM = DEFAULT_MODEL):
        super().__init__(
            name="Clarity_of_Expression_gpt-4o-mini",
            description="**Clarity of Expression** Simplified sentences should clearly convey the message without losing the original meaning.",
            axis="**Clarity of Expression** Simplified sentences should clearly convey the message without losing the original meaning.",
            model=model,
            task_description="Given a complicated original sentence, simplify it in a way such that a broader audience could easily understand it.",
            metric_card="provided",
            max_workers=32,
        )

    def __repr__(self):
        return f"Clarity_of_Expression_gpt_4o_mini_LLMJudge(model=dspy.LM(model='openai/gpt-4o-mini', api_key=os.getenv(\"OPENAI_API_KEY\")))"

