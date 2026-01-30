---
# Metric Card for MathProcessRewardModel (Qwen2.5-Math-PRM-7B)

MathProcessRewardModel is a process-level reward model that evaluates each intermediate step in a multi-step mathematical reasoning problem. Rather than scoring the final answer alone, it provides token-level feedback across a reasoning chain, identifying helpful versus unhelpful steps using a learned binary classifier. This allows for granular supervision of multi-hop reasoning in LLMs and is particularly effective in domains where correctness must be verified incrementally.

## Metric Details

### Metric Description

MathProcessRewardModel evaluates step-by-step mathematical reasoning by assigning a reward score to each reasoning step in a sequence. The model inserts a special token (`<extra_0>`) after each reasoning step and computes the probability that the token is classified as “positive” using a softmax over logits. This yields a scalar between 0 and 1 indicating how helpful or correct the step is deemed to be. The model is trained on labels derived from whether a step leads to a correct solution trajectory, allowing it to generalize to unseen reasoning processes.

- **Metric Type:** Semantic Similarity, Reference-Free, Faithfulness
- **Range:** [0, 1]
- **Higher is Better?:** Yes
- **Reference-Based?:** No
- **Input-Required?:** Yes

### Formal Definition

Let $x$ be a problem prompt, and $z_1, z_2, \dots, z_T$ be a sequence of reasoning steps. Let $<\!extra_0\!>$ be a separator token inserted after each step. Let $s_i$ denote the model’s score for step $z_i$.

For each step:
$$
s_i = P(\text{label} = \text{positive} \mid z_i)
$$

This is computed via softmax over the model's logits:
$$
s_i = \text{softmax}(l_i)[\text{positive\_class}]
$$

where $l_i$ are the logits at the token position corresponding to $<\!extra_0\!>$ following step $z_i$.

### Inputs and Outputs

- **Inputs:**  
  - Problem prompt (e.g., math word problem)  
  - Step-by-step reasoning sequence (as multiple text spans, each ending with `<extra_0>`)  

- **Outputs:**  
  - List of step-level scores (floats between 0 and 1), one for each reasoning step

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation
- **Tasks:** Mathematical Reasoning, Step-by-Step Problem Solving, Chain-of-Thought Evaluation

### Applicability and Limitations

- **Best Suited For:**  
  Evaluation of mathematical reasoning chains, especially in contexts where intermediate steps need supervision (e.g., tutoring systems, math QA).

- **Not Recommended For:**  
  Open-ended creative generation tasks, final-answer-only assessments, or settings where reasoning steps are implicit or uninterpretable.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**
  - Hugging Face Transformers ([Model Page](https://huggingface.co/Qwen/Qwen2.5-Math-PRM-7B))
  - Source example and logic: see [Qwen2.5 PRM README](https://huggingface.co/Qwen/Qwen2.5-Math-PRM-7B)

### Computational Complexity

- **Efficiency:**  
  Inference is linear in the number of steps and token length; a forward pass over the model is required per input with all reasoning steps and inserted tokens.

- **Scalability:**  
  Scales well for moderate-length reasoning chains (~4–20 steps), but cost grows with step count due to softmax computation per `<extra_0>` token.

## Known Limitations

- **Biases:**  
  Needs more information.

- **Task Misalignment Risks:**  
  May be less applicable to domains where step-wise correctness is hard to define or subjective.

- **Failure Cases:**  
  Needs more information.

## Related Metrics

- **Outcome Reward Models (ORM):** Only evaluate final answer correctness.
- **BERTScore:** Evaluates semantic similarity but not process reasoning.
- **Verifier Models:** Sometimes used to verify entire chains rather than local steps.

## Further Reading

- **Papers:**  
  - [Zhang et al. (2025) - The Lessons of Developing Process Reward Models in Mathematical Reasoning](https://arxiv.org/abs/2501.07301)

- **Blogs/Tutorials:**  
  - [Stephen Diehl – Process Reward Models](https://www.stephendiehl.com/posts/process-reward-models.html)

## Citation

```
@misc{zhang2025lessonsdevelopingprocessreward,
      title={The Lessons of Developing Process Reward Models in Mathematical Reasoning}, 
      author={Zhenru Zhang and Chujie Zheng and Yangzhen Wu and Beichen Zhang and Runji Lin and Bowen Yu and Dayiheng Liu and Jingren Zhou and Junyang Lin},
      year={2025},
      eprint={2501.07301},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.07301}, 
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from generative AI. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu