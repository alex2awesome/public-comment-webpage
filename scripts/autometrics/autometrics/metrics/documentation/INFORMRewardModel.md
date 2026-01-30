---
# Metric Card for INFORM Reward Model 70B

The INFORM Reward Model 70B (INF-ORM-Llama3.1-70B) is a large-scale outcome reward model designed to evaluate the quality of generated conversational responses. It predicts scalar reward scores for response texts, supporting preference-based fine-grained evaluations without requiring a reference response. The model is finetuned from the Llama-3.1-70B-Instruct backbone using preference-labeled datasets, employing scaled Bradley-Terry loss to incorporate preference magnitudes.

## Metric Details

### Metric Description

INFORM Reward Model 70B measures the quality of generated responses by assigning scalar reward scores. It uses a fine-tuned Llama 3.1-70B-Instruct model trained on paired comparisons, with annotated preference magnitudes indicating how much better one response is than another. A modified score head projects the hidden states to reward scores, and the model employs a scaled Bradley-Terry loss to better reflect differences in human preference strengths.

During training, human preference annotations originally assigned discrete scores of 1, 2, or 3 (for slight, better, or much better). These were **rescaled** during dataset preparation to magnitudes of 1, 3, and 10 respectively, amplifying stronger preferences to better guide the model's optimization.

- **Metric Type:** Reference-Free
- **Range:** Unbounded (observed values typically between approximately -33 and +3)
- **Higher is Better?:** Yes
- **Reference-Based?:** No
- **Input-Required?:** Yes

### Formal Definition

Given a prompt and a candidate response $y$, the INFORM Reward Model predicts a scalar reward $r(x, y)$.

For training, it optimizes the **Scaled Bradley-Terry loss**:

$$
L_{\text{Scaled-BT}} = -d \log(\sigma(r(x, y_{\text{chosen}}) - r(x, y_{\text{rejected}})))
$$

where:
- $d$ is the magnitude of preference between the chosen and rejected responses (scaled to values like 1, 3, or 10),
- $\sigma$ is the sigmoid function,
- $r(x, y)$ is the predicted reward for response $y$ given prompt $x$.

### Inputs and Outputs

- **Inputs:**  
  - Conversation history including user input and model response (as tokenized chat sequences).
  
- **Outputs:**  
  - A scalar reward score (floating point) indicating the response quality. Higher values indicate better responses.

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation, Dialogue Systems
- **Tasks:** Dialogue Generation, Response Generation

### Applicability and Limitations

- **Best Suited For:**  
  - Comparing the quality of candidate responses in dialogue or conversation settings, particularly for tasks where reference outputs are unavailable.
  - Reward modeling for RLHF (Reinforcement Learning from Human Feedback) setups.

- **Not Recommended For:**  
  - Tasks requiring direct evaluation against reference answers (e.g., machine translation).
  - Evaluation scenarios where absolute calibration of scores is necessary.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**
  - Hugging Face Transformers (custom Llama3.1-70B model checkpoint: [https://huggingface.co/infly/INF-ORM-Llama3.1-70B](https://huggingface.co/infly/INF-ORM-Llama3.1-70B))
  
### Computational Complexity

- **Efficiency:**  
  Efficient at inference time; a single forward pass per response is needed. Complexity is dominated by a transformer pass ($O(n \cdot d)$ where $n$ is sequence length and $d$ is hidden dimension).

- **Scalability:**  
  Scales linearly with batch size and input length; requires significant memory (70B parameters). Intended for GPU-based inference.

## Known Limitations

- **Biases:**  
  - Potential biases inherited from training datasets, including topic or stylistic biases present in the preference judgments.
  - Training rescaling of preference magnitudes (1 → 1, 2 → 3, 3 → 10) may amplify annotator subjectivity and increase sensitivity to preference errors.

- **Task Misalignment Risks:**  
  - May not align well with human preferences for tasks outside of the dialogue domain.
  - Risk of misinterpreting slight vs. large differences in quality due to score scaling.

- **Failure Cases:**  
  - Struggles in evaluating extremely diverse or creative responses where strict preference orders are unclear.
  - Calibration across very different prompt domains is not guaranteed.

## Related Metrics

- **Bradley-Terry Loss Models:** Standard Bradley-Terry models trained without magnitude scaling.
- **Scaled BT Models:** Models using magnitude information outside the log-sigmoid, as explored in [HelpSteer2-Preference](https://arxiv.org/pdf/2410.01257).
- **RewardBench Metrics:** INFORM Reward Model was benchmarked on RewardBench and compared against other reward models.

## Further Reading

- **Papers:**  
  - [INF-ORM-Llama3.1-70B Model Card on Hugging Face](https://huggingface.co/infly/INF-ORM-Llama3.1-70B)  
  - [HELPSTEER2-PREFERENCE: COMPLEMENTING RATINGS WITH PREFERENCES (ICLR 2025)](https://arxiv.org/pdf/2410.01257) (influential but not official paper)

- **Blogs/Tutorials:**  
  - None officially provided. (Needs more information)

## Citation

```
@misc{INF-ORM-Llama3.1-70B, 
      url={[https://huggingface.co/infly/INF-ORM-Llama3.1-70B](https://huggingface.co/infly/INF-ORM-Llama3.1-70B)},
      title={INF-ORM-Llama3.1-70B},
      year={2024},
      author={Minghao Yang, Chao Qu, Xiaoyu Tan}
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from OpenAI's ChatGPT, based on user-provided inputs and reference documentation. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu