---
# Metric Card for LDL Reward Model 27B

LDL Reward Model 27B is a large-scale, reference-free reward model designed for evaluating generative model outputs across multiple sub-dimensions such as helpfulness, correctness, coherence, and others. It is built on top of the **Gemma 2-27B** transformer model, with additional modules for reward prediction using **Label Distribution Learning (LDL)**. LDL models human rating uncertainty as probability distributions rather than single point estimates, improving robustness to subjective human annotations. The model outputs a final scalar reward score that can be used for ranking or fine-tuning generative models.

## Metric Details

### Metric Description

LDL Reward Model 27B predicts sub-score distributions for various dimensions of quality (e.g., helpfulness, correctness, complexity) by applying Label Distribution Learning (LDL). In the first stage, the model learns to predict label distributions using a regression layer based on the last hidden token representation from Gemma 2. In the second stage, a gating mechanism aggregates these sub-score distributions into a final scalar reward, trained via Bradley-Terry modeling on pairwise preference data. This approach allows the model to capture uncertainty in human feedback while providing a single, interpretable reward score for evaluation or training.

- **Metric Type:** Reference-Free
- **Range:** Needs more information (example outputs observed between approximately -3.5 and 7.0)
- **Higher is Better?:** Yes
- **Reference-Based?:** No
- **Input-Required?:** Yes

### Formal Definition

First-stage sub-score prediction (for each sub-score $s$):

Given a hidden state vector $h$, the regression layer predicts a label distribution $D_s$ modeled as a discrete Gaussian over possible scores:

$$
D_s(h) = \text{Regression}(h)
$$

Second-stage score aggregation:

A gating layer computes weights $w_s$ over the sub-scores:

$$
w = \text{Gating}(h)
$$

The final reward distribution $D_{\text{final}}$ is computed as a weighted combination:

$$
D_{\text{final}} = \sum _{s} w_s \cdot D_s
$$

The final scalar reward $r$ is then obtained by taking the expectation over $D_{\text{final}}$:

$$
r = \sum _{i} p_i \cdot v_i
$$

where $p_i$ are probabilities from $D_{\text{final}}$ and $v_i$ are the associated score values (typically discretized between 0 and 1).

Pairwise comparison is trained with a Bradley-Terry model:

$$
\mathbb{P}_{\theta}(a_1 > a_2 | x, a_1, a_2) = \sigma(r_1 - r_2)
$$

where $\sigma$ is the sigmoid function, and $r_1$, $r_2$ are the final reward scores of answers $a_1$, $a_2$ respectively.

### Inputs and Outputs

- **Inputs:**  
  - Prompt and generated response (formatted as conversation history).
  
- **Outputs:**  
  - Scalar reward score (real-valued).

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation, Code Generation, Dialogue Systems
- **Tasks:** Response Generation, Dialogue Generation, Code Completion

### Applicability and Limitations

- **Best Suited For:**  
  Evaluating and fine-tuning models for response quality in dialogue, general language generation, or code generation settings where human preference or judgment is a relevant evaluation signal.

- **Not Recommended For:**  
  Tasks requiring strict factual verification or tasks outside general conversational and code-related domains. Also not recommended when highly domain-specific expertise (e.g., medical, legal) is required without further adaptation.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**
  - [Hugging Face Model Page](https://huggingface.co/ShikaiChen/LDL-Reward-Gemma-2-27B-v0.1)

### Computational Complexity

- **Efficiency:**  
  Inference involves a forward pass through a large model (27B parameters) and multiple small neural network layers. Efficiency is similar to other large reward models; inference is relatively fast for single comparisons but costly for large-scale evaluations.

- **Scalability:**  
  Scalability is dependent on hardware; intended for GPU acceleration, with support for device mapping and memory optimization (e.g., using FlashAttention-2).

## Known Limitations

- **Biases:**  
  Needs more information.

- **Task Misalignment Risks:**  
  Potential risk if deployed directly to domains very different from the training data distribution (e.g., highly technical, legal, or medical content).

- **Failure Cases:**  
  Needs more information.

## Related Metrics

- Skywork Reward Models (Skywork-Reward-Gemma-2-27B)  
- INF-ORM Llama 3.1-70B (similar architecture using label distribution learning)

## Further Reading

- **Papers:**  
  - Tech Report (Coming Soon)

- **Blogs/Tutorials:**  
  - [Hugging Face Model Page](https://huggingface.co/ShikaiChen/LDL-Reward-Gemma-2-27B-v0.1)

## Citation

```
@misc{chen2025ldl,
  author       = {Shikai Chen and Jin Yuan and Yang Zhang and Zhongchao Shi and Jianping Fan and Xin Geng and Yong Rui},
  title        = {LDL-Reward-Gemma-2-27B-v0.1},
  year         = {2025},
  howpublished = {\url{https://huggingface.co/ShikaiChen/LDL-Reward-Gemma-2-27B-v0.1}},
  note         = {Hugging Face Model Repository},
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from OpenAI's ChatGPT, based on user-provided inputs and relevant documentation. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu