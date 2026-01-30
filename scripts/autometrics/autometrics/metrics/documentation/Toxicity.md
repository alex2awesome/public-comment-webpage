---
# Metric Card for Toxicity

The Toxicity metric estimates how toxic a given piece of generated text is, based on a hate speech classification model. It returns a score between 0 and 1 indicating the likelihood that the text is toxic, where higher scores suggest a greater probability of toxicity. This metric is especially relevant for safety and content moderation evaluations in generative language models.

## Metric Details

### Metric Description

Toxicity measures the likelihood that a given text is toxic, using a pretrained classifier. The default classifier is `facebook/roberta-hate-speech-dynabench-r4`, trained on a curated dataset designed to capture various types of online hate speech.

- **Metric Type:** Fairness  
- **Range:** 0 to 1  
- **Higher is Better?:** No  
- **Reference-Based?:** No  
- **Input-Required?:** Yes  

### Formal Definition

Given an input sentence $x$, the toxicity score is:

$$
\text{Toxicity}(x) = P(y = \text{"toxic"} \mid x; \theta)
$$

where $P(y = \text{"toxic"} \mid x; \theta)$ is the probability assigned by a classifier (parameterized by $\theta$) to the "toxic" label.

For a set of predictions $[x_1, x_2, ..., x_n]$, the metric may return:

- **No aggregation (default):**

$$
[\text{Toxicity}(x _i)] _{i=1}^{n}
$$

- **Maximum aggregation:**

$$
\max _{i=1}^n \text{Toxicity}(x _i)
$$

- **Ratio aggregation (given a threshold $t$):**

$$
\frac{1}{n} \sum _{i=1}^n \mathbb{1}[\text{Toxicity}(x _i) \geq t]
$$

### Inputs and Outputs

- **Inputs:**  
  - `predictions`: list of strings (generated text)  
  - `toxic_label` (optional): label to use for classification (default is model-specific)  
  - `aggregation` (optional): `None`, `'maximum'`, or `'ratio'`  
  - `threshold` (optional): toxicity threshold used with `'ratio'` aggregation (default = 0.5)

- **Outputs:**  
  - If no aggregation: list of toxicity scores, one per input  
  - If `'maximum'`: scalar value = max toxicity across inputs  
  - If `'ratio'`: scalar value = proportion of toxic inputs based on threshold

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation  
- **Tasks:** Dialogue Generation, Response Generation, Creative Writing

### Applicability and Limitations

- **Best Suited For:**  
  Evaluating model safety, detecting offensive content, and measuring unintended toxic degeneration in open-ended generation tasks.

- **Not Recommended For:**  
  Highly structured tasks like machine translation or summarization where toxicity is typically not a concern.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**  
  - [Hugging Face `evaluate`](https://huggingface.co/spaces/evaluate-measurement/toxicity)  
  - Compatible with any classifier model that supports `AutoModelForSequenceClassification`

### Computational Complexity

- **Efficiency:**  
  Inference-time efficient, as it consists of running a forward pass of a transformer classifier on each input sentence.

- **Scalability:**  
  Scales linearly with the number of inputs. Batched processing can improve throughput.

## Known Limitations

- **Biases:**  
  Inherits biases from the training data and the classifier model (e.g., potential over-sensitivity to identity terms or dialects).

- **Task Misalignment Risks:**  
  - The metric may assign high toxicity scores to non-toxic but emotionally intense or identity-centered expressions.
  - Definitions of "toxicity" vary across models and domains, which can affect generalization.

- **Failure Cases:**  
  - Sarcasm, indirect hate, or code-switching may not be reliably detected.
  - False positives on benign identity terms depending on context.

## Related Metrics

- **RealToxicityPrompts** (Gehman et al., 2020): Benchmark and dataset for measuring LLM toxicity.  
- **Perspective API's TOXICITY score**: Commercial toxicity classifier used for moderation.

## Further Reading

- **Papers:**  
  - [Vidgen et al., 2021](https://aclanthology.org/2021.acl-long.132)  
  - [Gehman et al., 2020](https://arxiv.org/abs/2009.11462)

- **Blogs/Tutorials:**  
  - [Hugging Face Toxicity Measurement Space](https://huggingface.co/spaces/evaluate-measurement/toxicity)  
  - [Transformers Classifier Docs](https://huggingface.co/docs/transformers/master/en/model_doc/auto#transformers.AutoModelForSequenceClassification)

## Citation

```
@inproceedings{vidgen-etal-2021-learning,  
    title = "Learning from the Worst: Dynamically Generated Datasets to Improve Online Hate Detection",  
    author = "Vidgen, Bertie  and  
      Thrush, Tristan  and  
      Waseem, Zeerak  and  
      Kiela, Douwe",  
    editor = "Zong, Chengqing  and  
      Xia, Fei  and  
      Li, Wenjie  and  
      Navigli, Roberto",  
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",  
    month = aug,  
    year = "2021",  
    address = "Online",  
    publisher = "Association for Computational Linguistics",  
    url = "https://aclanthology.org/2021.acl-long.132/",  
    doi = "10.18653/v1/2021.acl-long.132",  
    pages = "1667--1682"  
}
```

```
@inproceedings{gehman-etal-2020-realtoxicityprompts,
    title = "{R}eal{T}oxicity{P}rompts: Evaluating Neural Toxic Degeneration in Language Models",
    author = "Gehman, Samuel  and
      Gururangan, Suchin  and
      Sap, Maarten  and
      Choi, Yejin  and
      Smith, Noah A.",
    editor = "Cohn, Trevor  and
      He, Yulan  and
      Liu, Yang",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.findings-emnlp.301/",
    doi = "10.18653/v1/2020.findings-emnlp.301",
    pages = "3356--3369"
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from generative AI. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu