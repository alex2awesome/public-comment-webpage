---
# Metric Card for FastTextEducationalValue

FastTextEducationalValue is a reference-free classification-based metric that evaluates the educational quality of generated text. It uses a FastText classifier trained to predict three levels of educational value—Low, Mid, and High—and outputs an expected value score by taking a weighted sum over the classifier’s label probabilities. This metric is particularly useful for content filtering, ranking, or prioritizing educational materials in generative settings, especially when no reference output is available.

## Metric Details

### Metric Description

FastTextEducationalValue operates by predicting the probability distribution over three educational value labels: `__label__Low` (0), `__label__Mid` (1), and `__label__High` (2). The final score is computed as the expected value of this distribution, effectively producing a scalar in the range [0, 2] indicating the overall educational quality of the text. The classifier was trained using the FastText library on a custom dataset of educational and non-educational text, with a focus on fast CPU-based inference suitable for large-scale use.

- **Metric Type:** Fairness  
- **Range:** $[0, 2]$  
- **Higher is Better?:** Yes  
- **Reference-Based?:** No  
- **Input-Required?:** Optional  

### Formal Definition

Let $p_0$, $p_1$, and $p_2$ be the probabilities predicted by the classifier for the labels `Low`, `Mid`, and `High` respectively. Then the educational value score is:

$$
\text{Score} = 0 \cdot p_0 + 1 \cdot p_1 + 2 \cdot p_2 = p_1 + 2p_2
$$

### Inputs and Outputs

- **Inputs:**  
  - Generated text (e.g., from a language model)  
  - Input prompt (optional; not used in scoring)  

- **Outputs:**  
  - Scalar value in $[0, 2]$  
    - 0 = Low educational value  
    - 1 = Mid educational value  
    - 2 = High educational value  
    - Values in between represent the expected educational quality

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation, Dialogue Systems, Educational Content Generation  
- **Tasks:** Response Generation, Educational QA, Summarization, Content Ranking  

### Applicability and Limitations

- **Best Suited For:**  
  - Ranking or filtering generated text for educational content  
  - Benchmarking generative models on ability to produce high-value educational output  
  - Fast, large-scale evaluation of generated content

- **Not Recommended For:**  
  - Fine-grained subject classification or curriculum alignment  
  - Use on non-educational domains without adaptation  
  - Multilingual or out-of-distribution content without retraining

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**  
  - [Facebook fastText](https://github.com/facebookresearch/fastText)  
  - [Hugging Face Hub model repo](https://huggingface.co/kenhktsui/llm-data-textbook-quality-fasttext-classifer-v2)  
  - Available via `autometrics` Python module

### Computational Complexity

- **Efficiency:**  
  - Highly efficient; inference on CPU using hierarchical softmax and bag-of-ngrams  
  - Single-pass, low-latency scoring for individual or batched inputs  

- **Scalability:**  
  - Suitable for large-scale deployments, including dataset filtering pipelines and streaming evaluation  
  - Model loading is lightweight and fast due to compact binary format  

## Known Limitations

- **Biases:**  
  - [Needs more information]  

- **Task Misalignment Risks:**  
  - May underperform on open-ended or creative writing tasks where educational value is subjective or ambiguous  
  - Not suitable for measuring depth of reasoning or factual accuracy  

- **Failure Cases:**  
  - [Needs more information]  

## Related Metrics

- **FastTextToxicity** – Related FastText-based classifier metric for harmfulness  
- **EDU-level Detectors** – Heuristics or classifiers trained on educational standards  
- **LM-Based Rubric Scorers** – More complex scoring using LLMs conditioned on human-written rubrics

## Further Reading

- **Papers:**  
  - [Low Latency CPU Based Educational Value Classifier With Generic Educational Value (Tsui & Nguyen, 2024)]  
  - [Bag of Tricks for Efficient Text Classification (Joulin et al., 2017)](https://aclanthology.org/E17-2068/)

- **Blogs/Tutorials:**  
  - [More Information Needed]

## Citation

Educational Value Classifier
```  
@misc{ktsui2024cpueduvalue,  
  title={Low Latency CPU Based Educational Value Classifier With Generic Educational Value},  
  author={Ken Tsui and Huu Nguyen},  
  year={2024},  
}
```

FastText
```
@inproceedings{joulin-etal-2017-bag,  
  title = "Bag of Tricks for Efficient Text Classification",  
  author = "Joulin, Armand  and Grave, Edouard  and Bojanowski, Piotr  and Mikolov, Tomas",  
  editor = "Lapata, Mirella  and Blunsom, Phil  and Koller, Alexander",  
  booktitle = "Proceedings of the 15th Conference of the {E}uropean Chapter of the Association for Computational Linguistics: Volume 2, Short Papers",  
  month = apr,  
  year = "2017",  
  address = "Valencia, Spain",  
  publisher = "Association for Computational Linguistics",  
  url = "https://aclanthology.org/E17-2068/",  
  pages = "427--431"  
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from OpenAI's ChatGPT, based on user-provided inputs and referenced documentation. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu