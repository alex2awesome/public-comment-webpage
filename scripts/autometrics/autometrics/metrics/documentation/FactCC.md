---
# Metric Card for FactCC

FactCC is a reference-free metric designed to evaluate the factual consistency of summaries with respect to source documents. It uses a fine-tuned BERT-based sequence classification model trained on weakly supervised data to determine whether a summary sentence is factually supported by its source. The model was specifically developed to overcome the limitations of traditional summarization metrics that fail to capture factual inconsistencies.

## Metric Details

### Metric Description

FactCC formulates factual consistency as a binary classification task: given a source document and a candidate summary, the model predicts whether the summary is factually consistent (`CORRECT`) or inconsistent (`INCORRECT`) with the source. To train the model, synthetic inconsistent examples are generated using rule-based transformations such as entity swaps, negation, and date/number perturbations. The model is then fine-tuned using this weakly supervised dataset and evaluated against human-annotated summaries.

- **Metric Type:** Faithfulness  
- **Range:** [0, 1] (probability of label; final output is binary `CORRECT` or `INCORRECT`)  
- **Higher is Better?:** Yes (with respect to correctness probability or accuracy over a dataset)  
- **Reference-Based?:** No  
- **Input-Required?:** Yes  

### Formal Definition

Let $x$ denote the source document and $\hat{x}$ the generated summary. FactCC uses a BERT-based classifier to compute:

$$
\hat{y} = \arg\max _{y \in \{\text{CORRECT}, \text{INCORRECT}\}} f(x, \hat{x})
$$

where $f$ is the classification function learned by the model, trained on synthetic (x, summary, label) triples generated from CNN/DailyMail.

### Inputs and Outputs

- **Inputs:**  
  - Source document (string)  
  - Summary (string)  

- **Outputs:**  
  - Label: `CORRECT` or `INCORRECT`  
  - Confidence score (optional): probability assigned to the predicted label

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation  
- **Tasks:** Summarization  

### Applicability and Limitations

- **Best Suited For:**  
  Factual consistency evaluation of abstractive summaries, particularly when reference summaries are unavailable or unreliable.

- **Not Recommended For:**  
  Tasks requiring nuanced semantic judgments beyond factual consistency, such as coherence, fluency, or stylistic alignment.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**  
  - [Hugging Face Transformers](https://huggingface.co/manueldeprada/FactCC)  
  - [Original GitHub Repository](https://github.com/salesforce/factCC) (archived)  

### Computational Complexity

- **Efficiency:**  
  Comparable to standard BERT-based classification; inference time is linear in input length and batch size.  

- **Scalability:**  
  Scales well with GPU acceleration and batched inference; model supports max input lengths up to 512 tokens and can be used with pipelines for large-scale evaluation.

## Known Limitations

- **Biases:**  
  Trained primarily on CNN/DailyMail, which may bias performance toward that domain. Synthetic negative examples may not reflect real-world inconsistencies in all domains.  

- **Task Misalignment Risks:**  
  May fail to detect hallucinations that are not captured by training transformations. Not suitable for evaluating semantic quality or fluency.  

- **Failure Cases:**  
  - May misclassify paraphrased or implicature-based summaries as incorrect  
  - Sensitive to token truncation due to 512-token max input length  

## Related Metrics

- **QAGS:** Uses QA-based probing to evaluate consistency  
- **SummaQA:** Another question-answering-based factuality metric  
- **BERTScore:** Captures semantic similarity but not factuality  
- **DAE (Dependency Arc Entailment):** Focuses on syntactic entailment between summary and source  

## Further Reading

- **Papers:**  
  - [Kryściński et al., 2020 (EMNLP)](https://aclanthology.org/2020.emnlp-main.750/)  
  - [Original ArXiv Preprint](https://arxiv.org/abs/1910.12840)  

- **Blogs/Tutorials:**  
  - [More Information Needed]

## Citation

```
@inproceedings{kryscinski-etal-2020-evaluating,  
  title = "Evaluating the Factual Consistency of Abstractive Text Summarization",  
  author = "Kryscinski, Wojciech  and  
    McCann, Bryan  and  
    Xiong, Caiming  and  
    Socher, Richard",  
  editor = "Webber, Bonnie  and  
    Cohn, Trevor  and  
    He, Yulan  and  
    Liu, Yang",  
  booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",  
  month = nov,  
  year = "2020",  
  address = "Online",  
  publisher = "Association for Computational Linguistics",  
  url = "https://aclanthology.org/2020.emnlp-main.750/",  
  doi = "10.18653/v1/2020.emnlp-main.750",  
  pages = "9332--9346"  
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from generative AI. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu