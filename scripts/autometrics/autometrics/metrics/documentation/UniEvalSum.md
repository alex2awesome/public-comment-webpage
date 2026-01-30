---
# Metric Card for UniEvalSum

UniEvalSum is a multi-dimensional evaluation metric designed specifically for text summarization. It assesses summaries across four key dimensions: coherence, consistency, fluency, and relevance. The metric formulates evaluation as a Boolean Question Answering (QA) task, where a pre-trained language model predicts scores based on Yes/No answers to predefined evaluation questions.

## Metric Details

### Metric Description

UniEvalSum evaluates text summaries by converting evaluation into a Boolean QA problem. The model is guided by questions tailored to specific evaluation dimensions, enabling it to assess coherence, consistency, fluency, and relevance. It leverages a pre-trained T5 model and intermediate learning techniques to improve evaluation robustness. Unlike traditional similarity-based metrics (e.g., ROUGE, BLEU), UniEvalSum does not rely solely on reference texts and can function in a reference-free manner except for the relevance dimension.

- **Metric Type:** Semantic Similarity, Reference-Free, Multi-Dimensional Evaluation
- **Range:** [0,1] for all dimensions  
- **Higher is Better?:** Yes  
- **Reference-Based?:** Yes (Reference required for relevance, but not for other dimensions)  
- **Input-Required?:** Yes  

### Formal Definition

Given a generated summary $s$, a reference summary $r$, and a source document $d$, UniEvalSum evaluates four dimensions using a pre-trained T5 model in a Boolean QA format:

$$
\text{Score}_{dim} = \frac{P(\text{"Yes"} \mid s, d, r, q)}{P(\text{"Yes"} \mid s, d, r, q) + P(\text{"No"} \mid s, d, r, q)}
$$

where $q$ represents the evaluation question for a given dimension (e.g., "Is this a coherent summary to the document?"). The final overall score is computed as the average of the four dimension scores.

### Inputs and Outputs

- **Inputs:**  
  - Generated summary
  - Reference summary (only for relevance dimension)
  - Source document

- **Outputs:**  
  - Scores for coherence, consistency, fluency, and relevance (range: [0,1])  
  - Overall score (default: average of all dimension scores)  

### Prompt Formulation for Evaluation Dimensions

UniEvalSum structures evaluation by generating **Boolean QA-style prompts** for each dimension:

1. **Coherence** (Evaluates whether the summary is well-structured and logically consistent)
   - **Prompt Template:**  
     ```
     question: Is this a coherent summary to the document? </s> summary: {system_output} </s> document: {source}
     ```
   - **Inputs Required:** Source document, generated summary  

2. **Consistency** (Measures factual consistency with the source document)
   - **Prompt Template:**  
     ```
     question: Is this claim consistent with the document? </s> claim: {system_output} </s> document: {source}
     ```
   - **Inputs Required:** Source document, generated summary  

3. **Fluency** (Evaluates linguistic fluency and grammatical correctness)
   - **Prompt Template:**  
     ```
     question: Is this a fluent paragraph? </s> paragraph: {system_output}
     ```
   - **Inputs Required:** Generated summary  

4. **Relevance** (Measures how well the summary captures the key information from the reference summary)
   - **Prompt Template:**  
     ```
     question: Is this summary relevant to the reference? </s> summary: {system_output} </s> reference: {reference}
     ```
   - **Inputs Required:** Generated summary, reference summary  

These prompts are tokenized and passed into the **UniEvalSum** model, which then predicts Yes/No probabilities, converting them into scores between 0 and 1.

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation  
- **Tasks:** Summarization  

### Applicability and Limitations

- **Best Suited For:**  
  - Evaluating abstractive and extractive summaries in a structured manner  
  - Tasks requiring explainable and interpretable evaluation dimensions  

- **Not Recommended For:**  
  - Open-ended creative writing evaluation  
  - Tasks where diversity and novelty are primary concerns  

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**
  - [Hugging Face Model: MingZhong/unieval-sum](https://huggingface.co/MingZhong/unieval-sum)
  - [GitHub Repository: UniEval](https://github.com/maszhongming/UniEval)

### Computational Complexity

- **Efficiency:**  
  The model requires encoding the input and computing probability distributions over the Yes/No outputs. While more computationally expensive than simple n-gram-based metrics, it remains feasible for large-scale summarization evaluations.

- **Scalability:**  
  UniEvalSum scales well with dataset size, but evaluation cost grows linearly with the number of summaries due to separate model calls for each dimension.

## Known Limitations

- **Biases:**  
  - May inherit biases from the pre-trained T5 model.  
  - Performance may degrade if the summary style deviates significantly from the model's training data.  

- **Task Misalignment Risks:**  
  - While designed for summarization, results may not generalize to highly domain-specific summarization tasks (e.g., scientific summarization).  

- **Failure Cases:**  
  - In cases where reference summaries are poor, the relevance dimension may not be a reliable measure.  
  - Evaluation scores may not correlate well with human judgment for summaries that use complex paraphrasing.  

## Related Metrics

- **ROUGE:** Lexical similarity-based evaluation metric commonly used for summarization.  
- **BERTScore:** Embedding-based similarity metric for text evaluation.  
- **CTC (Compression, Transduction, Creation):** A framework for evaluating NLG tasks across multiple dimensions.  

## Further Reading

- **Papers:**  
  - [Towards a Unified Multi-Dimensional Evaluator for Text Generation (Zhong et al., 2022)](https://aclanthology.org/2022.emnlp-main.131.pdf)  

- **Blogs/Tutorials:**  
  - [UniEval GitHub Documentation](https://github.com/maszhongming/UniEval)  

## Citation

```
@inproceedings{zhong-etal-2022-towards,
    title = "Towards a Unified Multi-Dimensional Evaluator for Text Generation",
    author = "Zhong, Ming  and
      Liu, Yang  and
      Yin, Da  and
      Mao, Yuning  and
      Jiao, Yizhu  and
      Liu, Pengfei  and
      Zhu, Chenguang  and
      Ji, Heng  and
      Han, Jiawei",
    editor = "Goldberg, Yoav  and
      Kozareva, Zornitsa  and
      Zhang, Yue",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.131/",
    doi = "10.18653/v1/2022.emnlp-main.131",
    pages = "2023--2038"
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from generative AI. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu  