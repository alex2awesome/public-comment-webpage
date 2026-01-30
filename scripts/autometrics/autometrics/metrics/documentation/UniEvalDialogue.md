---
# Metric Card for UniEvalDialogue

UniEvalDialogue is a multi-dimensional evaluation metric designed specifically for **dialogue response generation**. It assesses responses across five key dimensions: **naturalness, coherence, engagingness, groundedness, and understandability**. The metric formulates evaluation as a **Boolean Question Answering (QA) task**, where a pre-trained language model predicts scores based on Yes/No answers to predefined evaluation questions.

## Metric Details

### Metric Description

UniEvalDialogue evaluates dialogue responses by **converting evaluation into a Boolean QA problem**. The model is prompted with questions tailored to specific evaluation dimensions, allowing it to assess **fluency, informativeness, factual grounding, and coherence** in open-ended dialogue tasks. Unlike traditional metrics that rely on reference responses, UniEvalDialogue is primarily **reference-free**, except when additional factual grounding is required (e.g., engagingness evaluation).

- **Metric Type:** Semantic Similarity, Reference-Free, Multi-Dimensional Evaluation  
- **Range:** [0,1] for all dimensions  
- **Higher is Better?:** Yes  
- **Reference-Based?:** Yes (groundedness and engagingness require factual context)
- **Input-Required?:** Yes  

### Formal Definition

Given a **generated response** $r$, a **dialogue history** $h$, and optionally a **factual grounding document** $f$, UniEvalDialogue evaluates five dimensions using a **pre-trained T5 model** in a Boolean QA format:

$$
\text{Score}_{dim} = \frac{P(\text{"Yes"} \mid r, h, f, q)}{P(\text{"Yes"} \mid r, h, f, q) + P(\text{"No"} \mid r, h, f, q)}
$$

where $q$ represents the evaluation question for a given dimension (e.g., "Is this response coherent given the dialogue history?"). The final **overall score** is computed as the **average** of the five dimension scores.

### Inputs and Outputs

- **Inputs:**  
  - Generated response  
  - Dialogue history  
  - Optional factual context (for groundedness and engagingness evaluation)  

- **Outputs:**  
  - Scores for **naturalness, coherence, engagingness, groundedness, and understandability** (range: [0,1])  
  - Overall score (default: **average of all dimension scores**)  

### Prompt Formulation for Evaluation Dimensions

UniEvalDialogue structures evaluation using **Boolean QA-style prompts** for each dimension:

1. **Naturalness** (Measures how human-like the response sounds)
   - **Prompt Template:**  
     ```
     question: Is this a natural response in the dialogue? </s> response: {system_output}
     ```
   - **Inputs Required:** Generated response  

2. **Coherence** (Checks logical consistency with prior turns in the dialogue)
   - **Prompt Template:**  
     ```
     question: Is this a coherent response given the dialogue history? </s> response: {system_output} </s> dialogue history: {source}
     ```
   - **Inputs Required:** Dialogue history, generated response  

3. **Engagingness** (Evaluates informativeness and conversational depth, requires factual grounding)
   - **Prompt Template:**  
     ```
     question: Is this an engaging and informative response according to the dialogue history and fact? </s> response: {system_output} </s> dialogue history: {source} </s> fact: {context}
     ```
   - **Inputs Required:** Dialogue history, generated response, factual grounding  

4. **Groundedness** (Assesses factual accuracy against a known knowledge base)
   - **Prompt Template:**  
     ```
     question: Is this response consistent with knowledge in the fact? </s> response: {system_output} </s> fact: {context}
     ```
   - **Inputs Required:** Generated response, factual grounding  

5. **Understandability** (Determines if the response is clear and interpretable)
   - **Prompt Template:**  
     ```
     question: Is this an understandable response in the dialogue? </s> response: {system_output}
     ```
   - **Inputs Required:** Generated response  

These prompts are tokenized and passed into the **UniEvalDialogue** model, which then predicts **Yes/No probabilities**, converting them into scores between **0 and 1**.

## Intended Use

### Domains and Tasks

- **Domain:** Dialogue Systems, Text Generation  
- **Tasks:** Dialogue Response Generation  

### Applicability and Limitations

- **Best Suited For:**  
  - Evaluating **open-domain** and **task-oriented** dialogue systems  
  - Systems where **coherence, fluency, and engagingness** are critical metrics  
  - Settings where **reference-free evaluation** is preferred  

- **Not Recommended For:**  
  - Evaluating **long-form creative writing** (e.g., storytelling, novel generation)  
  - Contexts where **fact verification requires external retrieval** (e.g., legal or medical dialogue)  

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**
  - [Hugging Face Model: MingZhong/unieval-dialog](https://huggingface.co/MingZhong/unieval-dialog)
  - [GitHub Repository: UniEval](https://github.com/maszhongming/UniEval)

### Computational Complexity

- **Efficiency:**  
  The model requires encoding the input and computing probability distributions over the **Yes/No outputs**. While more computationally expensive than simple token-overlap metrics, it remains efficient for evaluating dialogue at **scale**.

- **Scalability:**  
  UniEvalDialogue scales well with dataset size, but evaluation cost **grows linearly** with the number of responses due to separate model calls for each **evaluation dimension**.

## Known Limitations

- **Biases:**  
  - May inherit **biases** from the pre-trained T5 model.  
  - May **underestimate response diversity**, favoring safe, generic dialogue responses.  

- **Task Misalignment Risks:**  
  - While designed for dialogue, results may not **generalize well to highly structured** domains like legal or scientific chatbots.  

- **Failure Cases:**  
  - **Groundedness evaluation may fail** if factual sources are noisy or ambiguous.  
  - Model-generated scores may **not correlate well with human judgments** in highly subjective dialogue tasks.  

## Related Metrics

- **USR (Unsupervised and Reference-Free Dialogue Evaluation):** A multi-dimensional metric for dialogue quality.  
- **FED (Fine-grained Evaluation of Dialogue):** Uses a similar reference-free approach with multiple scoring dimensions.  
- **BLEU & METEOR:** Traditional n-gram overlap metrics (less relevant for dialogue).  

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