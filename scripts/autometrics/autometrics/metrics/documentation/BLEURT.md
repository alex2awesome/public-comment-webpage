---
# Metric Card for BLEURT

BLEURT (Bilingual Evaluation Understudy with Representations from Transformers) is a learned evaluation metric for natural language generation. It combines BERT-based contextual embeddings with a regression model fine-tuned on human-annotated data. BLEURT produces scalar scores that aim to reflect fluency, grammaticality, and semantic adequacy by measuring the similarity between generated and reference texts. It has demonstrated strong correlation with human judgment in machine translation and summarization tasks.

## Metric Details

### Metric Description

BLEURT is a regression-based evaluation metric trained to approximate human judgment of text quality. It leverages transfer learning in multiple stages: beginning with a pretrained BERT model, then performing further pretraining on synthetically noised data to improve robustness, and finally fine-tuning on human-labeled ratings from the WMT Metrics Shared Tasks.

The metric compares a generated sentence (candidate) to a human-written sentence (reference) and produces a scalar score indicating how well the candidate matches the reference in terms of fluency and adequacy. Different BLEURT checkpoints (e.g., BLEURT-20, BLEURT-20-D12) vary in size, accuracy, and multilingual support.

- **Metric Type:** Semantic Similarity  
- **Range:** Approximately 0 to 1 (but may exceed 1 or fall below 0 depending on checkpoint and inputs)  
- **Higher is Better?:** Yes  
- **Reference-Based?:** Yes  
- **Input-Required?:** No  

### Formal Definition

Let $x$ be the reference sentence and $\hat{x}$ the candidate sentence. BLEURT computes a learned score:

$$
\text{BLEURT}(x, \hat{x}) = f_\theta(x, \hat{x})
$$

where $f_\theta$ is a regression model (typically based on BERT or RemBERT) fine-tuned to predict human ratings of text similarity. The model is trained to minimize mean squared error on labeled sentence pairs.

### Inputs and Outputs

- **Inputs:**  
  - Generated text (candidate sentence)  
  - Reference text (reference sentence)  
  - BLEURT checkpoint (e.g., `BLEURT-20`, `bleurt-base-128`, etc.)

- **Outputs:**  
  - A scalar score for each sentence pair, representing similarity and adequacy  

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation  
- **Tasks:** Machine Translation, Summarization, Paraphrasing, Data-to-Text Generation  

### Applicability and Limitations

- **Best Suited For:**  
  Sentence-level evaluation where reference-based semantic adequacy is important. Particularly effective for machine translation, summarization, and tasks with high-quality references.

- **Not Recommended For:**  
  - Reference-free evaluation settings  
  - Tasks with highly diverse valid outputs (e.g., open-ended dialogue or storytelling) where many correct outputs may not resemble the reference lexically or structurally

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**  
  - [google-research/bleurt](https://github.com/google-research/bleurt)  
  - [Hugging Face Evaluate: BLEURT](https://huggingface.co/spaces/evaluate-metric/bleurt)  
  - [lucadiliello/bleurt-pytorch](https://github.com/lucadiliello/bleurt-pytorch)

### Computational Complexity

- **Efficiency:**  
  BLEURT is significantly more computationally expensive than n-gram based metrics like BLEU. It requires loading a large transformer model and computing contextual embeddings and a regression head per sentence pair.

- **Scalability:**  
  BLEURT supports batched inference and length-based batching to speed up evaluation on large corpora. Distilled checkpoints (e.g., BLEURT-20-D6) provide faster and smaller alternatives.

## Known Limitations

- BLEURT scores may vary significantly across different checkpoints; scores are not directly comparable across models.
- Output scores may fall outside the nominal 0–1 range.
- BLEURT may reflect biases present in the pretrained models and training data.
- BLEURT primarily supports English; while BLEURT-20 supports several other languages (e.g., French, Chinese, German), its performance in low-resource or code-mixed settings is less validated.

- **Biases:**  
  BLEURT inherits biases from BERT/RemBERT and from the WMT training annotations. Its judgments may reflect linguistic and cultural norms present in the training data.

- **Task Misalignment Risks:**  
  BLEURT assumes a single reference and may unfairly penalize valid alternative phrasings or creative outputs not matching the reference closely.

- **Failure Cases:**  
  BLEURT may overvalue surface similarity or penalize outputs that are fluent but structurally divergent from the reference. It may also be unreliable on very short or very long sequences.

## Related Metrics

- **BERTScore:** Also uses contextual embeddings but focuses on token-level similarity using cosine similarity.
- **COMET:** Another learned metric based on multilingual encoder-decoder architecture trained on direct assessment data.
- **METEOR:** Incorporates synonym matching and paraphrase tables but is not learned.
- **BLEU/ROUGE:** Surface-level overlap metrics commonly used for baseline evaluation.

## Further Reading

- **Papers:**  
  - [BLEURT: Learning Robust Metrics for Text Generation (Sellam et al., 2020)](https://aclanthology.org/2020.acl-main.704/)  
  - [Learning Compact Metrics for MT (Pu et al., 2021)](https://arxiv.org/abs/2110.06341)

- **Blogs/Tutorials:**  
  - [Google AI Blog: Evaluating Natural Language Generation with BLEURT](https://ai.googleblog.com/2020/05/evaluating-natural-language-generation.html)

## Citation

```
@inproceedings{sellam-etal-2020-bleurt,  
 title = "{BLEURT}: Learning Robust Metrics for Text Generation",  
 author = "Sellam, Thibault  and  
  Das, Dipanjan  and  
  Parikh, Ankur",  
 editor = "Jurafsky, Dan  and  
  Chai, Joyce  and  
  Schluter, Natalie  and  
  Tetreault, Joel",  
 booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",  
 month = jul,  
 year = "2020",  
 address = "Online",  
 publisher = "Association for Computational Linguistics",  
 url = "https://aclanthology.org/2020.acl-main.704/",  
 doi = "10.18653/v1/2020.acl-main.704",  
 pages = "7881--7892"  
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from generative AI. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu