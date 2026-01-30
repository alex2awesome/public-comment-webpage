---
# Metric Card for FastTextToxicity

FastTextToxicity is a reference-free classification metric based on a FastText binary classifier trained to detect toxic content in text. It was developed as part of the AI2 Dolma project to filter large-scale web data for pretraining language models. The classifier outputs a scalar toxicity score for a given model-generated output, without requiring gold-standard references. This makes it efficient and suitable for real-time or large-scale generative evaluations.

## Metric Details

### Metric Description

FastTextToxicity assigns a score to generated text based on whether it is classified as toxic or non-toxic. The score is the classifier’s confidence in the predicted label, negated if the prediction is “toxic.” This allows for a continuous scalar signal indicating both toxicity and confidence. The model is a FastText linear classifier trained on the Jigsaw Toxic Comment Classification dataset, using a bag-of-words representation with optional n-gram features.

The binary classification task was defined as follows:
- **Positive class ("toxic")**: Examples from the Jigsaw dataset that contain at least one label excluding “obscene” (e.g., “toxic”, “severe toxic”, “threat”, “insult”, “identity hate”).
- **Negative class ("non-toxic")**: Examples with no labels at all.
- **Excluded from training**: Examples labeled *only* as “obscene” were not used in training for this classifier.

- **Metric Type:** Fairness  
- **Range:** $[-1.0, 1.0]$  
- **Higher is Better?:** No (more negative implies more toxic)  
- **Reference-Based?:** No  
- **Input-Required?:** Optional  

### Formal Definition

Let $\hat{y}$ be the predicted label from the FastText classifier and $p(\hat{y}|\hat{x})$ its confidence score for generated text $\hat{x}$. Let $L_\text{toxic}$ denote the "toxic" label. Then the FastTextToxicity score is:

$$
s(\hat{x}) = 
\begin{cases}
-p(\hat{y}|\hat{x}) & \text{if } \hat{y} = L_\text{toxic} \\
\phantom{-}p(\hat{y}|\hat{x}) & \text{otherwise}
\end{cases}
$$

### Inputs and Outputs

- **Inputs:**  
  - Generated text (model output)  
  - Input prompt (optional, unused in classification)  

- **Outputs:**  
  - Scalar score in $[-1.0, 1.0]$  
    - Negative values indicate toxic content with higher toxicity closer to -1  
    - Positive values indicate non-toxic content with higher confidence

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation, Dialogue Systems  
- **Tasks:** Dialogue Generation, Response Generation, Content Moderation  

### Applicability and Limitations

- **Best Suited For:**  
  - Large-scale reference-free toxicity screening  
  - Fast toxicity filtering in data pipelines for LLM pretraining  
  - Use in fairness- or harm-focused evaluation benchmarks

- **Not Recommended For:**  
  - Fine-grained harm type classification (e.g., distinguishing identity hate from threats)  
  - Sensitive, nuanced contexts where subjective judgment or social context is important  
  - General-purpose hate speech detection without retraining or calibration

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**  
  - [Facebook fastText](https://github.com/facebookresearch/fastText)  
  - AI2’s internal implementation (`FastTextClassifier` in `autometrics`)  
  - Requires loading the FastText binary model trained on Jigsaw

### Computational Complexity

- **Efficiency:**  
  - Very fast; designed for CPU inference using bag-of-ngrams and hierarchical softmax.  
  - Suitable for batch and streaming evaluation.

- **Scalability:**  
  - Scales to millions of examples.  
  - Model weights are compact and load quickly, enabling low-latency prediction.

## Known Limitations

- **Biases:**  
  - Inherits annotation biases from the Jigsaw dataset (e.g., potential over-labeling of minority dialects).  
  - Can encode Western-centric norms of toxicity unless calibrated.

- **Task Misalignment Risks:**  
  - Does not distinguish between different types or severities of harm.  
  - May misclassify sarcasm, coded language, or culturally embedded expressions.

- **Failure Cases:**  
  - Lacks robustness to adversarial rewordings.  
  - Performance may degrade significantly on out-of-distribution text or non-English content unless retrained.

## Related Metrics

- **Perspective API Toxicity** (Google Jigsaw)  
- **HateXplain Score**  
- **RealToxicityPrompts**  
- **NSFW Classifier (Dolma)** — companion FastText-based filter trained on NSFW data

## Further Reading

- **Papers:**  
  - [Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research (Soldaini et al., 2024)](https://aclanthology.org/2024.acl-long.840/)  
  - [Bag of Tricks for Efficient Text Classification (Joulin et al., 2017)](https://aclanthology.org/E17-2068/)

- **Blogs/Tutorials:**  
  - [FastText Text Classification Tutorial](https://fasttext.cc/docs/en/supervised-tutorial.html)

## Citation

Dolma
```
@inproceedings{soldaini-etal-2024-dolma,  
  title = "Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research",  
  author = "Soldaini, Luca and Kinney, Rodney and Bhagia, Akshita and Schwenk, Dustin and Atkinson, David and Authur, Russell and Bogin, Ben and Chandu, Khyathi and Dumas, Jennifer and Elazar, Yanai and Hofmann, Valentin and Jha, Ananya and Kumar, Sachin and Lucy, Li and Lyu, Xinxi and Lambert, Nathan and Magnusson, Ian and Morrison, Jacob and Muennighoff, Niklas and Naik, Aakanksha and Nam, Crystal and Peters, Matthew and Ravichander, Abhilasha and Richardson, Kyle and Shen, Zejiang and Strubell, Emma and Subramani, Nishant and Tafjord, Oyvind and Walsh, Evan and Zettlemoyer, Luke and Smith, Noah and Hajishirzi, Hannaneh and Beltagy, Iz and Groeneveld, Dirk and Dodge, Jesse and Lo, Kyle",  
  editor = "Ku, Lun-Wei and Martins, Andre and Srikumar, Vivek",  
  booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",  
  month = aug,  
  year = "2024",  
  address = "Bangkok, Thailand",  
  publisher = "Association for Computational Linguistics",  
  url = "https://aclanthology.org/2024.acl-long.840/",  
  doi = "10.18653/v1/2024.acl-long.840",  
  pages = "15725--15788"  
}
```

FastText
```
@inproceedings{joulin-etal-2017-bag,  
  title = "Bag of Tricks for Efficient Text Classification",  
  author = "Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Mikolov, Tomas",  
  editor = "Lapata, Mirella and Blunsom, Phil and Koller, Alexander",  
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