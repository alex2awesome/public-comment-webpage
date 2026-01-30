from autometrics.metrics.reference_free.FastTextClassifier import FastTextClassifier
from typing import ClassVar

class FastTextNSFW(FastTextClassifier):
    """---
# Metric Card for FastTextNSFW

FastTextNSFW is a reference-free binary classification metric for evaluating the presence of not-safe-for-work (NSFW) content in generated text. It is based on a FastText linear classifier trained on the Jigsaw Toxic Comment Classification dataset, using the "obscene" label as a proxy for NSFW content. This metric was developed as part of the AI2 Dolma project to enable scalable filtering of large-scale web text data during corpus construction for language model pretraining. It outputs a signed scalar score indicating both the classification result and the model's confidence.

## Metric Details

### Metric Description

FastTextNSFW uses a FastText classifier to assess whether generated text is NSFW. The classifier was trained using the Jigsaw Toxic Comment Classification dataset, where comments labeled "obscene" were treated as positive examples (NSFW), and comments with **no labels** were treated as negative examples (non-NSFW). Comments labeled only with other toxicity labels (e.g., "threat", "insult", "identity hate") were **excluded** from training.

The model outputs a label ("nsfw" or "non-nsfw") along with a confidence score. The final metric score is the model's confidence, negated if the predicted label is "nsfw". This design allows the score to capture both the label and its confidence in a single continuous value.

- **Metric Type:** Fairness  
- **Range:** $[-1.0, 1.0]$  
- **Higher is Better?:** No (more negative implies more NSFW)  
- **Reference-Based?:** No  
- **Input-Required?:** Optional  

### Formal Definition

Let $\hat{y} \in \{\text{nsfw}, \text{non-nsfw}\}$ be the predicted label from the classifier for a generated output $\hat{x}$, and let $p(\hat{y}|\hat{x})$ be the classifier's confidence in that label. Then the FastTextNSFW score is:

$$
s(\hat{x}) = 
\begin{cases}
-p(\hat{y}|\hat{x}) & \text{if } \hat{y} = \text{nsfw} \\
\phantom{-}p(\hat{y}|\hat{x}) & \text{if } \hat{y} = \text{non-nsfw}
\end{cases}
$$

### Inputs and Outputs

- **Inputs:**  
  - Generated text (output from a model)  
  - Input prompt (optional; unused by the classifier)  

- **Outputs:**  
  - Scalar score in $[-1.0, 1.0]$  
    - Negative values indicate NSFW content (closer to -1 means higher confidence)  
    - Positive values indicate non-NSFW content (closer to 1 means higher confidence)

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation, Dialogue Systems  
- **Tasks:** Dialogue Generation, Response Generation, Content Moderation  

### Applicability and Limitations

- **Best Suited For:**  
  - Reference-free NSFW content detection in generative text  
  - Large-scale content filtering for pretraining datasets  
  - Safety alignment evaluations for LLMs  

- **Not Recommended For:**  
  - Fine-grained classification of NSFW subtypes (e.g., distinguishing between sexual content and offensive jokes)  
  - Use cases requiring contextual or cultural nuance  
  - Multilingual content detection without retraining  

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**  
  - [Facebook fastText](https://github.com/facebookresearch/fastText)  
  - AI2's `autometrics` library (`FastTextClassifier`)  
  - Requires loading model from:  
    `https://dolma-artifacts.org/fasttext_models/jigsaw_fasttext_bigrams_20230515/jigsaw_fasttext_bigrams_nsfw_final.bin`

### Computational Complexity

- **Efficiency:**  
  - Extremely fast inference using bag-of-ngrams and hierarchical softmax  
  - Designed for CPU inference and scalable evaluation  

- **Scalability:**  
  - Easily scales to millions of examples  
  - Small memory footprint enables real-time or batch use  

## Known Limitations

- **Biases:**  
  - Reflects annotation biases in the Jigsaw dataset (e.g., cultural interpretations of obscenity)  
  - May underrepresent some NSFW categories if not labeled as "obscene" in training  

- **Task Misalignment Risks:**  
  - Cannot distinguish harmful from benign NSFW (e.g., medical vs. explicit content)  
  - Prone to false negatives/positives in ambiguous or sarcastic language  

- **Failure Cases:**  
  - Vulnerable to adversarial manipulation (e.g., obfuscation, leetspeak)  
  - Less effective on out-of-domain or multilingual inputs unless retrained  

## Related Metrics

- **FastTextToxicity** – Similar classifier trained on other toxic labels from Jigsaw  
- **Perspective API (NSFW)** – External API for similar moderation use cases  
- **HateXplain Score** – Annotated dataset for hate and toxicity classification  
- **RealToxicityPrompts** – Benchmark for evaluating unsafe completions in LLMs  

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
- **Contact:** mryan0@stanford.edu"""
    
    # Resource usage statistics (in megabytes)
    gpu_mem: ClassVar[float] = 0.0  # in MB
    cpu_mem: ClassVar[float] = 1709.95703125  # in MB
    description: ClassVar[str] = "FastTextNSFW is a reference-free binary classification metric for evaluating the presence of not-safe-for-work (NSFW) content in generated text. It is based on a FastText linear classifier trained on the Jigsaw Toxic Comment Classification dataset, using the \"obscene\" label as a proxy for NSFW content. This metric was developed as part of the AI2 Dolma project to enable scalable filtering of large-scale web text data during corpus construction for language model pretraining. It outputs a signed scalar score indicating both the classification result and the model's confidence."

    def __init__(
        self,
        persistent: bool = True,
        data_dir: str = None,
        model_url: str = "https://dolma-artifacts.org/fasttext_models/jigsaw_fasttext_bigrams_20230515/jigsaw_fasttext_bigrams_nsfw_final.bin",
        negative_label: str = "nsfw",
        **kwargs
    ):
        super().__init__(
            name="FastTextNSFW",
            description="FastTextNSFW is a reference-free binary classification metric for evaluating the presence of not-safe-for-work (NSFW) content in generated text. It is based on a FastText linear classifier trained on the Jigsaw Toxic Comment Classification dataset, using the \"obscene\" label as a proxy for NSFW content. This metric was developed as part of the AI2 Dolma project to enable scalable filtering of large-scale web text data during corpus construction for language model pretraining. It outputs a signed scalar score indicating both the classification result and the model's confidence.",
            model_url=model_url,
            negative_label=negative_label,
            persistent=persistent,
            data_dir=data_dir,
            **kwargs
        ) 