---
# Metric Card for LENS_SALSA

LENS_SALSA is a reference-free metric designed to evaluate the overall quality of text simplification outputs. It leverages the SALSA (Simplification Analysis via Lexical and Structural Alignment) framework introduced by Heineman et al. (2023), which analyzes edits at the word level to assess whether a simplification succeeds or fails. The LENS_SALSA model uses these insights to produce a scalar simplification quality score based on input-output pairs, with no need for reference texts. This makes it particularly useful in settings where reference simplifications are unavailable or unreliable.

## Metric Details

### Metric Description

LENS_SALSA is a neural, reference-free metric for evaluating sentence-level simplification quality. It builds on the SALSA framework, which identifies and categorizes the types of edits performed when transforming a complex sentence into a simplified one. SALSA aligns input and output tokens using an alignment algorithm and labels each word-level edit with one of several tags—e.g., deletion, substitution, or addition—and further classifies the edit as a *success* or *failure* based on its impact on fluency, adequacy, and simplicity. These labels are derived from a manually annotated corpus.

The LENS_SALSA model is trained using these edit-level annotations. It learns to aggregate the local edit patterns into a global simplification quality score using a supervised regression objective. Crucially, this scoring process does not require reference simplifications at inference time, making LENS_SALSA a practical tool in real-world simplification pipelines.

- **Metric Type:** Reference-Free
- **Range:** Unbounded (empirically observed in [0, 100] scale)
- **Higher is Better?:** Yes
- **Reference-Based?:** No
- **Input-Required?:** Yes

### Formal Definition

The LENS_SALSA score is generated via a neural model trained on edit-labeled simplification data. The core of the system is the SALSA framework, which performs alignment and tagging of edits between a complex input sentence $x$ and a simplified candidate $\hat{x}$.

Let:
- $A = \text{Align}(x, \hat{x})$ be the alignment between input and output tokens,
- $E(x, \hat{x}, A)$ be the set of word-level edits extracted from the alignment,
- $T(e)$ be the success/failure tag for an edit $e$ (as determined by the SALSA labeling scheme),
- $f(E)$ be the feature vector summarizing the counts and types of edit tags in $E$.

Then, LENS_SALSA computes the final score using a regression model (MLP):

$$
\text{LENS SALSA}(x, \hat{x}) = \text{MLP}(f(E(x, \hat{x}, A)))
$$

The model is trained using human-annotated quality scores from simplification corpora.

### Inputs and Outputs

- **Inputs:**  
  - Input text (original complex sentence)  
  - Output text (candidate simplified sentence)  
  - (Optional) Reference text(s), used only during training or secondary analysis  
  
- **Outputs:**  
  - Scalar simplification score (float, typically between 0 and 100)  
  - (Optional) Word-level edit tags indicating success/failure for interpretability

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation
- **Tasks:** Text Simplification

### Applicability and Limitations

- **Best Suited For:**  
  Evaluating text simplification outputs where reference simplifications are unavailable, unreliable, or highly variable.  
  Particularly effective for sentence-level simplification tasks focused on fluency, adequacy, and simplicity.

- **Not Recommended For:**  
  - Tasks outside of simplification, such as summarization or paraphrasing  
  - Long-form or document-level generation  
  - Settings where simplification quality depends heavily on context beyond a single sentence

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**  
  - [Hugging Face: davidheineman/lens-salsa](https://huggingface.co/davidheineman/lens-salsa)  
  - `autometrics` (custom wrapper around LENS_SALSA model for reference-free evaluation)

### Computational Complexity

- **Efficiency:**  
  Relatively efficient inference via PyTorch-based model. Overhead comes from computing alignment-based features and scoring.

- **Scalability:**  
  Scales to batched input using the `calculate_batched` method. Memory usage depends on model size and batch configuration.

## Known Limitations

- **Biases:**  
  Needs more information.

- **Task Misalignment Risks:**  
  Designed specifically for simplification; using it for other tasks may result in misleading evaluations.

- **Failure Cases:**  
  - Very long input texts may cause padding errors in the model
  - For best results, texts should be sentence-level rather than long passages

## Related Metrics

- **SARI:** Reference-based simplification metric often used alongside LENS_SALSA.  
- **BERTScore (adapted to simplification):** Captures semantic similarity between input and output.  
- **LENS Framework:** Edit-level analysis from which LENS_SALSA is derived.

## Further Reading

- **Papers:**  
  - [Dancing Between Success and Failure: Edit-level Simplification Evaluation using SALSA](https://aclanthology.org/2023.emnlp-main.211/)  

- **Blogs/Tutorials:**  
  Needs more information.

## Citation

```
@inproceedings{heineman-etal-2023-dancing,
    title = "Dancing Between Success and Failure: Edit-level Simplification Evaluation using {SALSA}",
    author = "Heineman, David  and
      Dou, Yao  and
      Maddela, Mounica  and
      Xu, Wei",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.211/",
    doi = "10.18653/v1/2023.emnlp-main.211",
    pages = "3466--3495"
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from generative AI. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu