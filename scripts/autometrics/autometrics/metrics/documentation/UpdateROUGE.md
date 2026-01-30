---
# Metric Card for UpdateROUGE

UpdateROUGE is a variant of the ROUGE metric tailored for update-style generation tasks, where the goal is to revise or augment an existing source text. Rather than computing n-gram overlap over the entire generated and reference texts, UpdateROUGE isolates and evaluates only the newly added content—i.e., the parts of the text that differ from the shared source. This makes the metric especially suitable for tasks like information updating or document revision, where most of the source is preserved and only a subset is meaningfully changed.

## Metric Details

### Metric Description

UpdateROUGE evaluates the quality of additions made in a model’s output by comparing them to the additions in a human-authored reference, relative to a shared source input. This helps avoid artificially inflated scores in tasks where simply copying unchanged content from the source would otherwise result in high overlap. Additions are identified heuristically as complete sentences in the reference or output that do not appear in the source. ROUGE scores (ROUGE-1, ROUGE-2, and ROUGE-Lsum) are then computed using F1 over these addition segments.

- **Metric Type:** Surface-Level Similarity  
- **Range:** 0 to 1  
- **Higher is Better?:** Yes  
- **Reference-Based?:** Yes  
- **Input-Required?:** Yes

### Formal Definition

Let $S$ denote the source text, $R$ the reference text, and $\hat{Y}$ the model-generated output. Define $\Delta(R, S)$ as the set of sentence-level additions in $R$ not present in $S$, and similarly $\Delta(\hat{Y}, S)$ for the generation.

Then the UpdateROUGE score for ROUGE-$n$ is:

$$
\text{UpdateROUGE}_n = \text{ROUGE}_n(\Delta(\hat{Y}, S), \Delta(R, S)) \quad \text{for } n \in \{1, 2, \text{Lsum} \}
$$

The additions $\Delta(R, S)$ and $\Delta(\hat{Y}, S)$ are computed via heuristic sentence segmentation (e.g., via regex) followed by sentence-level exclusion against the source.

### Inputs and Outputs

- **Inputs:**  
  - Source text (input)  
  - Generated text (model output)  
  - Reference text (gold output)

- **Outputs:**  
  - `update_rouge1`: ROUGE-1 F1 score over additions  
  - `update_rouge2`: ROUGE-2 F1 score over additions  
  - `update_rougeLsum`: ROUGE-Lsum F1 score over additions  
  - `_target_diff_len`: number of characters in reference additions  
  - `_prediction_diff_len`: number of characters in prediction additions

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation  
- **Tasks:** Summarization, Document Revision, Update Generation

### Applicability and Limitations

- **Best Suited For:**  
  Scenarios in which the task involves making targeted updates to an existing document while retaining most of the original content. Examples include updating news articles, knowledge base entries, or summarizations that incorporate new facts.

- **Not Recommended For:**  
  Tasks where the entire output is expected to be novel or where there is no shared source (e.g., standalone generation, translation, or creative writing).

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**  
  - Custom implementation available in the [`autometrics`](https://github.com/XenonMolecule/autometrics) repository

### Computational Complexity

- **Efficiency:**  
  UpdateROUGE uses the `rouge_score` library internally and adds a preprocessing step for extracting additions. The computation is efficient and comparable to standard ROUGE.

- **Scalability:**  
  Scales well for typical document lengths. Performance may degrade if sentence segmentation or comparison becomes costly in very large documents.

## Known Limitations

- **Biases:**  
  The sentence-level extraction approach may miss partial or semantically rephrased updates, leading to false negatives or underestimation of quality.

- **Task Misalignment Risks:**  
  In tasks involving fine-grained edits within otherwise unchanged sentences, the metric may fail to capture meaningful changes.

- **Failure Cases:**  
  - If both reference and output share no additions (i.e., all text is from the source), the metric defaults to a high score.  
  - Models that make accurate sub-sentence updates without changing sentence boundaries may receive low scores despite correct updates.

## Related Metrics

- **ROUGE (Lin, 2004):** The foundational n-gram overlap metric that UpdateROUGE modifies to focus on additions.  
- **Entity Precision/Recall (from FRUIT):** Evaluate correctness and completeness of new entities in the update, serving as complementary signal to n-gram overlap.

## Further Reading

- **Papers:**  
  - Iv et al., 2022. “FRUIT: Faithfully Reflecting Updated Information in Text.” NAACL 2022.  
    [https://aclanthology.org/2022.naacl-main.269/](https://aclanthology.org/2022.naacl-main.269/)

- **Blogs/Tutorials:**  
  Needs more information

## Citation

```
@inproceedings{iv-etal-2022-fruit,
    title = "{FRUIT}: Faithfully Reflecting Updated Information in Text",
    author = "Iv, Robert  and
      Passos, Alexandre  and
      Singh, Sameer  and
      Chang, Ming-Wei",
    editor = "Carpuat, Marine  and
      de Marneffe, Marie-Catherine  and
      Meza Ruiz, Ivan Vladimir",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.269/",
    doi = "10.18653/v1/2022.naacl-main.269",
    pages = "3670--3686"
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from OpenAI's ChatGPT, based on user-provided inputs and relevant documentation. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu