---
# Metric Card for DistinctNGrams

DistinctNGrams is a simple, reference-free metric that quantifies the lexical diversity of generated text by measuring the proportion of distinct n-grams within a sentence or a corpus. It is often used to evaluate the diversity of outputs in generative tasks such as dialogue and storytelling, particularly to counteract issues with generic or repetitive responses.

## Metric Details

### Metric Description

The DistinctNGrams metric calculates the number of distinct $n$-grams in the generated output, normalized by either the length of the sentence or the total number of words across a corpus. Two common variants are used:

- **Sentence-level Distinct-N**: Measures the proportion of unique $n$-grams within a single sentence.
- **Corpus-level Distinct-N**: Averages the sentence-level Distinct-N scores across multiple sentences in a dataset.

This metric was introduced to address the lack of diversity in sequence-to-sequence models, particularly in dialogue generation where models often default to bland, generic responses. Higher values indicate more diverse output.

- **Metric Type:** Diversity
- **Range:** $[0, 1]$
- **Higher is Better?:** Yes
- **Reference-Based?:** No
- **Input-Required?:** No

### Formal Definition

Let $T = [w_1, w_2, \dots, w_m]$ be a generated text sequence of length $m$.

- For a given $n$, the set of $n$-grams is defined as:

$$
\text{ngrams}(T, n) = \{(w_i, w_{i+1}, \dots, w_{i+n-1}) \mid 1 \leq i \leq m - n + 1\}
$$

- The sentence-level Distinct-N is then:

$$
\text{DistinctN}_{\text{sent}}(T, n) = \frac{|\text{ngrams}(T, n)|}{m}
$$

- The corpus-level Distinct-N for a set of sentences $\{T_1, \dots, T_k\}$ is:

$$
\text{DistinctN}_{\text{corpus}}(n) = \frac{1}{k} \sum_{j=1}^{k} \text{DistinctN}_{\text{sent}}(T_j, n)
$$

### Inputs and Outputs

- **Inputs:**  
  - Generated sentence(s) represented as lists of tokens.
  - An integer $n$ specifying the n-gram size.

- **Outputs:**  
  - A scalar score in $[0, 1]$ representing the proportion of unique $n$-grams.

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation, Dialogue Systems, Storytelling
- **Tasks:** Dialogue Generation, Storytelling, Creative Writing

### Applicability and Limitations

- **Best Suited For:**  
  Evaluating output diversity in generative tasks where repetitive responses are common, such as chatbot dialogue or story generation.

- **Not Recommended For:**  
  Tasks requiring semantic adequacy, factual correctness, or alignment with input prompts, such as summarization or translation.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**
  - [distinct-n (custom Python utility)](https://github.com/jiweil/Neural-Dialogue-Generation/blob/master/utils/eval_utils.py)  
  - Simple implementations available in many research repos; not typically part of major NLP libraries.

### Computational Complexity

- **Efficiency:**  
  Efficient to compute with complexity $O(m)$ for a sentence of length $m$, since $n$-gram extraction and set operations are linear in the number of tokens.

- **Scalability:**  
  Scales well to large corpora due to its low per-sentence cost and ease of parallelization.

## Known Limitations

- **Biases:**  
  Tends to favor longer sentences since they allow for more unique $n$-grams, even if the added diversity is trivial or incoherent.

- **Task Misalignment Risks:**  
  May misrepresent quality in tasks where repetition is appropriate or coherence is more important than diversity.

- **Failure Cases:**  
  Can be artificially inflated by nonsensical or disfluent outputs with rare words or tokens; insensitive to semantic meaning.

## Related Metrics

- **Self-BLEU:** Measures how similar samples from the same system are to each other—lower is better for diversity.  
- **Entropy (of n-grams):** Measures the unpredictability or richness of a model’s output vocabulary.  
- **Distinct-N (un-normalized):** Variant that counts total number of distinct $n$-grams across the corpus, rather than normalizing per sentence.

## Further Reading

- **Papers:**  
  - [A Diversity-Promoting Objective Function for Neural Conversation Models (Li et al., 2016)](https://aclanthology.org/N16-1014/)  
- **Blogs/Tutorials:**  
  - [More Information Needed]

## Citation

```
@inproceedings{li-etal-2016-diversity,
    title = "A Diversity-Promoting Objective Function for Neural Conversation Models",
    author = "Li, Jiwei  and Galley, Michel  and Brockett, Chris  and Gao, Jianfeng  and Dolan, Bill",
    editor = "Knight, Kevin  and Nenkova, Ani  and Rambow, Owen",
    booktitle = "Proceedings of the 2016 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2016",
    address = "San Diego, California",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/N16-1014/",
    doi = "10.18653/v1/N16-1014",
    pages = "110--119"
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from OpenAI's ChatGPT, based on user-provided inputs and relevant documentation. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu