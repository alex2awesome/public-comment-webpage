---
# Metric Card for FKGL (Flesch-Kincaid Grade Level)

The Flesch-Kincaid Grade Level (FKGL) is a readability metric designed to evaluate the complexity of English-language texts. FKGL scores correspond to U.S. school grade levels, making it easy for educators, writers, and practitioners to understand the level of education required to comprehend a given text.

## Metric Details

### Metric Description

The FKGL metric calculates readability using the average sentence length (words per sentence) and the average syllables per word. It is widely used to assess the difficulty of documents in fields such as education, technical communication, and public policy. Lower FKGL scores indicate easier-to-read material, while higher scores signify increased complexity.

- **Metric Type:** Fluency
- **Range:** No theoretical upper bound; typical range is approximately -3.4 to above 20.
- **Higher is Better?:** No
- **Reference-Based?:** No
- **Input-Required?:** Yes

### Formal Definition

The FKGL formula is:

$$
\text{FKGL} = 0.39 \left( \frac{\text{total words}}{\text{total sentences}} \right) + 11.8 \left( \frac{\text{total syllables}}{\text{total words}} \right) - 15.59
$$

where:
- $\text{total words}$ is the number of words in the text,
- $\text{total sentences}$ is the number of sentences in the text,
- $\text{total syllables}$ is the number of syllables in the text.

The result corresponds to a U.S. grade level.

### Inputs and Outputs

- **Inputs:**  
  - Text (string) to analyze.  

- **Outputs:**  
  - A scalar score representing the U.S. school grade level required to understand the input text.

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation, Education, Technical Communication  
- **Tasks:** Readability assessment, document simplification, educational content evaluation  

### Applicability and Limitations

- **Best Suited For:**  
  - Analyzing educational materials, technical manuals, and legal documents to ensure they meet readability standards.  
  - Simplifying public-facing content such as insurance policies or government forms.  

- **Not Recommended For:**  
  - Tasks involving creative or highly contextual text, where readability depends on subjective factors.  

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**  
  - [TextStat](https://github.com/shivam5992/textstat): A Python library implementing FKGL and other readability formulas.  
  - [Microsoft Word Readability Statistics](https://support.microsoft.com/en-us/office/get-your-document-s-readability-and-level-statistics-85b4969e-e80a-4777-8dd3-f7fc3c8b3fd2): FKGL is included in Word's readability analysis tool.  

### Computational Complexity

- **Efficiency:**  
  FKGL is computationally efficient, with a complexity approximately linear in the number of words and sentences.  
- **Scalability:**  
  FKGL scales well for texts of varying lengths and can handle large datasets with appropriate preprocessing.

## Known Limitations

- **Biases:**  
  - FKGL does not account for the semantic content, context, or layout of the text, which may impact readability.  
  - Polysyllabic words disproportionately influence scores, potentially overestimating difficulty in texts with technical or specialized vocabulary.  

- **Task Misalignment Risks:**  
  - May fail to accurately represent the reading comprehension difficulty for non-native speakers or readers with diverse literacy levels.  

- **Failure Cases:**  
  - Poorly segmented texts (e.g., incorrect sentence splitting) can lead to inaccurate FKGL scores.  

## Related Metrics

- **Flesch Reading Ease (FRE):** A complementary metric providing a score from 0 to 100 for readability, inversely correlated with FKGL.  
- **Gunning Fog Index:** Another grade-level readability formula focusing on sentence length and complex words.  
- **Automated Readability Index (ARI):** Similar in purpose but uses character counts instead of syllables.

## Further Reading

- **Papers:**  
  - [Kincaid et al. (1975)](https://apps.dtic.mil/sti/pdfs/ADA006655.pdf): "Derivation of New Readability Formulas (Automated Readability Index, Fog Count, and Flesch Reading Ease Formula) for Navy Enlisted Personnel."

- **Blogs/Tutorials:**  
  - [TextStat Documentation](https://github.com/shivam5992/textstat)

## Citation

```
@article{kincaid1975derivation,
  title={Derivation of new readability formulas (automated readability index, fog count and flesch reading ease formula) for navy enlisted personnel},
  author={Kincaid, J Peter and Fishburne Jr, Robert P and Rogers, Richard L and Chissom, Brad S},
  year={1975},
  publisher={Institute for Simulation and Training, University of Central Florida}
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from OpenAI's ChatGPT, based on user-provided inputs and relevant documentation. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu