# ------------------------------------------------------------------------------
# This implementation of the CIDEr metric is based on:
#   Vedantam, Zitnick, and Parikh (2015), "CIDEr: Consensus-based Image Description Evaluation"
# and incorporates elements from the reference implementation available in the coco-caption repository:
#   https://github.com/tylin/coco-caption/blob/master/pycocoevalcap
#
# Credit to Ramakrishna Vedantam <vrama91@vt.edu> and Tsung-Yi Lin <tl483@cornell.edu> for most of the original code.
# ------------------------------------------------------------------------------

import math
import numpy as np
import copy
from collections import defaultdict
from autometrics.metrics.reference_based.ReferenceBasedMetric import ReferenceBasedMetric
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from collections import Counter
from typing import ClassVar

# ----------------------------
# Internal Functions and Classes
# ----------------------------

def precook(s, n=4, out=False):
    """
    Convert a sentence string into a dictionary of n-gram counts.
    
    Args:
        s (str): The input sentence.
        n (int): Maximum n-gram length.
        out (bool): Unused flag for compatibility.
        
    Returns:
        dict: Mapping from n-gram tuple to count.
    """
    words = s.split()
    counts = defaultdict(int)
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1
    return counts

def cook_refs(refs, n=4):
    """
    Process a list of reference sentences.
    
    Args:
        refs (list of str): Reference sentences.
        n (int): Maximum n-gram length.
        
    Returns:
        list of dict: List of n-gram count dictionaries for each reference.
    """
    return [precook(ref, n) for ref in refs]

def cook_test(test, n=4):
    """
    Process a single candidate sentence.
    
    Args:
        test (str): Candidate sentence.
        n (int): Maximum n-gram length.
        
    Returns:
        dict: n-gram count dictionary for the candidate.
    """
    return precook(test, n, True)

class CiderScorer(object):
    """
    Class for computing the CIDEr score.
    """
    def __init__(self, test=None, refs=None, n=4, sigma=6.0):
        self.n = n
        self.sigma = sigma
        self.crefs = []
        self.ctest = []
        self.document_frequency = defaultdict(float)
        self.cook_append(test, refs)
        self.ref_len = None

    def cook_append(self, test, refs):
        """
        Append a candidate and its references to the internal lists.
        """
        if refs is not None:
            self.crefs.append(cook_refs(refs, self.n))
            if test is not None:
                self.ctest.append(cook_test(test, self.n))
            else:
                self.ctest.append(None)

    def __iadd__(self, other):
        """
        Overload the += operator to add a candidate and references tuple.
        """
        if isinstance(other, tuple):
            self.cook_append(other[0], other[1])
        else:
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)
        return self

    def compute_doc_freq(self):
        """
        Compute document frequencies for all n-grams in the references.
        """
        for refs in self.crefs:
            for ref in refs:
                for (ngram, count) in ref.items():
                    self.document_frequency[ngram] += 1

    def compute_cider(self):
        """
        Compute the CIDEr score for each candidate.
        
        Returns:
            list: CIDEr score for each candidate.
        """
        def counts2vec(cnts):
            """
            Map n-gram counts to a TF-IDF vector.
            
            Returns:
                vec: List of dicts (one per n-gram length).
                norm: List of vector norms.
                length: Total count for bigrams (used for length penalty).
            """
            vec = [defaultdict(float) for _ in range(self.n)]
            norm = [0.0 for _ in range(self.n)]
            length = 0
            for (ngram, term_freq) in cnts.items():
                # Calculate inverse document frequency (IDF)
                df = np.log(max(1.0, self.document_frequency[ngram]))
                index = len(ngram) - 1  # index 0 for unigrams, etc.
                vec[index][ngram] = float(term_freq) * (self.ref_len - df)
                norm[index] += vec[index][ngram] ** 2
                # For length penalty, using bigrams
                if len(ngram) == 2:
                    length += term_freq
            norm = [np.sqrt(x) for x in norm]
            return vec, norm, length

        def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
            """
            Compute cosine similarity with a Gaussian length penalty.
            """
            delta = float(length_hyp - length_ref)
            sim_scores = np.array([0.0 for _ in range(self.n)])
            for i in range(self.n):
                for (ngram, value) in vec_hyp[i].items():
                    ref_val = vec_ref[i].get(ngram, 0.0)
                    sim_scores[i] += min(value, ref_val) * ref_val
                if norm_hyp[i] != 0 and norm_ref[i] != 0:
                    sim_scores[i] /= (norm_hyp[i] * norm_ref[i])
                sim_scores[i] *= np.exp(- (delta ** 2) / (2 * self.sigma ** 2))
            return sim_scores

        self.ref_len = np.log(float(len(self.crefs)))
        scores = []
        for test, refs in zip(self.ctest, self.crefs):
            vec, norm, length = counts2vec(test)
            score = np.array([0.0 for _ in range(self.n)])
            for ref in refs:
                vec_ref, norm_ref, length_ref = counts2vec(ref)
                score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            score_avg = np.mean(score) / len(refs)
            score_avg *= 10.0  # Scaling factor for numerical consistency
            scores.append(score_avg)
        return scores

    def compute_score(self, option=None, verbose=0):
        """
        Compute the overall CIDEr score.
        
        Returns:
            (float, np.array): Mean CIDEr score and array of individual scores.
        """
        self.compute_doc_freq()
        score = self.compute_cider()
        return np.mean(np.array(score)), np.array(score)

class CiderImpl:
    """
    Internal implementation of CIDEr that mimics the original 'Cider' class.
    """
    def __init__(self, n=4, sigma=6.0):
        self._n = n
        self._sigma = sigma

    def compute_score(self, gts, res):
        """
        Compute the CIDEr score given ground truth and candidate dictionaries.
        
        Args:
            gts (dict): Keys are IDs, values are lists of reference sentences.
            res (dict): Keys are IDs, values are lists containing a single candidate sentence.
        
        Returns:
            (float, np.array): Mean CIDEr score and array of individual scores.
        """
        assert(gts.keys() == res.keys())
        imgIds = list(gts.keys())
        cider_scorer = CiderScorer(n=self._n, sigma=self._sigma)
        for imgId in imgIds:
            hypo = res[imgId]
            ref = gts[imgId]
            # Sanity checks
            assert(isinstance(hypo, list))
            assert(len(hypo) == 1)
            assert(isinstance(ref, list))
            assert(len(ref) > 0)
            cider_scorer += (hypo[0], ref)
        score, scores = cider_scorer.compute_score()
        return score, scores

    def method(self):
        return "CIDEr"

# ----------------------------
# Public CIDEr Metric Class
# ----------------------------

class CIDEr(ReferenceBasedMetric):
    """---
# Metric Card for CIDEr

CIDEr (Consensus-based Image Description Evaluation) measures the similarity between a candidate image caption and a set of human-generated reference captions. It leverages TF-IDF weighted n-gram matching to capture consensus in both content and linguistic style, making it especially useful for benchmarking image captioning systems.

## Metric Details

### Metric Description

CIDEr evaluates how well a candidate caption matches the consensus of human descriptions by comparing n-gram counts weighted by TF-IDF. In this process, each sentence is decomposed into n-grams (typically from unigrams up to four-grams), and each n-gram is weighted according to its term frequency in the sentence and its inverse document frequency across a dataset of images. The final score is the weighted average of cosine similarities computed for each n-gram length. An extended version, CIDEr-D, incorporates additional mechanisms (e.g., clipping and a Gaussian length penalty) to mitigate potential gaming of the metric.

- **Metric Type:** Surface-Level Similarity
- **Range:** Typically 0 to approximately 10 (higher values indicate better consensus)
- **Higher is Better?:** Yes
- **Reference-Based?:** Yes
- **Input-Required?:** No

### Formal Definition

For each n-gram $\omega_k$ in a reference sentence $s_{ij}$, the TF-IDF weighting is computed as:

$$
g_k(s_{ij}) = \frac{h_k(s_{ij})}{\sum_{\omega_l \in \Omega} h_l(s_{ij})} \log \left( \frac{|I|}{\sum_{p \in I} \min \left(1, \sum_q h_k(s_{pq}) \right)} \right)
$$

where:
- $h_k(s_{ij})$ is the count of n-gram $\omega_k$ in $s_{ij}$,
- $\Omega$ is the vocabulary of all n-grams,
- $|I|$ is the total number of images in the dataset, and
- the denominator in the log computes the document frequency of $\omega_k$.

The n-gram level CIDEr score is defined as:

$$
CIDEr_n(c_i, S_i) = \frac{1}{m} \sum_j \frac{g_n(c_i) \cdot g_n(s_{ij})}{\| g_n(c_i) \| \, \| g_n(s_{ij}) \|}
$$

where:
- $c_i$ is the candidate caption,
- $S_i = \{ s_{i1}, s_{i2}, \dots, s_{im} \}$ is the set of reference captions,
- $g_n(c_i)$ is the TF-IDF vector for n-grams of length $n$ from $c_i$, and
- $\| \cdot \|$ denotes the Euclidean norm.

The overall CIDEr score aggregates the n-gram scores:

$$
CIDEr(c_i, S_i) = \sum_{n=1}^N w_n \, CIDEr_n(c_i, S_i)
$$

with uniform weights $w_n = \frac{1}{N}$ (typically $N=4$).

### Inputs and Outputs

- **Inputs:**  
  - Candidate caption (a string)  
  - A set of reference captions (a list of strings)

- **Outputs:**  
  - A scalar CIDEr score representing the degree of consensus between the candidate caption and the references

## Intended Use

### Domains and Tasks

- **Domain:** Image Captioning, Multimodal Generation
- **Tasks:** Image Description Evaluation

### Applicability and Limitations

- **Best Suited For:**  
  Evaluating image captioning systems where a rich set of human reference captions (e.g., 50 per image) is available. CIDEr is particularly effective for assessing how well a candidate caption captures the consensus of human descriptions.

- **Not Recommended For:**  
  Open-ended generation tasks (e.g., creative storytelling or dialogue) where high lexical diversity is expected, or scenarios lacking sufficient reference data.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**  
  - [coco-caption repository](https://github.com/tylin/coco-caption) (reference implementation)  

### Computational Complexity

- **Efficiency:**  
  CIDEr is efficient at the sentence level; it primarily involves n-gram extraction, TF-IDF weighting, and cosine similarity calculations.

- **Scalability:**  
  While the computation scales linearly with the number of reference captions, performance benefits from additional references up to a saturation point.

## Known Limitations

- **Biases:**  
  - May favor candidate captions that closely mimic the majority of reference texts, potentially penalizing valid paraphrases or creative expressions.
  
- **Task Misalignment Risks:**  
  - Less effective for tasks with high variability in acceptable outputs, such as creative writing or dialogue generation.
  
- **Failure Cases:**  
  - When candidate captions use synonyms or alternative phrasings not reflected in the reference set, CIDEr may under-score these valid variations.

## Related Metrics

- **BLEU:** Focuses on n-gram precision without TF-IDF weighting.  
- **ROUGE:** Emphasizes n-gram recall.  
- **METEOR:** Incorporates stemming and synonym matching for enhanced semantic alignment.  
- **CIDEr-D:** A variant of CIDEr with added clipping and length penalty to reduce susceptibility to metric gaming.

## Further Reading

- **Papers:**  
  - Vedantam, R., Zitnick, C. L., & Parikh, D. (2015). *CIDEr: Consensus-based Image Description Evaluation*. [Original paper, arXiv:1411.5726](https://arxiv.org/abs/1411.5726)
  
- **Blogs/Tutorials:**  
  - Documentation and tutorials available on the coco-caption GitHub repository

## Citation

```
@INPROCEEDINGS{7299087,
  author={Vedantam, Ramakrishna and Zitnick, C. Lawrence and Parikh, Devi},
  booktitle={2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  title={CIDEr: Consensus-based image description evaluation}, 
  year={2015},
  volume={},
  number={},
  pages={4566-4575},
  keywords={Measurement;Protocols;Accuracy;Training;Testing;Silicon;Correlation},
  doi={10.1109/CVPR.2015.7299087}
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:** Portions of this metric card were drafted with assistance from OpenAI's ChatGPT (o3-mini-high), based on user-provided inputs and relevant documentation. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu"""

    # Resource usage statistics (in megabytes)
    gpu_mem: ClassVar[float] = 0.0  # in MB
    cpu_mem: ClassVar[float] = 726.43359375  # in MB
    description: ClassVar[str] = "CIDEr (Consensus-based Image Description Evaluation) measures the similarity between a candidate image caption and a set of human-generated reference captions. It leverages TF-IDF weighted n-gram matching to capture consensus in both content and linguistic style, making it especially useful for benchmarking image captioning systems."

    def __init__(self, n=4, sigma=6.0, name="CIDEr", description="CIDEr (Consensus-based Image Description Evaluation) measures the similarity between a candidate image caption and a set of human-generated reference captions. It leverages TF-IDF weighted n-gram matching to capture consensus in both content and linguistic style, making it especially useful for benchmarking image captioning systems.", **kwargs):
        super().__init__(name + "_n" + str(n) + "_sig" + str(sigma), description, n=n, sigma=sigma, **kwargs)
        self._n = n
        self._sigma = sigma

    def _calculate_impl(self, input, output, references=None, **kwargs):
        """
        Calculate the CIDEr score for a single candidate sentence.
        
        Args:
            input: (Unused) Placeholder for interface consistency.
            output (str): The candidate sentence.
            references (list of str): List of reference sentences.
        
        Returns:
            float: The computed CIDEr score.
        """
        if references is None:
            references = []
        # Wrap the candidate and references in dictionaries using a dummy key (0)
        gts = {0: references}
        res = {0: [output]}
        cider_impl = CiderImpl(n=self._n, sigma=self._sigma)
        score, _ = cider_impl.compute_score(gts, res)
        return score