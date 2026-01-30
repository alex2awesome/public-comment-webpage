# -*- coding: utf-8 -*-
# Natural Language Toolkit: GLEU Score
#
# Copyright (C) 2001-2017 NLTK Project
# Authors:
# Contributors: Mike Schuster, Michael Wayne Goodman, Liling Tan
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT

""" GLEU score implementation. """
from __future__ import division
from collections import Counter
from typing import ClassVar

import nltk
from nltk import word_tokenize

from autometrics.metrics.reference_based.ReferenceBasedMetric import ReferenceBasedMetric


def sentence_gleu(references, hypothesis, min_len=1, max_len=4):
	"""
	Calculates the sentence level GLEU (Google-BLEU) score described in

		Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi,
		Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey,
		Jeff Klingner, Apurva Shah, Melvin Johnson, Xiaobing Liu, Lukasz Kaiser,
		Stephan Gouws, Yoshikiyo Kato, Taku Kudo, Hideto Kazawa, Keith Stevens,
		George Kurian, Nishant Patil, Wei Wang, Cliff Young, Jason Smith,
		Jason Riesa, Alex Rudnick, Oriol Vinyals, Greg Corrado, Macduff Hughes,
		Jeffrey Dean. (2016) Google's Neural Machine Translation System:
		Bridging the Gap between Human and Machine Translation.
		eprint arXiv:1609.08144. https://arxiv.org/pdf/1609.08144v2.pdf
		Retrieved on 27 Oct 2016.

	From Wu et al. (2016):
		"The BLEU score has some undesirable properties when used for single
		 sentences, as it was designed to be a corpus measure. We therefore
		 use a slightly different score for our RL experiments which we call
		 the 'GLEU score'. For the GLEU score, we record all sub-sequences of
		 1, 2, 3 or 4 tokens in output and target sequence (n-grams). We then
		 compute a recall, which is the ratio of the number of matching n-grams
		 to the number of total n-grams in the target (ground truth) sequence,
		 and a precision, which is the ratio of the number of matching n-grams
		 to the number of total n-grams in the generated output sequence. Then
		 GLEU score is simply the minimum of recall and precision. This GLEU
		 score's range is always between 0 (no matches) and 1 (all match) and
		 it is symmetrical when switching output and target. According to
		 our experiments, GLEU score correlates quite well with the BLEU
		 metric on a corpus level but does not have its drawbacks for our per
		 sentence reward objective."

	Note: The initial implementation only allowed a single reference, but now
		  a list of references is required (which is consistent with
		  bleu_score.sentence_bleu()).

	The infamous "the the the ... " example

		>>> ref = 'the cat is on the mat'.split()
		>>> hyp = 'the the the the the the the'.split()
		>>> sentence_gleu([ref], hyp)  # doctest: +ELLIPSIS
		0.0909...

	An example to evaluate normal machine translation outputs

		>>> ref1 = str('It is a guide to action that ensures that the military '
		...            'will forever heed Party commands').split()
		>>> hyp1 = str('It is a guide to action which ensures that the military '
		...            'always obeys the commands of the party').split()
		>>> hyp2 = str('It is to insure the troops forever hearing the activity '
		...            'guidebook that party direct').split()
		>>> sentence_gleu([ref1], hyp1) # doctest: +ELLIPSIS
		0.4393...
		>>> sentence_gleu([ref1], hyp2) # doctest: +ELLIPSIS
		0.1206...

	:param references: a list of reference sentences
	:type references: list(list(str))
	:param hypothesis: a hypothesis sentence
	:type hypothesis: list(str)
	:param min_len: The minimum order of n-gram this function should extract.
	:type min_len: int
	:param max_len: The maximum order of n-gram this function should extract.
	:type max_len: int
	:return: the sentence level GLEU score.
	:rtype: float
	"""
	return corpus_gleu(
		[references],
		[hypothesis],
		min_len=min_len,
		max_len=max_len
	)


def corpus_gleu(list_of_references, hypotheses, min_len=1, max_len=4):
	"""
	Calculate a single corpus-level GLEU score (aka. system-level GLEU) for all
	the hypotheses and their respective references.

	Instead of averaging the sentence level GLEU scores (i.e. macro-average
	precision), Wu et al. (2016) sum up the matching tokens and the max of
	hypothesis and reference tokens for each sentence, then compute using the
	aggregate values.

	From Mike Schuster (via email):
		"For the corpus, we just add up the two statistics n_match and
		 n_all = max(n_all_output, n_all_target) for all sentences, then
		 calculate gleu_score = n_match / n_all, so it is not just a mean of
		 the sentence gleu scores (in our case, longer sentences count more,
		 which I think makes sense as they are more difficult to translate)."

	>>> hyp1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which',
	...         'ensures', 'that', 'the', 'military', 'always',
	...         'obeys', 'the', 'commands', 'of', 'the', 'party']
	>>> ref1a = ['It', 'is', 'a', 'guide', 'to', 'action', 'that',
	...          'ensures', 'that', 'the', 'military', 'will', 'forever',
	...          'heed', 'Party', 'commands']
	>>> ref1b = ['It', 'is', 'the', 'guiding', 'principle', 'which',
	...          'guarantees', 'the', 'military', 'forces', 'always',
	...          'being', 'under', 'the', 'command', 'of', 'the', 'Party']
	>>> ref1c = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the',
	...          'army', 'always', 'to', 'heed', 'the', 'directions',
	...          'of', 'the', 'party']

	>>> hyp2 = ['he', 'read', 'the', 'book', 'because', 'he', 'was',
	...         'interested', 'in', 'world', 'history']
	>>> ref2a = ['he', 'was', 'interested', 'in', 'world', 'history',
	...          'because', 'he', 'read', 'the', 'book']

	>>> list_of_references = [[ref1a, ref1b, ref1c], [ref2a]]
	>>> hypotheses = [hyp1, hyp2]
	>>> corpus_gleu(list_of_references, hypotheses) # doctest: +ELLIPSIS
	0.5673...

	The example below show that corpus_gleu() is different from averaging
	sentence_gleu() for hypotheses

	>>> score1 = sentence_gleu([ref1a], hyp1)
	>>> score2 = sentence_gleu([ref2a], hyp2)
	>>> (score1 + score2) / 2 # doctest: +ELLIPSIS
	0.6144...

	:param list_of_references: a list of reference sentences, w.r.t. hypotheses
	:type list_of_references: list(list(list(str)))
	:param hypotheses: a list of hypothesis sentences
	:type hypotheses: list(list(str))
	:param min_len: The minimum order of n-gram this function should extract.
	:type min_len: int
	:param max_len: The maximum order of n-gram this function should extract.
	:type max_len: int
	:return: The corpus-level GLEU score.
	:rtype: float
	"""
	# sanity check
	assert len(list_of_references) == len(
		hypotheses), "The number of hypotheses and their reference(s) should be the same"

	# sum matches and max-token-lengths over all sentences
	corpus_n_match = 0
	corpus_n_all = 0

	for references, hypothesis in zip(list_of_references, hypotheses):
		hyp_ngrams = Counter(nltk.everygrams(hypothesis, min_len, max_len))
		tpfp = sum(hyp_ngrams.values())  # True positives + False positives.

		hyp_counts = []
		for reference in references:
			ref_ngrams = Counter(nltk.everygrams(reference, min_len, max_len))
			tpfn = sum(ref_ngrams.values())  # True positives + False negatives.

			overlap_ngrams = ref_ngrams & hyp_ngrams
			tp = sum(overlap_ngrams.values())  # True positives.

			# While GLEU is defined as the minimum of precision and
			# recall, we can reduce the number of division operations by one by
			# instead finding the maximum of the denominators for the precision
			# and recall formulae, since the numerators are the same:
			#     precision = tp / tpfp
			#     recall = tp / tpfn
			#     gleu_score = min(precision, recall) == tp / max(tpfp, tpfn)
			n_all = max(tpfp, tpfn)

			if n_all > 0:
				hyp_counts.append((tp, n_all))

		# use the reference yielding the highest score
		if hyp_counts:
			n_match, n_all = max(hyp_counts, key=lambda hc: hc[0] / hc[1])
			corpus_n_match += n_match
			corpus_n_all += n_all

	# corner case: empty corpus or empty references---don't divide by zero!
	if corpus_n_all == 0:
		gleu_score = 0.0
	else:
		gleu_score = corpus_n_match / corpus_n_all

	return gleu_score


class GLEU(ReferenceBasedMetric):
	"""---
# Metric Card for GLEU

GLEU (Google-BLEU) is a metric introduced to address limitations of BLEU for sentence-level evaluation. It is designed to compute recall and precision over n-grams for hypotheses and references, taking the minimum of these two values as the final score. The metric is symmetrical and ranges from 0 (no match) to 1 (perfect match). It was initially proposed in Google's Neural Machine Translation (GNMT) system for reinforcement learning experiments.

## Metric Details

### Metric Description

GLEU computes sentence-level evaluation scores by comparing n-grams (of lengths 1 to 4) in the hypothesis and reference sentences. It calculates the precision (matching n-grams over total n-grams in the hypothesis) and recall (matching n-grams over total n-grams in the reference) and uses the minimum of the two values to determine the GLEU score. This approach avoids issues with BLEU's sentence-level evaluation while maintaining a high correlation with corpus-level BLEU scores.

- **Metric Type:** Surface-Level Similarity  
- **Range:** 0 to 1  
- **Higher is Better?:** Yes  
- **Reference-Based?:** Yes  
- **Input-Required?:** No  

### Formal Definition

The GLEU score for a hypothesis $h$ and a set of reference sentences $\{r_1, r_2, \ldots, r_n\}$ is defined as:

$$
GLEU(h, R) = \min \left( \text{precision}, \text{recall} \right)
$$

Where:
- **Precision:** $\frac{\text{Number of matching n-grams}}{\text{Total n-grams in hypothesis}}$  
- **Recall:** $\frac{\text{Number of matching n-grams}}{\text{Total n-grams in reference}}$  

The final score is symmetrical with respect to hypothesis and reference, making it robust for single-sentence evaluation.

### Inputs and Outputs

- **Inputs:**  
  - Hypothesis sentence (generated text)  
  - Reference sentence(s) (gold-standard text)  

- **Outputs:**  
  - A scalar score in the range [0, 1], where 1 indicates a perfect match.

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation  
- **Tasks:** Machine Translation, Summarization  

### Applicability and Limitations

- **Best Suited For:**  
  Sentence-level evaluation in structured tasks where precision and recall over n-grams are meaningful indicators of quality, such as translation.  

- **Not Recommended For:**  
  Creative or open-ended text generation tasks where semantic similarity or diversity is more relevant than surface-level n-gram overlap.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**  
  - [NLTK GLEU Implementation](https://github.com/nltk/nltk/blob/develop/nltk/translate/gleu_score.py)  

### Computational Complexity

- **Efficiency:**  
  Efficient for sentence-level evaluation, as it requires simple n-gram matching and aggregation.

- **Scalability:**  
  Scales well for batch evaluations but may be computationally expensive for larger corpora due to repeated n-gram matching.

## Known Limitations

- **Biases:**  
  Penalizes valid paraphrases or semantically equivalent outputs that differ in n-gram overlap.  

- **Task Misalignment Risks:**  
  Designed for tasks with a single correct output structure; performs poorly for evaluating diverse or creative responses.  

- **Failure Cases:**  
  - GLEU may not adequately evaluate cases where semantic preservation is more important than lexical overlap.

## Related Metrics

- **BLEU:** GLEU is inspired by BLEU but designed for sentence-level evaluation.  
- **METEOR:** Aims to improve on BLEU by incorporating synonym matching.  
- **BERTScore:** Evaluates semantic similarity using contextual embeddings.  

## Further Reading

- **Papers:**  
  - [Google's Neural Machine Translation System (Wu et al., 2016)](https://arxiv.org/pdf/1609.08144v2.pdf)  

- **Blogs/Tutorials:**  
  - Needs more information  

## Citation

```
@misc{wu2016googlesneuralmachinetranslation,
      title={Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation}, 
      author={Yonghui Wu and Mike Schuster and Zhifeng Chen and Quoc V. Le and Mohammad Norouzi and Wolfgang Macherey and Maxim Krikun and Yuan Cao and Qin Gao and Klaus Macherey and Jeff Klingner and Apurva Shah and Melvin Johnson and Xiaobing Liu and Łukasz Kaiser and Stephan Gouws and Yoshikiyo Kato and Taku Kudo and Hideto Kazawa and Keith Stevens and George Kurian and Nishant Patil and Wei Wang and Cliff Young and Jason Smith and Jason Riesa and Alex Rudnick and Oriol Vinyals and Greg Corrado and Macduff Hughes and Jeffrey Dean},
      year={2016},
      eprint={1609.08144},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/1609.08144}, 
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from OpenAI's ChatGPT, based on user-provided inputs and relevant documentation. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu"""

	# Resource usage statistics (in megabytes)
	gpu_mem: ClassVar[float] = 0.0  # in MB
	cpu_mem: ClassVar[float] = 727.953125  # in MB
	description: ClassVar[str] = "GLEU (Google-BLEU) is a metric introduced to address limitations of BLEU for sentence-level evaluation. It is designed to compute recall and precision over n-grams for hypotheses and references, taking the minimum of these two values as the final score. The metric is symmetrical and ranges from 0 (no match) to 1 (perfect match). It was initially proposed in Google's Neural Machine Translation (GNMT) system for reinforcement learning experiments."

	# GLEU is fast enough without caching
	DEFAULT_USE_CACHE = False

	def __init__(self, name="GLEU", description="GLEU (Google-BLEU) is a metric introduced to address limitations of BLEU for sentence-level evaluation. It is designed to compute recall and precision over n-grams for hypotheses and references, taking the minimum of these two values as the final score. The metric is symmetrical and ranges from 0 (no match) to 1 (perfect match). It was initially proposed in Google's Neural Machine Translation (GNMT) system for reinforcement learning experiments.", **kwargs):
		super().__init__(name, description, **kwargs)
		
	def _calculate_impl(self, input, output, references=None, **kwargs):
		"""
		Actual implementation of the GLEU metric
		"""
		if references is None:
			references = [] 
			
		refs = [word_tokenize(ref.lower()) for ref in references]
		hyp = word_tokenize(output.lower())
		return sentence_gleu(refs, hyp)


