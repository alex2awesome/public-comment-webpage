from __future__ import division
from collections import Counter
from typing import ClassVar

from nltk import word_tokenize

from autometrics.metrics.reference_based.ReferenceBasedMultiMetric import ReferenceBasedMultiMetric
from autometrics.metrics.reference_based.ReferenceBasedMetric import ReferenceBasedMetric

"""
Based on code by Mounica Maddela, Yao Dou, and David Heineman
https://github.com/Yao-Dou/LENS/blob/master/experiments/meta_evaluation/metrics/sari.py

Based on code by Wei Xu
https://github.com/cocoxu/simplification
"""


def is_subsequence(str1, str2):
	m = len(str1)
	n = len(str2)
	i, j = 0, 0
	while j < m and i < n:
		if str1[j] == str2[i]:
			j = j + 1
		i = i + 1
	return j == m


def SARIngram(sgrams, cgrams, rgramslist, numref, complex):
	rgramsall = [rgram for rgrams in rgramslist for rgram in rgrams]
	rgramcounter = Counter(rgramsall)

	sgramcounter = Counter(sgrams)
	sgramcounter_rep = Counter()
	for sgram, scount in sgramcounter.items():
		sgramcounter_rep[sgram] = scount * numref

	cgramcounter = Counter(cgrams)
	cgramcounter_rep = Counter()
	for cgram, ccount in cgramcounter.items():
		cgramcounter_rep[cgram] = ccount * numref

	# KEEP
	keepgramcounter_rep = sgramcounter_rep & cgramcounter_rep
	keepgramcountergood_rep = keepgramcounter_rep & rgramcounter
	keepgramcounterall_rep = sgramcounter_rep & rgramcounter

	keeptmpscore1 = 0
	keeptmpscore2 = 0
	for keepgram in keepgramcountergood_rep:
		keeptmpscore1 += keepgramcountergood_rep[keepgram] / keepgramcounter_rep[keepgram]
		keeptmpscore2 += keepgramcountergood_rep[keepgram] / keepgramcounterall_rep[keepgram]
	# print "KEEP", keepgram, keepscore, cgramcounter[keepgram], sgramcounter[keepgram], rgramcounter[keepgram]
	keepscore_precision = 0
	if len(keepgramcounter_rep) > 0:
		keepscore_precision = keeptmpscore1 / len(keepgramcounter_rep)

	# if keeptmpscore1 == 0 and len(keepgramcounter_rep) == 0:
	# 	keepscore_precision = 1.0

	keepscore_recall = 0
	if len(keepgramcounterall_rep) > 0:
		keepscore_recall = keeptmpscore2 / len(keepgramcounterall_rep)

	# if keeptmpscore2 == 0 and len(keepgramcounterall_rep) == 0:
	# 	keepscore_precision = 1.0

	keepscore = 0
	if keepscore_precision > 0 or keepscore_recall > 0:
		keepscore = 2 * keepscore_precision * keepscore_recall / (keepscore_precision + keepscore_recall)


	# DELETION
	delgramcounter_rep = sgramcounter_rep - cgramcounter_rep
	delgramcountergood_rep = delgramcounter_rep - rgramcounter
	delgramcounterall_rep = sgramcounter_rep - rgramcounter
	deltmpscore1 = 0
	deltmpscore2 = 0
	for delgram in delgramcountergood_rep:
		deltmpscore1 += delgramcountergood_rep[delgram] / delgramcounter_rep[delgram]
		deltmpscore2 += delgramcountergood_rep[delgram] / delgramcounterall_rep[delgram]
	delscore_precision = 0
	if len(delgramcounter_rep) > 0:
		delscore_precision = deltmpscore1 / len(delgramcounter_rep)

	delscore_recall = 0
	if len(delgramcounterall_rep) > 0:
		delscore_recall = deltmpscore1 / len(delgramcounterall_rep)

	# if deltmpscore1 == 0 and len(delgramcounter_rep) == 0:
	# 	delscore_precision = 1.0

	delscore = 0
	if delscore_precision > 0 or delscore_recall > 0:
		delscore = 2 * delscore_precision * delscore_recall / (delscore_precision + delscore_recall)

	# ADDITION
	addgramcounter = set(cgramcounter) - set(sgramcounter)
	addgramcountergood = set(addgramcounter) & set(rgramcounter)
	addgramcounterall = set(rgramcounter) - set(sgramcounter)

	sgrams_set = set()
	for gram in sgrams:
		sgrams_set.update(gram.split())

	addgramcountergood_new = set()
	for gram in addgramcountergood:
		if any([tok not in sgrams_set for tok in gram.split()]) or not is_subsequence(gram.split(), complex.split()):
			addgramcountergood_new.add(gram)
	addgramcountergood = addgramcountergood_new

	addtmpscore = 0
	for _ in addgramcountergood:
		addtmpscore += 1

	addscore_precision = 0
	addscore_recall = 0
	if len(addgramcounter) > 0:
		addscore_precision = addtmpscore / len(addgramcounter)

	# if addtmpscore == 0 and len(addgramcounter) == 0:
	# 	addscore_precision = 1.0

	if len(addgramcounterall) > 0:
		addscore_recall = addtmpscore / len(addgramcounterall)

	# if addtmpscore == 0 and len(addgramcounterall) == 0:
	# 	addscore_recall = 1.0

	addscore = 0
	if addscore_precision > 0 or addscore_recall > 0:
		addscore = 2 * addscore_precision * addscore_recall / (addscore_precision + addscore_recall)



	return (keepscore, (delscore_precision, delscore_recall, delscore), addscore)


def SARIsent(ssent, csent, rsents):
	numref = len(rsents)

	s1grams = word_tokenize(ssent.lower())
	c1grams = word_tokenize(csent.lower())
	s2grams = []
	c2grams = []
	s3grams = []
	c3grams = []
	s4grams = []
	c4grams = []

	r1gramslist = []
	r2gramslist = []
	r3gramslist = []
	r4gramslist = []

	for rsent in rsents:
		r1grams = word_tokenize(rsent.lower())
		r2grams = []
		r3grams = []
		r4grams = []
		r1gramslist.append(r1grams)
		for i in range(0, len(r1grams) - 1):
			if i < len(r1grams) - 1:
				r2gram = r1grams[i] + " " + r1grams[i + 1]
				r2grams.append(r2gram)
			if i < len(r1grams) - 2:
				r3gram = r1grams[i] + " " + r1grams[i + 1] + " " + r1grams[i + 2]
				r3grams.append(r3gram)
			if i < len(r1grams) - 3:
				r4gram = r1grams[i] + " " + r1grams[i + 1] + " " + r1grams[i + 2] + " " + r1grams[i + 3]
				r4grams.append(r4gram)
		r2gramslist.append(r2grams)
		r3gramslist.append(r3grams)
		r4gramslist.append(r4grams)

	for i in range(0, len(s1grams) - 1):
		if i < len(s1grams) - 1:
			s2gram = s1grams[i] + " " + s1grams[i + 1]
			s2grams.append(s2gram)
		if i < len(s1grams) - 2:
			s3gram = s1grams[i] + " " + s1grams[i + 1] + " " + s1grams[i + 2]
			s3grams.append(s3gram)
		if i < len(s1grams) - 3:
			s4gram = s1grams[i] + " " + s1grams[i + 1] + " " + s1grams[i + 2] + " " + s1grams[i + 3]
			s4grams.append(s4gram)

	for i in range(0, len(c1grams) - 1):
		if i < len(c1grams) - 1:
			c2gram = c1grams[i] + " " + c1grams[i + 1]
			c2grams.append(c2gram)
		if i < len(c1grams) - 2:
			c3gram = c1grams[i] + " " + c1grams[i + 1] + " " + c1grams[i + 2]
			c3grams.append(c3gram)
		if i < len(c1grams) - 3:
			c4gram = c1grams[i] + " " + c1grams[i + 1] + " " + c1grams[i + 2] + " " + c1grams[i + 3]
			c4grams.append(c4gram)


	(keep1score, del1score, add1score) = SARIngram(s1grams, c1grams, r1gramslist, numref, ssent)
	(keep2score, del2score, add2score) = SARIngram(s2grams, c2grams, r2gramslist, numref, ssent)
	(keep3score, del3score, add3score) = SARIngram(s3grams, c3grams, r3gramslist, numref, ssent)
	(keep4score, del4score, add4score) = SARIngram(s4grams, c4grams, r4gramslist, numref, ssent)

	del1p, del1r, del1f = del1score
	del2p, del2r, del2f = del2score
	del3p, del3r, del3f = del3score
	del4p, del4r, del4f = del4score

	avgkeepscore = sum([keep1score, keep2score, keep3score, keep4score]) / 4
	avgdelpscore = sum([del1p, del2p, del3p, del4p]) / 4
	avgdelrscore = sum([del1r, del2r, del3r, del4r]) / 4
	avgdelfscore = sum([del1f, del2f, del3f, del4f]) / 4
	avgaddscore = sum([add1score, add2score, add3score, add4score]) / 4
	finalpscore = (avgkeepscore + avgdelpscore + avgaddscore) / 3
	finalfscore = (avgkeepscore + avgdelfscore + avgaddscore) / 3
	return avgkeepscore, (avgdelpscore, avgdelrscore, avgdelfscore), avgaddscore, (finalpscore, finalfscore)


class SARI(ReferenceBasedMultiMetric):
	"""---
# Metric Card for SARI (System output Against References and against the Input sentence)

SARI is a metric designed specifically for evaluating text simplification systems. It measures the quality of a simplification by comparing it to both the original complex text and reference simplifications. The metric considers three operations: keeping important words, deleting unnecessary words, and adding new words.

## Metric Details

### Metric Description

SARI evaluates text simplification by considering three types of operations: additions, deletions, and retention of n-grams. It computes precision and recall for these operations by comparing the system output to both the input and the reference simplified texts. SARI is particularly suited for simplification tasks as it explicitly rewards edits that improve readability while maintaining semantic correctness.

- **Metric Type:** Surface-Level Similarity
- **Range:** 0 to 1
- **Higher is Better?:** Yes
- **Reference-Based?:** Yes
- **Input-Required?:** Yes

### Formal Definition

SARI is computed as the arithmetic mean of F-scores for the addition and retention operations, along with the precision of the deletion operation:

$$
SARI = \frac{1}{3}(F_{add} + F_{keep} + P_{del})
$$

Where:
- $F_{add}$: F-score for addition operations
- $F_{keep}$: F-score for keeping relevant text
- $P_{del}$: Precision for deletion operations  

Each F-score or precision is computed based on the comparison of n-grams in the input, system output, and references.

### Inputs and Outputs

- **Inputs:**  
- Source text (original, complex input text)  
- Candidate text (simplified text from the system)  
- Reference texts (simplified human-created texts)  

- **Outputs:**  
- Scalar SARI score (range: 0 to 1)

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation  
- **Tasks:**  
- Text Simplification  

### Applicability and Limitations

- **Best Suited For:**  
- Text simplification tasks where changes to the text, such as paraphrasing, deletions, or additions, are expected to enhance readability.  

- **Not Recommended For:**  
- Open-ended or creative text generation tasks where diversity and semantic similarity matter more than lexical transformation.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**  
- [SARI Implementation in LENS Repository](https://github.com/Yao-Dou/LENS/blob/master/experiments/meta_evaluation/metrics/sari.py)  

### Computational Complexity

- **Efficiency:**  
SARI is computationally efficient, with complexity similar to BLEU, as it involves n-gram extraction and comparison.  

- **Scalability:**  
SARI scales well across datasets with multiple references, leveraging n-gram matching for simplicity evaluation.

## Known Limitations

- **Biases:**  
- SARI may over-penalize outputs that do not align well with reference texts, particularly in cases where valid simplifications are not covered by references.  

- **Task Misalignment Risks:**  
- SARI is unsuitable for tasks that emphasize semantic similarity over structural changes, such as summarization or machine translation.  

- **Failure Cases:**  
- It can struggle with highly creative or diverse simplifications where multiple equally valid outputs are possible.

## Related Metrics

- **BLEU:** Measures surface similarity but does not compare outputs with the input text.  
- **FKBLEU:** Combines BLEU with the Flesch-Kincaid readability metric for simplification tasks.  
- **ROUGE:** Suitable for summarization but less relevant for simplification.  

## Further Reading

- **Papers:**  
- [Optimizing Statistical Machine Translation for Text Simplification (Xu et al., 2016)](https://github.com/cocoxu/simplification/)  

## Citation

```
@article{Xu-EtAl:2016:TACL,
  author = {Wei Xu and Courtney Napoles and Ellie Pavlick and Quanze Chen and Chris Callison-Burch},
  title = {Optimizing Statistical Machine Translation for Text Simplification},
  journal = {Transactions of the Association for Computational Linguistics},
  volume = {4},
  year = {2016},
  url = {https://cocoxu.github.io/publications/tacl2016-smt-simplification.pdf},
  pages = {401--415}
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
Portions of this metric card were drafted with assistance from OpenAI's ChatGPT, based on user-provided inputs and relevant documentation. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu
	"""
	
	# Resource usage statistics (in megabytes)
	gpu_mem: ClassVar[float] = 0.0  # in MB
	cpu_mem: ClassVar[float] = 727.91015625  # in MB
	description: ClassVar[str] = "SARI is a metric designed specifically for evaluating text simplification systems. It measures the quality of a simplification by comparing it to both the original complex text and reference simplifications. The metric considers three operations: keeping important words, deleting unnecessary words, and adding new words."
	
	# SARI is fast enough without caching
	DEFAULT_USE_CACHE = False
	
	def __init__(self, name="SARI", description="SARI is a metric designed specifically for evaluating text simplification systems. It measures the quality of a simplification by comparing it to both the original complex text and reference simplifications. The metric considers three operations: keeping important words, deleting unnecessary words, and adding new words.", **kwargs):
		super().__init__(name, description, submetric_names=["SARI_P", "SARI_F"], **kwargs)
		
	def _calculate_impl(self, input, output, references=None, **kwargs):
		"""
		Actual implementation of the SARI metric
		"""
		if references is None:
			references = []
		
		keep, dels, add, final = SARIsent(input, output, references)
		
		return final