from autometrics.metrics.reference_based.ReferenceBasedMetric import ReferenceBasedMetric
import re
from collections import defaultdict
from itertools import chain
from operator import itemgetter
import math
import difflib
from typing import List, Tuple, Union, Dict, Any, ClassVar

class CharCut(ReferenceBasedMetric):
    """---
# Metric Card for CharCut

CharCut is a character-based evaluation metric for machine translation that measures the similarity between candidate and reference translations using a human-targeted string difference algorithm. It identifies and scores "loose differences" through an iterative process of extracting long common substrings (LCSubstrs), designed to reduce noise and better align with human perception of meaningful edits. Unlike traditional edit distances, CharCut emphasizes user-aligned visualization and scoring of string differences, offering both evaluation and diagnostic utility.

## Metric Details

### Metric Description

CharCut compares generated text to a reference by iteratively identifying the longest common substrings (LCSubstrs) between the candidate and reference segments, under a threshold to avoid short, noisy matches. After extracting all LCSubstrs that meet a length-based threshold (typically ≥3 characters), the remaining non-matching substrings are categorized as "loose differences." CharCut also detects and handles shifts (reordered substrings) and assigns penalties for insertions, deletions, and shifts, yielding a normalized similarity score.

This metric is both interpretable and efficient, with results shown to correlate strongly with human judgments in WMT16 system- and segment-level evaluations. CharCut is especially useful when both automated scoring and human-readable highlighting of differences are needed.

- **Metric Type:** Surface-Level Similarity  
- **Range:** 0 to 1  
- **Higher is Better?:** No (lower scores indicate higher similarity)  
- **Reference-Based?:** Yes  
- **Input-Required?:** No

### Formal Definition

Let $C_0$ and $R_0$ be the candidate and reference segments, respectively. CharCut proceeds as follows:

1. **Segmentation Phase:**
   - Iteratively extract the longest common substrings (LCSubstrs) between $C_n$ and $R_n$, cutting them from both strings until no LCSubstr exceeds a predefined length threshold (typically 3 characters).
   - Add longest common prefix and suffix (if applicable) to the LCSubstr set.

2. **Shift Detection:**
   - Identify re-ordered substrings (shifts) among LCSubstrs not in the longest common subsequence (LCS) of the ordered match sets.

3. **Scoring Phase:**
   - Assign cost 1 to each inserted, deleted, or shifted character.
   - Compute normalized score using the following formula:

$$
\text{CharCut}(C_0, R_0) = \min \left(1, \frac{\#\text{deletions} + \#\text{insertions} + \#\text{shifts}}{2 \cdot |C_0|} \right)
$$

A score of 0 indicates perfect match; 1 indicates maximal divergence.

### Inputs and Outputs

- **Inputs:**  
  - Predictions (candidate translations): list of strings  
  - References (gold-standard translations): list of strings  

- **Outputs:**  
  - `charcut_mt`: a float score between 0 and 1 (lower is better)

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation  
- **Tasks:** Machine Translation

### Applicability and Limitations

- **Best Suited For:**  
  - Machine translation evaluation, especially when visual inspection of differences is also required.  
  - Languages with alphabetic or subword representations (e.g., Byte Pair Encoding).  

- **Not Recommended For:**  
  - Evaluation of highly diverse or creative tasks (e.g., storytelling, open-ended generation) where character-level overlap is not informative.  
  - Scenarios requiring semantic similarity or meaning preservation beyond surface-level matches.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**
  - [Hugging Face Evaluate: `charcut`](https://huggingface.co/spaces/evaluate-metric/CharCut)
  - [Original GitHub Repository (alardill/CharCut)](https://github.com/alardill/CharCut)
  - [Repackaged GitHub (BramVanroy/CharCut)](https://github.com/BramVanroy/CharCut)

### Computational Complexity

- **Efficiency:**  
  - Processes ~260 segment pairs/second on a 2.8 GHz processor using a Python implementation.  
  - More efficient than CharacTER and comparable in speed to chrF.

- **Scalability:**  
  - Suitable for medium to large-scale evaluations. May be slower than embedding-based methods for very large datasets.

## Known Limitations

- **Biases:**  
  - Sensitive to character-level surface variations (e.g., spelling differences, inflections), which may not reflect true semantic error.  

- **Task Misalignment Risks:**  
  - Inappropriate for evaluating tasks requiring semantic alignment or abstraction.  

- **Failure Cases:**  
  - May fail to identify semantic equivalence in paraphrased or reordered content when surface form diverges significantly.  
  - Shifts are detected without considering shift *distance*, so distant reordering incurs the same cost as nearby shifts.

## Related Metrics

- **chrF:** Another character-level metric, but based on precision and recall of character n-grams.  
- **CharacTER:** Related metric that penalizes edit distance on characters with additional weighting for shifts.  
- **TER:** Word-based edit distance metric.  
- **BLEU:** n-gram overlap metric with brevity penalty (much less sensitive to character-level variation).

## Further Reading

- **Papers:**  
  - [CharCut: Human-Targeted Character-Based MT Evaluation with Loose Differences (Lardilleux & Lepage, 2017)](https://aclanthology.org/2017.iwslt-1.20)  

- **Blogs/Tutorials:**  
  - [More Information Needed]

## Citation

```
@inproceedings{lardilleux-lepage-2017-charcut,  
  title = "{CHARCUT}: Human-Targeted Character-Based {MT} Evaluation with Loose Differences",  
  author = "Lardilleux, Adrien  and  
    Lepage, Yves",  
  booktitle = "Proceedings of the 14th International Conference on Spoken Language Translation",  
  month = dec # " 14-15",  
  year = "2017",  
  address = "Tokyo, Japan",  
  publisher = "International Workshop on Spoken Language Translation",  
  url = "https://aclanthology.org/2017.iwslt-1.20",  
  pages = "146--153",  
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from generative AI. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu"""
    
    # Resource usage statistics (in megabytes)
    gpu_mem: ClassVar[float] = 0.0  # in MB
    cpu_mem: ClassVar[float] = 727.6953125  # in MB
    description: ClassVar[str] = "CharCut is a character-based evaluation metric for machine translation that measures the similarity between candidate and reference translations using a human-targeted string difference algorithm. It identifies and scores 'loose differences' through an iterative process of extracting long common substrings (LCSubstrs), designed to reduce noise and better align with human perception of meaningful edits. Unlike traditional edit distances, CharCut emphasizes user-aligned visualization and scoring of string differences, offering both evaluation and diagnostic utility."

    def __init__(
        self,
        name: str = "CharCut",
        description: str = "CharCut is a character-based evaluation metric for machine translation that measures the similarity between candidate and reference translations using a human-targeted string difference algorithm. It identifies and scores 'loose differences' through an iterative process of extracting long common substrings (LCSubstrs), designed to reduce noise and better align with human perception of meaningful edits. Unlike traditional edit distances, CharCut emphasizes user-aligned visualization and scoring of string differences, offering both evaluation and diagnostic utility.",
        match_size: int = 3,
        alt_norm: bool = False,
        **kwargs
    ):
        super().__init__(
            name=name,
            description=description,
            match_size=match_size,
            alt_norm=alt_norm,
            **kwargs
        )
        self.match_size = match_size
        self.alt_norm = alt_norm

    # Utility functions for CharCut implementation
    def _iter_common_substrings(self, seq1, seq2, start_pos1, start_pos2, min_match_size, add_fix):
        """
        Iterates over common substrings between two sequences, looking at specific start positions.
        """
        n1 = len(seq1)
        n2 = len(seq2)
        # Parameters to the "recursive function". 3rd one is the offset.
        todo = [(start_pos1, start_pos2, 0)]
        while todo:
            pos1, pos2, offset = todo.pop()
            # Associate to each token the list of positions it appears at
            tokens1 = defaultdict(list)
            tokens2 = defaultdict(list)
            for i in pos1:
                if i + offset < n1:
                    tokens1[seq1[i + offset]].append(i)
            for i in pos2:
                if i + offset < n2:
                    tokens2[seq2[i + offset]].append(i)
            # Take intersection of the two token sets
            for token, ok_pos1 in tokens1.items():
                ok_pos2 = tokens2.get(token)
                if ok_pos2:
                    first_pos = ok_pos1[0]
                    substr = "".join(seq1[first_pos : first_pos + offset + 1])
                    if len(substr) >= min_match_size:
                        yield substr, ok_pos1, ok_pos2
                    elif add_fix and 0 in ok_pos1 and 0 in ok_pos2:  # common prefix
                        yield substr, [0], [0]
                    elif add_fix and n1 - 1 - offset in ok_pos1 and n2 - 1 - offset in ok_pos2:  # common suffix
                        yield substr, [n1 - 1 - offset], [n2 - 1 - offset]
                    todo.append((ok_pos1, ok_pos2, offset + 1))

    WORD_RE = re.compile(r"(\W)", re.UNICODE)
    
    def _word_split(self, seq):
        """
        Prepares a sequence of characters for the search of inter-words common substrings.
        """
        pos = 0
        for elt in self.WORD_RE.split(seq):
            if elt:
                yield pos, elt
                pos += len(elt)

    def _word_based_matches(self, seq1, seq2, min_match_size):
        """Iterator over all word-based common substrings between seq1 and seq2."""
        starts1, words1 = list(zip(*self._word_split(seq1))) if seq1 else ([], [])
        starts2, words2 = list(zip(*self._word_split(seq2))) if seq2 else ([], [])
        it = self._iter_common_substrings(
            words1, words2, list(range(len(words1))), list(range(len(words2))), min_match_size, True
        )
        for substr, pos1, pos2 in it:
            # Replace positions in words with positions in characters
            yield substr, [starts1[i] for i in pos1], [starts2[i] for i in pos2]

    def _start_pos(self, words):
        """Iterator over start positions of a list of words (cumulative lengths)."""
        pos = 0
        for elt in words:
            yield pos
            pos += len(elt)

    CHAR_RE = re.compile(r"(\w+)", re.UNICODE)

    def _char_split(self, seq, sep_sign):
        """
        Prepares a sequence of characters for the search of intra-words common substrings.
        """
        split = self.CHAR_RE.split(seq)
        # Fix in case seq contains only non-word characters
        tokens = ["", split[0], ""] if len(split) == 1 else split
        # "tokens" alternate actual words and runs of non-word characters
        starts = list(self._start_pos(tokens))
        for i in range(0, len(tokens) - 2, 2):
            # insert unique separator to prevent common substrings to span multiple words
            if i:
                yield None, i * sep_sign, False
            for j in range(i, i + 3):
                is_start_pos = j != i + 2
                for k, char in enumerate(tokens[j], starts[j]):
                    yield k, char, is_start_pos

    def _char_based_matches(self, seq1, seq2, min_match_size):
        """Iterator over all intra-word character-based common substrings between seq1 and seq2."""
        starts1, chars1, is_start1 = list(zip(*self._char_split(seq1, 1))) if seq1 else ([], [], [])
        starts2, chars2, is_start2 = list(zip(*self._char_split(seq2, -1))) if seq2 else ([], [], [])
        start_pos1 = [i for i, is_start in enumerate(is_start1) if is_start]
        start_pos2 = [i for i, is_start in enumerate(is_start2) if is_start]
        ics = self._iter_common_substrings(chars1, chars2, start_pos1, start_pos2, min_match_size, False)
        for substr, pos1, pos2 in ics:
            # Replace positions with those from the original sequences
            yield substr, [starts1[i] for i in pos1], [starts2[i] for i in pos2]

    def _order_key(self, match):
        """Sort key for common substrings: longest first, plus a few heuristic comparisons."""
        substr, pos1, pos2 = match
        return -len(substr), len(pos1) == len(pos2), len(pos1) + len(pos2), pos1

    def _clean_match_list(self, match_list, mask1, mask2):
        """
        Filter list of common substrings: remove those for which at least one character
        has already been covered (specified by the two masks).
        """
        for substr, pos1, pos2 in match_list:
            k = len(substr)
            clean_pos1 = [i for i in pos1 if all(mask1[i : i + k])]
            if clean_pos1:
                clean_pos2 = [i for i in pos2 if all(mask2[i : i + k])]
                if clean_pos2:
                    yield substr, clean_pos1, clean_pos2

    def _residual_diff(self, mask):
        """
        Factor successive 0's from a mask.
        Returns list of pairs: (start position, length)
        """
        buf = []
        for i, elt in enumerate(mask):
            if elt:
                buf.append(i)
            elif buf:
                yield buf[0], len(buf)
                buf = []
        if buf:
            yield buf[0], len(buf)

    def _greedy_matching(self, seq1, seq2, min_match_size):
        """
        Greedy search for common substrings between seq1 and seq2.
        """
        assert min_match_size > 0
        # Special case: if either sequence is empty, handle explicitly
        if not seq1 or not seq2:
            if seq1:  # seq2 is empty
                yield 0, -1, seq1
            elif seq2:  # seq1 is empty
                yield -1, 0, seq2
            return
            
        retained_matches = []
        # Indicate for each character if it is already covered by a match
        mask1 = [1] * len(seq1)
        mask2 = [1] * len(seq2)

        # List *all* common substrings and sort them (mainly) by length.
        match_it = chain(self._word_based_matches(seq1, seq2, min_match_size), 
                         self._char_based_matches(seq1, seq2, min_match_size))
        dedup = {match[0]: match for match in match_it}
        match_list = sorted(dedup.values(), key=self._order_key)

        # Consume all common substrings, longest first
        while match_list:
            substr, pos1, pos2 = match_list[0]
            i, j = pos1[0], pos2[0]
            retained_matches.append((i, j, substr))
            size = len(substr)
            # Update masks with newly retained characters
            mask1[i : i + size] = [0] * size
            mask2[j : j + size] = [0] * size
            # Eliminate common substrings for which at least one char is already covered
            match_list = list(self._clean_match_list(match_list, mask1, mask2))

        # Output matches
        for match in retained_matches:
            yield match
        # Output deletions
        for pos, size in self._residual_diff(mask1):
            yield pos, -1, seq1[pos : pos + size]
        # Output insertions
        for pos, size in self._residual_diff(mask2):
            yield -1, pos, seq2[pos : pos + size]

    def _find_regular_matches(self, ops):
        """
        Find the set of regular (non-shift) matches from the list of operations.
        """
        matches1 = sorted(m for m in ops if m[0] != -1 and m[1] != -1)
        matches2 = sorted(matches1, key=lambda match: match[1])
        
        # Return empty set if no matches
        if not matches1:
            return set()
            
        # Search for the longest common subsequence in characters
        # Expand "string" matches into "character" matches
        char_matches1 = [(m, i) for m in matches1 for i in range(len(m[2]))]
        char_matches2 = [(m, i) for m in matches2 for i in range(len(m[2]))]
        sm = difflib.SequenceMatcher(None, char_matches1, char_matches2, autojunk=False)
        return {m for a, _, size in sm.get_matching_blocks() for m, _ in char_matches1[a : a + size]}

    def _eval_shift_distance(self, shift, reg_matches):
        """
        Compute the distance in characters a match has been shifted over.
        """
        mid_matches = sorted(
            m for m in reg_matches if (m[0] < shift[0] and m[1] > shift[1]) or (m[0] > shift[0] and m[1] < shift[1])
        )
        if not mid_matches:
            return 0
        return (
            -(shift[0] - mid_matches[0][0])
            if mid_matches[0][0] < shift[0]
            else (mid_matches[-1][0] + len(mid_matches[-1][2]) - (shift[0] + len(shift[2])))
        )

    def _add_shift_distance(self, ops, reg_matches):
        """
        Decorate the list of operations with the shift distance.
        """
        for op in ops:
            alo, blo, slice = op
            if alo == -1 or blo == -1 or op in reg_matches:
                yield op + (0,)
            else:  # shift
                dist = self._eval_shift_distance(op, reg_matches)
                # Heuristic: the shorter a string,
                # the shorter the distance it is allowed to travel
                if math.exp(len(slice)) >= abs(dist):
                    yield op + (dist,)
                else:  # replace shift with deletion + insertion
                    yield -1, blo, slice, 0
                    yield alo, -1, slice, 0

    def _merge_adjacent_diffs_aux(self, diffs):
        prev_start = 0
        prev_substr = ""
        for start, substr in diffs:
            if start == prev_start + len(prev_substr):
                prev_substr += substr
            else:
                if prev_substr:
                    yield prev_start, prev_substr
                prev_start = start
                prev_substr = substr
        if prev_substr:
            yield prev_start, prev_substr

    def _merge_adjacent_diffs(self, ops):
        """Final cleaning: merge adjacent deletions or insertions into a single operation."""
        matches = [op for op in ops if op[0] != -1 and op[1] != -1]
        deletions = sorted((alo, substr) for alo, blo, substr, _ in ops if blo == -1)
        insertions = sorted((blo, substr) for alo, blo, substr, _ in ops if alo == -1)
        for op in matches:
            yield op
        for alo, substr in self._merge_adjacent_diffs_aux(deletions):
            yield alo, -1, substr, 0
        for blo, substr in self._merge_adjacent_diffs_aux(insertions):
            yield -1, blo, substr, 0

    def _compare_segments(self, cand, ref, min_match_size):
        """
        Main segment comparison function.
        """
        base_ops = list(self._greedy_matching(cand, ref, min_match_size))
        reg_matches = self._find_regular_matches(base_ops)
        return list(self._merge_adjacent_diffs(list(self._add_shift_distance(base_ops, reg_matches))))

    def _score_pair(self, cand, ref, ops, alt_norm):
        """Score a single candidate/reference pair."""
        ins_cost = sum(len(slice) for _, _, slice, _ in ops if _ == -1)
        del_cost = sum(len(slice) for _, blo, slice, _ in ops if blo == -1)
        # shifts are identical in cand and ref
        shift_cost = sum(len(slice) for alo, blo, slice, dist in ops if alo != -1 and blo != -1 and dist)
        cost = ins_cost + del_cost + shift_cost
        
        # Handle empty strings correctly
        if not cand and not ref:
            return 0.0  # Perfect match for empty strings
        elif not cand:
            return 1.0  # Maximum score if candidate is empty
        elif not ref:
            return 1.0  # Maximum score if reference is empty
            
        div = 2 * len(cand) if alt_norm else len(cand) + len(ref)
        # Prevent scores > 100%
        bounded_cost = min(cost, div)
        return bounded_cost / div if div else 0.0

    def _calculate_impl(self, input, output, references, **kwargs):
        """
        Calculate CharCut score for a single input/output pair with references.
        """
        if not references:
            raise ValueError("CharCut requires at least one reference")
        
        # For multiple references, compute score for each and take the minimum (best) score
        scores = []
        for ref in references:
            ops = self._compare_segments(output, ref, self.match_size)
            score = self._score_pair(output, ref, ops, self.alt_norm)
            scores.append(score)
        
        return min(scores)  # Lower is better for CharCut
        
    def _calculate_batched_impl(self, inputs, outputs, references=None, **kwargs):
        """
        Calculate CharCut scores for a batch of inputs/outputs with references.
        """
        results = []
        for i, (input, output, refs) in enumerate(zip(inputs, outputs, references)):
            results.append(self._calculate_impl(input, output, refs, **kwargs))
        return results 