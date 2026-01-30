import spacy
import torch
import warnings
from transformers import BertTokenizerFast, BertForQuestionAnswering
from collections import Counter
import re, string
from autometrics.metrics.reference_free.ReferenceFreeMultiMetric import ReferenceFreeMultiMetric
from typing import List, Tuple, Dict, ClassVar


def normalize_answer(s: str) -> str:
    """Lower text, remove punctuation, articles and extra whitespace."""
    def remove_articles(text: str) -> str:
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def remove_punc(text: str) -> str:
        return ''.join(ch for ch in text if ch not in set(string.punctuation))
    def white_space_fix(text: str) -> str:
        return ' '.join(text.split())
    def lower(text: str) -> str:
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute F1 score between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


class QA_Bert:
    """
    BERT-based QA model (SQuAD) to answer cloze questions.
    """
    def __init__(self, device=None):
        # Use the same model name for both tokenizer and model to ensure compatibility
        model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
        # Suppress tokenizer overflow warnings which are expected with truncation
        warnings.filterwarnings('ignore', message='.*overflowing tokens.*')
        warnings.filterwarnings('ignore', category=UserWarning, message='.*truncate.*longest_first.*')
        # Use fast tokenizer for offsets and robust decoding
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        
        # Set device (use GPU if available by default)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load on CPU first to avoid meta/cuda init issues, then move
        self.model = BertForQuestionAnswering.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        
        self.model.eval()
        self.max_seq_length = 512  # BERT's maximum sequence length
        
        # Debug: Print device information
        print(f"    🔧 SummaQA QA_Bert initialized on device: {self.device}")
        print(f"    🔧 Model device: {next(self.model.parameters()).device}")

    def predict(self, question: str, text: str) -> Tuple[str, float]:
        """
        Return the predicted answer string and its probability.
        Handles long sequences by truncating to fit BERT's max length.
        """
        def _predict_impl(device: torch.device) -> Tuple[str, float]:
            # Pair encode; prefer truncating the context (second sequence) only, with safe fallback
            try:
                encoded_dict = self.tokenizer(
                    question,
                    text,
                    add_special_tokens=True,
                    max_length=self.max_seq_length,
                    padding='max_length',
                    truncation='only_second',
                    return_tensors='pt',
                    return_token_type_ids=True,
                    return_offsets_mapping=True,
                )
            except Exception as e:
                msg = str(e).lower()
                if 'truncation error' in msg or 'too short to respect the provided max_length' in msg:
                    warnings.warn("[SummaQA] Falling back to longest_first truncation due to tokenizer truncation edge case.")
                    encoded_dict = self.tokenizer(
                        question,
                        text,
                        add_special_tokens=True,
                        max_length=self.max_seq_length,
                        padding='max_length',
                        truncation='longest_first',
                        return_tensors='pt',
                        return_token_type_ids=True,
                        return_offsets_mapping=True,
                    )
                else:
                    raise

            input_ids = encoded_dict['input_ids'].to(device, non_blocking=True)
            token_type_ids = encoded_dict['token_type_ids'].to(device, non_blocking=True)
            attention_mask = encoded_dict['attention_mask'].to(device, non_blocking=True)

            with torch.inference_mode():
                outputs = self.model(
                    input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                )
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits

                # Valid context positions: segment_id == 1 and attention_mask == 1
                # Ensure shapes align with logits (batch, seq_len)
                context_mask = (token_type_ids == 1) & (attention_mask == 1)
                if context_mask.sum().item() == 0:
                    return "", 0.0

                # Mask invalid positions with large negative before softmax
                very_neg = torch.finfo(start_logits.dtype).min if start_logits.dtype.is_floating_point else -1e9
                context_mask_float = context_mask.to(dtype=start_logits.dtype)
                start_logits = start_logits.masked_fill(~context_mask, very_neg)
                end_logits = end_logits.masked_fill(~context_mask, very_neg)

                start_probs = torch.softmax(start_logits, dim=-1)
                end_probs = torch.softmax(end_logits, dim=-1)

            start_index = int(torch.argmax(start_probs[0]).item())
            end_index = int(torch.argmax(end_probs[0]).item())
            if end_index < start_index:
                end_index = start_index

            # Decode span cleanly
            span_ids = input_ids[0, start_index:end_index + 1].tolist()
            answer = self.tokenizer.decode(span_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            prob = float(start_probs[0, start_index] * end_probs[0, end_index])
            return answer, prob

        try:
            return _predict_impl(self.device)
        except RuntimeError as e:
            if 'device-side assert triggered' in str(e) or 'indexSelectLargeIndex' in str(e):
                print("Error in QA_Bert.predict on GPU; retrying on CPU due to CUDA assert...")
                
                cpu_device = torch.device('cpu')
                self.model = self.model.to(cpu_device)
                return _predict_impl(cpu_device)


class QG_masked:
    """
    Cloze-style question generator based on spaCy NER.
    """
    def __init__(self, spacy_model: str = 'en_core_web_sm'):
        self.nlp = spacy.load(spacy_model)

    def get_questions(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Generate masked-sentence questions and their ground truth entity answers.
        """
        questions: List[str] = []
        answers: List[str] = []
        for sent in self.nlp(text).sents:
            for ent in sent.ents:
                start = ent.start_char - sent.start_char
                end = start + len(ent.text)
                masked = sent.text[:start] + 'MASKED' + sent.text[end:]
                questions.append(masked)
                answers.append(ent.text)
        return questions, answers


class QA_Metric:
    """
    Computes average answer probability and F1 over a set of cloze questions.
    """
    def __init__(self, model: QA_Bert = None, device=None):
        self.model = model or QA_Bert(device=device)

    def compute(self, questions: List[str], true_asws: List[str], evaluated_text: str) -> Dict[str, float]:
        if not questions:
            return {'avg_prob': 0.0, 'avg_fscore': 0.0}
        total_prob = 0.0
        total_f = 0.0
        for q, gt in zip(questions, true_asws):
            pred, prob = self.model.predict(q, evaluated_text)
            total_prob += prob
            total_f += f1_score(pred, gt)
        n = len(questions)
        return {'avg_prob': total_prob / n, 'avg_fscore': total_f / n}


class SummaQA(ReferenceFreeMultiMetric):
    """---
# Metric Card for SummaQA

SummaQA is a reference-free evaluation metric for summarization based on question answering (QA). It evaluates a generated summary by measuring its ability to correctly answer cloze-style questions derived from named entities in the source document. The metric leverages a BERT-based QA model to estimate both the probability of predicted answers and their F1 score against ground-truth answers masked from the original text.

## Metric Details

### Metric Description

SummaQA evaluates the content preservation of a generated summary without requiring a reference summary. It uses a two-step process:
1. **Question Generation (QG):** Named entities in the source text are masked to create cloze-style questions.
2. **Question Answering (QA):** A pretrained BERT QA model attempts to answer these questions using the generated summary.

The metric reports two sub-scores:
- **Average Answer Probability (avg_prob):** Likelihood of the predicted answer span under the QA model.
- **Average F1 Score (avg_fscore):** Overlap between predicted and ground-truth answers.

These scores reflect the factual consistency and informativeness of the summary with respect to the source.

- **Metric Type:** Faithfulness
- **Range:** 0 to 1 (for both submetrics)
- **Higher is Better?:** Yes
- **Reference-Based?:** No
- **Input-Required?:** Yes

### Formal Definition

Let $Q = \{(q_i, a_i)\}_{i=1}^N$ be a set of cloze questions $q_i$ and ground-truth answers $a_i$ extracted from the source document. Let $s$ be the generated summary. For each $(q_i, a_i)$:

- The QA model predicts answer $\hat{a}_i$ and confidence $p_i$ from $s$.
- Compute F1 score between $\hat{a}_i$ and $a_i$:
  
$$
\text{F1}_i = \frac{2 \cdot \text{precision}_i \cdot \text{recall}_i}{\text{precision}_i + \text{recall}_i}
$$

- The overall metric returns:

$$
\text{avg\_prob} = \frac{1}{N} \sum _{i=1}^{N} p_i, \quad \text{avg\_fscore} = \frac{1}{N} \sum _{i=1}^{N} \text{F1}_i
$$

### Inputs and Outputs

- **Inputs:**  
  - Input document (used for generating cloze questions)  
  - Generated summary (evaluated as a source of answers to cloze questions)

- **Outputs:**  
  - Two scalar values:
    - `avg_prob`: average answer probability under the QA model
    - `avg_fscore`: average F1 score between predicted and gold answers

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation
- **Tasks:** Summarization

### Applicability and Limitations

- **Best Suited For:**  
  - Evaluating factual consistency and information preservation in abstractive summarization.
  - Scenarios where no reference summaries are available.
  
- **Not Recommended For:**  
  - Non-extractive or highly abstractive generation without surface-level entity mentions.
  - Tasks involving creative generation or summarization of texts with sparse named entities.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**
  - [Official GitHub Repository](https://github.com/ThomasScialom/summa-qa)  
  - [SummaQA Paper on ACL Anthology](https://aclanthology.org/D19-1320/)  

### Computational Complexity

- **Efficiency:**  
  The metric requires:
  - spaCy NER for question generation (linear in text length),
  - Transformer-based QA inference per question (expensive for large $N$).
  Overall cost scales with number of entity-based questions.

- **Scalability:**  
  Suitable for small to medium batch evaluation. Preloading models and disabling persistence can reduce memory usage. Less scalable for long documents or large corpora without parallelization.

## Known Limitations

- **Biases:**  
  - Relies on named entity recognition (NER); may neglect important non-entity content.
  - Biased toward facts recoverable via NER-based questions, overlooking stylistic or inferential aspects.

- **Task Misalignment Risks:**  
  - Poor alignment with abstractive summarization models that paraphrase or omit named entities.
  - Vulnerable to hallucinated but answerable spans in generated summaries.

- **Failure Cases:**  
  - Low F1 despite semantically correct paraphrases.
  - High scores if hallucinated content coincides with correct QA patterns.

## Related Metrics

- **QAGS:** Also uses QA to evaluate factual consistency in summaries but supports sentence-level scoring and manual questions.  
- **FEQA:** Focuses on factual consistency via QA pairs and calculates cosine similarity in embedding space.  
- **BERTScore:** Measures semantic similarity but not fact retention.  

## Further Reading

- **Papers:**  
  - [Scialom et al. (2019) "Answers Unite! Unsupervised Metrics for Reinforced Summarization Models"](https://aclanthology.org/D19-1320/)  
  - [arXiv version](https://arxiv.org/abs/1909.01610)

- **Blogs/Tutorials:**  
  - [SummaQA GitHub Quickstart Guide](https://github.com/ThomasScialom/summa-qa#quickstart)

## Citation

```
@inproceedings{scialom-etal-2019-answers,
    title = "Answers Unite! Unsupervised Metrics for Reinforced Summarization Models",
    author = "Scialom, Thomas  and
      Lamprier, Sylvain  and
      Piwowarski, Benjamin  and
      Staiano, Jacopo",
    editor = "Inui, Kentaro  and
      Jiang, Jing  and
      Ng, Vincent  and
      Wan, Xiaojun",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D19-1320/",
    doi = "10.18653/v1/D19-1320",
    pages = "3246--3256",
    abstract = "Abstractive summarization approaches based on Reinforcement Learning (RL) have recently been proposed to overcome classical likelihood maximization. RL enables to consider complex, possibly non differentiable, metrics that globally assess the quality and relevance of the generated outputs. ROUGE, the most used summarization metric, is known to suffer from bias towards lexical similarity as well as from sub-optimal accounting for fluency and readability of the generated abstracts. We thus explore and propose alternative evaluation measures: the reported human-evaluation analysis shows that the proposed metrics, based on Question Answering, favorably compare to ROUGE {--} with the additional property of not requiring reference summaries. Training a RL-based model on these metrics leads to improvements (both in terms of human or automated metrics) over current approaches that use ROUGE as reward."
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from generative AI. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu"""

    # Resource usage statistics (in megabytes)
    gpu_mem: ClassVar[float] = 1283.37548828125  # in MB
    cpu_mem: ClassVar[float] = 1542.46484375  # in MB
    description: ClassVar[str] = 'SummaQA is a reference-free evaluation metric for summarization based on question answering (QA). It evaluates a generated summary by measuring its ability to correctly answer cloze-style questions derived from named entities in the source document. The metric leverages a BERT-based QA model to estimate both the probability of predicted answers and their F1 score against ground-truth answers masked from the original text.'

    def __init__(
        self,
        name: str = 'SummaQA',
        description: str = 'SummaQA is a reference-free evaluation metric for summarization based on question answering (QA). It evaluates a generated summary by measuring its ability to correctly answer cloze-style questions derived from named entities in the source document. The metric leverages a BERT-based QA model to estimate both the probability of predicted answers and their F1 score against ground-truth answers masked from the original text.',
        spacy_model: str = 'en_core_web_sm',
        persistent: bool = True,
        device=None,
        **kwargs
    ):
        super().__init__(name, description, submetric_names=['summaqa_avg_prob', 'summaqa_avg_fscore'], **kwargs)
        self.spacy_model = spacy_model
        self.persistent = persistent
        self.device = device
        self.qg: QG_masked = None
        self.qa: QA_Metric = None

        self.exclude_from_cache_key('persistent')
        self.exclude_from_cache_key('device')

    def _init_models(self):
        self.qg = QG_masked(self.spacy_model)
        self.qa = QA_Metric(device=self.device)

    def _unload_models(self):
        # Clear heavy models
        self.qg = None
        self.qa = None

    def _calculate_impl(self, input_text: str, output: str, references=None, **kwargs) -> Tuple[float, float]:
        # Lazy init
        if self.qg is None or self.qa is None:
            self._init_models()

        input_text = str(input_text) if input_text is not None else ""
        output = str(output) if output is not None else ""

        # Generate questions from original document
        questions, answers = self.qg.get_questions(input_text)
        # Compute QA scores on the summary
        scores = self.qa.compute(questions, answers, output)
        avg_prob = scores['avg_prob']
        avg_fscore = scores['avg_fscore']
        # Optionally unload
        if not self.persistent:
            self._unload_models()
        return (avg_prob, avg_fscore)

    def _calculate_batched_impl(
        self,
        inputs: List[str],
        outputs: List[str],
        references=None,
        **kwargs
    ) -> List[List[float]]:
        """
        Batch calculation for SummaQA: returns two lists [avg_probances, avg_fscores].
        """
        # Lazy init models once
        if self.qg is None or self.qa is None:
            self._init_models()
        probs: List[float] = []
        fscores: List[float] = []
        for inp, out in zip(inputs, outputs):
            inp = str(inp) if inp is not None else ""
            out = str(out) if out is not None else ""
            # Generate QA pairs on source document
            questions, answers = self.qg.get_questions(inp)
            # Score against summary
            scores = self.qa.compute(questions, answers, out)
            probs.append(scores['avg_prob'])
            fscores.append(scores['avg_fscore'])
        # Optionally unload models
        if not self.persistent:
            self._unload_models()
        # return per-submetric lists: [avg_prob_list, avg_fscore_list]
        return [probs, fscores] 