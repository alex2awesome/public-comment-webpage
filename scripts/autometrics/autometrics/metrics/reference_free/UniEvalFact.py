from autometrics.metrics.reference_free.ReferenceFreeMultiMetric import ReferenceFreeMultiMetric

from autometrics.metrics.unieval.utils import convert_to_json
# Removing evaluator import to use lazy loading
# from autometrics.metrics.unieval.evaluator import get_evaluator

import torch
from typing import ClassVar

class UniEvalFact(ReferenceFreeMultiMetric):
    """---
# Metric Card for UniEvalFact

UniEvalFact is a multi-dimensional evaluation metric designed specifically for **factual consistency detection** in generated text. It assesses whether claims made in a generated text align with facts provided in a **source document**. The metric formulates evaluation as a **Boolean Question Answering (QA) task**, where a pre-trained language model predicts scores based on Yes/No answers to predefined evaluation questions.

## Metric Details

### Metric Description

UniEvalFact evaluates the **factual accuracy** of generated claims by **converting evaluation into a Boolean QA problem**. The model is prompted with a **fact-checking question**, enabling it to determine whether a claim is **consistent** with a supporting document. Unlike traditional **n-gram-based metrics** (e.g., ROUGE), UniEvalFact directly **compares factual consistency** instead of lexical similarity.

- **Metric Type:** Factual Consistency, Reference-Free Evaluation  
- **Range:** [0,1]  
- **Higher is Better?:** Yes  
- **Reference-Based?:** No  
- **Input-Required?:** Yes (source document)  

### Formal Definition

Given a **generated claim** $c$ and a **source document** $d$, UniEvalFact evaluates factual consistency using a **pre-trained T5 model** in a Boolean QA format:

$$
\text{Score}_{\text{fact}} = \frac{P(\text{"Yes"} \mid c, d, q)}{P(\text{"Yes"} \mid c, d, q) + P(\text{"No"} \mid c, d, q)}
$$

where $q$ represents the evaluation question (e.g., "Is this claim consistent with the document?"). The final **factual consistency score** is computed as the **average of sentence-level consistency scores**.

### Inputs and Outputs

- **Inputs:**  
  - Generated claim  
  - Source document  

- **Outputs:**  
  - **Factual consistency score** (range: [0,1])  

### Prompt Formulation for Factual Consistency

UniEvalFact structures evaluation using a **Boolean QA-style prompt**:

1. **Factual Consistency** (Measures whether the generated claim is factually consistent with the reference document)
   - **Prompt Template:**  
     ```
     question: Is this claim consistent with the document? </s> claim: {system_output} </s> document: {source}
     ```
   - **Inputs Required:** Source document, generated claim  

This prompt is tokenized and passed into the **UniEvalFact** model, which then predicts **Yes/No probabilities**, converting them into scores between **0 and 1**.

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation, Fact Verification  
- **Tasks:** Fact Checking, Factual Consistency Evaluation  

### Applicability and Limitations

- **Best Suited For:**  
  - Evaluating **factual consistency** in summarization and knowledge-grounded text generation  
  - **Reference-free** fact verification tasks  

- **Not Recommended For:**  
  - Evaluating **coherence, fluency, or linguistic quality**  
  - Tasks where **multiple plausible answers exist**, as the metric strictly checks consistency  

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**
  - [Hugging Face Model: MingZhong/unieval-fact](https://huggingface.co/MingZhong/unieval-fact)
  - [GitHub Repository: UniEval](https://github.com/maszhongming/UniEval)

### Computational Complexity

- **Efficiency:**  
  The model requires encoding the input and computing probability distributions over the **Yes/No outputs**. While more computationally expensive than simple token-overlap metrics, it remains efficient for evaluating factual consistency at **scale**.

- **Scalability:**  
  UniEvalFact scales well with dataset size, but evaluation cost **grows linearly** with the number of claims due to separate model calls for each **sentence-level evaluation**.

## Known Limitations

- **Biases:**  
  - May inherit **biases** from the pre-trained T5 model.  
  - Performance may degrade if the **claim style** deviates significantly from the model's training data.  

- **Task Misalignment Risks:**  
  - While designed for **fact checking**, results may not generalize well to **highly domain-specific fact verification** (e.g., legal, medical claims).  

- **Failure Cases:**  
  - **Context ambiguity**: If the source document is **incomplete or vague**, the metric may provide misleading consistency scores.  
  - **Paraphrased claims**: The model may **overpenalize** valid claims that do not **exactly match** reference wording.  

## Related Metrics

- **FactCC:** A factual consistency metric designed for text summarization.  
- **FEQA:** An **extractive QA-based** metric that evaluates factual correctness.  
- **QAGS:** Uses **question generation** to check factual consistency.  

## Further Reading

- **Papers:**  
  - [Towards a Unified Multi-Dimensional Evaluator for Text Generation (Zhong et al., 2022)](https://aclanthology.org/2022.emnlp-main.131.pdf)

- **Blogs/Tutorials:**  
  - [UniEval GitHub Documentation](https://github.com/maszhongming/UniEval)

## Citation

```
@inproceedings{zhong-etal-2022-towards,
    title = "Towards a Unified Multi-Dimensional Evaluator for Text Generation",
    author = "Zhong, Ming  and
      Liu, Yang  and
      Yin, Da  and
      Mao, Yuning  and
      Jiao, Yizhu  and
      Liu, Pengfei  and
      Zhu, Chenguang  and
      Ji, Heng  and
      Han, Jiawei",
    editor = "Goldberg, Yoav  and
      Kozareva, Zornitsa  and
      Zhang, Yue",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.131/",
    doi = "10.18653/v1/2022.emnlp-main.131",
    pages = "2023--2038"
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from generative AI. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu  """

    # Resource usage statistics (in megabytes)
    gpu_mem: ClassVar[float] = 3140.60546875  # in MB
    cpu_mem: ClassVar[float] = 3167.10546875  # in MB
    description: ClassVar[str] = "UniEvalFact is a multi-dimensional evaluation metric designed specifically for **factual consistency detection** in generated text. It assesses whether claims made in a generated text align with facts provided in a **source document**. The metric formulates evaluation as a **Boolean Question Answering (QA) task**, where a pre-trained language model predicts scores based on Yes/No answers to predefined evaluation questions."

    def __init__(self, device: str = "cuda", persistent: bool = True, **kwargs):
        name = "UniEvalFact"
        description = "UniEvalFact is a multi-dimensional evaluation metric designed specifically for **factual consistency detection** in generated text. It assesses whether claims made in a generated text align with facts provided in a **source document**. The metric formulates evaluation as a **Boolean Question Answering (QA) task**, where a pre-trained language model predicts scores based on Yes/No answers to predefined evaluation questions."
        self.submetrics = ["consistency"]

        self.task = 'fact'
        self.device = torch.device(device)
        self.persistent = persistent
        self.evaluator = None
        
        super().__init__(name, description, ["UniEvalFact-" + submetric for submetric in self.submetrics], 
                          device=device, persistent=persistent, **kwargs)
        
        self.exclude_from_cache_key('persistent')

    def _load_model(self):
        """Load the UniEval model if not already loaded."""
        if self.evaluator is None:
            from autometrics.metrics.unieval.evaluator import get_evaluator
            try:
                self.evaluator = get_evaluator(self.task, device=self.device)
            except RuntimeError as e:
                # If CUDA asserts or device issues, retry on CPU
                if 'device-side assert' in str(e).lower() or 'cuda' in str(e).lower():
                    self.evaluator = get_evaluator(self.task, device='cpu')
                else:
                    raise
            
    def _unload_model(self):
        """Unload model to free resources."""
        if self.evaluator is not None:
            del self.evaluator
            torch.cuda.empty_cache()
            self.evaluator = None
    
    def _parse_unieval(self, result):
      results = [result[submetric] for submetric in self.submetrics]
      return results
    
    def _calculate_impl(self, input, output, references=None, **kwargs):
        """
        Calculate UniEvalFact scores for the given input and output.
        """
        if self.evaluator is None:
            self._load_model()
            
        # Prepare data for pre-trained evaluators
        data = convert_to_json(output_list=[output], src_list=[input])

        # Get multi-dimensional evaluation scores
        try:
            eval_scores = self.evaluator.evaluate(data)
        except RuntimeError as e:
            # Fallback to CPU execution if GPU path fails during evaluation
            if 'device-side assert' in str(e).lower() or 'expected device meta' in str(e).lower():
                # Recreate evaluator on CPU and retry once
                self._unload_model()
                from autometrics.metrics.unieval.evaluator import get_evaluator
                self.evaluator = get_evaluator(self.task, device='cpu')
                eval_scores = self.evaluator.evaluate(data)
            else:
                raise
        
        result = self._parse_unieval(eval_scores[0])
        
        if not self.persistent:
            self._unload_model()
            
        return result
    
    def _calculate_batched_impl(self, inputs, outputs, references=None, **kwargs):
        """
        Calculate UniEvalFact scores for the given inputs and outputs in batches.
        """
        if self.evaluator is None:
            self._load_model()

        # Prepare data for pre-trained evaluators
        data = convert_to_json(output_list=outputs, src_list=inputs)

        # Get multi-dimensional evaluation scores
        try:
            eval_scores = self.evaluator.evaluate(data)
        except RuntimeError as e:
            if 'device-side assert' in str(e).lower() or 'expected device meta' in str(e).lower():
                self._unload_model()
                from autometrics.metrics.unieval.evaluator import get_evaluator
                self.evaluator = get_evaluator(self.task, device='cpu')
                eval_scores = self.evaluator.evaluate(data)
            else:
                raise
        
        results = [self._parse_unieval(eval_score) for eval_score in eval_scores]

        # unzip the results
        results = list(zip(*results))

        # Convert to list of lists
        results = [list(result) for result in results]
        
        if not self.persistent:
            self._unload_model()

        return results

if __name__ == "__main__":
    # Example usage
    unieval = UniEvalFact()
    input = "Peter and Elizabeth took a taxi to attend the night party in the city. \
             While in the party, Elizabeth collapsed and was rushed to the hospital."
    output = "Tom was rushed to hospital."

    scores = unieval.calculate(input, output)
    print("UniEvalFact scores:", scores)

    # Test batch processing
    inputs = [
        "Peter and Elizabeth took a taxi to attend the night party in the city. \
             While in the party, Elizabeth collapsed and was rushed to the hospital.",
        "Pancakes are a type of flatbread made from flour, water, and milk."
    ]
    outputs = [
        "Tom was rushed to hospital.",
        "Pancakes can be made with flour, water, and milk."
    ]
    scores = unieval.calculate_batched(inputs, outputs)
    print("UniEvalFact batch scores:", scores)

