from autometrics.metrics.reference_based.ReferenceBasedMultiMetric import ReferenceBasedMultiMetric

from autometrics.metrics.unieval.utils import convert_to_json
# Removing evaluator import to use lazy loading
# from autometrics.metrics.unieval.evaluator import get_evaluator

import torch
from typing import ClassVar

class UniEvalDialogue(ReferenceBasedMultiMetric):
    """---
# Metric Card for UniEvalDialogue

UniEvalDialogue is a multi-dimensional evaluation metric designed specifically for **dialogue response generation**. It assesses responses across five key dimensions: **naturalness, coherence, engagingness, groundedness, and understandability**. The metric formulates evaluation as a **Boolean Question Answering (QA) task**, where a pre-trained language model predicts scores based on Yes/No answers to predefined evaluation questions.

## Metric Details

### Metric Description

UniEvalDialogue evaluates dialogue responses by **converting evaluation into a Boolean QA problem**. The model is prompted with questions tailored to specific evaluation dimensions, allowing it to assess **fluency, informativeness, factual grounding, and coherence** in open-ended dialogue tasks. Unlike traditional metrics that rely on reference responses, UniEvalDialogue is primarily **reference-free**, except when additional factual grounding is required (e.g., engagingness evaluation).

- **Metric Type:** Semantic Similarity, Reference-Free, Multi-Dimensional Evaluation  
- **Range:** [0,1] for all dimensions  
- **Higher is Better?:** Yes  
- **Reference-Based?:** Yes (groundedness and engagingness require factual context)
- **Input-Required?:** Yes  

### Formal Definition

Given a **generated response** $r$, a **dialogue history** $h$, and optionally a **factual grounding document** $f$, UniEvalDialogue evaluates five dimensions using a **pre-trained T5 model** in a Boolean QA format:

$$
\text{Score}_{dim} = \frac{P(\text{"Yes"} \mid r, h, f, q)}{P(\text{"Yes"} \mid r, h, f, q) + P(\text{"No"} \mid r, h, f, q)}
$$

where $q$ represents the evaluation question for a given dimension (e.g., "Is this response coherent given the dialogue history?"). The final **overall score** is computed as the **average** of the five dimension scores.

### Inputs and Outputs

- **Inputs:**  
  - Generated response  
  - Dialogue history  
  - Optional factual context (for groundedness and engagingness evaluation)  

- **Outputs:**  
  - Scores for **naturalness, coherence, engagingness, groundedness, and understandability** (range: [0,1])  
  - Overall score (default: **average of all dimension scores**)  

### Prompt Formulation for Evaluation Dimensions

UniEvalDialogue structures evaluation using **Boolean QA-style prompts** for each dimension:

1. **Naturalness** (Measures how human-like the response sounds)
   - **Prompt Template:**  
     ```
     question: Is this a natural response in the dialogue? </s> response: {system_output}
     ```
   - **Inputs Required:** Generated response  

2. **Coherence** (Checks logical consistency with prior turns in the dialogue)
   - **Prompt Template:**  
     ```
     question: Is this a coherent response given the dialogue history? </s> response: {system_output} </s> dialogue history: {source}
     ```
   - **Inputs Required:** Dialogue history, generated response  

3. **Engagingness** (Evaluates informativeness and conversational depth, requires factual grounding)
   - **Prompt Template:**  
     ```
     question: Is this an engaging and informative response according to the dialogue history and fact? </s> response: {system_output} </s> dialogue history: {source} </s> fact: {context}
     ```
   - **Inputs Required:** Dialogue history, generated response, factual grounding  

4. **Groundedness** (Assesses factual accuracy against a known knowledge base)
   - **Prompt Template:**  
     ```
     question: Is this response consistent with knowledge in the fact? </s> response: {system_output} </s> fact: {context}
     ```
   - **Inputs Required:** Generated response, factual grounding  

5. **Understandability** (Determines if the response is clear and interpretable)
   - **Prompt Template:**  
     ```
     question: Is this an understandable response in the dialogue? </s> response: {system_output}
     ```
   - **Inputs Required:** Generated response  

These prompts are tokenized and passed into the **UniEvalDialogue** model, which then predicts **Yes/No probabilities**, converting them into scores between **0 and 1**.

## Intended Use

### Domains and Tasks

- **Domain:** Dialogue Systems, Text Generation  
- **Tasks:** Dialogue Response Generation  

### Applicability and Limitations

- **Best Suited For:**  
  - Evaluating **open-domain** and **task-oriented** dialogue systems  
  - Systems where **coherence, fluency, and engagingness** are critical metrics  
  - Settings where **reference-free evaluation** is preferred  

- **Not Recommended For:**  
  - Evaluating **long-form creative writing** (e.g., storytelling, novel generation)  
  - Contexts where **fact verification requires external retrieval** (e.g., legal or medical dialogue)  

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**
  - [Hugging Face Model: MingZhong/unieval-dialog](https://huggingface.co/MingZhong/unieval-dialog)
  - [GitHub Repository: UniEval](https://github.com/maszhongming/UniEval)

### Computational Complexity

- **Efficiency:**  
  The model requires encoding the input and computing probability distributions over the **Yes/No outputs**. While more computationally expensive than simple token-overlap metrics, it remains efficient for evaluating dialogue at **scale**.

- **Scalability:**  
  UniEvalDialogue scales well with dataset size, but evaluation cost **grows linearly** with the number of responses due to separate model calls for each **evaluation dimension**.

## Known Limitations

- **Biases:**  
  - May inherit **biases** from the pre-trained T5 model.  
  - May **underestimate response diversity**, favoring safe, generic dialogue responses.  

- **Task Misalignment Risks:**  
  - While designed for dialogue, results may not **generalize well to highly structured** domains like legal or scientific chatbots.  

- **Failure Cases:**  
  - **Groundedness evaluation may fail** if factual sources are noisy or ambiguous.  
  - Model-generated scores may **not correlate well with human judgments** in highly subjective dialogue tasks.  

## Related Metrics

- **USR (Unsupervised and Reference-Free Dialogue Evaluation):** A multi-dimensional metric for dialogue quality.  
- **FED (Fine-grained Evaluation of Dialogue):** Uses a similar reference-free approach with multiple scoring dimensions.  
- **BLEU & METEOR:** Traditional n-gram overlap metrics (less relevant for dialogue).  

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
- **Contact:** mryan0@stanford.edu"""

    # Resource usage statistics (in megabytes
    gpu_mem: ClassVar[float] = 3140.60546875  # in MB
    cpu_mem: ClassVar[float] = 3169.94140625  # in MB
    description: ClassVar[str] = "UniEvalDialogue is a multi-dimensional evaluation metric designed specifically for **dialogue response generation**. It assesses responses across five key dimensions: **naturalness, coherence, engagingness, groundedness, and understandability**. The metric formulates evaluation as a **Boolean Question Answering (QA) task**, where a pre-trained language model predicts scores based on Yes/No answers to predefined evaluation questions."

    def __init__(self, persistent=True, **kwargs):
        name = "UniEvalDialogue"
        description = "UniEvalDialogue is a multi-dimensional evaluation metric designed specifically for **dialogue response generation**. It assesses responses across five key dimensions: **naturalness, coherence, engagingness, groundedness, and understandability**. The metric formulates evaluation as a **Boolean Question Answering (QA) task**, where a pre-trained language model predicts scores based on Yes/No answers to predefined evaluation questions."
        self.submetrics = ["naturalness", "coherence", "engagingness", "groundedness", "understandability"]

        self.task = 'dialogue'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.persistent = persistent
        self.evaluator = None
        
        super().__init__(name, description, ["UniEvalDialogue-" + submetric for submetric in self.submetrics], 
                         persistent=persistent, **kwargs)
        
        self.exclude_from_cache_key('persistent')

    def _load_model(self):
        """Load the UniEval model if not already loaded."""
        if self.evaluator is None:
            from autometrics.metrics.unieval.evaluator import get_evaluator
            self.evaluator = get_evaluator(self.task, device=self.device)
            
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
        Calculate UniEvalSum scores for the given input and output.
        """
        if self.evaluator is None:
            self._load_model()
            
        if references is None:
            references = []

        if len(references) > 1:
            references = [references[0]]

        # Prepare data for pre-trained v
        data = convert_to_json(output_list=[output], src_list=[input], context_list=references)

        # Get multi-dimensional evaluation scores
        eval_scores = self.evaluator.evaluate(data)
        
        result = self._parse_unieval(eval_scores[0])
        
        if not self.persistent:
            self._unload_model()
            
        return result
    
    def _calculate_batched_impl(self, inputs, outputs, references=None, **kwargs):
        """
        Calculate UniEvalSum scores for the given inputs and outputs in batches.
        """
        if self.evaluator is None:
            self._load_model()
            
        if references is None:
            references = [[] for _ in range(len(inputs))]

        # Prepare data for pre-trained evaluators
        data = convert_to_json(output_list=outputs, src_list=inputs, context_list=[reference[0] for reference in references])

        # Get multi-dimensional evaluation scores
        eval_scores = self.evaluator.evaluate(data)
        
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
    unieval = UniEvalDialogue()
    input = "hi , do you know much about the internet ? \n i know a lot about different sites and some website design , how about you ? \n\n"
    output = "i do too . did you know the 3 horizontal line menu on apps and websites is called the hamburger button ?"
    references = ["the 3 horizontal line menu on apps and websites is called a hamburger button .\n"]
    scores = unieval.calculate(input, output, references)
    print("UniEvalDialogue scores:", scores)
    # Test batch processing
    inputs = [input, "hey, what do you think about the weather today?"]
    outputs = [output, "it's sunny and warm outside."]
    references = [references, ["the weather is nice today."]]
    scores = unieval.calculate_batched(inputs, outputs, references)
    print("UniEvalDialogue batch scores:", scores)
