---
# Metric Card for {{ metric_name | default("Metric Name", true) }}

{{ metric_summary | default("A brief description of the metric and its purpose.", true) }}

## Metric Details

### Metric Description

{{ metric_description | default("Detailed explanation of the metric, including how it is calculated and what it measures.", true) }}

- **Metric Type:** {{ metric_type | default("[More Information Needed]", true) }}
- **Range:** {{ metric_range | default("[More Information Needed]", true) }}
- **Higher is Better?:** {{ higher_is_better | default("[More Information Needed]", true) }}
- **Reference-Based?:** {{ reference_based | default("[More Information Needed]", true) }}
- **Input-Required?:** {{ input_required | default("[More Information Needed]", true) }}

### Formal Definition

{{ metric_definition | default("Mathematical formula or detailed algorithmic definition.", true) }}

### Inputs and Outputs

- **Inputs:**  
  {{ metric_inputs | default("Description of required inputs (e.g., generated text, reference text, input prompt).", true) }}
  
- **Outputs:**  
  {{ metric_outputs | default("Description of the metric output (e.g., scalar score, distribution).", true) }}

## Intended Use

### Domains and Tasks

- **Domain:** {{ domain | default("[More Information Needed]", true) }}
- **Tasks:** {{ tasks | default("[More Information Needed]", true) }}

### Applicability and Limitations

- **Best Suited For:** {{ best_suited_for | default("[More Information Needed]", true) }}
- **Not Recommended For:** {{ not_recommended_for | default("[More Information Needed]", true) }}

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:** {{ libraries | default("[More Information Needed]", true) }}

### Computational Complexity

- **Efficiency:** {{ efficiency | default("[More Information Needed]", true) }}
- **Scalability:** {{ scalability | default("[More Information Needed]", true) }}

## Known Limitations

{{ known_limitations | default("[More Information Needed]", true) }}

- **Biases:** {{ biases | default("Potential biases inherent in the metric.", true) }}
- **Task Misalignment Risks:** {{ task_misalignment | default("[More Information Needed]", true) }}
- **Failure Cases:** {{ failure_cases | default("[More Information Needed]", true) }}

## Related Metrics

{{ related_metrics | default("[More Information Needed]", true) }}

## Further Reading

- **Papers:** {{ papers | default("[More Information Needed]", true) }}
- **Blogs/Tutorials:** {{ blogs | default("[More Information Needed]", true) }}

## Citation

{{ bibtex_citation | default("[More Information Needed]", true) }}

## Metric Card Authors

- **Authors:** {{ metric_authors | default("[More Information Needed]", true) }}  
- **Acknowledgment of AI Assistance:**  
  {{ ai_assistance | default("Portions of this metric card were drafted with assistance from generative AI. All content has been reviewed and curated by the author to ensure accuracy.", true) }}  
- **Contact:** {{ metric_contact | default("[More Information Needed]", true) }}