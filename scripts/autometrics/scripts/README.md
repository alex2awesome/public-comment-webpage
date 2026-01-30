### Autometrics EPA Runner Overview

The scaffolding runner is still in-progress. Normal usage will be:

```bash
python scripts/autometrics/scripts/run_epa_rubric_pipeline.py \
  --initial-core-metrics 4 \
  --scaffolding-metrics-per-iteration 4 \
  --max-new-metrics-per-iteration 2 \
  --max-metric-ceiling 8 \
  --data-path notebooks/full_matched_comment_df__epa.csv \
  --sample-size 200 \
  --verbose
```
