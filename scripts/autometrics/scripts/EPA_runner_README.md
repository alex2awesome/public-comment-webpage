# Autometrics EPA Scaffolding Runner

`run_epa_rubric_pipeline.py` is the scriptized version of the EPA notebook workflow. It loads the EPA comment dataset, filters/downs a balanced sample, and drives Autometrics with the residual-driven scaffolding loop so you can generate metrics without opening a notebook.

## How it works

1. **Data prep:** load `notebooks/full_matched_comment_df__epa.csv`, filter by word count, downsample to balance the `c` label, and sample `--sample-size` rows.
2. **LLM setup:** instantiate `generator_llm` / `judge_llm` using `--generator-model` and `--judge-model`.
3. **Scaffolding metrics phase (default):**
   - **Initial rubrics:** RFC metric generators (DSPy rubric + LLM-as-a-judge) produce the first `--initial-core-metrics` features that anchor the regression.
   - **Residual mining & prompts:** after each regression fit, the loop ranks examples by signed residual. The top failures (stratified by label) are summarized—including target/pred/residual and a short text snippet—and, if `--explain-residuals` is enabled, the generator LLM writes a brief explanation of why the current metrics missed them. That short summary is appended to the next metric-generation prompt so the LLM “knows” which failure modes to address.
   - **Scaffolding cycle:** for every iteration, request `--scaffolding-metrics-per-iteration` new metric ideas. Regression then evaluates the union of old + new metrics, but only up to `--max-new-metrics-per-iteration` can be adopted (non-zero weight) in that round. The ensemble size never exceeds `--max-metric-ceiling`, so the loop cannot overgrow the support.
   - **Retiring legacy metrics:** by default, legacy metrics stay unless you set `--allow-retire-old-metrics`. Even then, a legacy metric is dropped only if a replacement survives `--retire-persistence` consecutive accepted iterations *and* the original metric’s importance decays below `--retire-importance-eps`. This makes retirement conservative and guards against one-off noise.
4. **Regression & export:** retrieve `--num-to-retrieve` bank metrics, fit ElasticNet, and emit the final ensemble. Each run now writes to a unique subfolder under `scripts/autometrics/scaffolding_runs/<dataset>/<target>/run_<id>` so repeated executions don’t overwrite prior results; inside each run you’ll find the usual `metrics/iteration_XX` directories, prompts, and residual logs.

Use `--verbose` to enable tqdm progress bars and per-iteration logs (metric adoption/rejection counts, retirements, ensemble size). Add `--log-prompts` if you want the full metric-generation prompt payloads.

## Sample command

```bash
python scripts/autometrics/scripts/run_epa_rubric_pipeline.py \
  --generation-mode scaffolding \
  --initial-core-metrics 6 \
  --scaffolding-metrics-per-iteration 4 \
  --max-new-metrics-per-iteration 2 \
  --max-metric-ceiling 8 \
  --allow-retire-old-metrics \
  --retire-persistence 2 \
  --retire-importance-eps 0.05 \
  --retire-persistence 2 \
  --retire-importance-eps 0.05 \
  --explain-residuals \
  --verbose \
  --data-path notebooks/full_matched_comment_df__epa.csv \
  --sample-size 300
```

This command seeds six core metrics, lets two new metrics be adopted per residual loop (capped at eight total), requires replacements to prove themselves for two accepted iterations with a 0.05 importance drop before retiring old metrics, and logs residual explanations plus tqdm status. To resume a previous run, pass `--resume-from-run run_ab12cd34` (see `scaffolding_runs/<dataset>/<target>/` for IDs); you can also point directly at a `checkpoint.json` with `--resume-checkpoint`.
