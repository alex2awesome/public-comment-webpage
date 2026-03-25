# AI Detection — Experimental Versions

## v4
- **Mode**: pooled (no hierarchical κ at inference time; hierarchical flag on estimate just saves agency counts for future use)
- **public_submission**: uses v3_cleaned AI corpus (truncated rewrites, dropped long outliers), deduped human corpus
- **rule / notice / proposed_rule**: uses v3 AI corpus (raw), full human corpus (no dedup)
- **What changed vs v3**: more data spread across more agencies from additional .htm/.html files incorporated into the source CSVs

## v3
- **Mode**: pooled
- **AI corpus**: v3 generate with `--sample-per-agency 500 --sample-proportional --sample-agency-floor 20`
- **Results**: `data/v3/ai_usage_results.csv.gz`

## v2
- **Mode**: pooled
- **AI corpus**: earlier generate run, fewer agencies/samples
- **Results**: `data/v2/ai_usage_results.csv`
