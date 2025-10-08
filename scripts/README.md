# Embedding Utilities

`scripts/embed.py` builds embeddings for regulations and collaborator profiles, then writes the artifacts consumed by the frontend fixtures.

## Usage

Run the script from the repository root so relative paths resolve correctly:

```bash
python scripts/embed.py \
  --csv-glob "data/federal_rulemaking_*.csv" \
  --profiles-dir data/user_profiles \
  --model hashing \
  --output-dir data/vector_store \
  --recommendations-fixture app/src/fixtures/recommendations.json \
  --profiles-fixture app/src/fixtures/userProfiles.json \
  --profile-summary-length 400 \
  --regulation-text-length 1200 \
  --top-k 15
```

Arguments mirror the command line interface:

- `--profile-summary-length`: maximum characters to keep in collaborator profile summaries (`<=0` keeps the full text).
- `--regulation-text-length`: maximum characters from each regulation considered during embedding (`<=0` keeps the full text).
- `--model`: embedding backend (`hashing`, `tfidf`, or any Sentence Transformers identifier).
- `--csv-glob`, `--profiles-dir`, `--output-dir`: source regulation snapshots, collaborator profile directory, and destination for vector artifacts.
- `--recommendations-fixture`, `--profiles-fixture`: output paths for frontend fixtures.
- `--top-k`: number of regulations to retain per collaborator when constructing recommendations.

All arguments have sensible defaults; pass only the flags you need to override.
