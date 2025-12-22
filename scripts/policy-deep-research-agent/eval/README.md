# Offline Evaluation

`offline_eval.py` scores cached trajectories deterministically using the same reward function that runs inside the environment. Provide a JSONL file with entries of the form:

```json
{"task_id": "policy_001", "memo": "...final memo text...", "bib": [{"paperId": "abc", "title": "..."}], "step_count": 9}
```

Usage:

```bash
python eval/offline_eval.py --trajectories ../data/cache/rollouts.jsonl --tasks ../data/tasks/policy_questions.jsonl
```

The script will enrich records with task questions (if missing), recompute reward components, and print a summary table so you can compare different checkpoints.
