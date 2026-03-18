# Directive: Adaptive Comment-Matching Pipeline

## Overview

Build a single Python script `match_pipeline.py` that, for each chunked comment file, automatically selects the best strategy to match public comments to government responses. The script orchestrates three models (base mpnet, a pre-trained SBERT model, and optionally a per-file fine-tuned model) and an LLM labeler, choosing the cheapest strategy that achieves acceptable accuracy.

## Inputs

- `proc_response_df`: The processed government response DataFrame (loaded from `2026-02-10__comment-response-cache.csv` via the existing parsing logic).
- `chunk_files`: List of `*_chunked.csv` paths found by globbing `../data/bulk_downloads/**/*_chunked.csv`.
- Pre-trained SBERT model at path `sbert-model-training/final`.
- Base model: `all-MiniLM-L6-v2` (the base scorer used throughout the notebook).
- OpenAI API access for `gpt-5.2` via the existing `prompt_utils.process_batch()` async function.

## Output

For each chunk file, produce:
1. A CSV of all comment-response pairs with a `final_label` column (`yes`/`no`) and a `strategy` column describing how the label was determined.
2. A per-file JSON log entry appended to `match_pipeline_log.jsonl`.

---

## Architecture

```
match_pipeline.py          # main entry point, async
├── imports & config
├── load_response_df()     # Cell 18+19 logic
├── score_file()           # Cell 23 logic (called twice per file, once per model)
├── sample_for_labeling()  # Cell 28 logic
├── label_pairs()          # Cell 30 logic (LLM call)
├── find_optimal_threshold()  # Cell 55 logic
├── build_triplets()       # Cell 41 logic
├── finetune_model()       # Cell 31 logic
├── run_pipeline()         # main orchestrator loop
└── prompt_utils           # existing module, import as-is
```

---

## Detailed Implementation Steps

### 1. Imports and Config

```python
from __future__ import annotations
import asyncio, glob, json, logging, random, os, re, sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

import prompt_utils
from utils import robust_json_load

# Config
BASE_MODEL_NAME = 'all-MiniLM-L6-v2'         # base scorer used in notebook
SBERT_MODEL_PATH = 'sbert-model-training/final'
FINETUNE_BASE = 'microsoft/mpnet-base'
LLM_MODEL = 'gpt-5.2'
F1_THRESHOLD = 0.7            # if best model F1 >= this, use threshold labeling
MAX_LLM_SAMPLE = 1000         # max rows to send to LLM
SCORE_SAMPLE_MIN = 0.1        # for sample_for_labeling
SCORE_SAMPLE_MAX = 0.9
LOG_FILE = 'match_pipeline_log.jsonl'
OUTPUT_DIR = Path('matched_outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)
```

### 2. Loading Response Data

Copy directly from notebook cells 18 and 19. The function should:
- Find the response cache CSV
- Parse `summarized_response` via `robust_json_load`
- Explode and flatten into `proc_response_df` with columns including `Agency ID`, `Docket ID`, `content_of_comment`, `summarized_content_of_comment`

```python
def load_response_df() -> pd.DataFrame:
    """Cell 18+19 logic exactly."""
    PROJECT_ROOT = Path.cwd().resolve()
    RESPONSE_CACHE_CANDIDATES = [
        Path('2026-02-10__comment-response-cache.csv'),
        PROJECT_ROOT / '2026-02-10__comment-response-cache.csv',
        PROJECT_ROOT / 'notebooks/2026-02-10__comment-response-cache.csv',
    ]
    for candidate in RESPONSE_CACHE_CANDIDATES:
        if candidate.exists():
            RESPONSE_CACHE_PATH = candidate
            break
    else:
        RESPONSE_CACHE_PATH = RESPONSE_CACHE_CANDIDATES[0]

    orig_response_df = (
        pd.read_csv(RESPONSE_CACHE_PATH, index_col=0)
            .assign(parsed_response=lambda df: df['summarized_response'].apply(robust_json_load))
            .drop(columns='summarized_response')
    )
    proc_response_df = (
        orig_response_df
            .assign(parsed_response=lambda df: df['parsed_response'].apply(lambda x: x if isinstance(x, list) else [x]))
            .loc[lambda df: df['parsed_response'].str.len() > 0]
            .explode('parsed_response')
            .reset_index(drop=True)
            .assign(parsed_response=lambda df: df['parsed_response'].apply(lambda x: x[0] if isinstance(x, list) else x))
            .pipe(lambda df: pd.concat([df[['Agency ID', 'Docket ID']], pd.DataFrame(df['parsed_response'].tolist())], axis=1))
            .drop(columns=['error', 'detail', 'commenter_identifiers_Text'], errors='ignore')
    )
    return proc_response_df
```

### 3. Embedding Model Cache

```python
_embedding_models = {}
def get_embedding_model(model_name: str) -> SentenceTransformer:
    if model_name not in _embedding_models:
        _embedding_models[model_name] = SentenceTransformer(model_name)
    return _embedding_models[model_name]
```

### 4. Scoring Functions

`match_comments_to_responses` — copied from cell 18:

```python
def match_comments_to_responses(df, response_col=['content_of_comment', 'summarized_content_of_comment'],
                                 comment_col='comment_text', model_name='all-MiniLM-L6-v2',
                                 threshold=0.4, verbose=False):
    model = get_embedding_model(model_name)
    if isinstance(response_col, str):
        responses = df[response_col].tolist()
    else:
        responses = df.fillna('').apply(
            lambda x: ' '.join([x[y] for y in response_col]), axis=1
        ).tolist()
    comments = df[comment_col].tolist()
    resp_emb = model.encode(responses, normalize_embeddings=True, show_progress_bar=verbose)
    comm_emb = model.encode(comments, normalize_embeddings=True, show_progress_bar=verbose)
    sims = np.sum(resp_emb * comm_emb, axis=1)
    df = df.copy()
    df['scores'] = sims
    df['match'] = (df['scores'] >= threshold)
    return df
```

`score_file` — adapted from cell 23, calls scoring twice:

```python
def score_file(chunk_file: Path, response_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Merge chunk file with responses and compute scores from both models."""
    # 1. Read and prep chunk_df (cell 23 top half)
    chunk_df = pd.read_csv(chunk_file)
    if chunk_df.empty:
        return None
    chunk_df['chunk_file'] = str(Path(chunk_file).resolve())
    chunk_df['document_id'] = chunk_df['document_id'].astype(str)
    chunk_df['docket_id'] = chunk_df['document_id'].str.rsplit('-', n=1).str[0]
    chunk_df['agency_id'] = chunk_df['docket_id'].str.split('-').str[0]
    chunk_df = chunk_df.rename(columns={'chunk_text': 'comment_text'})

    # 2. Filter responses and merge
    file_dockets = set(chunk_df['docket_id'].astype(str).unique())
    resp_subset = response_df.loc[lambda df: df['Docket ID'].astype(str).isin(file_dockets)]
    if resp_subset.empty:
        return None
    merged = resp_subset.merge(chunk_df, left_on='Docket ID', right_on='docket_id',
                               how='inner', suffixes=('_resp', '_comment'))
    if merged.empty:
        return None

    # 3. Score with base model
    merged = match_comments_to_responses(merged, model_name=BASE_MODEL_NAME, verbose=False)
    merged = merged.rename(columns={'scores': 'base_scores', 'match': 'base_match'})

    # 4. Score with SBERT model
    merged = match_comments_to_responses(merged, model_name=SBERT_MODEL_PATH, verbose=False)
    merged = merged.rename(columns={'scores': 'sbert_scores', 'match': 'sbert_match'})

    return merged
```

### 5. `sample_for_labeling(df, max_rows=1000)` → DataFrame

Adapted from cell 28 (`sample_response_comment_batches`):

```python
def sample_for_labeling(df: pd.DataFrame, max_rows: int = MAX_LLM_SAMPLE) -> pd.DataFrame:
    """Cell 28 logic. Use base_scores for sampling band."""
    candidates = (
        df.dropna(subset=['content_of_comment', 'comment_text'])
          .loc[lambda d: d['base_scores'].between(SCORE_SAMPLE_MIN, SCORE_SAMPLE_MAX)]
          .copy()
    )
    if candidates.empty:
        # Fallback: widen the band
        candidates = df.dropna(subset=['content_of_comment', 'comment_text']).copy()

    candidates['response_text'] = (
        candidates.fillna('')
        .pipe(lambda d: d['content_of_comment'] + ' ' + d['summarized_content_of_comment'])
        .str.strip()
    )
    candidates['response_id'] = (
        candidates['Agency ID'].astype(str)
        + '|' + candidates['Docket ID'].astype(str)
        + '|' + candidates['content_of_comment'].fillna('').astype(str)
    )

    # Filter to groups with >= 4 comments per response
    eligible = candidates.groupby('response_id').filter(lambda g: len(g) >= 4)
    if eligible.empty:
        eligible = candidates  # fallback: use all

    # Downsample to max 30 per response
    sampled = (
        eligible.groupby('response_id', group_keys=False)
        .apply(lambda g: g.sample(min(len(g), 30)))
        .reset_index(drop=True)
    )

    # Global cap at max_rows
    if len(sampled) > max_rows:
        sampled = sampled.sample(max_rows, random_state=42)

    return sampled
```

### 6. `label_pairs(sampled_df)` → DataFrame

Copy from notebook cell 30 exactly:

```python
MATCHING_PROMPT = """You are an expert legal assistant.
I am analyzing government responses to comments submitted during the notes & comment process.
I will show you a comment and a government response to comments. You will tell me whether the response is responding to this comment:
either directly an individual comment or as part of a larger group.
Be careful: even comments that are not being responded to are likely to be semantically similar, so really read them carefully.
Ignore any "official"-seeming correlates, like letterhead, signatures, citations of evidence in the comment.
Only look directly at the content of the comment and whether the response is responding to it.
Answer with "yes" or "no". Don't say anything else.

Here is a comment:

<comment>
{comment}
</comment>

<response>
{response}
</response>

Your response:
"""

async def label_pairs(sampled_df: pd.DataFrame) -> pd.DataFrame:
    prompts = (
        sampled_df
        .apply(lambda row: MATCHING_PROMPT.format(
            comment=row['comment_text'], response=row['response_text']
        ), axis=1)
        .tolist()
    )
    labels = await prompt_utils.process_batch(prompts=prompts, model=LLM_MODEL)
    labels = pd.Series(labels, index=sampled_df.index).str.strip().str.lower()
    return sampled_df.assign(llm_label=labels)
```

### 7. `find_optimal_threshold(labeled_df, score_col)` → dict

Copy from notebook cell 55:

```python
def find_optimal_threshold(labeled_df: pd.DataFrame, score_col: str) -> dict:
    df = labeled_df.loc[labeled_df['llm_label'].isin(['yes', 'no'])].copy()
    y_true = (df['llm_label'] == 'yes').astype(int)

    if y_true.sum() == 0 or (1 - y_true).sum() == 0:
        return {'f1': 0.0, 'threshold': 0.5, 'precision': 0.0, 'recall': 0.0,
                'report': 'No positive or negative samples'}

    thresholds = np.arange(df[score_col].min(), df[score_col].max(), 0.01)
    best_f1, best_t = 0, 0.5
    for t in thresholds:
        preds = (df[score_col] >= t).astype(int)
        f = f1_score(y_true, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t

    preds = (df[score_col] >= best_t).astype(int)
    report = classification_report(y_true, preds, target_names=['no', 'yes'], zero_division=0)
    return {
        'f1': best_f1,
        'threshold': best_t,
        'precision': precision_score(y_true, preds, zero_division=0),
        'recall': recall_score(y_true, preds, zero_division=0),
        'report': report,
    }
```

### 8. `build_triplets(labeled_df)` → DataFrame

Copy from cell 41 (`triplets_with_neg_lists`), then explode per cell 59:

```python
def build_triplets(labeled_df: pd.DataFrame) -> pd.DataFrame:
    """Cell 41 logic -> Cell 59 explode to single negative."""
    df = labeled_df.copy()
    df = df.loc[df['comment_text'].str.len() > 10]
    df = df.loc[df['response_text'].str.len() > 10]
    df = df.loc[df['comment_text'].apply(lambda x: isinstance(x, str))]
    df = df.loc[df['llm_label'].isin(['yes', 'no'])]

    records = []
    for rid, group in df.groupby(['Docket ID', 'response_id', 'response_text']):
        pos_df = group.query("llm_label == 'yes'")
        neg_df = group.query("llm_label == 'no'").copy()
        if pos_df.empty or neg_df.empty:
            continue
        neg_df['word_len'] = neg_df['comment_text'].str.split().str.len()
        for _, pos_row in pos_df.iterrows():
            pos = pos_row['comment_text']
            pos_len = len(pos.split())
            lower = pos_len * 0.7
            upper = pos_len * 1.3
            candidates = neg_df.loc[
                (neg_df['word_len'] >= lower) & (neg_df['word_len'] <= upper)
            ]
            if len(candidates) < 2:
                candidates = neg_df.iloc[
                    np.argsort(np.abs(neg_df['word_len'].values - pos_len))[:10]
                ]
            top_negs = candidates.sample(n=min(2, len(candidates)))['comment_text'].tolist()
            if top_negs:
                records.append({
                    'anchor': rid[2],     # response_text
                    'positive': pos,
                    'negative': top_negs,
                })

    triplet_df = pd.DataFrame(records)
    if triplet_df.empty:
        return triplet_df
    # Explode to one negative per row (cell 59 pattern)
    triplet_df = (
        triplet_df
        .assign(negative=lambda df: df['negative'].str[0])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    return triplet_df
```

### 9. `finetune_model(triplets_df, output_dir)` → (model, pre_eval, final_eval)

Copy from cell 31 (`finetune_response_matcher`), with exact same training settings:

```python
def finetune_model(triplets_df: pd.DataFrame, output_dir: str) -> tuple:
    """Cell 31 logic. Returns (model, pre_eval, final_eval)."""
    from sentence_transformers import (
        SentenceTransformerTrainer, SentenceTransformerTrainingArguments,
        SentenceTransformerModelCardData
    )
    from sentence_transformers.losses import TripletLoss, TripletDistanceMetric
    from sentence_transformers.evaluation import TripletEvaluator
    from datasets import Dataset
    from transformers import TrainerCallback

    class EvalCallback(TrainerCallback):
        def __init__(self, evaluator, eval_steps=50):
            self.evaluator = evaluator
            self.eval_steps = eval_steps
            self.results = []
        def on_step_end(self, args, state, control, model=None, **kwargs):
            if state.global_step % self.eval_steps == 0:
                scores = self.evaluator(model)
                self.results.append((state.global_step, scores))
                logger.info(f"Step {state.global_step}: {scores}")

    cols = ['anchor', 'positive', 'negative']
    df = triplets_df[cols].reset_index(drop=True)
    eval_df = df.sample(frac=0.1, random_state=42)
    train_df = df.drop(eval_df.index).reset_index(drop=True)
    eval_df = eval_df.reset_index(drop=True)

    model = SentenceTransformer(FINETUNE_BASE)
    train_loss = TripletLoss(model, distance_metric=TripletDistanceMetric.COSINE, triplet_margin=0.3)

    evaluator = None
    eval_callback = None
    pre_eval = None
    if len(eval_df) > 0:
        evaluator = TripletEvaluator(
            anchors=eval_df['anchor'].tolist(),
            positives=eval_df['positive'].tolist(),
            negatives=eval_df['negative'].tolist(),
            show_progress_bar=True,
        )
        pre_eval = evaluator(model)
        logger.info(f"Pre-training eval: {pre_eval}")
        eval_callback = EvalCallback(evaluator, eval_steps=50)

    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=4,
        per_device_train_batch_size=8,
        learning_rate=5e-6,
        gradient_accumulation_steps=4,
        warmup_ratio=0.2,
        fp16=True,
        eval_strategy='no',
        save_strategy='epoch',
        logging_steps=50,
        run_name=Path(output_dir).name,
    )
    trainer = SentenceTransformerTrainer(
        model=model, args=args,
        train_dataset=Dataset.from_pandas(train_df),
        loss=train_loss,
        callbacks=[eval_callback] if eval_callback else [],
    )
    trainer.train()

    final_eval = evaluator(model) if evaluator else None
    if final_eval and pre_eval:
        logger.info(f"Accuracy change: {pre_eval['cosine_accuracy']:.4f} -> {final_eval['cosine_accuracy']:.4f}")

    model.save_pretrained(Path(output_dir) / 'final')
    return model, pre_eval, final_eval
```

### 10. Logging Helper

```python
def log_result(entry: dict):
    """Append one JSON line to the log file."""
    entry['timestamp'] = datetime.now().isoformat()
    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps(entry, default=str) + '\n')
    logger.info(json.dumps(entry, indent=2, default=str))
```

### 11. Main Pipeline: `process_chunk_file()`

This is the core orchestrator. For each chunk file:

```python
async def process_chunk_file(chunk_file: Path, response_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    file_id = chunk_file.stem
    logger.info(f"Processing {chunk_file}")
    log_entry = {'file': str(chunk_file), 'file_id': file_id}

    # ── Step 1: Score with both models ──
    scored_df = score_file(chunk_file, response_df)
    if scored_df is None or scored_df.empty:
        log_entry['status'] = 'skipped_no_data'
        log_result(log_entry)
        return None

    total_rows = len(scored_df)
    log_entry['total_rows'] = total_rows

    # ── Step 2: Sample and LLM-label ──
    sampled = sample_for_labeling(scored_df, max_rows=MAX_LLM_SAMPLE)
    if sampled.empty:
        log_entry['status'] = 'skipped_no_samples'
        log_result(log_entry)
        return None

    log_entry['sampled_rows'] = len(sampled)
    labeled = await label_pairs(sampled)
    labeled = labeled.loc[labeled['llm_label'].isin(['yes', 'no'])]
    log_entry['labeled_rows'] = len(labeled)
    log_entry['label_dist'] = labeled['llm_label'].value_counts().to_dict()

    # ── Step 3: If total_rows <= 1000, LLM labels cover everything ──
    if total_rows <= MAX_LLM_SAMPLE:
        logger.info(f"Small dataset ({total_rows} rows). Using LLM labels directly.")
        # Merge LLM labels back into scored_df
        scored_df = scored_df.merge(
            labeled[['llm_label']].rename_axis('_idx'),
            left_index=True, right_index=True, how='left'
        )
        # For any rows not in the sample (edge case), label by best threshold
        base_eval = find_optimal_threshold(labeled, 'base_scores')
        unlabeled_mask = scored_df['llm_label'].isna()
        if unlabeled_mask.any():
            scored_df.loc[unlabeled_mask, 'llm_label'] = np.where(
                scored_df.loc[unlabeled_mask, 'base_scores'] >= base_eval['threshold'],
                'yes', 'no'
            )
        scored_df['final_label'] = scored_df['llm_label']
        scored_df['strategy'] = 'llm_direct'

        log_entry['strategy'] = 'llm_direct'
        log_entry['base_f1'] = base_eval['f1']
        log_entry['base_threshold'] = base_eval['threshold']
        log_result(log_entry)
        _save_output(scored_df, file_id)
        return scored_df

    # ── Step 4: Large dataset — evaluate both models ──
    base_eval = find_optimal_threshold(labeled, 'base_scores')
    sbert_eval = find_optimal_threshold(labeled, 'sbert_scores')

    log_entry['base_f1'] = base_eval['f1']
    log_entry['base_threshold'] = base_eval['threshold']
    log_entry['base_precision'] = base_eval['precision']
    log_entry['base_recall'] = base_eval['recall']
    log_entry['sbert_f1'] = sbert_eval['f1']
    log_entry['sbert_threshold'] = sbert_eval['threshold']
    log_entry['sbert_precision'] = sbert_eval['precision']
    log_entry['sbert_recall'] = sbert_eval['recall']

    logger.info(f"Base F1: {base_eval['f1']:.3f} @ {base_eval['threshold']:.3f}")
    logger.info(f"SBERT F1: {sbert_eval['f1']:.3f} @ {sbert_eval['threshold']:.3f}")
    logger.info(f"Base report:\n{base_eval['report']}")
    logger.info(f"SBERT report:\n{sbert_eval['report']}")

    # ── Step 5: If either model F1 > 0.7, use that model's threshold ──
    if base_eval['f1'] >= F1_THRESHOLD or sbert_eval['f1'] >= F1_THRESHOLD:
        if sbert_eval['f1'] >= base_eval['f1']:
            best_name, best_eval, best_col = 'sbert', sbert_eval, 'sbert_scores'
        else:
            best_name, best_eval, best_col = 'base', base_eval, 'base_scores'

        scored_df['final_label'] = np.where(
            scored_df[best_col] >= best_eval['threshold'], 'yes', 'no'
        )
        scored_df['strategy'] = f'threshold_{best_name}'

        log_entry['strategy'] = f'threshold_{best_name}'
        log_entry['chosen_model'] = best_name
        log_entry['chosen_threshold'] = best_eval['threshold']
        log_entry['chosen_f1'] = best_eval['f1']
        log_result(log_entry)
        _save_output(scored_df, file_id)
        return scored_df

    # ── Step 6: Neither model good enough — fine-tune a custom model ──
    logger.info("Both models below F1 threshold. Fine-tuning custom model...")

    triplets = build_triplets(labeled)
    log_entry['triplet_count'] = len(triplets)

    if len(triplets) < 20:
        # Not enough triplets to train. Fall back to best available model.
        logger.warning(f"Only {len(triplets)} triplets. Falling back to best model.")
        if sbert_eval['f1'] >= base_eval['f1']:
            best_name, best_eval, best_col = 'sbert', sbert_eval, 'sbert_scores'
        else:
            best_name, best_eval, best_col = 'base', base_eval, 'base_scores'
        scored_df['final_label'] = np.where(
            scored_df[best_col] >= best_eval['threshold'], 'yes', 'no'
        )
        scored_df['strategy'] = f'threshold_{best_name}_fallback'
        log_entry['strategy'] = f'threshold_{best_name}_fallback'
        log_result(log_entry)
        _save_output(scored_df, file_id)
        return scored_df

    ft_output_dir = f'sbert-model-training/custom_{file_id}'
    ft_model, pre_eval, final_eval = finetune_model(triplets, ft_output_dir)

    log_entry['finetune_pre_eval'] = pre_eval
    log_entry['finetune_final_eval'] = final_eval

    # Re-score the labeled subset with the fine-tuned model to find threshold
    ft_model_path = str(Path(ft_output_dir) / 'final')
    _embedding_models.pop(ft_model_path, None)  # clear cache for fresh load
    labeled_rescored = match_comments_to_responses(
        labeled, model_name=ft_model_path, verbose=False
    )
    labeled_rescored = labeled_rescored.rename(columns={'scores': 'custom_scores'})
    custom_eval = find_optimal_threshold(labeled_rescored, 'custom_scores')

    # Re-score the full dataset with the fine-tuned model
    scored_df = match_comments_to_responses(
        scored_df, model_name=ft_model_path, verbose=False
    )
    scored_df = scored_df.rename(columns={'scores': 'custom_scores', 'match': 'custom_match'})

    log_entry['custom_f1'] = custom_eval['f1']
    log_entry['custom_threshold'] = custom_eval['threshold']
    log_entry['custom_precision'] = custom_eval['precision']
    log_entry['custom_recall'] = custom_eval['recall']

    logger.info(f"Custom F1: {custom_eval['f1']:.3f} @ {custom_eval['threshold']:.3f}")
    logger.info(f"Custom report:\n{custom_eval['report']}")

    # Pick the best of all three
    all_evals = {
        'base': (base_eval, 'base_scores'),
        'sbert': (sbert_eval, 'sbert_scores'),
        'custom': (custom_eval, 'custom_scores'),
    }
    best_name = max(all_evals, key=lambda k: all_evals[k][0]['f1'])
    best_eval, best_col = all_evals[best_name]

    scored_df['final_label'] = np.where(
        scored_df[best_col] >= best_eval['threshold'], 'yes', 'no'
    )
    scored_df['strategy'] = f'threshold_{best_name}_finetuned'

    log_entry['strategy'] = f'threshold_{best_name}_finetuned'
    log_entry['chosen_model'] = best_name
    log_entry['chosen_threshold'] = best_eval['threshold']
    log_entry['chosen_f1'] = best_eval['f1']
    log_result(log_entry)
    _save_output(scored_df, file_id)
    return scored_df
```

### 12. Output Helpers and Entry Point

```python
def _save_output(df: pd.DataFrame, file_id: str):
    out_path = OUTPUT_DIR / f'{file_id}_matched.csv'
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df)} rows to {out_path}")


async def run_pipeline():
    response_df = load_response_df()
    chunk_files = sorted(Path('../data/bulk_downloads').rglob('*_chunked.csv'))
    logger.info(f"Found {len(chunk_files)} chunk files")

    for chunk_file in tqdm(chunk_files, desc='Pipeline'):
        try:
            await process_chunk_file(chunk_file, response_df)
        except Exception as e:
            logger.error(f"Failed on {chunk_file}: {e}", exc_info=True)
            log_result({'file': str(chunk_file), 'status': 'error', 'error': str(e)})


if __name__ == '__main__':
    asyncio.run(run_pipeline())
```

---

## Decision Flow Summary

```
For each chunk file:
│
├─ Score with base model → base_scores
├─ Score with SBERT model → sbert_scores
├─ Sample ≤1000 rows → LLM label with gpt-5.2
│
├─ IF total_rows ≤ 1000:
│   └─ Use LLM labels directly → strategy = "llm_direct"
│
├─ ELSE IF max(base_f1, sbert_f1) ≥ 0.7:
│   └─ Use best model's threshold → strategy = "threshold_{model}"
│
├─ ELSE:
│   ├─ Build triplets from LLM labels
│   ├─ Fine-tune mpnet from scratch (NOT continue training sbert — see note below)
│   ├─ Re-score labeled subset with custom model
│   ├─ Find optimal threshold for custom model
│   └─ Pick best of {base, sbert, custom} by F1
│       └─ strategy = "threshold_{best}_finetuned"
│
└─ Save: matched CSV + log entry
```

## Log Schema (`match_pipeline_log.jsonl`)

Each line is a JSON object with these fields:

```json
{
  "timestamp": "2026-02-16T...",
  "file": "/path/to/chunked.csv",
  "file_id": "agency_chunked",
  "total_rows": 15000,
  "sampled_rows": 1000,
  "labeled_rows": 980,
  "label_dist": {"yes": 120, "no": 860},
  "base_f1": 0.45,
  "base_threshold": 0.52,
  "base_precision": 0.40,
  "base_recall": 0.51,
  "sbert_f1": 0.72,
  "sbert_threshold": 0.61,
  "sbert_precision": 0.68,
  "sbert_recall": 0.77,
  "custom_f1": null,
  "custom_threshold": null,
  "triplet_count": null,
  "finetune_pre_eval": null,
  "finetune_final_eval": null,
  "strategy": "threshold_sbert",
  "chosen_model": "sbert",
  "chosen_threshold": 0.61,
  "chosen_f1": 0.72
}
```

Fields are present only when relevant (e.g., `custom_*` and `finetune_*` only when fine-tuning was triggered).

## Important Implementation Notes

1. **Fine-tune from scratch, not continue training SBERT.** The pre-trained SBERT was trained on a different data distribution. For a dataset where it's failing, starting fresh from `microsoft/mpnet-base` avoids inheriting bad representations. The SBERT model is still used as a scorer (step 1) — it just isn't the starting point for fine-tuning.

2. **The `response_text` column** must be constructed before `build_triplets` or `label_pairs` are called. It's built inside `sample_for_labeling` as `content_of_comment + ' ' + summarized_content_of_comment`. Make sure this column is present on `labeled` before passing to `build_triplets`.

3. **When re-scoring with custom model in step 6**, you must also re-score the `labeled` subset (not just the full `scored_df`) to compute `custom_eval`. The labeled subset needs `custom_scores` to find the optimal threshold.

4. **Error handling**: Wrap each file in try/except. Log errors and continue. Don't let one bad file kill the pipeline.

5. **Memory**: Clear `_embedding_models` cache between files if memory is tight: `_embedding_models.clear()` at the end of each file. Definitely clear after fine-tuning to avoid stale models.

6. **The `scores` column gets overwritten** by `match_comments_to_responses`. That's why we immediately rename to `base_scores`/`sbert_scores`/`custom_scores` after each call.

7. **`prompt_utils.process_batch`** is an existing async utility. Import it as-is. It handles batching and rate limits internally.

8. **The `match` column also gets overwritten** — rename it alongside `scores` each time.
