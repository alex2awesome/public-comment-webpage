"""Utilities for loading and running the cross-encoder reranker.

Provides a cached model loader and a batch scoring function for use
in the main matching pipeline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_cross_encoder_cache: dict = {}


def load_cross_encoder(model_path: str, max_length: int = 4096):
    """Load a CrossEncoder model, caching it for reuse across directories."""
    if model_path in _cross_encoder_cache:
        return _cross_encoder_cache[model_path]

    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_path, max_length=max_length)
    _cross_encoder_cache[model_path] = model
    logger.info("Loaded cross-encoder from %s (max_length=%d)", model_path, max_length)
    return model


def load_optimal_threshold(model_path: str) -> float:
    """Load the optimal threshold saved during training."""
    threshold_path = Path(model_path) / "optimal_threshold.json"
    if threshold_path.exists():
        with open(threshold_path) as f:
            data = json.load(f)
        threshold = data["threshold"]
        logger.info(
            "Loaded optimal threshold %.3f from %s", threshold, threshold_path
        )
        return threshold
    logger.warning("No optimal_threshold.json at %s, using 0.5", model_path)
    return 0.5


def rerank_pairs(
    model,
    pairs_df: pd.DataFrame,
    response_col: str = "response_text",
    candidate_col: str = "candidate_text",
    batch_size: int = 64,
) -> np.ndarray:
    """Score (response, candidate) pairs with the cross-encoder.

    Returns an array of sigmoid scores in [0, 1], one per row.
    """
    pairs = list(
        zip(
            pairs_df[response_col].tolist(),
            pairs_df[candidate_col].tolist(),
        )
    )
    scores = model.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=True,
    )
    return np.array(scores)
