"""Utilities for clustering argument records."""

from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
from tqdm.auto import tqdm

from openai import OpenAI
from collections import Counter
import itertools
import networkx as nx


def embed_arguments(
    texts: Iterable[str],
    client: OpenAI,
    model: str = "text-embedding-3-large",
    batch_size: int = 64,
) -> np.ndarray:
    """Return embeddings for the given texts as a float32 numpy array."""
    vectors: List[List[float]] = []
    texts = list(texts)
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        vectors.extend(item.embedding for item in response.data)
    return np.asarray(vectors, dtype=np.float32)


def evaluate_kmeans(
    embeddings: np.ndarray,
    k_values: Iterable[int],
    random_state: int = 42,
    n_init: int = 10,
) -> pd.DataFrame:
    """Compute inertia and silhouette scores for multiple k."""
    inertias: List[float] = []
    silhouettes: List[float] = []
    fitted_models: List[Optional[KMeans]] = []

    best_silhouette = float("-inf")
    best_model: Optional[KMeans] = None

    for k in tqdm(k_values, desc='kmeans value sweep'):
        model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = model.fit_predict(embeddings)
        inertias.append(model.inertia_)
        if len(set(labels)) > 1:
            score = silhouette_score(embeddings, labels)
        else:
            score = float("nan")
        silhouettes.append(score)
        fitted_models.append(model)

        if not np.isnan(score) and score > best_silhouette:
            best_silhouette = score
            best_model = model

    summary = pd.DataFrame({
        "k": list(k_values),
        "inertia": inertias,
        "silhouette": silhouettes,
    })
    if best_model is not None:
        summary["best"] = summary["k"] == best_model.n_clusters
    else:
        summary["best"] = False
    summary.attrs["best_model"] = best_model
    return summary


def assign_kmeans_clusters(
    argument_records: pd.DataFrame,
    embeddings: np.ndarray,
    kmeans_model: KMeans,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Assign clusters and compute representative rows for kmeans results."""
    if "cluster" in argument_records.columns:
        argument_records = argument_records.drop(columns=["cluster"], errors="ignore")
    labels = kmeans_model.labels_
    centroids = kmeans_model.cluster_centers_

    argument_records = argument_records.copy()
    argument_records["cluster"] = labels
    argument_records["centroid_distance"] = np.linalg.norm(
        embeddings - centroids[labels], axis=1
    )

    cluster_summary = (
        argument_records
        .groupby("cluster")
        .agg(
            n_arguments=("argument", "size"),
            dominant_labels=("label", lambda x: x.value_counts().head(3).to_dict()),
        )
        .sort_values("n_arguments", ascending=False)
    )

    rep_idx = argument_records.groupby("cluster")["centroid_distance"].idxmin()
    representatives = (
        argument_records.loc[rep_idx]
        .set_index("cluster")[
            ["argument", "label", "doc_id", "source", "centroid_distance"]
        ]
        .sort_index()
    )

    return cluster_summary, representatives


def setup_argument_records(
    text_df: pd.DataFrame,
    text_df_with_labels: pd.DataFrame,
) -> pd.DataFrame:
    """Explode argument strings into individual rows with metadata."""
    records = (
        text_df_with_labels[["doc_id", "source", "collection_id", "label"]]
        .merge(text_df[["doc_id", "arguments"]], on="doc_id", how="left")
        .dropna(subset=["arguments"])
        .assign(argument=lambda df: df["arguments"].str.split("\n\n"))
        .explode("argument")
        .assign(argument=lambda df: df["argument"].str.strip())
        .loc[lambda df: df["argument"].str.len() > 0]
        .reset_index(drop=True)
    )
    records["argument_id"] = records.index
    return records


def downsample_argument_records(
    argument_records: pd.DataFrame,
    max_samples: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Downsample a copy of ``argument_records`` if it exceeds ``max_samples``."""
    if max_samples is None or len(argument_records) <= max_samples:
        return argument_records.copy(), argument_records.index.to_numpy()

    rng = np.random.default_rng(random_state)
    sample_idx = np.sort(
        rng.choice(len(argument_records), size=max_samples, replace=False)
    )
    sampled = argument_records.iloc[sample_idx].reset_index(drop=True)
    return sampled, sample_idx


def run_umap_bgmm(
    embeddings: np.ndarray,
    n_components: int,
    random_state: int = 42,
    weight_concentration_prior: float = 0.1,
    umap_kwargs: Optional[dict] = None,
    bgmm_kwargs: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Reduce embeddings with UMAP then cluster with Bayesian GMM."""
    try:
        from umap import UMAP  # type: ignore
    except ImportError:  # pragma: no cover - compatibility path
        from umap.umap_ import UMAP  # type: ignore

    from sklearn.mixture import BayesianGaussianMixture

    umap_kwargs = umap_kwargs or {}
    umap_model = UMAP(random_state=random_state, **umap_kwargs)
    reduced = umap_model.fit_transform(embeddings)

    bgmm_kwargs = bgmm_kwargs or {}
    bgmm = BayesianGaussianMixture(
        n_components=n_components,
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=weight_concentration_prior,
        covariance_type="full",
        max_iter=1000,
        random_state=random_state,
        **bgmm_kwargs,
    )
    bgmm.fit(reduced)

    labels = bgmm.predict(reduced)
    probabilities = bgmm.predict_proba(reduced).max(axis=1)

    active_components = np.where(bgmm.weights_ > 1e-3)[0]
    mask_active = np.isin(labels, active_components)

    silhouette = float("nan")
    if mask_active.sum() > 1 and len(active_components) > 1:
        silhouette = silhouette_score(reduced[mask_active], labels[mask_active])

    return reduced, labels, probabilities, silhouette


def summarize_bgmm(
    argument_records: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
    min_weight: float = 1e-3,
) -> pd.DataFrame:
    """Build a summary table for Bayesian GMM clusters."""
    argument_records = argument_records.copy()
    argument_records["bgmm_cluster"] = labels
    argument_records["bgmm_probability"] = probabilities

    active_mask = labels != -1

    if active_mask.sum() == 0:
        return pd.DataFrame()

    summary = (
        argument_records.loc[active_mask]
        .groupby("bgmm_cluster")
        .agg(
            n_arguments=("argument", "size"),
            dominant_labels=("label", lambda x: x.value_counts().head(3).to_dict()),
            mean_probability=("bgmm_probability", "mean"),
            median_probability=("bgmm_probability", "median"),
        )
        .sort_values("n_arguments", ascending=False)
    )

    return summary


def build_contrast_pairs(
    representatives: pd.DataFrame,
    centroids: np.ndarray,
    top_n: int = 8,
) -> Tuple[List[str], List[dict]]:
    """Prepare representative statement pairs for contrast scoring."""
    from sklearn.metrics.pairwise import cosine_distances

    cluster_ids = representatives.index.tolist()
    pair_inputs: List[str] = []
    pair_metadata: List[dict] = []

    for i, cluster_a in enumerate(cluster_ids):
        for cluster_b in cluster_ids[i + 1 :]:
            distance = float(
                cosine_distances(
                    centroids[cluster_a].reshape(1, -1),
                    centroids[cluster_b].reshape(1, -1),
                )[0, 0]
            )
            pair_metadata.append({
                "distance": distance,
                "cluster_a": int(cluster_a),
                "cluster_b": int(cluster_b),
                "label_a": representatives.loc[cluster_a, "label"],
                "label_b": representatives.loc[cluster_b, "label"],
                "argument_a": representatives.loc[cluster_a, "argument"],
                "argument_b": representatives.loc[cluster_b, "argument"],
            })

    pair_metadata = sorted(pair_metadata, key=lambda x: x["distance"], reverse=True)[:top_n]

    for meta in pair_metadata:
        pair_inputs.append(
            f"Statement A:\n{meta['argument_a']}\n\nStatement B:\n{meta['argument_b']}"
        )

    return pair_inputs, pair_metadata


def cluster_centroid_distance_matrix(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    metric: str = "cosine",
    include_noise: bool = False,
    centroids: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute pairwise distances between cluster centroids.

    Parameters
    ----------
    embeddings:
        Array of shape (n_samples, n_features) containing the sample embeddings.
    labels:
        Cluster labels for each embedding. Noise points can be marked with ``-1``.
    metric:
        Distance metric passed to :func:`sklearn.metrics.pairwise_distances`.
    include_noise:
        Whether to include clusters labelled ``-1`` when computing distances.
    centroids:
        Optional array of precomputed centroids with shape (n_clusters, n_features).
        When ``None`` the centroid for each cluster is computed as the mean of the
        embeddings assigned to that cluster. This is equivalent to the centroids used
        by KMeans and works for methods such as BGMM that may not expose explicit
        cluster centres.

    Returns
    -------
    cluster_ids:
        Array of cluster ids associated with the rows/columns in the distance matrix.
    distance_matrix:
        Square matrix of pairwise centroid distances.
    """
    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels)

    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
        raise ValueError("labels must be a 1D array aligned with embeddings")

    if centroids is None:
        centroid_vectors: List[np.ndarray] = []
        cluster_ids: List[int] = []
        for cluster_id in np.unique(labels):
            if cluster_id < 0 and not include_noise:
                continue
            mask = labels == cluster_id
            if not mask.any():
                continue
            centroid_vectors.append(embeddings[mask].mean(axis=0))
            cluster_ids.append(int(cluster_id))
        if not centroid_vectors:
            return np.asarray([]), np.empty((0, 0))
        centroid_matrix = np.vstack(centroid_vectors)
    else:
        if isinstance(centroids, dict):
            ordered = []
            cluster_ids = []
            for cid in sorted(centroids.keys()):
                if not include_noise and cid < 0:
                    continue
                ordered.append(np.asarray(centroids[cid]))
                cluster_ids.append(int(cid))
            if not ordered:
                return np.asarray([]), np.empty((0, 0))
            centroid_matrix = np.vstack(ordered)
        else:
            centroid_matrix = np.asarray(centroids)
            cluster_ids = sorted({int(c) for c in labels if include_noise or c >= 0})
            if centroid_matrix.shape[0] != len(cluster_ids):
                raise ValueError(
                    "centroids array must have the same number of rows as clusters"
                )

    distances = pairwise_distances(centroid_matrix, metric=metric)
    return np.asarray(cluster_ids), distances


def compute_cluster_cooccurrence_graph(
    argument_records: pd.DataFrame,
    *,
    cluster_col: str = "cluster",
    doc_col: str = "doc_id",
    min_weight: float = 1.0,
    weight_metric: str = "lift",
) -> nx.Graph:
    """Build a co-occurrence graph of clusters based on document membership.

    Parameters
    ----------
    argument_records:
        DataFrame containing at least ``cluster_col`` and ``doc_col`` columns.
    cluster_col:
        Column name holding cluster labels (integers).
    doc_col:
        Column name indicating the document grouping.
    min_weight:
        Minimum weight required to retain an edge in the returned graph.
    weight_metric:
        One of ``{"count", "lift"}``. ``"count"`` uses raw co-occurrence counts;
        ``"lift"`` divides the observed co-occurrence by the expected count based
        on marginal frequencies.

    Returns
    -------
    G : networkx.Graph
        Undirected graph whose nodes are cluster ids and edges encode co-occurrence
        weights between cluster pairs.
    """
    if weight_metric not in {"count", "lift"}:
        raise ValueError("weight_metric must be 'count' or 'lift'")

    docs = (
        argument_records[[doc_col, cluster_col]]
        .dropna(subset=[doc_col, cluster_col])
        .groupby(doc_col)[cluster_col]
        .apply(lambda s: sorted({int(c) for c in s if int(c) >= 0}))
    )

    total_docs = len(docs)
    co_counts = Counter()
    cluster_counts = Counter()

    for clusters in docs:
        for c in clusters:
            cluster_counts[c] += 1
        for a, b in itertools.combinations(clusters, 2):
            co_counts[(a, b)] += 1
            co_counts[(b, a)] += 1

    G = nx.Graph()
    for cluster_id in cluster_counts:
        G.add_node(cluster_id, count=cluster_counts[cluster_id])

    for (a, b), obs in co_counts.items():
        if a >= b:
            continue
        if weight_metric == "lift":
            expected = (cluster_counts[a] * cluster_counts[b]) / max(total_docs, 1)
            weight = obs / expected if expected else 0.0
        else:
            weight = float(obs)
        if weight >= min_weight:
            G.add_edge(a, b, weight=weight, count=obs)

    return G
