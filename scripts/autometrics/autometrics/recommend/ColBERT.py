from typing import List, Type, Optional
from autometrics.metrics.Metric import Metric
from autometrics.dataset.Dataset import Dataset
from autometrics.recommend.MetricRecommender import MetricRecommender


import os
from platformdirs import user_data_dir
from pylate import indexes, models, retrieve
try:
    import torch
except Exception:
    torch = None

class ColBERT(MetricRecommender):
    """Metric recommender that leverages Modern ColBERT via the PyLate package.

    The first time the class is instantiated (or when *force_reindex* is ``True``)
    we create a ColBERT index over the provided ``metric_classes``.  Subsequent
    instantiations simply load the existing index which keeps start-up time low.
    """

    def __init__(
        self,
        metric_classes: List[Type[Metric]],
        index_path: Optional[str] = user_data_dir("autometrics", "colbert"),
        force_reindex: bool = False,
        use_description_only: bool = False,
    ) -> None:
        # Store the metric classes that will be indexed/searched
        self.metric_classes: List[Type[Metric]] = metric_classes

        self.index_path: str = index_path
        self.force_reindex: bool = force_reindex
        self.index_name: str = "colbert_index" # Causes too many issues when this is a parameter.  Just use the index path to determine the index name.
        self.use_description_only: bool = use_description_only

        # Initialize the ColBERT model
        self.model = models.ColBERT(
            model_name_or_path="lightonai/GTE-ModernColBERT-v1",
        )

        # If the index is missing (or the user explicitly asked for a rebuild)
        # we need to construct it from scratch.
        if force_reindex or not os.path.exists(self.index_path):
            self._build_index()
        
        # Initialize the PLAID index
        self.index = indexes.PLAID(
            index_folder=self.index_path,
            index_name=self.index_name,
            override=False,
        )
        
        # Initialize the retriever
        self.retriever = retrieve.ColBERT(index=self.index)

        super().__init__(metric_classes, index_path, force_reindex)

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _build_index(self) -> None:
        """Create a ColBERT index for *metric_classes* at *index_path*."""
        # Make sure the parent directory exists
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        # Prepare the documents: one document per metric class.
        # If use_description_only is True, prefer class-level description if available; otherwise use docstring.
        # If neither is available, fall back to empty string to keep ordering consistent.
        metric_ids: List[str] = [mc.__name__ for mc in self.metric_classes]
        metric_docs: List[str] = []
        for mc in self.metric_classes:
            if self.use_description_only:
                # Prefer class attribute 'description' if present (many metrics define it as ClassVar)
                desc = getattr(mc, 'description', None)
                if desc is None:
                    desc = mc.__doc__ or ""
                metric_docs.append(str(desc))
            else:
                metric_docs.append(mc.__doc__ or "")

        print("[ColBERT] Metric IDs: ", metric_ids)

        print(
            f"Building ColBERT index at {self.index_path} for {len(metric_ids)} metrics..."
        )

        # Encode the documents
        documents_embeddings = self.model.encode(
            metric_docs,
            batch_size=32,
            is_query=False,
            show_progress_bar=True,
        )

        # Create and initialize the PLAID index
        index = indexes.PLAID(
            index_folder=self.index_path,
            index_name=self.index_name,
            override=True,
        )

        # Add the documents to the index
        index.add_documents(
            documents_ids=metric_ids,
            documents_embeddings=documents_embeddings,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def recommend(
        self, dataset: Dataset, target_measurement: str, k: int = 20
    ) -> List[Type[Metric]]:
        """Return the *k* most relevant metrics for *(dataset, target_measurement)*.

        The query template follows the pattern we established for the BM25
        recommender so that both systems are directly comparable.
        """
        # Robustly obtain a human-readable task description from the dataset
        task_desc = dataset.get_task_description()

        task_desc = task_desc or dataset.get_name() if hasattr(dataset, "get_name") else ""

        query = (
            f'I am looking for a metric to evaluate the following task: "{task_desc}" '
            f' In particular I care about "{target_measurement}".'
        )

        # Encode the query
        query_embeddings = self.model.encode(
            [query],
            batch_size=1,
            is_query=True,
            show_progress_bar=False,
        )

        # Retrieve results with recovery for broken index on GPU machines
        def _has_gpu() -> bool:
            try:
                return torch is not None and torch.cuda.is_available()
            except Exception:
                return False

        try:
            scores = self.retriever.retrieve(
                queries_embeddings=query_embeddings,
                k=k,
            )
        except Exception as e:
            message = str(e)
            if "NoneType" in message and "has no attribute 'search'" in message:
                # Likely a broken PLAID index; rebuild if a GPU is available
                if _has_gpu():
                    print("[ColBERT] Detected broken index during retrieval. Rebuilding index on GPU...")
                    self._build_index()
                    # Reinitialize index and retriever after rebuild
                    self.index = indexes.PLAID(
                        index_folder=self.index_path,
                        index_name=self.index_name,
                        override=False,
                    )
                    self.retriever = retrieve.ColBERT(index=self.index)
                    # Retry once
                    scores = self.retriever.retrieve(
                        queries_embeddings=query_embeddings,
                        k=k,
                    )
                else:
                    # CPU-only machine: re-raise as before
                    raise e
            else:
                # Other errors bubble up
                raise e

        print("[ColBERT] Scores: ", scores)

        print("[ColBERT] Retrieved metrics: ", [self.metric_name_to_class(hit["id"]) for hit in scores[0]])

        # Extract document IDs from the first query's results
        return [self.metric_name_to_class(hit["id"]) for hit in scores[0]]
