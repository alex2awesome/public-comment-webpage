from typing import List, Type, Optional
import os
import json
import subprocess

from platformdirs import user_data_dir
from pyserini.search.faiss import FaissSearcher
from pyserini.encode import DocumentEncoder, QueryEncoder

from autometrics.metrics.Metric import Metric
from autometrics.dataset.Dataset import Dataset
from autometrics.recommend.MetricRecommender import MetricRecommender


class Faiss(MetricRecommender):
    """Metric recommender based on dense retrieval with a Faiss index built via Pyserini.

    This class mirrors the logic of `BM25` and `ColBERT`, but relies on Pyserini's
    dense encoders and a Faiss inner-product index.  The first instantiation will
    build the index (unless it already exists or *force_reindex* is *False*).  All
    subsequent instantiations reuse the persisted index, making start-up fast.
    """

    def __init__(
        self,
        metric_classes: List[Type[Metric]],
        index_path: str | None = None,
        encoder_name: str = "facebook/dpr-question_encoder-multiset-base",
        force_reindex: bool = False,
    ) -> None:
        self.metric_classes = metric_classes
        # ------------------------------------------------------------------
        # Determine root path for index storage. If the caller did not provide
        # an explicit *index_path*, place it under the platform-specific user
        # data dir: ~/.local/share/autometrics/faiss
        # ------------------------------------------------------------------
        if index_path is None:
            index_path = os.path.join(user_data_dir("autometrics"), "faiss")

        self.root_path = index_path
        self.encoder_name = encoder_name
        self.force_reindex = force_reindex

        # Paths
        # Root directory that will hold both the collection JSON and the Faiss
        # index files.  Pyserini expects **the directory** that contains the
        # index file named "index". Therefore we keep this directory path
        # separate and *do not* append another "/index" when instantiating the
        # searcher (to avoid the previously observed '/index/index' look-up
        # error).
        self.collection_path = os.path.join(self.root_path, "collection")
        self.index_dir = self.root_path  # Directory passed to FaissSearcher
        self.index_file = os.path.join(self.index_dir, "index")  # Actual index file created by Pyserini

        # Build the index if requested or if it does not yet exist.
        if force_reindex or not os.path.exists(self.index_file):
            self._build_index()

        # ------------------------------------------------------------------
        # Initialise the Pyserini Faiss searcher.
        # ------------------------------------------------------------------
        self.searcher = FaissSearcher(self.index_dir, query_encoder=self.encoder_name)

        super().__init__(metric_classes, index_path, force_reindex)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_index(self) -> None:
        """Create a Faiss flat IP index for the metric docstrings."""
        print(
            f"[Faiss] Building dense index for {len(self.metric_classes)} metrics at {self.index_dir} …"
        )

        # Clean slate
        import shutil
        if os.path.exists(self.root_path):
            shutil.rmtree(self.root_path)
        os.makedirs(self.collection_path, exist_ok=True)

        # ---------------------------------------------
        # 1) Write collection to JSONL (docs.jsonl)
        # ---------------------------------------------
        docs_file = os.path.join(self.collection_path, "docs.jsonl")
        with open(docs_file, "w", encoding="utf-8") as f:
            for cls in self.metric_classes:
                doc = {
                    "id": cls.__name__,
                    "text": cls.__doc__ or "",
                }
                json.dump(doc, f, ensure_ascii=False)
                f.write("\n")

        # ---------------------------------------------
        # 2) Encode and directly build Faiss flat index
        # ---------------------------------------------
        # We leverage Pyserini's CLI which handles batching, GPU utilisation, etc.
        # The `--to-faiss` flag converts vectors directly into a flat IP index.
        encode_cmd = [
            "python",
            "-m",
            "pyserini.encode",
            "input",
            "--corpus",
            docs_file,
            "--fields",
            "text",
            "output",
            "--embeddings",
            self.index_dir,
            "--to-faiss",
            "encoder",
            "--encoder",
            self.encoder_name,
            "--batch-size",
            "32",
        ]

        # ------------------------------------------------------------------
        # Device handling: default to CPU if GPU not available.  Using FP16 on
        # CPU is not supported.
        # ------------------------------------------------------------------
        try:
            import torch
            gpu_available = torch.cuda.is_available()
        except (ImportError, RuntimeError):
            gpu_available = False

        if gpu_available:
            encode_cmd.extend(["--device", "cuda:0", "--fp16"])
        else:
            encode_cmd.extend(["--device", "cpu"])

        result = subprocess.run(encode_cmd)
        if result.returncode != 0:
            # delete the index directory
            shutil.rmtree(self.root_path)
            raise RuntimeError("[Faiss] Failed to encode documents and build Faiss index.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def recommend(self, dataset: Dataset, target_measurement: str, k: int = 20) -> List[Type[Metric]]:
        task_desc = dataset.get_task_description()
        query = (
            f'I am looking for a metric to evaluate the following task: "{task_desc}" '
            f' In particular I care about "{target_measurement}".'
        )
        
        # Directly search with raw query string; FaissSearcher will encode internally
        hits = self.searcher.search(query, k=k)
        return [self.metric_name_to_class(hit.docid) for hit in hits]
