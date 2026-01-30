from typing import List, Type
from autometrics.metrics.Metric import Metric
from autometrics.dataset.Dataset import Dataset
from autometrics.recommend.MetricRecommender import MetricRecommender
import os
from platformdirs import user_data_dir
import json
import subprocess
import warnings
from pyserini.search.lucene import LuceneSearcher

class BM25(MetricRecommender):
    def __init__(self, metric_classes: List[Type[Metric]], index_path: str = user_data_dir("autometrics", "bm25"), force_reindex: bool = False, use_description_only: bool = False):
        self.metric_classes = metric_classes
        # Root directory that will hold both collection and index
        self.root_path = index_path
        self.force_reindex = force_reindex
        self.use_description_only = use_description_only

        # ------------------------------------------------------------------
        # Directory layout
        #   <root_path>/collection/docs.jsonl   → input to indexer
        #   <root_path>/index/                 → Lucene index written here
        # ------------------------------------------------------------------
        self.collection_path = os.path.join(self.root_path, "collection")
        self.lucene_index_path = os.path.join(self.root_path, "index")

        # (Re-)build index if needed
        if force_reindex or not os.path.exists(self.lucene_index_path):
            # --------------------------------------------------------------
            # Java check (Pyserini/Lucene requires Java 21)
            # --------------------------------------------------------------
            def _java_major_version() -> int | None:
                try:
                    res = subprocess.run(["java", "-version"], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
                    out = res.stderr.decode() if res.stderr else res.stdout.decode()
                    # Expect lines like: openjdk version "21.0.x"
                    if 'version "' in out:
                        ver = out.split('version "', 1)[1].split('"', 1)[0]
                        major = ver.split('.')[0]
                        return int(major)
                except Exception:
                    return None
                return None

            _major = _java_major_version()
            if _major is None or _major < 21:
                warnings.warn(
                    "[BM25] Java 21+ is required by Pyserini/Lucene.\n"
                    "Install Java 21 and ensure `java -version` reports 21+, e.g.:\n"
                    "  - Ubuntu/Debian: sudo apt install openjdk-21-jdk\n"
                    "  - macOS: brew install openjdk@21\n"
                    "  - Windows: https://www.oracle.com/java/technologies/downloads/#java21",
                    RuntimeWarning,
                )

            print(
                f"Building BM25 index in {self.lucene_index_path} for {len(metric_classes)} metrics …"
            )

            # Clean slate
            import shutil
            if os.path.exists(self.root_path):
                shutil.rmtree(self.root_path)
            os.makedirs(self.collection_path, exist_ok=True)

            # Write docs.jsonl
            metric_names = [m.__name__ for m in metric_classes]
            metric_docs = []
            for m in metric_classes:
                if self.use_description_only:
                    desc = getattr(m, 'description', None)
                    if desc is None:
                        desc = m.__doc__ or ""
                    metric_docs.append(str(desc))
                else:
                    metric_docs.append(m.__doc__ or "")

            docs_file = os.path.join(self.collection_path, "docs.jsonl")
            with open(docs_file, "w", encoding="utf-8") as f:
                for name, doc in zip(metric_names, metric_docs):
                    json.dump({"id": name, "contents": doc}, f)
                    f.write("\n")

            # Invoke Pyserini indexer
            result = subprocess.run([
                "python", "-m", "pyserini.index.lucene",
                "--collection", "JsonCollection",
                "--input", self.collection_path,
                "--index", self.lucene_index_path,
                "--generator", "DefaultLuceneDocumentGenerator",
                "--threads", "1",
                "--storePositions", "--storeDocvectors", "--storeRaw",
            ])

            if result.returncode != 0:
                raise RuntimeError("Failed to build BM25 index. Ensure Java 21+ is installed and on PATH.")

        # ------------------------------------------------------------------
        # Initialise Lucene searcher on the freshly built (or cached) index.
        # ------------------------------------------------------------------
        self.searcher = LuceneSearcher(self.lucene_index_path)
        self.searcher.set_language('en')

        super().__init__(metric_classes, index_path, force_reindex)

    def recommend(self, dataset: Dataset, target_measurement: str, k: int = 20) -> List[Type[Metric]]:
        query = f'I am looking for a metric to evaluate the following task: "{dataset.get_task_description()}"  In particular I care about "{target_measurement}".'
        hits = self.searcher.search(query, k=k)
        return [self.metric_name_to_class(hit.docid) for hit in hits]