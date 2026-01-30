import os
import fasttext
import requests
from platformdirs import user_data_dir
from autometrics.metrics.reference_free.ReferenceFreeMetric import ReferenceFreeMetric
from typing import List
try:
    from huggingface_hub.constants import HF_HUB_CACHE
except ImportError:
    HF_HUB_CACHE = None

class FastTextClassifier(ReferenceFreeMetric):
    """
    Base class for fastText-based reference-free classification metrics.
    Downloads and loads a fastText model binary, then predicts labels and flips score for specific labels.
    """
    def __init__(
        self,
        name: str,
        description: str,
        model_url: str,
        negative_label: str,
        persistent: bool = True,
        data_dir: str = None,
        **kwargs
    ):
        # Pass ALL parameters to parent constructor
        super().__init__(
            name=name,
            description=description,
            model_url=model_url,
            negative_label=negative_label,
            persistent=persistent,
            data_dir=data_dir,
            **kwargs
        )
        self.model_url = model_url
        self.negative_label = negative_label
        self.persistent = persistent
        # Determine data directory: use provided data_dir,
        # else prefer HuggingFace hub cache if available, else fallback to user_data_dir
        if data_dir:
            base_dir = data_dir
        else:
            hf_cache_root = HF_HUB_CACHE
            if hf_cache_root:
                base_dir = os.path.join(hf_cache_root, "autometrics")
            else:
                base_dir = user_data_dir("autometrics")
        os.makedirs(base_dir, exist_ok=True)
        self.model_path = os.path.join(base_dir, os.path.basename(self.model_url))
        self.model = None
        
        # Exclude parameters that don't affect results from cache key
        self.exclude_from_cache_key('persistent', 'data_dir')

    def _download_model(self):
        resp = requests.get(self.model_url, stream=True)
        resp.raise_for_status()
        with open(self.model_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    def _load_model(self):
        if not os.path.isfile(self.model_path):
            self._download_model()
        self.model = fasttext.load_model(self.model_path)

    def _unload_model(self):
        # fastText has no explicit unload; dereference
        self.model = None

    def _calculate_impl(self, input_text: str, output: str, references=None, **kwargs) -> float:
        # Lazy load if needed
        if self.model is None:
            self._load_model()
        text = output.replace("\n", " ")
        labels, probs = self.model.predict(text)
        label = labels[0].replace("__label__", "")
        score = float(probs[0])
        if label == self.negative_label:
            score = -score
        if not self.persistent:
            self._unload_model()
        return score 
    
    def _calculate_batched_impl(self, input_texts: List[str], output_texts: List[str], references=None, **kwargs) -> List[float]:
        # Lazy load if needed
        if self.model is None:
            self._load_model()
        
        results = []
        for output in output_texts:
            text = output.replace("\n", " ")
            labels, probs = self.model.predict(text)
            label = labels[0].replace("__label__", "")
            score = float(probs[0])
            if label == self.negative_label:
                score = -score
            results.append(score)
            
        if not self.persistent:
            self._unload_model()
            
        return results