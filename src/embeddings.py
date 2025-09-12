# src/embeddings.py
import os
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import openai
except Exception:
    openai = None

MODEL_NAME_HF = "all-mpnet-base-v2"  # fast, high-quality SBERT model

class Embeddings:
    def __init__(self, backend="hf", openai_api_key=None):
        """
        backend: "hf" or "openai"
        """
        self.backend = backend
        if backend == "hf":
            self.model = SentenceTransformer(MODEL_NAME_HF)
        elif backend == "openai":
            if openai is None:
                raise RuntimeError("openai package not available")
            if openai_api_key is None:
                openai_api_key = os.getenv("OPENAI_API_KEY")
            openai.api_key = openai_api_key
        else:
            raise ValueError("backend must be 'hf' or 'openai'")

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.backend == "hf":
            embs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return embs
        else:
            # OpenAI embeddings (example using text-embedding-3-small)
            resp = openai.Embedding.create(model="text-embedding-3-small", input=texts)
            embs = [e["embedding"] for e in resp["data"]]
            return np.array(embs, dtype=np.float32)
