# src/retriever.py
import faiss
import pickle
import numpy as np
from embeddings import Embeddings

INDEX_PATH = "data/faiss_index.bin"
META_PATH = "data/meta.pkl"

class Retriever:
    def __init__(self, embed_backend="hf", openai_key=None):
        self.embed_backend = embed_backend
        self.embedder = Embeddings(backend=embed_backend, openai_api_key=openai_key)
        self.index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "rb") as f:
            self.meta = pickle.load(f)

    def query(self, query_text, top_k=5):
        q_emb = self.embedder.encode([query_text]).astype("float32")
        faiss.normalize_L2(q_emb)
        scores, ids = self.index.search(q_emb, top_k)
        results = []
        for s, idx in zip(scores[0], ids[0]):
            if idx < 0: continue
            meta = self.meta[idx]
            results.append({"score": float(s), "text": meta["text"], "source_question": meta["source_question"], "source_answer": meta["source_answer"]})
        return results
