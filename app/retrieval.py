import pickle
import numpy as np
import faiss
from huggingface_hub import InferenceClient
from typing import List, Dict

class VectorRetriever:
    def __init__(self, faiss_path: str, chunk_path: str, model_name: str, hf_token: str):
        # HuggingFace Inference API client
        self.client = InferenceClient(model_name, token=hf_token)

        # FAISS index load
        self.index = faiss.read_index(faiss_path)

        # Chunks load
        with open(chunk_path, "rb") as f:
            self.chunks = pickle.load(f, encoding="utf-8")

    def _embed_query(self, query: str) -> np.ndarray:
        query = "query: " + query
        # Remote embedding via HF API
        vector = self.client.feature_extraction(query)
        return np.array([vector], dtype="float32")

    def search(self, query: str, top_k: int = 20) -> List[Dict]:
        q_vec = self._embed_query(query)
        _, indices = self.index.search(q_vec, top_k)

        results = []
        for idx in indices[0]:
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            text_value = chunk.get("text")
            if isinstance(text_value, bytes):
                text_value = text_value.decode("utf-8", errors="ignore")
            results.append({
                "text": text_value,
                "video_title": chunk.get("title"),
                "youtube_url": chunk.get("play_url"),
                "start": chunk.get("start"),
                "end": chunk.get("end"),
            })
        return results
