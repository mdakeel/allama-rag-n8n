# retrieval.py
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict

class VectorRetriever:
    def __init__(
        self,
        faiss_path: str,
        chunk_path: str,
        model_name: str,
    ):
        # Step 1: Model load
        self.model = SentenceTransformer(model_name)

        # Step 2: FAISS index load
        self.index = faiss.read_index(faiss_path)

        # Step 3: Chunks load (UTF-8 safe)
        with open(chunk_path, "rb") as f:
            self.chunks = pickle.load(f, encoding="utf-8")

        # Step 4: Validation
        if self.index.d == 0 or len(self.chunks) == 0:
            raise RuntimeError("FAISS index or chunks not loaded properly")

    def _embed_query(self, query: str) -> np.ndarray:
        query = "query: " + query
        vector = self.model.encode(
            [query],
            normalize_embeddings=True
        )
        return np.array(vector).astype("float32")

    def search(self, query: str, top_k: int = 20) -> List[Dict]:
        q_vec = self._embed_query(query)
        _, indices = self.index.search(q_vec, top_k)

        results = []
        for idx in indices[0]:
            if idx == -1:
                continue

            chunk = self.chunks[idx]

            # Text को UTF-8 safe decode करो
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


# #  Direct test run (same file में)
# if __name__ == "__main__":
#     faiss_path = "faiss_index.bin"
#     chunk_path = "chunks.pkl"
#     model_name = "sentence-transformers/all-MiniLM-L6-v2"

#     retriever = VectorRetriever(faiss_path, chunk_path, model_name)

#     query = "huroof e muqattat"
#     results = retriever.search(query, top_k=10)

#     print(f"Query: {query}\n")
#     for i, r in enumerate(results, 1):
#         print(f"Chunk {i}:")
#         print(f"Text: {r['text']}")
#         print(f"Source: {r['video_title']}")
#         print(f"URL: {r['youtube_url']} ({r['start']} - {r['end']})\n")
