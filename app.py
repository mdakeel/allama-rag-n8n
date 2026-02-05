from fastapi import FastAPI
import faiss, pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ----------------
FAISS_PATH = r"data/vector_store/faiss.index"
CHUNK_PATH = r"data/vector_store/chunks.pkl"
MODEL_NAME = "intfloat/multilingual-e5-large"
TOP_K = 5
# ----------------------------------------

app = FastAPI()

print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

print("Loading FAISS index...")
index = faiss.read_index(FAISS_PATH)

print("Loading chunks...")
with open(CHUNK_PATH, "rb") as f:
    chunks = pickle.load(f)

print(f"FAISS index dim: {index.d}")
print(f"Total chunks: {len(chunks)}")

# ---------- SEARCH FUNCTION -------------
def embed_query(text: str):
    text = "query: " + text
    vec = model.encode([text], normalize_embeddings=True)
    return np.array(vec).astype("float32")

# -------------- API ---------------------
@app.post("/search")
def search(query: str, top_k: int = TOP_K):
    q_vec = embed_query(query)
    distances, indices = index.search(q_vec, top_k)

    results = []
    for idx in indices[0]:
        if idx == -1:
            continue
        chunk = chunks[idx]
        results.append({
            "text": chunk["text"],
            "video_title": chunk["title"],
            "youtube_url": chunk["play_url"],
            "start": chunk["start"],
            "end": chunk["end"]
        })

    return {
        "query": query,
        "results": results
    }
