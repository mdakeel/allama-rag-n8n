# from fastapi import FastAPI
# import faiss, pickle
# import numpy as np
# from sentence_transformers import SentenceTransformer

# # ---------------- CONFIG ----------------
# FAISS_PATH = r"data/vector_store/faiss.index"
# CHUNK_PATH = r"data/vector_store/chunks.pkl"
# MODEL_NAME = "intfloat/multilingual-e5-large"
# TOP_K = 5
# # ----------------------------------------

# app = FastAPI()

# print("Loading embedding model...")
# model = SentenceTransformer(MODEL_NAME)

# print("Loading FAISS index...")
# index = faiss.read_index(FAISS_PATH)

# print("Loading chunks...")
# with open(CHUNK_PATH, "rb") as f:
#     chunks = pickle.load(f)

# print(f"FAISS index dim: {index.d}")
# print(f"Total chunks: {len(chunks)}")

# # ---------- SEARCH FUNCTION -------------
# def embed_query(text: str):
#     text = "query: " + text
#     vec = model.encode([text], normalize_embeddings=True)
#     return np.array(vec).astype("float32")

# # -------------- API ---------------------
# @app.post("/search")
# def search(query: str, top_k: int = TOP_K):
#     q_vec = embed_query(query)
#     distances, indices = index.search(q_vec, top_k)

#     results = []
#     for idx in indices[0]:
#         if idx == -1:
#             continue
#         chunk = chunks[idx]
#         results.append({
#             "text": chunk["text"],
#             "video_title": chunk["title"],
#             "youtube_url": chunk["play_url"],
#             "start": chunk["start"],
#             "end": chunk["end"]
#         })

#     return {
#         "query": query,
#         "results": results
#     }



from fastapi import FastAPI
import faiss, pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------- CONFIG ----------------
FAISS_PATH = r"data/vector_store/faiss.index"
CHUNK_PATH = r"data/vector_store/chunks.pkl"

EMBED_MODEL = "intfloat/multilingual-e5-large"
GEN_MODEL = "google/flan-t5-large"

TOP_K = 5
# ----------------------------------------

app = FastAPI()

# ---------- LOAD MODELS -----------------
print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)

print("Loading generation model...")
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)

print("Loading FAISS index...")
index = faiss.read_index(FAISS_PATH)

print("Loading chunks...")
with open(CHUNK_PATH, "rb") as f:
    chunks = pickle.load(f)

print(f"FAISS index dim: {index.d}")
print(f"Total chunks: {len(chunks)}")

# ---------- EMBED QUERY -----------------
def embed_query(text: str):
    text = "query: " + text
    vec = embedder.encode([text], normalize_embeddings=True)
    return np.array(vec).astype("float32")

# ---------- GENERATE ANSWER -------------
def generate_answer(question: str, context_chunks: list):
    context_text = "\n".join(context_chunks)

    prompt = f"""
You are an Islamic knowledge assistant.

The context is in URDU.
The question is in ENGLISH.

Your task:
- Understand the Urdu context
- Answer the question in SIMPLE ENGLISH
- If partially relevant, explain based on Islamic meaning

Context:
{context_text}

Question:
{question}

Answer in English:
"""


    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    outputs = gen_model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------------- API ---------------------
@app.post("/ask")
def ask(query: str):
    # 1️⃣ FAISS search
    q_vec = embed_query(query)
    distances, indices = index.search(q_vec, TOP_K)

    retrieved_chunks = []
    response_chunks = []

    for idx in indices[0]:
        if idx == -1:
            continue
        chunk = chunks[idx]

        retrieved_chunks.append(chunk["text"])

        response_chunks.append({
            "text": chunk["text"],
            "video_title": chunk["title"],
            "youtube_url": chunk["play_url"],
            "start": chunk["start"],
            "end": chunk["end"]
        })

    # 2️⃣ Generate final answer
    answer = generate_answer(query, retrieved_chunks)

    return {
        "query": query,
        "answer": answer,
        "sources": response_chunks
    }
