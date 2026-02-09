import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from huggingface_hub import hf_hub_download
from app.retrieval import VectorRetriever
from app.schemas import SearchRequest, SearchResponse

# HuggingFace dataset repo (तुम्हारा repo ID)
REPO_ID = "aakiltayyab/quran-faiss-index"

# Model name (same as FAISS embeddings)
MODEL_NAME = "intfloat/multilingual-e5-large"

# HuggingFace token (optional if repo is public)
HF_TOKEN = os.getenv("HF_TOKEN")

app = FastAPI(
    title="Quran RAG Retrieval API",
    version="1.0.0",
    default_response_class=JSONResponse
)

retriever: VectorRetriever | None = None

@app.on_event("startup")
def load_resources():
    global retriever

    # HuggingFace Hub से files download करो
    FAISS_PATH = hf_hub_download(
        repo_id=REPO_ID,
        filename="faiss.index",
        repo_type="dataset"
    )

    CHUNK_PATH = hf_hub_download(
        repo_id=REPO_ID,
        filename="chunks.pkl",
        repo_type="dataset"
    )

    # Retriever initialize करो
    retriever = VectorRetriever(
        faiss_path=FAISS_PATH,
        chunk_path=CHUNK_PATH,
        model_name=MODEL_NAME,
        hf_token=HF_TOKEN
    )

@app.post("/retrieve", response_model=SearchResponse)
def retrieve(req: SearchRequest):
    if not retriever:
        raise HTTPException(status_code=500, detail="Retriever not initialized")

    results = retriever.search(query=req.query, top_k=req.top_k)
    return {"query": req.query, "results": results}
