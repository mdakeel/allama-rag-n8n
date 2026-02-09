import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from app.retrieval import VectorRetriever
from app.schemas import SearchRequest, SearchResponse

FAISS_URL = "https://drive.google.com/uc?export=download&id=1rmVnQWDCwv8u0XpZautijetlDpKn1LBr"
CHUNKS_URL = "https://drive.google.com/uc?export=download&id=1DBocNIeO5nhxDwPpEAVOR55iyCqkqaRs"

FAISS_PATH = "/tmp/faiss.index"
CHUNK_PATH = "/tmp/chunks.pkl"

MODEL_NAME = "intfloat/multilingual-e5-large"
HF_TOKEN = os.getenv("HF_TOKEN")

app = FastAPI(title="Quran RAG Retrieval API", version="1.0.0", default_response_class=JSONResponse)

retriever: VectorRetriever | None = None

def download_file(url: str, path: str):
    r = requests.get(url)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)

@app.on_event("startup")
def load_resources():
    global retriever
    download_file(FAISS_URL, FAISS_PATH)
    download_file(CHUNKS_URL, CHUNK_PATH)
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
