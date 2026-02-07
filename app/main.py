# main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from app.retrieval import VectorRetriever
from app.schemas import SearchRequest, SearchResponse

FAISS_PATH = "data/vector_store/faiss.index"
CHUNK_PATH = "data/vector_store/chunks.pkl"
MODEL_NAME = "intfloat/multilingual-e5-large"

app = FastAPI(
    title="Quran RAG Retrieval API",
    version="1.0.0",
    default_response_class=JSONResponse   #  ensures UTF-8 JSON globally
)

retriever: VectorRetriever | None = None

@app.on_event("startup")
def load_resources():
    global retriever
    retriever = VectorRetriever(
        faiss_path=FAISS_PATH,
        chunk_path=CHUNK_PATH,
        model_name=MODEL_NAME
    )

@app.post("/retrieve", response_model=SearchResponse)
def retrieve(req: SearchRequest):
    if not retriever:
        raise HTTPException(status_code=500, detail="Retriever not initialized")

    results = retriever.search(
        query=req.query,
        top_k=req.top_k
    )

    # Explicit JSONResponse with UTF-8 charset
    return JSONResponse(
        content={"query": req.query, "results": results},
        media_type="application/json; charset=utf-8"
    )
