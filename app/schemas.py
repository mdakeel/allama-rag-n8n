# schemas.py
from pydantic import BaseModel
from typing import List

class SearchRequest(BaseModel):
    query: str
    top_k: int = 20

class SourceChunk(BaseModel):
    text: str
    video_title: str
    youtube_url: str
    start: str
    end: str

class SearchResponse(BaseModel):
    query: str
    results: List[SourceChunk]
