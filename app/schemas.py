# app/schemas.py

from pydantic import BaseModel
from typing import List, Optional

class ArticleChunk(BaseModel):
    article_id: str          # e.g. "المادة 37"
    text: str                # full article text or relevant passage
    source: str              # e.g. "Saudi Labor Law 2023 PDF"
    score: Optional[float]   # relevance score after rerank

class AskRequest(BaseModel):
    question: str            # user question in Arabic or English

class AskResponse(BaseModel):
    answer: str              # final LLM answer (Arabic summary, English note, etc.)
    citations: List[ArticleChunk]

# internal helper model for reranker input/output if needed
class ScoredCandidate(BaseModel):
    text: str
    article_id: str
    source: str
    dense_score: float
    rerank_score: Optional[float] = None
