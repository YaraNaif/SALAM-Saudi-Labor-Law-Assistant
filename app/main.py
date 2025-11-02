# app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import AskRequest, AskResponse, ArticleChunk
from app.retriever import Retriever
from app.llm_engine import LLMEngine

app = FastAPI(
    title="SALAM Labor Law Assistant",
    description="Ask questions about Saudi labor law and get cited answers.",
    version="2.0.0",
)

# CORS so HTML / frontend can call it from localhost:8501 etc.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten later if you deploy
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# global singletons (loaded once)
retriever = Retriever()
llm = LLMEngine()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """
    1. Retrieve top law articles
    2. Generate structured legal answer w/ citations
    3. Return both
    """
    context_chunks: list[ArticleChunk] = retriever.get_context(req.question)
    answer_text: str = llm.generate_answer(req.question, context_chunks)

    return AskResponse(
        answer=answer_text,
        citations=context_chunks
    )
