# app/config.py

import os
from pathlib import Path

class Config:
    # === Paths ===
    ROOT_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = ROOT_DIR / "data"
    RAW_DOCS_DIR = DATA_DIR / "raw"            # where the PDFs is / text sources
    INDEX_DIR = ROOT_DIR / "index"
    FAISS_DIR = INDEX_DIR / "faiss"
    META_PATH = INDEX_DIR / "meta.json"        # metadata (article_id, source, etc.)

    # directories 
    os.makedirs(RAW_DOCS_DIR, exist_ok=True)
    os.makedirs(FAISS_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)

    # === Embedding model for dense retrieval ===
    # multilingual, good for Arabic/English
    EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

    # top K from vector search before rerank
    CANDIDATE_K = 20
    # final K after rerank
    FINAL_K = 5

    # === Reranker model ===
    # multilingual cross-encoder reranker
    RERANKER_MODEL_NAME = "jinaai/jina-reranker-v2-base-multilingual"

    # === Generation model (LLM) ===
    #  Qwen instruct model fits in GPU/RAM
    QWEN_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

    # quantization / device config
    LOAD_IN_4BIT = False # use 4-bit quant to save VRAM
    DEVICE_MAP = {"": "cpu"}  # transformers will decide gpu/cpu placement
    MAX_NEW_TOKENS = 512
    TEMPERATURE = 0.2    # more deterministic since it's law

    # === Chunking strategy ===
    # We split by article, not by sliding windows, so we don't need token chunk size.
    # We still keep max length sanity just in case.
    MAX_CHUNK_CHARS = 4000  # safety cutoff if an article is insanely long

CONFIG = Config()

