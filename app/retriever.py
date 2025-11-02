# app/retriever.py

import json
from pathlib import Path
from typing import List

import faiss
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from app.config import CONFIG
from app.schemas import ArticleChunk, ScoredCandidate


class DenseRetriever:
    """
    Loads:
    - FAISS index of article embeddings
    - metadata for each vector
    Also keeps an embedding model to embed new queries.
    """

    def __init__(self):
        # load metadata
        with open(CONFIG.META_PATH, "r", encoding="utf-8") as f:
            self.articles = json.load(f)
        # load faiss
        index_path = CONFIG.FAISS_DIR / "index.faiss"
        if not index_path.exists():
            raise RuntimeError("FAISS index not found. Run build_index.py first.")
        self.index = faiss.read_index(str(index_path))

        # load embedding model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embed_model = SentenceTransformer(CONFIG.EMBEDDING_MODEL_NAME, device=device)

    def search(self, query: str, top_k: int) -> List[ScoredCandidate]:
        """
        1. Embed the query
        2. Nearest neighbors from FAISS
        3. Return candidates with dense_score
        """

        q_emb = self.embed_model.encode(
            [query],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )  # shape (1, dim)

        q_emb = q_emb.astype(np.float32)

        scores, idxs = self.index.search(q_emb, top_k)
        # scores: shape (1, top_k)
        # idxs:   shape (1, top_k)

        out: List[ScoredCandidate] = []
        for score, idx in zip(scores[0], idxs[0]):
            art = self.articles[idx]
            out.append(
                ScoredCandidate(
                    text=art["text"],
                    article_id=art["article_id"],
                    source=art["source"],
                    dense_score=float(score),
                )
            )
        return out


class Reranker:
    """
    Cross-encoder reranker.
    Scores (query, passage) pairs and gives you the most relevant.
    """

    def __init__(self):
        # trust_remote_code=True is needed for some Jina models.
        # This was a previous pain point, so we set it explicitly here.
        self.tokenizer = AutoTokenizer.from_pretrained(
            CONFIG.RERANKER_MODEL_NAME,
            trust_remote_code=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            CONFIG.RERANKER_MODEL_NAME,
            trust_remote_code=True
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.device = device
        self.model.eval()

    @torch.inference_mode()
    def rerank(self, query: str, candidates: List[ScoredCandidate]) -> List[ScoredCandidate]:
        """
        Returns same list but with rerank_score set and sorted descending by rerank_score.
        We'll batch for speed.
        """

        texts_a = [query] * len(candidates)
        texts_b = [c.text for c in candidates]

        enc = self.tokenizer(
            texts_a,
            texts_b,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        logits = self.model(**enc).logits  # shape (N, 1) or (N,) depending on model
        scores = logits.squeeze(-1).detach().cpu().tolist()

        # attach rerank score
        for cand, sc in zip(candidates, scores):
            cand.rerank_score = float(sc)

        # sort by rerank descending
        candidates_sorted = sorted(
            candidates,
            key=lambda x: x.rerank_score if x.rerank_score is not None else -1e9,
            reverse=True
        )
        return candidates_sorted


class Retriever:
    """
    High-level helper:
    - dense search topK
    - rerank
    - return final topK chunks for LLM
    """

    def __init__(self):
        self.dense = DenseRetriever()
        self.reranker = Reranker()

    def get_context(self, query: str) -> List[ArticleChunk]:
        # 1. dense retrieve
        cands = self.dense.search(query, CONFIG.CANDIDATE_K)

        # 2. rerank
        reranked = self.reranker.rerank(query, cands)

        # 3. take FINAL_K best
        best = reranked[: CONFIG.FINAL_K]

        # 4. convert to ArticleChunk for downstream usage
        chunks: List[ArticleChunk] = []
        for b in best:
            chunks.append(
                ArticleChunk(
                    article_id=b.article_id,
                    text=b.text,
                    source=b.source,
                    score=b.rerank_score
                )
            )
        return chunks