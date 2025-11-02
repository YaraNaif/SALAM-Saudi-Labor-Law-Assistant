# app/build_index.py

import json
from pathlib import Path
from typing import List, Dict, Tuple
import re

import faiss
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import fitz  # PyMuPDF

from app.config import CONFIG


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract all text from the English PDF and return one big string.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    pages_text = []
    for page in doc:
        pages_text.append(page.get_text("text"))
    doc.close()

    full_text = "\n".join(pages_text)

    # normalize spacing a bit
    full_text = re.sub(r"[ \t]+", " ", full_text)
    full_text = re.sub(r"\n\s+\n", "\n\n", full_text)

    return full_text.strip()


def split_into_articles(full_text: str) -> List[Dict]:
    """
    Parse English-only labor law PDF into article chunks.

    We assume headers like:
      "Article 116:"  OR  "Article (116)"  OR  "Article 116 -"

    We will:
      - detect each header
      - group lines until the next header
      - store as {article_id, text, lang="en", source="Labor.pdf"}
    """

    ARTICLE_PATTERNS = [
        r"^(Article\s+\d+)\s*[:：\-)]",
        r"^(Article\s*\(?\d+\)?)",
    ]
    header_regex = re.compile("|".join(ARTICLE_PATTERNS),
                              re.IGNORECASE | re.MULTILINE)

    lines = full_text.splitlines()

    articles_raw: List[Tuple[str, List[str]]] = []
    current_article_id = None
    current_buffer: List[str] = []

    def push_current():
        nonlocal current_article_id, current_buffer, articles_raw
        if current_article_id and current_buffer:
            body_text = "\n".join(current_buffer).strip()
            if body_text:
                body_text = body_text[: CONFIG.MAX_CHUNK_CHARS]
                articles_raw.append((current_article_id, body_text))
        current_article_id = None
        current_buffer = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        m = header_regex.match(line)
        if m:
            # start of a new article block
            push_current()

            # pick whichever group matched
            for g in m.groups():
                if g:
                    current_article_id = g.strip()
                    break

            remainder = line[m.end():].strip()
            current_buffer = []
            if remainder:
                current_buffer.append(remainder)
        else:
            if current_article_id is not None:
                current_buffer.append(line)

    push_current()

    structured: List[Dict] = []
    for art_id, art_text in articles_raw:
        structured.append({
            "article_id": art_id,      # e.g. "Article 116"
            "text": art_text,          # full English text
            "lang": "en",
            "source": "Labor.pdf"
        })
    return structured


def build_embeddings(articles: List[Dict]) -> np.ndarray:
    """
    Encode article text using bge-m3 and return embeddings.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(CONFIG.EMBEDDING_MODEL_NAME, device=device)

    texts = [a["text"] for a in articles]
    embeddings = model.encode(
        texts,
        batch_size=16,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine-ready
    )

    return embeddings


def save_faiss_index(embeddings: np.ndarray, articles: List[Dict]) -> None:
    """
    Save FAISS index and metadata.
    We use IndexFlatIP with normalized embeddings for cosine similarity.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    faiss.write_index(index, str(CONFIG.FAISS_DIR / "index.faiss"))

    with open(CONFIG.META_PATH, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)


def main():
    """
    1. Extract English text from Labor.pdf
    2. Split into articles
    3. Embed English only
    4. Save FAISS + meta.json

    After this:
      - index/faiss/index.faiss
      - index/meta.json
    """

    pdf_path = CONFIG.RAW_DOCS_DIR / "Labor.pdf"

    print("[1/4] Extracting text from PDF ...")
    full_text = extract_text_from_pdf(pdf_path)
    print(f"   Extracted {len(full_text)} characters.")

    print("[2/4] Splitting into English articles ...")
    english_articles = split_into_articles(full_text)
    print(f"   Found {len(english_articles)} English articles.")

    if not english_articles:
        raise RuntimeError(
            "No articles were detected from the PDF. Adjust split_into_articles() regex."
        )

    print("[3/4] Building embeddings ...")
    embeddings = build_embeddings(english_articles)
    print(f"   Embeddings shape: {embeddings.shape}")

    print("[4/4] Saving FAISS index and metadata ...")
    save_faiss_index(embeddings, english_articles)

    print("✅ Index build complete.")
    print(f"   Saved index to: {CONFIG.FAISS_DIR / 'index.faiss'}")
    print(f"   Saved meta to:  {CONFIG.META_PATH}")


if __name__ == "__main__":
    main()
