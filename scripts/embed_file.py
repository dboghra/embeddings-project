"""
Create embeddings for Trello cards or a PDF file.

Usage:
    pip install python-dotenv numpy sentence-transformers pypdf
    Add OPENAI_API_KEY to .env to use OpenAI; otherwise the script will use sentence-transformers.
    python scripts/embed_data.py --pdf data/myfile.pdf
"""
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from dotenv import load_dotenv

load_dotenv()

try:
    from sentence_transformers import SentenceTransformer  # optional fallback
except Exception:
    SentenceTransformer = None

try:
    # pypdf is a lightweight PDF reader
    from pypdf import PdfReader
except Exception:
    PdfReader = None


OUT_EMBED_NPY = Path("data/embeddings.npy")
OUT_IDS = Path("data/embedding_ids.json")


def load_pdf(path: Path) -> Tuple[List[str], List[str]]:
    """Return (ids, texts) for each page of the PDF.

    If `pypdf` is not installed the function will raise a RuntimeError prompting
    the user to install the package.
    """
    if PdfReader is None:
        raise RuntimeError("pypdf is required to read PDFs (pip install pypdf)")
    if not path.exists():
        raise SystemExit(f"PDF file not found: {path}")
    reader = PdfReader(str(path))
    ids: List[str] = []
    texts: List[str] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        # Use file name + page number as an id
        ids.append(f"{path.name}-page-{i+1}")
        texts.append(text.strip())
    return ids, texts


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Chunk a single string into a list of chunks by characters.

    If `chunk_size` <= 0, returns the original text as a single chunk.
    """
    if chunk_size <= 0:
        return [text]
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks: List[str] = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == L:
            break
        start = max(end - overlap, end) if overlap > 0 else end
    return chunks


def chunk_pdf_pages(ids: List[str], texts: List[str], chunk_size: int = 1000, overlap: int = 200) -> Tuple[List[str], List[str]]:
    """Chunk each page text into chunks and return new ids and texts.

    New ids will be of the form `<filename>-page-<n>-chunk-<m>`.
    """
    new_ids: List[str] = []
    new_texts: List[str] = []
    for pid, text in zip(ids, texts):
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for j, ch in enumerate(chunks, start=1):
            new_ids.append(f"{pid}-chunk-{j}")
            new_texts.append(ch)
    return new_ids, new_texts


# Removed compose_text (card formatting) — not needed for PDF-only workflow

batch_size = 64


def embed_with_local(texts: List[str], model_name: str = "all-MiniLM-L6-v2", batch_size: int = 64) -> List[List[float]]:
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed (pip install sentence-transformers)")
    model = SentenceTransformer(model_name)
    arr = model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
    return arr.tolist()


def embed_texts(texts: List[str], model_name: str = "all-MiniLM-L6-v2", batch_size: int = 64) -> List[List[float]]:
    """Wrapper for embedding a list of texts. Separated for testability.

    Returns a list of embedding vectors.
    """
    return embed_with_local(texts, model_name=model_name, batch_size=batch_size)


def save_embeddings(ids: List[str], texts: List[str], embeddings: List[List[float]]) -> None:
    # Deprecated: per-record JSON output removed. Save binary embeddings and ids only.
    OUT_EMBED_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_EMBED_NPY, np.array(embeddings, dtype=np.float32))
    with OUT_IDS.open("w", encoding="utf-8") as f:
        json.dump(ids, f)
    print(f"Wrote {len(ids)} embeddings to {OUT_EMBED_NPY} and ids to {OUT_IDS}")


# simple_search removed to keep script focused on PDF -> embeddings pipeline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--local-model", default="all-MiniLM-L6-v2")
    p.add_argument("--pdf", type=Path, required=True, help="Path to a PDF file to embed (required)")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--chunk-size", type=int, default=1000, help="Chunk size in characters; set <=0 to use pages as units")
    p.add_argument("--overlap", type=int, default=200, help="Overlap in characters between chunks")
    args = p.parse_args()
    ids, texts = load_pdf(args.pdf)
    if not texts:
        raise SystemExit("No text extracted from PDF; nothing to embed.")

    # chunk pages into smaller pieces if requested
    if args.chunk_size > 0:
        ids, texts = chunk_pdf_pages(ids, texts, chunk_size=args.chunk_size, overlap=args.overlap)

    embeddings = embed_texts(texts, model_name=args.local_model, batch_size=args.batch)

    save_embeddings(ids, texts, embeddings)


if __name__ == "__main__":
    main()
