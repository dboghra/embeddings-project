"""
Create embeddings chunks

Usage:
  pip install sentence-transformers
"""

from pathlib import Path
from typing import List

try:
    from sentence_transformers import SentenceTransformer  # optional fallback
except Exception:
    SentenceTransformer = None


def embed_with_local(texts: List[str], model_name: str = "all-MiniLM-L6-v2", batch_size: int = 64) -> List[List[float]]:
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed (pip install sentence-transformers)")
    model = SentenceTransformer(model_name)
    arr = model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
    return arr.tolist()

