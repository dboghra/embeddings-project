"""
Create embeddings for Trello cards.

Usage:
  pip install openai python-dotenv numpy sentence-transformers
  Add OPENAI_API_KEY to .env to use OpenAI; otherwise the script will use sentence-transformers.
  python scripts/embed_cards.py
"""
import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from dotenv import load_dotenv

load_dotenv()

try:
    from sentence_transformers import SentenceTransformer  # optional fallback
except Exception:
    SentenceTransformer = None


INPUT_CARDS = Path("data/cleaned_cards.json")
FALLBACK_CARDS = Path("data/cards.json")
OUT_EMBED_JSON = Path("data/embeddings.json")
OUT_EMBED_NPY = Path("data/embeddings.npy")
OUT_IDS = Path("data/embedding_ids.json")


def load_cards(path: Path = INPUT_CARDS) -> List[dict]:
    if path.exists():
        p = path
    elif FALLBACK_CARDS.exists():
        p = FALLBACK_CARDS
    else:
        raise SystemExit("No input cards file found (data/cleaned_cards.json or data/cards.json).")
    
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def compose_text(card: dict) -> str:
    parts = [card.get("name", ""), card.get("desc", "")]
    labels = ", ".join([lbl.get("name") for lbl in card.get("labels", []) if lbl.get("name")])
    if labels:
        parts.append(f"Labels: {labels}")
    members = ", ".join([m.get("fullName") for m in card.get("members", []) if m.get("fullName")])
    if members:
        parts.append(f"Members: {members}")
    short = card.get("shortUrl")
    if short:
        parts.append(f"Source: {short}")
    return "\n\n".join([p.strip() for p in parts if p and p.strip()])

batch_size = 64


def embed_with_local(texts: List[str], model_name: str = "all-MiniLM-L6-v2", batch_size: int = 64) -> List[List[float]]:
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed (pip install sentence-transformers)")
    model = SentenceTransformer(model_name)
    arr = model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
    return arr.tolist()


def save_embeddings(ids: List[str], texts: List[str], embeddings: List[List[float]]) -> None:
    OUT_EMBED_JSON.parent.mkdir(parents=True, exist_ok=True)
    records = [{"id": _id, "text": txt, "embedding": emb} for _id, txt, emb in zip(ids, texts, embeddings)]
    with OUT_EMBED_JSON.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    np.save(OUT_EMBED_NPY, np.array(embeddings, dtype=np.float32))
    with OUT_IDS.open("w", encoding="utf-8") as f:
        json.dump(ids, f)
    print(f"Wrote {len(ids)} embeddings to {OUT_EMBED_JSON} and {OUT_EMBED_NPY}")


def simple_search(query: str, top_k: int = 5, model: str = "openai") -> List[dict]:
    # load stored embeddings + ids and return top-k by cosine similarity
    if not OUT_EMBED_NPY.exists() or not OUT_IDS.exists():
        raise SystemExit("Run the embedding script to produce data/embeddings.npy and data/embedding_ids.json")
    embs = np.load(OUT_EMBED_NPY)
    with OUT_IDS.open("r", encoding="utf-8") as f:
        ids = json.load(f)
    if model == "openai":
        q_emb = embed_with_openai([query])[0]
    else:
        q_emb = embed_with_local([query])[0]
    q = np.array(q_emb, dtype=np.float32)
    dot = embs @ q
    embs_norm = np.linalg.norm(embs, axis=1)
    q_norm = np.linalg.norm(q)
    scores = dot / (embs_norm * q_norm + 1e-12)
    idxs = np.argsort(-scores)[:top_k]
    return [{"id": ids[i], "score": float(scores[i])} for i in idxs]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["openai", "local", "auto"], default="auto",
                   help="Embedding backend: openai, local (sentence-transformers), or auto (prefers OpenAI if key present)")
    #p.add_argument("--openai-model", default="text-embedding-3-small")
    p.add_argument("--local-model", default="all-MiniLM-L6-v2")
    p.add_argument("--batch", type=int, default=64)
    args = p.parse_args()

    cards = load_cards()
    
    ids = [c["id"] for c in cards]
    
    texts = [compose_text(c) for c in cards]
    #print(texts)

    embeddings = embed_with_local(texts, model_name="all-MiniLM-L6-v2", batch_size=64)

    save_embeddings(ids, texts, embeddings)

    simple_search("Can you find any duplicates cards?", 5, "all-MiniLM-L6-v2")


if __name__ == "__main__":
    main()
