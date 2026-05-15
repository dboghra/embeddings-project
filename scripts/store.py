"""
ChromaDB interface for storing and retrieving chunk embeddings.

Handles:
  - Creating / loading a persistent ChromaDB collection
  - Adding chunks with their embeddings and metadata
  - Querying by embedding vector to retrieve top-k similar chunks

Usage:
  pip install chromadb
"""

from pathlib import Path
from typing import List, Dict, Any

import chromadb


# Where ChromaDB persists its data to disk
DB_PATH = Path("data/chroma_db")


def get_collection(
    collection_name: str = "pdf_chunks",
    db_path: Path = DB_PATH,
) -> chromadb.Collection:
    """Load or create a persistent ChromaDB collection.

    ChromaDB will create the directory if it doesn't exist.
    If the collection already exists, it loads it — so this is
    safe to call multiple times without duplicating data.

    Args:
        collection_name: name for the collection
        db_path: where to persist the database on disk

    Returns:
        A ChromaDB Collection object
    """
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},  # use cosine similarity for ranking
    )
    return collection


def add_chunks(
    collection: chromadb.Collection,
    ids: List[str],
    texts: List[str],
    embeddings: List[List[float]],
    metadata: List[Dict[str, Any]] = None,
) -> None:
    """Store chunks, their embeddings, and optional metadata in ChromaDB.

    ChromaDB requires that ids are unique. If you re-run ingestion on
    the same PDF, call clear_collection() first to avoid duplicates.

    Args:
        collection: ChromaDB collection to write to
        ids: unique chunk ids (e.g. "myfile.pdf-page-2-chunk-1")
        texts: the raw chunk text strings (stored as documents)
        embeddings: embedding vectors from embed_data.embed_with_local()
        metadata: optional list of dicts with extra info per chunk
                  (e.g. source filename, page number)
    """
    if not ids:
        print("No chunks to add.")
        return

    if metadata is None:
        metadata = [{} for _ in ids]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadata,
    )
    print(f"Added {len(ids)} chunks to collection '{collection.name}'")


def query_collection(
    collection: chromadb.Collection,
    query_embedding: List[float],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Retrieve the top-k most similar chunks for a query embedding.

    Args:
        collection: ChromaDB collection to search
        query_embedding: embedding vector of the query text
        top_k: number of results to return

    Returns:
        List of dicts, each with keys:
          - id: chunk id
          - text: chunk text
          - score: cosine distance (lower = more similar)
          - metadata: dict of any stored metadata
    """
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances", "metadatas"],
    )

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "score": results["distances"][0][i],
            "metadata": results["metadatas"][0][i],
        })

    return hits


def clear_collection(collection: chromadb.Collection) -> None:
    """Delete all chunks from a collection.

    Useful when re-ingesting a PDF to avoid duplicate entries.
    """
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        print(f"Cleared {len(existing['ids'])} chunks from '{collection.name}'")
    else:
        print("Collection already empty.")