"""
Semantic search interface for querying embedded PDF chunks.

Takes a plain English query, embeds it, and retrieves the most
relevant chunks from ChromaDB using cosine similarity.

Usage:
  python src/query.py --query "What is the refund policy?"
  python src/query.py --query "What is the refund policy?" --top-k 10
"""

import argparse
from typing import List, Dict, Any

from embed_data import embed_with_local
from store import get_collection, query_collection


def search(
    query: str,
    top_k: int = 5,
    model_name: str = "all-MiniLM-L6-v2",
    collection_name: str = "pdf_chunks",
) -> List[Dict[str, Any]]:
    """Embed a query and retrieve the top-k most relevant chunks.

    Args:
        query: plain English question or search phrase
        top_k: number of results to return
        model_name: must match the model used during ingestion
        collection_name: ChromaDB collection to search

    Returns:
        List of result dicts with keys: id, text, score, metadata
    """
    # Embed the query using the same model used to embed the chunks
    # If the models don't match, similarity scores will be meaningless
    query_embedding = embed_with_local([query], model_name=model_name)[0]

    collection = get_collection(collection_name=collection_name)
    results = query_collection(collection, query_embedding, top_k=top_k)

    return results


def print_results(results: List[Dict[str, Any]]) -> None:
    """Pretty-print search results to the terminal."""
    if not results:
        print("No results found.")
        return

    for i, hit in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print(f"ID:    {hit['id']}")
        print(f"Score: {hit['score']:.4f}  (lower = more similar)")
        if hit.get("metadata"):
            print(f"Meta:  {hit['metadata']}")
        print(f"\n{hit['text']}")


def main():
    parser = argparse.ArgumentParser(description="Query embedded PDF chunks")
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Plain English query to search for"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model (must match ingestion model)"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="pdf_chunks",
        help="ChromaDB collection name (default: pdf_chunks)"
    )
    args = parser.parse_args()

    results = search(
        query=args.query,
        top_k=args.top_k,
        model_name=args.model,
        collection_name=args.collection,
    )

    print_results(results)


if __name__ == "__main__":
    main()