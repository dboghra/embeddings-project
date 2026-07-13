"""
PDF->ingest->clean->chunk->embed->store (ChromaDB)

Usage:
  pip install pypdf sentence-transformers chromadb python-dotenv

  # Ingest a PDF into the vector store
  python scripts/pipeline.py --pdf data/easy.pdf

  # Tune chunking + wipe any existing chunks first (avoids duplicate ids)
  python scripts/pipeline.py --pdf data/easy.pdf --chunk-size 800 --overlap 100 --reset

  # Ingest, then immediately ask a question
  python scripts/pipeline.py --pdf data/easy.pdf --query "When is the yard sale?"
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from ingest import load_pdf
from clean_data import clean_pages
from chunk_file import chunk_pdf_pages, save_chunks_json, print_chunk_stats
from embed_data import embed_with_local
from store import get_collection, add_chunks, query_collection, clear_collection

load_dotenv()


def build_metadata(chunk_ids: List[str], source: str) -> List[Dict[str, Any]]:
    """Derive per-chunk metadata from the structured chunk ids.

    Chunk ids follow the pattern "<filename>-page-<n>-chunk-<m>", so we can
    recover the source file and page number without threading extra state
    through the pipeline. Anything we can't parse just falls back to source.

    
    chunk_ids: ids from chunk_file.chunk_pdf_pages
    source: the PDF filename (stored on every chunk for filtering later)

    Returns: List of metadata dicts aligned 1:1 with chunk_ids.
    """
    metadata: List[Dict[str, Any]] = []
    for chunk_id in chunk_ids:
        entry: Dict[str, Any] = {"source": source}
        parts = chunk_id.split("-")
        # ["<file>.pdf", "page", "<n>", "chunk", "<m>"]  (filename may itself
        # contain dashes, so index from the right)
        if len(parts) >= 4 and parts[-4] == "page" and parts[-2] == "chunk":
            try:
                entry["page"] = int(parts[-3])
                entry["chunk"] = int(parts[-1])
            except ValueError:
                pass
        metadata.append(entry)
    return metadata


def ingest(
    pdf_path: Path,
    chunk_size: int = 1000,
    overlap: int = 200,
    model_name: str = "all-MiniLM-L6-v2",
    collection_name: str = "pdf_chunks",
    reset: bool = False,
    save_chunks: bool = True,
    chunks_path: Path = Path("data/chunks.json"),
) -> int:
    """Run the full ingest pipeline for one PDF and write it to ChromaDB.

    Returns:
        The number of chunks stored.
    """
    # 1) Extract raw text, one string per page
    print(f"[1/5] Loading PDF: {pdf_path}")
    page_ids, page_texts = load_pdf(pdf_path)
    print(f"      Extracted {len(page_ids)} page(s)")

    # 2) Clean pages (fix hyphenation, dedupe, strip headers, drop empties).
    #    This is the step the standalone chunker was skipping.
    print("[2/5] Cleaning pages")
    clean_ids, clean_texts = clean_pages(page_ids, page_texts)
    if not clean_ids:
        raise SystemExit("No usable text left after cleaning — nothing to embed.")
    print(f"      {len(clean_ids)} page(s) survived cleaning")

    # 3) Split each page into overlapping chunks
    print(f"[3/5] Chunking (size={chunk_size}, overlap={overlap})")
    chunk_ids, chunk_texts = chunk_pdf_pages(
        clean_ids, clean_texts, chunk_size=chunk_size, overlap=overlap
    )
    print_chunk_stats(chunk_ids, chunk_texts)
    if save_chunks:
        save_chunks_json(chunk_ids, chunk_texts, chunks_path)

    # 4) Embed every chunk with the local Sentence-BERT model
    print(f"[4/5] Embedding {len(chunk_texts)} chunk(s) with '{model_name}'")
    embeddings = embed_with_local(chunk_texts, model_name=model_name)

    # 5) Store in ChromaDB
    print(f"[5/5] Storing in collection '{collection_name}'")
    collection = get_collection(collection_name=collection_name)
    if reset:
        clear_collection(collection)
    metadata = build_metadata(chunk_ids, source=pdf_path.name)
    add_chunks(collection, chunk_ids, chunk_texts, embeddings, metadata=metadata)

    print("\n✓ Ingest complete")
    return len(chunk_ids)


def run_query(
    query: str,
    top_k: int = 5,
    model_name: str = "all-MiniLM-L6-v2",
    collection_name: str = "pdf_chunks",
) -> List[Dict[str, Any]]:
    """Embed a query and print the top-k most similar chunks."""
    print(f"\nQuery: {query!r}")
    query_embedding = embed_with_local([query], model_name=model_name)[0]
    collection = get_collection(collection_name=collection_name)
    results = query_collection(collection, query_embedding, top_k=top_k)

    if not results:
        print("No results found.")
        return results

    for i, hit in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print(f"ID:    {hit['id']}")
        print(f"Score: {hit['score']:.4f}  (lower = more similar)")
        if hit.get("metadata"):
            print(f"Meta:  {hit['metadata']}")
        print(f"\n{hit['text']}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run the full PDF-RAG pipeline (ingest and optionally query)."
    )
    parser.add_argument("--pdf", type=Path, help="Path to the PDF to ingest.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Target chunk size in characters (default: 1000).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=200,
        help="Overlap between chunks in characters (default: 200).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model (query must reuse the same one).",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="pdf_chunks",
        help="ChromaDB collection name (default: pdf_chunks).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear the collection before adding (avoids duplicate-id errors on re-ingest).",
    )
    parser.add_argument(
        "--no-save-chunks",
        action="store_true",
        help="Skip writing the intermediate data/chunks.json debug file.",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Optional question to run against the collection after ingesting.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return for --query (default: 5).",
    )
    args = parser.parse_args()

    if not args.pdf and not args.query:
        parser.error("provide --pdf to ingest, --query to search, or both.")

    if args.chunk_size > 0 and args.overlap >= args.chunk_size:
        parser.error(
            f"overlap ({args.overlap}) must be < chunk_size ({args.chunk_size})."
        )

    if args.pdf:
        ingest(
            pdf_path=args.pdf,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            model_name=args.model,
            collection_name=args.collection,
            reset=args.reset,
            save_chunks=not args.no_save_chunks,
        )

    if args.query:
        run_query(
            query=args.query,
            top_k=args.top_k,
            model_name=args.model,
            collection_name=args.collection,
        )


if __name__ == "__main__":
    main()
