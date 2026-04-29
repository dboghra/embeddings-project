"""
Process PDF files and chunk text for embedding.

This script extracts text from PDF files and chunks it according to configurable parameters.
Chunks are stored with their metadata for further processing (e.g., embedding).

Usage:
  pip install pypdf python-dotenv
  python scripts/chunk_file.py --pdf data/myfile.pdf
  python scripts/chunk_file.py --pdf data/myfile.pdf --chunk-size 800 --overlap 100
"""
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional

from dotenv import load_dotenv

load_dotenv()

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

#why are we returning a tuple? tuple is ordered, unchangeable and duplicats OK
def load_pdf(path: Path) -> Tuple[List[str], List[str]]:
    """Extract text from each page of a PDF.
    
    Args:
        path: Path to the PDF file
        
    Returns:
        Tuple of (page_ids, page_texts) where:
        - page_ids: list of strings like "filename-page-1", "filename-page-2", etc.
        - page_texts: list of extracted text from each page
        
    Raises:
        RuntimeError: if pypdf is not installed
        SystemExit: if PDF file not found
    """
    if PdfReader is None: #what does pfdfreader do? 
        raise RuntimeError("pypdf is required to read PDFs (pip install pypdf)")
    if not path.exists():
        raise SystemExit(f"PDF file not found: {path}")
    
    reader = PdfReader(str(path)) 
    ids: List[str] = []
    texts: List[str] = [] #why does ids and texts need to be immutable?
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        ids.append(f"{path.name}-page-{i+1}")
        texts.append(text.strip())
    
    return ids, texts

#why overlapping chunkcs by character count?
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Chunk a single string into overlapping chunks by character count.
    
    Args:
        text: source text to chunk
        chunk_size: maximum characters per chunk (if <= 0, returns [text])
        overlap: number of characters to overlap between chunks
        
    Returns:
        List of chunk strings. Empty chunks are filtered out.
        
    Raises:
        ValueError: if overlap >= chunk_size 
    """
    if chunk_size <= 0:
        return [text] if text.strip() else []
    
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    
    # why a list was chosen here? ordered and changeable duplicates ok
    start = 0
    L = len(text)
    
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end].strip() #strip removes leading and trailing whitespace
        
        if chunk:
            chunks.append(chunk)
        
        if end == L:
            break
        
        # Move start position: either by (chunk_size - overlap) or to chunk_size
        # this ensures we don't get stuck if overlap is 0
        # what does move start position mean?  It means advancing the starting index for the next chunk
        #why are we advancing the starting index for the next chunk?
        start = max(end - overlap, end) if overlap > 0 else end
    
    return chunks


def chunk_pdf_pages(
    ids: List[str], 
    texts: List[str], 
    chunk_size: int = 1000, 
    overlap: int = 200
) -> Tuple[List[str], List[str]]:
    """Chunk each PDF page into smaller chunks and assign unique ids.
    
    Args:
        ids: list of page identifiers (e.g., ["file.pdf-page-1", "file.pdf-page-2"])
        texts: list of page texts
        chunk_size: target characters per chunk
        overlap: overlap between chunks in characters
        
    Returns:
        Tuple of (chunk_ids, chunk_texts) where chunk ids follow the pattern:
        "<filename>-page-<n>-chunk-<m>"
    """
    # could i use a hashmap for this?
    new_ids: List[str] = []
    new_texts: List[str] = []
    
    for page_id, text in zip(ids, texts):
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for chunk_idx, chunk in enumerate(chunks, start=1):
            chunk_id = f"{page_id}-chunk-{chunk_idx}"
            new_ids.append(chunk_id)
            new_texts.append(chunk)
    
    return new_ids, new_texts


def save_chunks_json(ids: List[str], texts: List[str], output_path: Path) -> None:
    """Save chunks to a JSON file for inspection/debugging.
    
    Args:
        ids: list of chunk identifiers
        texts: list of chunk texts
        output_path: where to save the JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    chunks_data = [
        {"id": chunk_id, "text": chunk_text}
        for chunk_id, chunk_text in zip(ids, texts)
    ]
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(chunks_data)} chunks to {output_path}")


def print_chunk_stats(ids: List[str], texts: List[str]) -> None:
    """Print summary statistics about the chunks."""
    if not texts:
        print("No chunks to display.")
        return
    
    chunk_sizes = [len(t) for t in texts]
    total_chars = sum(chunk_sizes)
    
    print(f"\n=== Chunking Statistics ===")
    print(f"Total chunks: {len(texts)}")
    print(f"Total characters: {total_chars:,}")
    print(f"Min chunk size: {min(chunk_sizes):,} chars")
    print(f"Max chunk size: {max(chunk_sizes):,} chars")
    print(f"Avg chunk size: {total_chars / len(texts):.0f} chars")
    print(f"\nFirst 3 chunk IDs:")
    for chunk_id in ids[:3]:
        print(f"  - {chunk_id}")


def main():
    parser = argparse.ArgumentParser(description="Process PDF and chunk text for embeddings")
    parser.add_argument(
        "--pdf",
        type=Path,
        required=True,
        help="Path to PDF file to process"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Target chunk size in characters (default: 1000). Set to <=0 to disable chunking."
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=200,
        help="Overlap between chunks in characters (default: 200)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/chunks.json"),
        help="Output JSON file to save chunks (default: data/chunks.json)"
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Skip saving chunks to JSON file"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.chunk_size > 0 and args.overlap >= args.chunk_size:
        raise ValueError(f"overlap ({args.overlap}) must be < chunk_size ({args.chunk_size})")
    
    # Load PDF
    print(f"Loading PDF: {args.pdf}")
    page_ids, page_texts = load_pdf(args.pdf)
    print(f"Extracted {len(page_ids)} pages")
    
    # Chunk pages
    if args.chunk_size > 0:
        print(f"Chunking with size={args.chunk_size}, overlap={args.overlap}")
        chunk_ids, chunk_texts = chunk_pdf_pages(
            page_ids, page_texts,
            chunk_size=args.chunk_size,
            overlap=args.overlap
        )
    else:
        print("Chunking disabled (using pages as units)")
        chunk_ids, chunk_texts = page_ids, page_texts
    
    # Display stats
    print_chunk_stats(chunk_ids, chunk_texts)
    
    # Save chunks if requested
    if not args.no_output:
        save_chunks_json(chunk_ids, chunk_texts, args.output)
    
    print("\n✓ Processing complete")


if __name__ == "__main__":
    main()
















# ----------------------------------------------------------------------------------------------------------------
# """
# Process PDF files and chunk text for embedding.

# This script extracts text from PDF files and chunks it according to configurable parameters.
# Chunks are stored with their metadata for further processing (e.g., embedding).

# Usage:
#   pip install pypdf python-dotenv
#   python scripts/chunk_file.py --pdf data/myfile.pdf
#   python scripts/chunk_file.py --pdf data/myfile.pdf --chunk-size 800 --overlap 100
# """
# import argparse
# import json
# from pathlib import Path
# from typing import List, Tuple, Optional

# from dotenv import load_dotenv

# load_dotenv()

# try:
#     from pypdf import PdfReader
# except Exception:
#     PdfReader = None

# #why are we returning a tuple? tuple is ordered, unchangeable and duplicats OK
# def load_pdf(path: Path) -> Tuple[List[str], List[str]]:
#     """Extract text from each page of a PDF.
    
#     Args:
#         path: Path to the PDF file
        
#     Returns:
#         Tuple of (page_ids, page_texts) where:
#         - page_ids: list of strings like "filename-page-1", "filename-page-2", etc.
#         - page_texts: list of extracted text from each page
        
#     Raises:
#         RuntimeError: if pypdf is not installed
#         SystemExit: if PDF file not found
#     """
#     if PdfReader is None: #what does pfdfreader do? 
#         raise RuntimeError("pypdf is required to read PDFs (pip install pypdf)")
#     if not path.exists():
#         raise SystemExit(f"PDF file not found: {path}")
    
#     reader = PdfReader(str(path)) 
#     ids: List[str] = []
#     texts: List[str] = [] #why does ids and texts need to be immutable?
    
#     for i, page in enumerate(reader.pages):
#         text = page.extract_text() or ""
#         ids.append(f"{path.name}-page-{i+1}")
#         texts.append(text.strip())
    
#     return ids, texts

# #why overlapping chunkcs by character count?
# def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
#     """Chunk a single string into overlapping chunks by character count.
    
#     Args:
#         text: source text to chunk
#         chunk_size: maximum characters per chunk (if <= 0, returns [text])
#         overlap: number of characters to overlap between chunks
        
#     Returns:
#         List of chunk strings. Empty chunks are filtered out.
        
#     Raises:
#         ValueError: if overlap >= chunk_size 
#     """
#     if chunk_size <= 0:
#         return [text] if text.strip() else []
    
#     if overlap >= chunk_size:
#         raise ValueError("overlap must be smaller than chunk_size")
    
#     # why a list was chosen here? ordered and changeable duplicates ok
#     start = 0
#     L = len(text)
    
#     while start < L:
#         end = min(start + chunk_size, L)
#         chunk = text[start:end].strip() #strip removes leading and trailing whitespace
        
#         if chunk:
#             chunks.append(chunk)
        
#         if end == L:
#             break
        
#         # Move start position: either by (chunk_size - overlap) or to chunk_size
#         # this ensures we don't get stuck if overlap is 0
#         # what does move start position mean?  It means advancing the starting index for the next chunk
#         #why are we advancing the starting index for the next chunk?
#         start = max(end - overlap, end) if overlap > 0 else end
    
#     return chunks


# def chunk_pdf_pages(
#     ids: List[str], 
#     texts: List[str], 
#     chunk_size: int = 1000, 
#     overlap: int = 200
# ) -> Tuple[List[str], List[str]]:
#     """Chunk each PDF page into smaller chunks and assign unique ids.
    
#     Args:
#         ids: list of page identifiers (e.g., ["file.pdf-page-1", "file.pdf-page-2"])
#         texts: list of page texts
#         chunk_size: target characters per chunk
#         overlap: overlap between chunks in characters
        
#     Returns:
#         Tuple of (chunk_ids, chunk_texts) where chunk ids follow the pattern:
#         "<filename>-page-<n>-chunk-<m>"
#     """
#     # could i use a hashmap for this?
#     new_ids: List[str] = []
#     new_texts: List[str] = []
    
#     for page_id, text in zip(ids, texts):
#         chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
#         for chunk_idx, chunk in enumerate(chunks, start=1):
#             chunk_id = f"{page_id}-chunk-{chunk_idx}"
#             new_ids.append(chunk_id)
#             new_texts.append(chunk)
    
#     return new_ids, new_texts


# def save_chunks_json(ids: List[str], texts: List[str], output_path: Path) -> None:
#     """Save chunks to a JSON file for inspection/debugging.
    
#     Args:
#         ids: list of chunk identifiers
#         texts: list of chunk texts
#         output_path: where to save the JSON file
#     """
#     output_path.parent.mkdir(parents=True, exist_ok=True)
    
#     chunks_data = [
#         {"id": chunk_id, "text": chunk_text}
#         for chunk_id, chunk_text in zip(ids, texts)
#     ]
    
#     with output_path.open("w", encoding="utf-8") as f:
#         json.dump(chunks_data, f, ensure_ascii=False, indent=2)
    
#     print(f"Saved {len(chunks_data)} chunks to {output_path}")


# def print_chunk_stats(ids: List[str], texts: List[str]) -> None:
#     """Print summary statistics about the chunks."""
#     if not texts:
#         print("No chunks to display.")
#         return
    
#     chunk_sizes = [len(t) for t in texts]
#     total_chars = sum(chunk_sizes)
    
#     print(f"\n=== Chunking Statistics ===")
#     print(f"Total chunks: {len(texts)}")
#     print(f"Total characters: {total_chars:,}")
#     print(f"Min chunk size: {min(chunk_sizes):,} chars")
#     print(f"Max chunk size: {max(chunk_sizes):,} chars")
#     print(f"Avg chunk size: {total_chars / len(texts):.0f} chars")
#     print(f"\nFirst 3 chunk IDs:")
#     for chunk_id in ids[:3]:
#         print(f"  - {chunk_id}")


# def main():
#     parser = argparse.ArgumentParser(description="Process PDF and chunk text for embeddings")
#     parser.add_argument(
#         "--pdf",
#         type=Path,
#         required=True,
#         help="Path to PDF file to process"
#     )
#     parser.add_argument(
#         "--chunk-size",
#         type=int,
#         default=1000,
#         help="Target chunk size in characters (default: 1000). Set to <=0 to disable chunking."
#     )
#     parser.add_argument(
#         "--overlap",
#         type=int,
#         default=200,
#         help="Overlap between chunks in characters (default: 200)"
#     )
#     parser.add_argument(
#         "--output",
#         type=Path,
#         default=Path("data/chunks.json"),
#         help="Output JSON file to save chunks (default: data/chunks.json)"
#     )
#     parser.add_argument(
#         "--no-output",
#         action="store_true",
#         help="Skip saving chunks to JSON file"
#     )
    
#     args = parser.parse_args()
    
#     # Validate inputs
#     if args.chunk_size > 0 and args.overlap >= args.chunk_size:
#         raise ValueError(f"overlap ({args.overlap}) must be < chunk_size ({args.chunk_size})")
    
#     # Load PDF
#     print(f"Loading PDF: {args.pdf}")
#     page_ids, page_texts = load_pdf(args.pdf)
#     print(f"Extracted {len(page_ids)} pages")
    
#     # Chunk pages
#     if args.chunk_size > 0:
#         print(f"Chunking with size={args.chunk_size}, overlap={args.overlap}")
#         chunk_ids, chunk_texts = chunk_pdf_pages(
#             page_ids, page_texts,
#             chunk_size=args.chunk_size,
#             overlap=args.overlap
#         )
#     else:
#         print("Chunking disabled (using pages as units)")
#         chunk_ids, chunk_texts = page_ids, page_texts
    
#     # Display stats
#     print_chunk_stats(chunk_ids, chunk_texts)
    
#     # Save chunks if requested
#     if not args.no_output:
#         save_chunks_json(chunk_ids, chunk_texts, args.output)
    
#     print("\n✓ Processing complete")


# if __name__ == "__main__":
#     main()
