"""
Clean extracted PDF text before chunking and embedding.

Pipeline per page:
  1) Fix hyphenated line breaks
  2) Remove duplicate adjacent words
  3) Normalize whitespace
  4) Filter empty/near-empty pages

Optionally:
  5) Strip repeated headers/footers across pages
"""

import re
from typing import List, Tuple

# --- Regex compiled once at module level --- ???ask!

DUPLICATE_WORD_RE = re.compile(r"\b(\w+)(\s+\1\b)+", flags=re.IGNORECASE)


#cleaning functions
def fix_soft_hyphen_linebreaks(s: str) -> str:
    """Rejoin words split across lines with a hyphen.
    
    e.g. 'auto-\\n    scaling' -> 'auto-scaling'
    """
    if not s:
        return ""
    return re.sub(r"-\s*\n\s*", "-", s)


def dedupe_adjacent_words(s: str) -> str:
    """Remove immediately repeated words.

    e.g. 'the the cat' -> 'the cat'
    Leaves intentional repetition further apart untouched.
    """
    if not s:
        return ""
    return DUPLICATE_WORD_RE.sub(lambda m: m.group(1), s)


def normalize_whitespace(s: str) -> str:
    """Collapse whitespace while preserving paragraph breaks.

    Keeps double newlines (paragraph boundaries) intact so chunker
    can use them as natural split points. Collapses everything else.
    """
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)   #make excessive blank lines to one paragraph break
    s = re.sub(r"[^\S\n]+", " ", s)    # collapse spaces/tabs but leave newlines alone
    return s.strip()


def is_empty_page(text: str, min_chars: int = 50) -> bool:
    """Return True if a page has too little text to be useful.

    Catches title pages, blank pages, and image-only pages that
    pypdf extracts as empty or near-empty strings.
    """
    return len(text.strip()) < min_chars

#if there is a header/footer 
def detect_repeated_lines(pages: List[str], threshold: float = 0.6) -> set:
    """Find lines that appear on more than `threshold` fraction of pages.

    These are likely headers, footers, or page numbers that add noise
    without meaning.

    Args:
        pages: list of page text strings
        threshold: fraction of pages a line must appear on to be flagged

    Returns:
        Set of line strings to remove
    """
    if not pages:
        return set()

    from collections import Counter
    line_counts: Counter = Counter()

    for page in pages:
        # count each unique line once per page
        unique_lines = set(line.strip() for line in page.splitlines() if line.strip())
        line_counts.update(unique_lines)

    cutoff = threshold * len(pages)
    return {line for line, count in line_counts.items() if count >= cutoff}


def strip_repeated_lines(text: str, repeated: set) -> str:
    """Remove known header/footer lines from a page's text."""
    if not repeated:
        return text
    lines = text.splitlines()
    cleaned = [line for line in lines if line.strip() not in repeated]
    return "\n".join(cleaned)


#main cleaning functions
def clean_page(text: str, repeated_lines: set = None) -> str:
    """Full cleaning pipeline for a single page of PDF text.

    Args:
        text: raw extracted page text
        repeated_lines: set of header/footer lines to strip (from detect_repeated_lines)

    Returns:
        Cleaned text string, or empty string if nothing remains.
    """
    if not text:
        return ""
    if repeated_lines:
        text = strip_repeated_lines(text, repeated_lines)
    text = fix_soft_hyphen_linebreaks(text)
    text = dedupe_adjacent_words(text)
    text = normalize_whitespace(text)
    return text


def clean_pages(
    ids: List[str],
    texts: List[str],
    min_chars: int = 50,
    strip_headers: bool = True,
) -> Tuple[List[str], List[str]]:
    """Clean all pages and filter out empty ones.

    Args:
        ids: page id strings from ingest.load_pdf
        texts: raw page text strings from ingest.load_pdf
        min_chars: pages with fewer characters after cleaning are dropped
        strip_headers: whether to auto-detect and remove repeated lines

    Returns:
        Tuple of (cleaned_ids, cleaned_texts) with empty pages removed.
    """
    repeated: set = set()
    if strip_headers:
        repeated = detect_repeated_lines(texts)

    cleaned_ids = []
    cleaned_texts = []

    for page_id, text in zip(ids, texts):
        cleaned = clean_page(text, repeated_lines=repeated)
        if not is_empty_page(cleaned, min_chars=min_chars):
            cleaned_ids.append(page_id)
            cleaned_texts.append(cleaned)

    dropped = len(texts) - len(cleaned_texts)
    if dropped:
        print(f"Dropped {dropped} empty/near-empty page(s)")

    return cleaned_ids, cleaned_texts