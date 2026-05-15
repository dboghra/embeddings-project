from pathlib import Path
from typing import List, Tuple




try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

def load_pdf(path: Path) -> Tuple[List[str], List[str]]: #path to the pdf file
    """Extract text from each page of a PDF.
        
    Returns:
        Tuple of (page_ids, page_texts) where:
        - page_ids: list of strings like "filename-page-1", "filename-page-2", etc.
        - page_texts: list of extracted text from each page
        
    Raises:
        RuntimeError: if pypdf is not installed
        SystemExit: if PDF file not found
    """
    if PdfReader is None: 
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
