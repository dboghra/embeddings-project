# 🔍 PDF RAG — Retrieval-Augmented Generation from Scratch

A self-directed deep dive into **RAG (Retrieval-Augmented Generation)** — built to understand how modern AI pipelines work under the hood, from raw text ingestion all the way to semantic search.

> **Status:** 🚧 In Development — architecture planned, implementation in progress

---

## 💡 What This Project Does

Upload any PDF and ask it questions in plain English. Rather than keyword matching, the system understands *meaning* — surfacing the most relevant passages using vector similarity search.

```
PDF → Parse → Chunk → Embed → Store in Vector DB → Query → Answer
```

This is the same fundamental pipeline behind tools like ChatGPT's file uploads, Notion AI, and enterprise document search systems.

---

## 🧱 System Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────────┐
│  PDF Input  │────▶│  Text Parser │────▶│   Chunking Engine   │
└─────────────┘     └──────────────┘     └──────────┬──────────┘
                                                     │
                                          ┌──────────▼──────────┐
                                          │  Embedding Model    │
                                          │  (Sentence-BERT)    │
                                          └──────────┬──────────┘
                                                     │
                                          ┌──────────▼──────────┐
                                          │  Vector Store       │
                                          │  (ChromaDB)         │
                                          └──────────┬──────────┘
                                                     │
                              ┌──────────────────────▼────────────────────────┐
                              │  Query Interface — Semantic Similarity Search  │
                              └───────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Tool | Why |
|---|---|---|
| PDF Parsing | `PyMuPDF` / `pdfplumber` | Robust text + layout extraction |
| Embeddings | `sentence-transformers` | Pretrained SBERT models, no fine-tuning needed |
| Vector Database | `ChromaDB` | Lightweight, local-first vector store |
| Language | Python 3.10+ | Industry standard for ML pipelines |

---

## 📐 Key Concepts Explored

- **Text Embeddings** — converting unstructured text into dense numerical vectors that encode semantic meaning
- **Chunking Strategy** — splitting documents into overlapping segments to preserve context across chunk boundaries
- **Cosine Similarity** — measuring vector proximity to rank retrieved passages by relevance
- **RAG Pipeline Design** — decoupling retrieval from generation; understanding where each component lives in the stack
- **Vector Databases** — indexing, storing, and querying high-dimensional embeddings efficiently

---

## 🗂️ Project Structure

```
pdf-rag/
├── data/               # Sample PDFs for testing
├── src/
│   ├── ingest.py       # PDF loading and text extraction
│   ├── chunker.py      # Text splitting logic
│   ├── embed.py        # Embedding generation via sentence-transformers
│   ├── store.py        # ChromaDB read/write interface
│   └── query.py        # Similarity search and result ranking
├── notebooks/          # Exploratory work and model comparisons
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

```bash
git clone https://github.com/your-username/pdf-rag.git
cd pdf-rag
pip install -r requirements.txt

# Ingest a PDF
python src/ingest.py --file data/sample.pdf

# Query it
python src/query.py --question "What are the main findings?"
```

---

## 🗺️ Roadmap

- [ ] PDF ingestion and text extraction
- [ ] Chunking with overlap
- [ ] Embedding pipeline with `sentence-transformers`
- [ ] ChromaDB integration
- [ ] Semantic query interface
- [ ] Notebook with embedding visualizations (UMAP/t-SNE)
- [ ] Swap and benchmark multiple SBERT models
- [ ] Optional: lightweight frontend (Streamlit or React)

---

## 🎯 Why I Built This

RAG is one of the most widely deployed patterns in production AI systems today. Rather than treating it as a black box, I wanted to build each layer myself — from how embeddings encode meaning, to how vector databases enable fast approximate nearest-neighbor search — so I genuinely understand what's happening when I use or build AI-powered tools.

---

## 📬 Contact

**[Your Name]** — [your-email@northeastern.edu] — [LinkedIn](https://linkedin.com/in/yourprofile) — [GitHub](https://github.com/your-username)
