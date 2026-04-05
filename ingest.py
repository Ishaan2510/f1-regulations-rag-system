"""
ingest.py
---------
Offline ingestion pipeline. Run this ONCE to build the FAISS index.
Think of this as the "compile" step — it transforms raw PDFs into a
searchable vector index saved to disk.

Flow: PDF files → pages → text → chunks → embeddings → FAISS index
"""

import os
import pickle
import time
from pathlib import Path

import fitz  # PyMuPDF — C++ extension wrapped in Python
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# --- Configuration constants (think: #define in C++) ---
PDF_DIR = Path("data/pdfs")
INDEX_DIR = Path("index")
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
CHUNKS_PATH = INDEX_DIR / "chunks.pkl"

# BGE-small produces 384-dim embeddings. Fast on CPU, ~10ms per chunk.
# "bge" = BAAI General Embedding. Instruction prefix improves retrieval quality.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Chunk size in characters. 512 chars ≈ 100-130 tokens for English text,
# well within the 512-token limit of bge-small.
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64  # Sliding window overlap to avoid boundary misses


def load_pdfs(pdf_dir: Path) -> list[dict]:
    """
    Parse all PDFs in pdf_dir and return a list of page records.
    Each record is a dict: { "text": str, "page": int, "source": str }

    In C++ terms: vector<PageRecord> load_pdfs(path dir)
    """
    pages = []

    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {pdf_dir}")

    for pdf_path in pdf_files:
        print(f"Parsing: {pdf_path.name}")
        # fitz.open() returns a Document object — think of it as
        # constructing a Document class that wraps the PDF binary
        doc = fitz.open(str(pdf_path))

        for page_num, page in enumerate(doc):
            # page.get_text() extracts plain text from the PDF page.
            # For FIA regs (single-column), this is clean. For multi-column
            # or scanned PDFs, you would need layout analysis or OCR.
            text = page.get_text("text").strip()

            # Skip blank pages — common in FIA PDFs (divider pages)
            if len(text) < 50:
                continue

            pages.append({
                "text": text,
                "page": page_num + 1,  # 1-indexed for human display
                "source": pdf_path.name
            })

        doc.close()

    print(f"Loaded {len(pages)} non-empty pages from {len(pdf_files)} PDF(s)")
    return pages


def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Split pages into fixed-size overlapping chunks.
    Returns list of chunk dicts with text + inherited metadata.

    In C++ terms: this is like calling std::string::substr() in a sliding
    window loop, but LangChain handles sentence-aware boundary detection
    so splits don't land mid-sentence.
    """
    # RecursiveCharacterTextSplitter tries to split on paragraph breaks (\n\n),
    # then sentence breaks (\n), then word breaks (" "), in that order.
    # This is smarter than a naive character-level split.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]  # priority-ordered split points
    )

    chunks = []
    for page in pages:
        # split_text returns a list of strings — each is one chunk
        texts = splitter.split_text(page["text"])

        for i, text in enumerate(texts):
            chunks.append({
                "text": text,
                "page": page["page"],
                "source": page["source"],
                "chunk_id": len(chunks)  # global monotonic ID, like an array index
            })

    print(f"Created {len(chunks)} chunks from {len(pages)} pages")
    return chunks


def embed_chunks(chunks: list[dict], model: SentenceTransformer) -> np.ndarray:
    """
    Embed all chunk texts into dense vectors.
    Returns a float32 numpy array of shape (n_chunks, embedding_dim).

    In C++ terms: float[n][384] embed_chunks(vector<Chunk> chunks, Model& m)
    The return value is the matrix you will insert into the FAISS index.
    """
    texts = [chunk["text"] for chunk in chunks]

    print(f"Embedding {len(texts)} chunks with {EMBEDDING_MODEL}...")
    start = time.perf_counter()

    # BGE models require an instruction prefix ONLY for queries, not documents.
    # For documents we embed them as-is. This is a documented BGE convention.
    embeddings = model.encode(
        texts,
        batch_size=32,        # Process 32 chunks at once — like a batch DMA transfer
        show_progress_bar=True,
        normalize_embeddings=True  # L2-normalize so dot product == cosine similarity
    )

    elapsed = time.perf_counter() - start
    print(f"Embedding complete: {elapsed:.2f}s ({elapsed/len(texts)*1000:.1f}ms per chunk)")

    # SentenceTransformer returns float32 numpy array by default
    # FAISS requires float32 — ensure this explicitly
    return embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS flat L2 index from the embedding matrix.

    IndexFlatL2 = brute-force exact nearest neighbor search.
    No approximation, no training required, deterministic results.
    For n < 10,000 chunks, this is fast enough (<1ms query time on CPU).

    In C++ terms: this is like constructing a sorted array that you will
    later binary-search — except the "search" here is a matrix multiply.
    """
    n_vectors, dim = embeddings.shape
    print(f"Building FAISS IndexFlatL2: {n_vectors} vectors of dim {dim}")

    # IndexFlatL2 computes squared Euclidean distance.
    # Since we L2-normalized embeddings, L2 distance ∝ cosine distance.
    # So this is effectively cosine similarity search.
    index = faiss.IndexFlatL2(dim)

    # add() copies the embedding matrix into the index's internal storage
    # After this call, index.ntotal == n_vectors
    index.add(embeddings)

    print(f"FAISS index built: {index.ntotal} vectors indexed")
    return index


def save_artifacts(index: faiss.Index, chunks: list[dict]) -> None:
    """
    Persist the FAISS index and chunk metadata to disk.
    The Streamlit app loads these at startup — no re-embedding on restart.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print(f"FAISS index saved: {FAISS_INDEX_PATH}")

    # chunks.pkl stores the list of dicts with text + metadata.
    # At query time, we use FAISS to get chunk_ids, then look up this list.
    # This is like a hash map: chunk_id → ChunkRecord.
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Chunk metadata saved: {CHUNKS_PATH}")


def main():
    print("=== TechReg Analyst — Ingestion Pipeline ===\n")

    # Load embedding model once. This downloads ~130MB on first run.
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    pages = load_pdfs(PDF_DIR)
    chunks = chunk_pages(pages)
    embeddings = embed_chunks(chunks, model)
    index = build_faiss_index(embeddings)
    save_artifacts(index, chunks)

    print("\n=== Ingestion complete. Run app.py to start the Streamlit app. ===")


if __name__ == "__main__":
    main()