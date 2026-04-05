"""
ingest.py
---------
Updated for multi-section FIA 2026 F1 Regulations (Sections A-F).
Parses all PDFs in data/pdfs/, chunks, embeds, and saves FAISS index.
"""

import os
import pickle
import time
from pathlib import Path

import fitz
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

PDF_DIR = Path("data/pdfs")
INDEX_DIR = Path("index")
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
CHUNKS_PATH = INDEX_DIR / "chunks.pkl"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# Maps filename keywords → human-readable section labels
# Used in source citations shown to the user
SECTION_LABELS = {
    "section_a": "Section A – General Provisions",
    "section_b": "Section B – Sporting Regulations",
    "section_c": "Section C – Technical Regulations",
    "section_d": "Section D – Financial (Teams)",
    "section_e": "Section E – Financial (Power Unit Manufacturers)",
    "section_f": "Section F – Operational Regulations",
}


def get_section_label(filename: str) -> str:
    """
    Extract a readable section label from the PDF filename.
    In C++ terms: this is a lookup in a std::map<string, string>.
    Falls back to the raw filename if no match found.
    """
    filename_lower = filename.lower()
    for key, label in SECTION_LABELS.items():
        if key in filename_lower:
            return label
    return filename  # fallback: use filename as-is


def load_pdfs(pdf_dir: Path) -> list[dict]:
    """
    Parse all PDFs and return page records with section labels.
    Each record: { text, page, source (filename), section (readable label) }
    """
    pages = []
    pdf_files = sorted(pdf_dir.glob("*.pdf"))  # sorted for deterministic ordering

    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir}")

    for pdf_path in pdf_files:
        section_label = get_section_label(pdf_path.name)
        print(f"Parsing: {pdf_path.name} → {section_label}")

        doc = fitz.open(str(pdf_path))
        page_count = len(doc)

        for page_num, page in enumerate(doc):
            text = page.get_text("text").strip()
            if len(text) < 50:
                continue

            pages.append({
                "text": text,
                "page": page_num + 1,
                "source": pdf_path.name,
                "section": section_label,   # NEW: human-readable section label
                "total_pages": page_count   # NEW: useful for context
            })

        doc.close()
        print(f"  → {page_count} pages loaded")

    print(f"\nTotal: {len(pages)} non-empty pages from {len(pdf_files)} PDFs")
    return pages


def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Chunk pages into overlapping windows, inheriting section metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = []
    for page in pages:
        texts = splitter.split_text(page["text"])
        for text in texts:
            chunks.append({
                "text": text,
                "page": page["page"],
                "source": page["source"],
                "section": page["section"],   # propagated from page record
                "chunk_id": len(chunks)
            })

    # Print per-section breakdown so you can verify coverage
    section_counts = {}
    for c in chunks:
        section_counts[c["section"]] = section_counts.get(c["section"], 0) + 1

    print(f"\nChunk breakdown by section:")
    for section, count in sorted(section_counts.items()):
        print(f"  {section}: {count} chunks")
    print(f"  TOTAL: {len(chunks)} chunks\n")

    return chunks


def embed_chunks(chunks: list[dict], model: SentenceTransformer) -> np.ndarray:
    texts = [chunk["text"] for chunk in chunks]
    print(f"Embedding {len(texts)} chunks (this takes ~2-5 min on CPU)...")
    start = time.perf_counter()

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    elapsed = time.perf_counter() - start
    print(f"Done: {elapsed:.1f}s ({elapsed/len(texts)*1000:.1f}ms/chunk avg)")
    return embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    n, dim = embeddings.shape
    print(f"Building IndexFlatL2: {n} vectors × {dim} dims")

    # At 4000 chunks, FlatL2 query time is still <5ms — no need for IVF
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    print(f"Index built: {index.ntotal} vectors")
    return index


def save_artifacts(index: faiss.Index, chunks: list[dict]) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved index → {FAISS_INDEX_PATH}")
    print(f"Saved chunks → {CHUNKS_PATH}")


def main():
    print("=== TechReg Analyst — Multi-Section Ingestion (FIA 2026 F1) ===\n")

    print(f"Loading model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    pages = load_pdfs(PDF_DIR)
    chunks = chunk_pages(pages)
    embeddings = embed_chunks(chunks, model)
    index = build_faiss_index(embeddings)
    save_artifacts(index, chunks)

    print("\n=== Ingestion complete. Run: streamlit run app.py ===")


if __name__ == "__main__":
    main()