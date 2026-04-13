"""
ingest.py
---------
One-time offline pipeline: 6 FIA 2026 F1 Regulation PDFs → FAISS index on disk.

Think of this as the "compile" step. The Streamlit app never reads PDFs directly —
it only reads the two output files this script produces:
  - index/faiss.index   (the vector search index)
  - index/chunks.pkl    (chunk text + metadata: page, section, source)

Run once with: python ingest.py
Re-run only if you add new PDFs or change chunk_size/overlap.

v2: Added text cleaning step to remove FIA page headers before chunking.

The FIA PDFs print a boilerplate header on every page:
  "SECTION X: SOMETHING REGULATIONS    X##  ..."
  "2026 Formula 1: ... ©2026 Fédération Internationale de l'Automobile"

Without cleaning, many chunks are entirely or mostly this header.
These header-chunks pollute retrieval — queries match the section name
instead of the actual regulation content.

Fix: strip the header pattern from each page's text before chunking.
"""

import pickle
import time
from pathlib import Path
import re        

import fitz          # PyMuPDF: C++ PDF parser with Python bindings
import faiss         # Facebook AI Similarity Search
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Configuration

PDF_DIR = Path("data/pdfs")
INDEX_DIR = Path("index")
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
CHUNKS_PATH = INDEX_DIR / "chunks.pkl"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # 384-dim, CPU-fast, free

# 512 chars ≈ 100-130 tokens for English text.
# bge-small's token limit is 512 tokens, so we're safely within bounds.
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64  # Characters repeated between adjacent chunks

# Maps filename substrings → human-readable labels for citations
# In C++ terms: std::unordered_map<string, string> SECTION_LABELS
SECTION_LABELS = {
    "section_a": "Section A – General Provisions",
    "section_b": "Section B – Sporting Regulations",
    "section_c": "Section C – Technical Regulations",
    "section_d": "Section D – Financial (Teams)",
    "section_e": "Section E – Financial (Power Unit Manufacturers)",
    "section_f": "Section F – Operational Regulations",
}

# Step 1: PDF Parsing

def get_section_label(filename: str) -> str:
    """
    Map a PDF filename to a human-readable section label.

    In C++ terms: this is a std::map lookup with a fallback.
    We search the filename (lowercased) for each key in SECTION_LABELS.
    First match wins. If no match, return the raw filename.

    Example:
      "section_c_technical.pdf" → "Section C – Technical Regulations"
    """
    filename_lower = filename.lower()
    for key, label in SECTION_LABELS.items():
        if key in filename_lower:
            return label
    # Fallback: raw filename. Means your PDF wasn't named with section_x prefix.
    print(f"  [WARNING] No section label matched for: {filename}. Using filename.")
    return filename


def load_pdfs(pdf_dir: Path) -> list[dict]:
    """
    Parse all PDFs in pdf_dir → list of page records.

    Each record is a dict (think: C++ struct):
      {
        "text":    str,   # Raw text extracted from this page
        "page":    int,   # 1-indexed page number (for human-readable citations)
        "source":  str,   # Filename (e.g. "section_c_technical.pdf")
        "section": str,   # Human label (e.g. "Section C – Technical Regulations")
      }

    Why per-page and not per-document?
    Because we need page numbers for citations. If we treated each PDF as one
    blob of text, we'd lose track of which page each chunk came from.
    """
    pages = []
    # sorted() ensures A→F order every time — deterministic behavior
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {pdf_dir}\n"
            f"Make sure your 6 PDFs are in: {pdf_dir.resolve()}"
        )

    for pdf_path in pdf_files:
        section_label = get_section_label(pdf_path.name)
        print(f"\nParsing: {pdf_path.name}")
        print(f"  Label: {section_label}")

        # fitz.open() is like calling new Document(filepath) in C++.
        # It memory-maps the PDF binary — doesn't load all pages at once.
        doc = fitz.open(str(pdf_path))
        page_count = len(doc)
        pages_added = 0

        for page_num, page in enumerate(doc):
            # get_text("text") returns plain text with newlines preserved.
            # "text" mode = reading order. Other modes: "html", "dict", "blocks".
            # For single-column FIA PDFs, "text" mode is clean and sufficient.
            text = page.get_text("text").strip()
            text = clean_page_text(text)  # Remove boilerplate headers/footers

            # Skip pages with very little content — covers pages, dividers,
            # blank pages, header-only pages. Threshold of 50 chars empirically works.
            if len(text) < 50:
                continue

            pages.append({
                "text": text,
                "page": page_num + 1,       # +1 because enumerate is 0-indexed
                "source": pdf_path.name,
                "section": section_label,
            })
            pages_added += 1

        doc.close()
        print(f"  Pages: {page_count} total, {pages_added} non-empty extracted")

    print(f"\n{'='*50}")
    print(f"Total pages loaded: {len(pages)} across {len(pdf_files)} PDFs")
    return pages

def clean_page_text(text: str) -> str:
    """
    Remove FIA PDF header/footer lines that appear on every page.
    These lines are noise — they match every query equally and dilute
    chunk quality.

    Patterns to remove:
      - Section codes like "C150", "F5", "B44" (letter + digits, standalone)
      - Copyright lines: "©2026 Fédération Internationale..."
      - Repeated title lines: "2026 Formula 1: Technical Regulations"
      - Issue/date lines: "Issue 06", "27 February 2026"
      - Section header lines: "SECTION C: TECHNICAL REGULATIONS"
    """
    lines = text.split("\n")
    cleaned = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Standalone page codes like "C150", "F5", "B44"
        if re.match(r'^[A-F]\d+$', line):
            continue
        # Copyright and organisation name
        if "©2026" in line or "Fédération Internationale" in line:
            continue
        # Repeated document title
        if "2026 Formula 1:" in line:
            continue
        # Section header lines
        if re.match(r'^SECTION [A-F]:', line):
            continue
        # Issue/date metadata
        if re.match(r'^Issue \d+', line) or re.match(r'^\d+ \w+ 20\d\d$', line):
            continue
        # Standalone integers (bare page numbers)
        if re.match(r'^\d+$', line):
            continue
        cleaned.append(line)

    return "\n".join(cleaned)

# Step 2: Chunking

def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Split page texts into overlapping fixed-size chunks.

    Why not just use full pages as retrieval units?
    1. Pages can be 500-800 words — exceeds bge-small's 512 token limit.
    2. A page covering 5 topics produces a diluted embedding.
       A chunk covering 1 topic produces a precise embedding → better retrieval.

    RecursiveCharacterTextSplitter splits in priority order:
      1. Paragraph breaks (\n\n) — best: keeps paragraphs intact
      2. Line breaks (\n)        — good: keeps sentences intact
      3. Spaces (" ")            — ok: word boundary
      4. Characters ("")         — last resort: hard cut

    This is smarter than naive character slicing because it avoids
    cutting mid-sentence when possible.

    Each chunk inherits its parent page's metadata (page number, section, source).
    chunk_id is a global monotonic counter — like an array index.
    FAISS returns chunk_ids at query time; we use them to look up this list.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = []

    for page in pages:
        # split_text() returns a list[str] of chunk texts
        texts = splitter.split_text(page["text"])

        for text in texts:
            chunks.append({
                "text": text,
                "page": page["page"],
                "source": page["source"],
                "section": page["section"],
                "chunk_id": len(chunks),    # monotonic: 0, 1, 2, ... N-1
            })

    # Print per-section breakdown — useful sanity check
    # In C++ terms: std::map<string, int> counts; for each chunk: counts[section]++
    section_counts: dict[str, int] = {}
    for chunk in chunks:
        sec = chunk["section"]
        section_counts[sec] = section_counts.get(sec, 0) + 1

    print(f"\n{'='*50}")
    print("Chunk breakdown by section:")
    for section in sorted(section_counts):
        print(f"  {section}: {section_counts[section]} chunks")
    print(f"  TOTAL: {len(chunks)} chunks")

    return chunks

# Step 3: Embedding

def embed_chunks(chunks: list[dict], model: SentenceTransformer) -> np.ndarray:
    """
    Convert chunk texts → 384-dimensional float32 vectors.

    What is an embedding?
    It's a function: f(text) → vector<float, 384>
    where semantically similar texts produce geometrically close vectors.

    "Maximum car width" and "overall vehicle breadth limit" share no words
    but will have cosine similarity ~0.9 because bge-small learned from
    millions of text pairs that these phrases mean the same thing.

    Why normalize_embeddings=True?
    L2 normalization makes every vector have length 1 (unit vector).
    After normalization: L2_distance(a, b) = 2 * (1 - cosine_similarity(a, b))
    So minimizing L2 distance is equivalent to maximizing cosine similarity.
    This lets us use FAISS's IndexFlatL2 as a cosine similarity search.

    Why BGE_QUERY_PREFIX is NOT used here:
    BGE models use asymmetric encoding — queries and documents are treated
    differently. Documents are embedded as-is. Queries get a prefix.
    This asymmetry improves retrieval precision. Applied here → documents only.
    """
    texts = [chunk["text"] for chunk in chunks]

    print(f"\n{'='*50}")
    print(f"Embedding {len(texts)} chunks...")
    print(f"Model: {EMBEDDING_MODEL}")
    print(f"Expected time: ~{len(texts) * 0.01:.0f}s on CPU (10ms/chunk)")

    start = time.perf_counter()

    # encode() with batch_size=32: processes 32 chunks per forward pass.
    # Like sending data to a function in batches of 32 instead of one at a time.
    # Larger batch = more RAM, slightly faster. 32 is safe for 8GB RAM.
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True   # L2-normalize for cosine similarity via L2 index
    )

    elapsed = time.perf_counter() - start
    print(f"\nEmbedding complete: {elapsed:.1f}s total")
    print(f"Per-chunk average: {elapsed / len(texts) * 1000:.1f}ms")
    print(f"Output shape: {embeddings.shape}")  # Should be (n_chunks, 384)

    return embeddings.astype(np.float32)  # FAISS requires float32 explicitly

# Step 4: FAISS Index

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS IndexFlatL2 from the embedding matrix.

    What is FAISS doing?
    It stores all your vectors and answers nearest-neighbor queries:
    "Given query vector q, which K stored vectors are closest?"

    IndexFlatL2 = brute-force exact search using L2 (Euclidean) distance.
    For each query, it computes distance to EVERY stored vector.
    This is a matrix multiply: O(n × d) where n=chunks, d=384.

    Why not use a more sophisticated index (IVF, HNSW)?
    - IVFFlat: clusters vectors into cells, only searches nearest cells.
      Faster for n > 50,000 but introduces approximation errors.
    - HNSWFlat: graph-based, very fast but uses 3-5x more RAM.
    At n=2,300 chunks, FlatL2 takes <2ms per query on CPU — no need
    for approximation. Correctness > marginal speed gain at this scale.

    When to switch: if you scale to >50,000 chunks, switch to IVFFlat.
    """
    n_vectors, dim = embeddings.shape
    print(f"\n{'='*50}")
    print(f"Building FAISS IndexFlatL2")
    print(f"  Vectors: {n_vectors}")
    print(f"  Dimensions: {dim}")

    index = faiss.IndexFlatL2(dim)  # Constructor takes only the dimension

    # add() copies the numpy array into FAISS's internal memory.
    # After this: index.ntotal == n_vectors
    index.add(embeddings)

    print(f"  Indexed: {index.ntotal} vectors ✓")
    return index

# Step 5: Save to Disk

def save_artifacts(index: faiss.Index, chunks: list[dict]) -> None:
    """
    Persist both artifacts to disk.

    faiss.write_index() serializes the index to a binary file.
    pickle.dump() serializes the Python list of chunk dicts.

    Why separate files?
    FAISS only stores vectors — not the text or metadata.
    At query time: FAISS returns indices (0, 1, 2...) → we look up
    the chunk dict at that index in chunks.pkl.
    This separation is like a database split: FAISS = index, pkl = data store.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print(f"\nFAISS index saved → {FAISS_INDEX_PATH}")

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Chunk metadata saved → {CHUNKS_PATH}")

    # Report file sizes so you know what's on disk
    index_size = FAISS_INDEX_PATH.stat().st_size / (1024 * 1024)
    chunks_size = CHUNKS_PATH.stat().st_size / (1024 * 1024)
    print(f"\nDisk usage:")
    print(f"  faiss.index: {index_size:.1f} MB")
    print(f"  chunks.pkl:  {chunks_size:.1f} MB")


def main():
    print("=" * 50)
    print("TechReg Analyst — Ingestion Pipeline")
    print("FIA 2026 F1 Regulations (Sections A–F)")
    print("=" * 50)

    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    print("(First run downloads ~130MB — subsequent runs use cache)\n")
    model = SentenceTransformer(EMBEDDING_MODEL)

    pages  = load_pdfs(PDF_DIR)
    chunks = chunk_pages(pages)
    embeddings = embed_chunks(chunks, model)
    index  = build_faiss_index(embeddings)
    save_artifacts(index, chunks)

    print("\n" + "=" * 50)
    print("Ingestion complete.")
    print("Next step: python retriever.py  (we'll build this together next)")
    print("=" * 50)


if __name__ == "__main__":
    main()