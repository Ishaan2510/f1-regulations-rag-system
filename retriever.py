"""
retriever.py
------------
Query-time retrieval: loads FAISS index + models once, serves queries fast.

Two-stage pipeline:
  Stage 1 (Bi-encoder):    query → embed → FAISS search → top-20 candidates
                           Optimizes for RECALL. Cheap: O(n×d) matrix multiply.
  Stage 2 (Cross-encoder): (query, chunk) pairs → relevance scores → rerank
                           Optimizes for PRECISION. Expensive: O(k × seq²).
                           Only runs on 20 candidates, not 3573.

In C++ terms:
  The Retriever class is a singleton with a heavy constructor (disk I/O +
  model loading) and a lightweight member function (retrieve) called per query.
  @st.cache_resource in app.py ensures the constructor runs exactly once.
"""

import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FAISS_INDEX_PATH = Path("index/faiss.index")
CHUNKS_PATH      = Path("index/chunks.pkl")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Applied to queries ONLY — not to documents during ingestion.
# BGE asymmetric training: query prefix shifts embedding toward "search intent".
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

BI_ENCODER_TOP_K = 20   # Candidates after FAISS (recall phase)
RERANKER_TOP_K   = 5    # Final results after cross-encoder (precision phase)

# When section_filter is active, many FAISS results get dropped.
# Fetch 3× more upfront so we still have 20 candidates after filtering.
FAISS_FETCH_MULTIPLIER = 2


# ---------------------------------------------------------------------------
# Output data structure
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    """
    One retrieved result — passed to chain.py for context building.

    In C++ terms: this is a POD struct (plain old data, no methods).
    dataclass auto-generates __init__, __repr__, __eq__ from the field list.

    Fields:
      text                 → the actual regulation text (fed to LLM)
      source               → filename (e.g. "section_c_technical.pdf")
      section              → human label (e.g. "Section C – Technical Regulations")
      page                 → 1-indexed page number (for citation display)
      chunk_id             → position in chunks.pkl (for debugging)
      bi_encoder_score     → L2 distance from FAISS (LOWER = more similar)
      cross_encoder_score  → relevance logit from CE (HIGHER = more relevant)
    """
    text:                str
    source:              str
    section:             str
    page:                int
    chunk_id:            int
    bi_encoder_score:    float
    cross_encoder_score: float


# ---------------------------------------------------------------------------
# Retriever class
# ---------------------------------------------------------------------------

class Retriever:
    """
    Stateful retriever. Heavy construction, lightweight per-query operation.

    Constructor loads:
      - FAISS index from disk (memory-mapped, fast)
      - chunks.pkl from disk (full list into RAM)
      - bge-small bi-encoder (from HuggingFace cache, ~130MB)
      - ms-marco cross-encoder (from HuggingFace cache, ~80MB)

    In C++ terms: this is a class where the constructor does all the
    expensive work (file I/O, model allocation) so that retrieve() is cheap.
    Like pre-allocating and loading a lookup table before processing begins.
    """

    def __init__(self):
        print("Initializing Retriever...")

        # --- Load FAISS index ---
        # faiss.read_index() deserializes the binary file written by ingest.py.
        # The index is loaded into RAM. For 3573 vectors × 384 dims × 4 bytes
        # = ~5.5MB — trivial memory usage.
        if not FAISS_INDEX_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {FAISS_INDEX_PATH}\n"
                "Run: python ingest.py"
            )
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))
        print(f"  FAISS index: {self.index.ntotal} vectors loaded")

        # --- Load chunk metadata ---
        # chunks.pkl is the parallel data store to the FAISS index.
        # FAISS gives us indices (0, 1, 2...); we use them to look up
        # chunk text + metadata in this list.
        # In C++ terms: vector<ChunkRecord> where FAISS gives the array index.
        if not CHUNKS_PATH.exists():
            raise FileNotFoundError(
                f"Chunk metadata not found at {CHUNKS_PATH}\n"
                "Run: python ingest.py"
            )
        with open(CHUNKS_PATH, "rb") as f:
            self.chunks: list[dict] = pickle.load(f)
        print(f"  Chunk metadata: {len(self.chunks)} records loaded")

        # --- Load bi-encoder ---
        # Used at query time to embed the user's question.
        # During ingestion, this same model embedded the chunks.
        # Query and chunks must use the SAME model for similarity to be meaningful.
        print(f"  Loading bi-encoder: {EMBEDDING_MODEL}")
        self.bi_encoder = SentenceTransformer(EMBEDDING_MODEL)

        # --- Load cross-encoder ---
        # CrossEncoder is a SEPARATE class from SentenceTransformer.
        # It takes (query, passage) pairs — NOT individual texts.
        # ms-marco-MiniLM-L-6-v2: 6-layer MiniLM, ~22M params, ~80MB.
        # Trained on MS MARCO: 500k+ (query, passage, relevance) triples.
        print(f"  Loading cross-encoder: {RERANKER_MODEL}")
        self.cross_encoder = CrossEncoder(RERANKER_MODEL)

        print("Retriever ready.\n")

    # -----------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        section_filter: str | None = None
    ) -> tuple[list[RetrievedChunk], dict]:
        """
        Full two-stage retrieval for a user query.

        Args:
          query:          Raw user question string.
          section_filter: If provided, restrict results to this section label.
                          e.g. "Section C – Technical Regulations"
                          None means search across all 6 sections.

        Returns:
          results:    list[RetrievedChunk], length <= RERANKER_TOP_K (5)
          latencies:  dict with keys:
                        "bi_encoder_ms"    — FAISS search time
                        "cross_encoder_ms" — reranking time
                        "total_retrieval_ms"

        In C++ terms:
          pair<vector<RetrievedChunk>, unordered_map<string, float>> retrieve(...)
        """
        latencies: dict[str, float] = {}

        # ===================================================================
        # STAGE 1: Bi-encoder query embedding + FAISS ANN search
        # ===================================================================
        t0 = time.perf_counter()

        # Apply BGE query prefix — CRITICAL.
        # Documents were embedded without this prefix.
        # Queries must use it to match the training distribution.
        prefixed_query = BGE_QUERY_PREFIX + query

        # encode() with a list of 1 string → shape (1, 384) float32.
        # normalize_embeddings=True: L2-normalize so dot product = cosine sim.
        # This must match ingest.py's normalize_embeddings=True.
        query_vec = self.bi_encoder.encode(
            [prefixed_query],
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype(np.float32)   # FAISS requires float32 explicitly

        # How many vectors to fetch from FAISS?
        # If filtering by section: fetch 3× more (many will be dropped).
        # If no filter: fetch exactly BI_ENCODER_TOP_K.
        fetch_k = (BI_ENCODER_TOP_K * FAISS_FETCH_MULTIPLIER
                   if section_filter else BI_ENCODER_TOP_K)

        # index.search(query_matrix, k) returns:
        #   distances: shape (1, k) — L2 distances, LOWER = more similar
        #   indices:   shape (1, k) — chunk_ids in chunks.pkl
        # We pass query_vec as shape (1, 384) — FAISS expects a 2D matrix
        # even for a single query (designed for batch queries).
        distances, indices = self.index.search(query_vec, fetch_k)

        t1 = time.perf_counter()
        latencies["bi_encoder_ms"] = (t1 - t0) * 1000

        # Build candidate list from FAISS results.
        # indices[0] because we have 1 query (batch size = 1).
        # FAISS returns -1 for "no result" if the index has fewer than k vectors.
        candidates = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue   # Fewer results than requested — skip sentinel

            chunk = self.chunks[idx]  # O(1) array lookup by index

            # Apply section filter — like a SQL WHERE clause.
            # chunk["section"] is the human-readable label propagated from ingest.py.
            if section_filter and chunk.get("section") != section_filter:
                continue   # Not from the requested section — drop it

            candidates.append({
                "chunk":    chunk,
                "bi_score": float(dist)   # L2 distance — lower is better
            })

            # Stop once we have enough candidates for the cross-encoder
            if len(candidates) >= BI_ENCODER_TOP_K:
                break

        # Edge case: no candidates after filtering
        # This happens if section_filter is very restrictive and the query
        # doesn't match any chunks in that section.
        if not candidates:
            return [], {
                "bi_encoder_ms":     latencies["bi_encoder_ms"],
                "cross_encoder_ms":  0.0,
                "total_retrieval_ms": latencies["bi_encoder_ms"]
            }

        # ===================================================================
        # STAGE 2: Cross-encoder reranking
        # ===================================================================
        t2 = time.perf_counter()

        # Build (query, passage) pairs for the cross-encoder.
        # CrossEncoder.predict() takes a list of 2-tuples.
        # It concatenates them internally as: [CLS] query [SEP] passage [SEP]
        # and runs a full transformer forward pass for each pair.
        pairs = [(query, c["chunk"]["text"]) for c in candidates]

        # predict() returns a 1D numpy array of raw logit scores.
        # These are NOT probabilities — no softmax applied.
        # Higher logit = more relevant. Negative logits are valid.
        # The model was trained to output high scores for relevant pairs
        # and low scores for irrelevant pairs.
        scores = self.cross_encoder.predict(
            pairs,
            show_progress_bar=False
        )

        t3 = time.perf_counter()
        latencies["cross_encoder_ms"] = (t3 - t2) * 1000
        latencies["total_retrieval_ms"] = (t3 - t0) * 1000

        # Attach cross-encoder scores to each candidate
        for i, score in enumerate(scores):
            candidates[i]["ce_score"] = float(score)

        # Sort descending by cross-encoder score — highest relevance first.
        # In C++ terms: std::sort with a comparator on ce_score, reversed.
        candidates.sort(key=lambda x: x["ce_score"], reverse=True)

        # Take top RERANKER_TOP_K after reranking
        top = candidates[:RERANKER_TOP_K]

        # Build RetrievedChunk dataclass instances for the output.
        # chain.py and app.py consume these — they don't touch raw dicts.
        results = []
        for c in top:
            chunk = c["chunk"]
            results.append(RetrievedChunk(
                text=chunk["text"],
                source=chunk["source"],
                section=chunk.get("section", chunk["source"]),
                page=chunk["page"],
                chunk_id=chunk["chunk_id"],
                bi_encoder_score=c["bi_score"],
                cross_encoder_score=c["ce_score"]
            ))

        return results, latencies


# ---------------------------------------------------------------------------
# Standalone test — run this file directly to verify retrieval works
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Retriever Standalone Test ===\n")

    retriever = Retriever()

    test_queries = [
        ("What is the maximum permitted overall width of the car?", None),
        ("What are the DRS activation rules?",                      None),
        ("What are the curfew regulations for teams?",              "Section F – Operational Regulations"),
        ("What are the restricted working hours at the circuit?", "Section F – Operational Regulations"),
    ]

    for query, section_filter in test_queries:
        print(f"Query: {query}")
        if section_filter:
            print(f"Filter: {section_filter}")

        results, latencies = retriever.retrieve(query, section_filter)

        print(f"Latency: bi-encoder={latencies['bi_encoder_ms']:.0f}ms | "
              f"cross-encoder={latencies['cross_encoder_ms']:.0f}ms | "
              f"total={latencies['total_retrieval_ms']:.0f}ms")
        print(f"Results: {len(results)} chunks returned")

        for i, chunk in enumerate(results):
            print(f"\n  [{i+1}] {chunk.section} | Page {chunk.page}")
            print(f"       CE score: {chunk.cross_encoder_score:.4f} | "
                  f"L2 dist: {chunk.bi_encoder_score:.4f}")
            # Print first 120 chars of chunk text as preview
            preview = chunk.text[:120].replace("\n", " ")
            print(f"       Preview: {preview}...")

        print("\n" + "="*60 + "\n")