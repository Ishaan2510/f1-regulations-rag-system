"""
retriever.py
------------
Query-time retrieval logic.

Two-stage pipeline:
  Stage 1: Bi-encoder + FAISS → top-20 candidates  (recall-optimized)
  Stage 2: Cross-encoder re-ranking → top-5 results (precision-optimized)

In C++ terms: this is a two-pass filter. Stage 1 is a cheap O(n) scan.
Stage 2 is an expensive O(k) scorer applied only to Stage 1's survivors.
"""

import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

FAISS_INDEX_PATH = Path("index/faiss.index")
CHUNKS_PATH = Path("index/chunks.pkl")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# BGE instruction prefix — applied to QUERIES only, not documents
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

BI_ENCODER_TOP_K = 20   # How many candidates to retrieve in Stage 1
RERANKER_TOP_K = 5       # How many to return after Stage 2 re-ranking


@dataclass
class RetrievedChunk:
    """
    Struct-like dataclass for a single retrieved result.
    In C++ this would be: struct RetrievedChunk { ... };
    """
    text: str
    source: str
    section: str
    page: int
    chunk_id: int
    bi_encoder_score: float   # L2 distance (lower = more similar)
    cross_encoder_score: float  # Relevance logit (higher = more relevant)


class Retriever:
    """
    Stateful retriever class — loads models and index once at construction,
    then serves queries. Designed for Streamlit's @st.cache_resource pattern.

    In C++ terms: this is a class with a constructor that does heavy I/O
    and member functions that do cheap query processing.
    """

    def __init__(self):
        print("Initializing Retriever...")

        # --- Load FAISS index from disk ---
        # This is a memory-mapped file load — fast, no re-computation
        if not FAISS_INDEX_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {FAISS_INDEX_PATH}. "
                "Run ingest.py first."
            )
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))
        print(f"FAISS index loaded: {self.index.ntotal} vectors")

        # --- Load chunk metadata (text + page numbers) ---
        with open(CHUNKS_PATH, "rb") as f:
            self.chunks: list[dict] = pickle.load(f)
        print(f"Loaded {len(self.chunks)} chunk records")

        # --- Load bi-encoder (for query embedding) ---
        self.bi_encoder = SentenceTransformer(EMBEDDING_MODEL)

        # --- Load cross-encoder (for re-ranking) ---
        # CrossEncoder is a separate class from SentenceTransformer.
        # It takes (query, passage) pairs, not individual texts.
        self.cross_encoder = CrossEncoder(RERANKER_MODEL)

        print("Retriever ready.")

    def retrieve(self, query: str, section_filter: str = None) -> tuple[list[RetrievedChunk], dict]:
        """
        section_filter: if provided, only retrieve chunks from this section.
        Example: section_filter="Section C – Technical Regulations"
        
        This is like a SQL WHERE clause applied after FAISS search.
        In C++ terms: filter the results vector by a predicate before re-ranking.
        """
        latencies = {}
        t0 = time.perf_counter()

        prefixed_query = BGE_QUERY_PREFIX + query
        query_embedding = self.bi_encoder.encode(
            [prefixed_query],
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype(np.float32)

        # Fetch more candidates when filtering, since some will be dropped
        fetch_k = BI_ENCODER_TOP_K * 3 if section_filter else BI_ENCODER_TOP_K
        distances, indices = self.index.search(query_embedding, fetch_k)

        t1 = time.perf_counter()
        latencies["bi_encoder_ms"] = (t1 - t0) * 1000

        candidates = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]

            # Apply section filter — skip chunks not from the selected section
            if section_filter and chunk.get("section") != section_filter:
                continue

            candidates.append({"chunk": chunk, "bi_score": float(dist)})

            # Stop once we have enough candidates for re-ranking
            if len(candidates) >= BI_ENCODER_TOP_K:
                break

        if not candidates:
            return [], {"bi_encoder_ms": latencies["bi_encoder_ms"],
                        "cross_encoder_ms": 0, "total_retrieval_ms": latencies["bi_encoder_ms"]}

        t2 = time.perf_counter()
        pairs = [(query, c["chunk"]["text"]) for c in candidates]
        scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
        t3 = time.perf_counter()
        latencies["cross_encoder_ms"] = (t3 - t2) * 1000

        for i, score in enumerate(scores):
            candidates[i]["ce_score"] = float(score)

        candidates.sort(key=lambda x: x["ce_score"], reverse=True)
        top_candidates = candidates[:RERANKER_TOP_K]

        results = []
        for c in top_candidates:
            chunk = c["chunk"]
            results.append(RetrievedChunk(
                text=chunk["text"],
                source=chunk["source"],
                page=chunk["page"],
                chunk_id=chunk["chunk_id"],
                section=chunk.get("section", chunk["source"]),   # NEW field
                bi_encoder_score=c["bi_score"],
                cross_encoder_score=c["ce_score"]
            ))

        latencies["total_retrieval_ms"] = (t3 - t0) * 1000
        return results, latencies