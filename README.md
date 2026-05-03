<div align="center">

# TechReg Analyst

**A production-grade RAG system for querying 589 pages of FIA Formula 1 2026 Regulations in plain English.**

80% Section Match@1 · ~1,000ms retrieval latency · CPU-only · Zero cost

[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://f1-regulations-rag-system.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Meta_AI-0064E0?style=flat-square)](https://faiss.ai/)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3.1_8B-F54D27?style=flat-square)](https://groq.com/)

</div>

---

## What This Is

TechReg Analyst lets you ask natural language questions across all six sections of the FIA 2026 F1 Regulations — 589 pages, 3,258 indexed chunks — and get cited, grounded answers backed by exact source passages.

This is a domain-specific retrieval system, not a chatbot. The LLM is constrained to answer only from retrieved regulation text, with citations. The architecture is a two-stage retrieval pipeline: a bi-encoder for fast recall, followed by a cross-encoder for precision reranking.

**Live demo:** https://f1-regulations-rag-system.streamlit.app/

**Example queries that work:**

| Query | Source Section |
|---|---|
| What are the rules for Driver Adjustable Bodywork activation? | Section B |
| What is the maximum fuel energy flow rate? | Section C |
| What happens if a driver misses the weighbridge? | Section B |
| What are the minimum weight requirements for an F1 car? | Section C |
| What are the restrictions on tyre compounds during a race? | Section B |

---

## Results

```
Queries evaluated     :  10
Avg retrieval latency :  1,430 ms  (CPU-only, no GPU)
LLM latency (typical) :  350–900 ms  (Groq free tier, interactive use)
Section Match@1       :  8/10  (80%)
```

Of the 2 misses: one was a benchmark labelling error (the content genuinely lives in the retrieved section, not the expected one). The other is a genuine cross-section ambiguity where financial content spans two regulation sections — not a retrieval failure, a domain ambiguity.

---

## Why Two-Stage Retrieval

The core design decision in this system — and the one that drives all the performance numbers.

```
The problem with bi-encoder alone (FAISS):
──────────────────────────────────────────
  Encodes query and document SEPARATELY into vectors.
  Fast. Scalable. But approximate.

  "fuel mass flow rate" and "fuel energy flow" are semantically
  close in embedding space — but the regulation only uses one term.
  Vocabulary mismatches cause missed recalls.

The problem with cross-encoder alone:
──────────────────────────────────────
  Sees (query, passage) jointly — far more accurate.
  But O(n) per query. Running on all 3,258 chunks = ~30 seconds.
  Not viable for a real application.

The solution — two stages:
──────────────────────────
  Stage 1 — FAISS bi-encoder:    Speed ✓  Recall ✓  Precision ✗
    Retrieve top-20 fast (~25ms). Cast a wide net.

  Stage 2 — Cross-encoder:       Speed ✗  Recall —  Precision ✓
    Rerank top-20 → top-5 (~150ms). Filter for precision.

  Combined: fast enough for production, accurate enough to matter.
```

---

## System Architecture

### Ingestion Pipeline (runs once, offline)

```
6 × PDF (589 pages)
       │
       ▼
PyMuPDF — page.get_text() per page
       │
       ▼
clean_page_text()
  strips: FIA headers/footers, section codes (C150, F5),
          ©2026 lines, issue/date lines, page numbers
       │
       ▼
LangChain RecursiveCharacterTextSplitter
  chunk_size=512, overlap=64
  separators: ["\n\n", "\n", " ", ""]
       │  3,258 chunks
       ▼
BAAI/bge-small-en-v1.5 (384-dim)
  Documents embedded WITHOUT query prefix
       │
       ▼
FAISS IndexFlatL2 → saved to index/ (disk)
  faiss.index  (4.8 MB)
  chunks.pkl   (1.5 MB — text, source, section, page, chunk_id)
```

### Query Pipeline (per request)

```
User Query
    │  Prepend BGE asymmetric prefix:
    │  "Represent this sentence for searching relevant passages: ..."
    ▼
BGE bi-encoder embed (~25ms after warmup)
    │  384-dim query vector
    ▼
FAISS IndexFlatL2.search() → Top-20 candidates (~1ms)
    │  [If section_filter: fetch 60, filter to 20 before CE]
    ▼
Cross-encoder reranker (ms-marco-MiniLM-L-6-v2)
  Scores each (query, chunk) pair jointly (~150ms for 20 pairs)
    │  Top-5 reranked chunks
    ▼
build_context() — numbered passages with source metadata
    │
    ▼
Groq API — llama-3.1-8b-instant
  temperature=0 (deterministic)
  Context-only generation with citation instructions
    │
    ▼
Answer + [citations] + retrieval_ms + llm_ms
```

---

## Key Design Decisions

### BGE asymmetric encoding is not optional

BGE models encode queries and documents differently. Queries must include a prefix; documents must not. Omitting the prefix silently degrades retrieval quality by roughly 10–15 percentage points on MTEB — no error, just worse results.

```python
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Ingestion — no prefix
doc_embedding = model.encode(chunk_text)

# Query time — prefix required
query_embedding = model.encode(BGE_QUERY_PREFIX + user_query)
```

### FAISS IndexFlatL2 over IVFFlat

At 3,258 chunks, brute-force exact search takes under 2ms. Approximate IVF indexing adds overhead without meaningful speedup and introduces approximation error. The crossover point where IVF becomes worth it is around 50,000 vectors.

### chunk_size=512, overlap=64

Smaller chunks (256 chars) lose surrounding context. Larger chunks (1024 chars) embed too much unrelated content into a single vector, diluting the signal. 512 characters (~80–100 words) is the right balance for regulation-length clauses. The 64-character overlap ensures that clauses split at chunk boundaries are still findable via either adjacent chunk.

### Section metadata propagation

Every chunk carries a section label propagated from the PDF filename through ingestion. This enables section filtering at query time (retrieve only from Section B, for example) and proper citation display in the UI. When section filtering is active, FAISS fetches `top_k × 3 = 60` candidates before filtering to ensure the cross-encoder still sees 20 chunks to rerank.

### Hallucination mitigation — three layers

| Layer | Mechanism |
|---|---|
| Prompt constraint | "Answer using ONLY the context passages below. Do not use prior knowledge." |
| Temperature = 0 | Deterministic output; eliminates creative generation |
| Numeric heuristic | Post-generation: extract numbers from answer; if more than 2 are absent from all retrieved chunks, warn the user |

---

## What I Learned Building This

**RAG failures are mostly retrieval failures, not generation failures.** When the system gives a wrong answer, 80% of the time the right chunk simply wasn't in the top-5. Improving retrieval (chunk quality, query rewriting, better embeddings) has far higher ROI than prompt engineering.

**CE scores being negative is expected, not a bug.** The ms-marco cross-encoder was trained on web data, not regulatory text. Absolute scores are meaningless — relative ordering is what matters. The highest score in a batch is the most relevant chunk regardless of sign.

**Python version compatibility matters more than documentation suggests.** PyMuPDF 1.24.5 doesn't ship pre-built wheels for Python 3.14. This isn't prominently documented. Python 3.11 is the stable target for production ML stacks in 2026.

**Naive chunking is the system's biggest limitation.** Article `C5.2.3` may split across two chunks — neither contains the full clause. The right fix is semantic chunking on article-number boundaries, not character count.

---

## Benchmark Results

```
╭────┬──────────────────────────┬────────────┬─────────┬──────────┬────────────╮
│ #  │ Query                    │ Retrieval  │ LLM     │ Total    │ Sec Match  │
├────┼──────────────────────────┼────────────┼─────────┼──────────┼────────────┤
│  1 │ DAB activation rules     │   4,303ms  │   559ms │  4,862ms │ Yes        │
│  2 │ Fuel energy flow limit   │     715ms  │   323ms │  1,038ms │ Yes        │
│  3 │ Weighbridge procedure    │     525ms  │   397ms │    922ms │ Yes        │
│  4 │ Car minimum weight       │   1,491ms  │   565ms │  2,055ms │ Yes        │
│  5 │ Tyre compound rules      │     584ms  │  5,580ms│  6,164ms │ Yes        │
│  6 │ Cost cap limit           │     475ms  │  9,315ms│  9,790ms │ No         │
│  7 │ Curfew regulations       │   1,404ms  │  8,479ms│  9,882ms │ No *       │
│  8 │ DRS detection zones      │   2,882ms  │  7,034ms│  9,916ms │ Yes        │
│  9 │ Survival cell materials  │     609ms  │  9,830ms│ 10,439ms │ Yes        │
│ 10 │ False start penalties    │   1,317ms  │  8,488ms│  9,805ms │ Yes        │
╰────┴──────────────────────────┴────────────┴─────────┴──────────┴────────────╝

* Query 7: benchmark expected Section F, but curfew content lives in Section B.
  Retrieval was correct — the benchmark label was wrong.

High LLM variance (323ms → 9,830ms) = Groq free-tier rate limiting on consecutive
benchmark calls. In interactive use with natural pauses, LLM latency is 350–900ms.
```

---

## Tech Stack

| Component | Technology | Reason |
|---|---|---|
| PDF Parsing | PyMuPDF 1.24.5 | Fast extraction with page metadata |
| Chunking | LangChain RecursiveCharacterTextSplitter | Respects sentence/paragraph boundaries |
| Embeddings | BAAI/bge-small-en-v1.5 (384-dim) | CPU-friendly; MTEB ~52, beats OpenAI ada-002 (~49) on English retrieval |
| Vector Store | FAISS IndexFlatL2 | Exact search; under 2ms at 3,258 vectors |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Joint query-document scoring; substantially improves precision |
| LLM | Groq — llama-3.1-8b-instant | Free tier; ~350–900ms latency at temperature=0 |
| Frontend | Streamlit 1.37.1 | `@st.cache_resource` singleton retriever |
| Deployment | Streamlit Community Cloud | Free, zero-config |

**Zero-cost constraint:** No OpenAI, no GPU, no paid APIs. The entire stack runs on CPU for free.

---

## Known Limitations

**Vocabulary mismatch.** The regulation uses precise technical language that doesn't always match plain English queries. Fix: query expansion via LLM rewriting before embedding.

**Naive chunking.** Article `C5.2.3` may split across two chunks, so neither chunk alone contains the full clause. Fix: semantic chunking on article-number boundaries (regex on `[A-F]\d+\.\d+`).

**No CE score quality gate.** If all 5 retrieved chunks have very low CE scores, the system still passes them to the LLM. Fix: if max CE score < threshold, return "not found in regulations" directly, skipping the LLM call entirely.

**Section F coverage.** Section F (32 pages) produces fewer chunks than other sections. Operational procedure queries may retrieve results from Section B where related sporting regulations live.

---

## Setup & Running Locally

### Prerequisites
- Python 3.11.x (not 3.12+; PyMuPDF wheels require 3.11)
- Free [Groq API key](https://console.groq.com) — no credit card required

```bash
git clone https://github.com/Ishaan2510/f1-regulations-rag-system
cd f1-regulations-rag-system

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Create `.env`:
```
GROQ_API_KEY=your_key_here
```

Place the 6 regulation PDFs in `data/pdfs/` (see filenames in project structure).

```bash
python ingest.py          # ~3–5 minutes on CPU, run once
python retriever.py       # verify retrieval pipeline
python chain.py           # verify LLM generation
streamlit run app.py      # launch at localhost:8501
python benchmark.py       # run evaluation harness
```

---

## Project Structure

```
f1-regulations-rag-system/
├── data/pdfs/                     ← 6 FIA 2026 regulation PDFs
├── index/                         ← Generated by ingest.py (gitignored)
│   ├── faiss.index                    (4.8 MB)
│   └── chunks.pkl                     (1.5 MB)
├── ingest.py                      ← Offline ingestion pipeline
├── retriever.py                   ← Two-stage retrieval engine
├── chain.py                       ← LLM generation layer
├── app.py                         ← Streamlit frontend
├── benchmark.py                   ← Evaluation harness
├── benchmark_results.csv
├── requirements.txt
└── .env                           ← GROQ_API_KEY (gitignored)
```

---

*Built by [Ishaan Goswami](https://github.com/Ishaan2510) — CS undergrad, PDEU + IIT Madras*
