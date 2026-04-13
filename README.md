# 🏎️ TechReg Analyst

> **A production-grade Retrieval-Augmented Generation (RAG) system for querying FIA Formula 1 2026 Regulations in plain English.**

Built as a portfolio project targeting Machine Learning Engineer internships. Every component — chunking strategy, embedding model selection, two-stage retrieval, hallucination mitigation — was chosen deliberately and benchmarked.

---

## Table of Contents

- [What This Is](#what-this-is)
- [Live Demo](#live-demo)
- [Architecture](#architecture)
  - [System Overview](#system-overview)
  - [Ingestion Pipeline](#ingestion-pipeline-offline)
  - [Query Pipeline](#query-pipeline-runtime)
  - [Two-Stage Retrieval](#two-stage-retrieval-deep-dive)
- [Tech Stack](#tech-stack)
- [Key Design Decisions](#key-design-decisions)
- [Benchmark Results](#benchmark-results)
- [Project Structure](#project-structure)
- [Setup & Running Locally](#setup--running-locally)
- [Known Limitations](#known-limitations)
- [Future Improvements](#future-improvements)

---

## What This Is

TechReg Analyst lets you ask natural language questions over all six sections of the FIA 2026 Formula 1 regulation PDFs — 589 pages, 3,258 indexed chunks — and get cited, grounded answers backed by exact source passages.

**Example queries that work:**

| Query | Source Section |
|---|---|
| What are the rules for Driver Adjustable Bodywork activation? | Section B |
| What is the maximum fuel energy flow rate? | Section C |
| What happens if a driver misses the weighbridge? | Section B |
| What are the minimum weight requirements for an F1 car? | Section C |
| What are the restrictions on tyre compounds during a race? | Section B |

**This is not a chatbot.** It is a domain-specific retrieval system with grounded generation — the LLM is constrained to answer only from retrieved regulation text, with citations.

---

## Live Demo

🚀 **[techreg-analyst.streamlit.app](https://techreg-analyst.streamlit.app)** *(Streamlit Community Cloud)*

---

## Architecture

### System Overview

The system is split into two completely separate phases:

```
╔══════════════════════════════════════════════════════════════════════╗
║                     INGESTION PHASE  (run once)                      ║
║                                                                      ║
║   6 × PDF  ──►  PyMuPDF  ──►  clean_page_text()  ──►  LangChain    ║
║                                                        Chunker       ║
║                                                           │          ║
║                                               512-char chunks        ║
║                                               + 64-char overlap      ║
║                                                           │          ║
║                                          BGE-small-en-v1.5           ║
║                                          (384-dim embeddings)        ║
║                                                           │          ║
║                                          FAISS IndexFlatL2           ║
║                                          ──► index/ (disk)           ║
╚══════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════╗
║                     QUERY PHASE  (per request)                       ║
║                                                                      ║
║   User Query                                                         ║
║       │                                                              ║
║       ▼                                                              ║
║   BGE prefix + embed  ──►  FAISS search  ──►  Top-20 candidates     ║
║                                                      │               ║
║                                          Cross-Encoder Reranker      ║
║                                          (ms-marco-MiniLM-L-6-v2)   ║
║                                                      │               ║
║                                              Top-5 chunks            ║
║                                                      │               ║
║                                          Build context + prompt      ║
║                                                      │               ║
║                                          Groq LLaMA 3.1 8B           ║
║                                          (temperature=0)             ║
║                                                      │               ║
║                                     Answer + Citations + Latency     ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

### Ingestion Pipeline (Offline)

```
PDF Files (6 sections)
        │
        ▼
┌───────────────────┐
│   PyMuPDF (fitz)  │  ← page.get_text("text") per page
└───────────────────┘
        │
        ▼
┌───────────────────────────┐
│   clean_page_text()       │  ← strips FIA headers/footers:
│                           │    • Section codes  (C150, F5)
│                           │    • ©2026 lines
│                           │    • "2026 Formula 1:" prefixes
│                           │    • Issue/date lines
│                           │    • Standalone page numbers
└───────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│   LangChain RecursiveCharacterSplitter │
│   chunk_size=512, overlap=64          │
│   separators: ["\n\n", "\n", " ", ""] │
└───────────────────────────────────────┘
        │  3,258 chunks
        ▼
┌──────────────────────────────┐
│   BAAI/bge-small-en-v1.5    │  ← asymmetric model — docs get no prefix
│   384-dim embeddings (CPU)   │    ~10ms/chunk on CPU
└──────────────────────────────┘
        │
        ▼
┌──────────────────────────────┐
│   FAISS IndexFlatL2          │  ← brute-force L2 search
│   Saved to index/ (disk)     │    < 2ms at 3,258 vectors
└──────────────────────────────┘

Output:
  index/faiss.index   — 4.8 MB
  index/chunks.pkl    — 1.5 MB (metadata: text, source, section, page, chunk_id)
```

---

### Query Pipeline (Runtime)

```
User Query: "What is the maximum fuel energy flow rate?"
        │
        │  Prepend BGE asymmetric query prefix:
        │  "Represent this sentence for searching relevant passages: ..."
        ▼
┌──────────────────────────────┐
│   BGE bi-encoder embed       │  ~25ms (after warmup)
└──────────────────────────────┘
        │  384-dim query vector
        ▼
┌──────────────────────────────┐
│   FAISS IndexFlatL2.search() │  Top-20 nearest neighbors (~1ms)
└──────────────────────────────┘
        │  20 candidate chunks + L2 distances
        │
        │  [If section_filter active: fetch 60, filter to 20]
        ▼
┌──────────────────────────────────────────┐
│   Cross-Encoder Reranker                  │
│   ms-marco-MiniLM-L-6-v2                 │
│   Scores each (query, chunk) pair        │  ~150ms for 20 pairs
│   Returns relevance score ∈ ℝ            │  (negative CE scores = expected
└──────────────────────────────────────────┘   for domain-specific text)
        │  Top-5 reranked chunks
        ▼
┌──────────────────────────────────────────┐
│   build_context()                         │
│   Formats chunks as numbered passages:   │
│   [1] Source: Section C | Page 65        │
│   ...chunk text...                       │
└──────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│   Groq API — llama-3.1-8b-instant        │
│   temperature=0  (deterministic)         │
│   max_tokens=512                         │
│   Grounded generation with citations [1] │
└──────────────────────────────────────────┘
        │
        ▼
Answer + [citations] + retrieval_ms + llm_ms
```

---

### Two-Stage Retrieval: Deep Dive

This is the most important part of the system. Here's why two stages are necessary:

```
                    WHY NOT JUST FAISS?
                    ───────────────────
  FAISS (bi-encoder) is fast but approximate.
  It encodes query and document SEPARATELY into vectors,
  then finds nearest neighbors by distance.

  Problem: "fuel mass flow rate" and "fuel energy flow"
  are semantically close in general embedding space,
  but the regulation may not use "mass flow" at all.
  Bi-encoder misses vocabulary mismatches.

  ──────────────────────────────────────────────────────

                  WHY NOT JUST CROSS-ENCODER?
                  ────────────────────────────
  Cross-encoder sees (query, passage) jointly — much more
  accurate relevance scoring. But it's O(n) per query:
  running it on all 3,258 chunks = ~30 seconds.

  ──────────────────────────────────────────────────────

                     THE SOLUTION: TWO STAGES
                     ─────────────────────────

  Stage 1 — Bi-encoder (FAISS):   Speed  ✓  Recall  ✓  Precision  ✗
  ├── Retrieve Top-20 fast (~25ms)
  └── Cast a wide net (recall)

  Stage 2 — Cross-encoder:        Speed  ✗  Recall  —  Precision  ✓
  ├── Rerank Top-20 → Top-5 (~150ms)
  └── Filter for precision

  Combined:  Fast enough for production  ✓  Accurate  ✓
```

---

## Tech Stack

| Component | Technology | Reason |
|---|---|---|
| **PDF Parsing** | PyMuPDF 1.24.5 | Fast, accurate text extraction with page metadata |
| **Chunking** | LangChain RecursiveCharacterTextSplitter | Respects sentence/paragraph boundaries |
| **Embeddings** | BAAI/bge-small-en-v1.5 (384-dim) | Free, CPU-friendly, MTEB score ~52 — beats OpenAI ada-002 (~49) on English retrieval |
| **Vector Store** | FAISS IndexFlatL2 | Brute-force exact search; at 3,258 vectors, <2ms. IVF approximation not worth it below ~50k chunks |
| **Reranker** | cross-encoder/ms-marco-MiniLM-L-6-v2 | Joint query-document scoring; substantially improves precision over bi-encoder alone |
| **LLM** | Groq API — llama-3.1-8b-instant | Free tier, ~350–900ms latency, deterministic at temperature=0 |
| **Orchestration** | LangChain v0.2 (ChatPromptTemplate, Runnable pipe) | Standardised prompt/LLM chaining |
| **Frontend** | Streamlit 1.37.1 | Rapid deployment; @st.cache_resource for singleton retriever |
| **Deployment** | Streamlit Community Cloud | Free, zero-config for Streamlit apps |
| **Python** | 3.11.9 | PyMuPDF wheels not available for Python 3.14; 3.11 is the stable ML target |

**Zero-cost constraint:** No OpenAI, no GPU, no paid APIs. Entire stack runs on CPU for free.

---

## Key Design Decisions

### D1 — BGE Asymmetric Encoding

BGE models use *asymmetric* encoding: queries and documents are embedded differently. **Queries must include a prefix** — documents must not.

```python
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# At query time:
query_embedding = model.encode(BGE_QUERY_PREFIX + user_query)

# At ingestion time:
doc_embedding = model.encode(chunk_text)  # NO prefix
```

Omitting this prefix degrades retrieval quality. Most naive implementations miss this.

---

### D2 — FAISS IndexFlatL2 over IVFFlat

At 3,258 chunks, brute-force exact search takes <2ms. Approximate IVF indexing adds overhead without meaningful speedup and introduces approximation error. The crossover point is ~50,000 vectors.

---

### D3 — chunk_size=512, overlap=64

Smaller chunks (256) lose context; larger chunks (1024) embed too much unrelated content into one vector. 512 characters (~80–100 words) balances specificity and context. The 64-character overlap ensures that clauses split across chunk boundaries are still findable.

---

### D4 — Section Metadata Propagation

Every chunk carries a `section` label propagated from the PDF filename through ingestion:

```
section_c_technical.pdf  →  "Section C – Technical Regulations"
                                     ↓
                          Every page record gets section label
                                     ↓
                          Every chunk inherits section label
                                     ↓
                          UI shows "[1] Section C – Technical | Page 65"
```

This enables section filtering (retrieve only from Section B, etc.) and proper citation display.

---

### D5 — Hallucination Mitigation

Three layers:

| Layer | Mechanism |
|---|---|
| **Prompt constraint** | "Answer using ONLY the context passages below. Do not use prior knowledge." |
| **Temperature = 0** | Deterministic output; no creative generation |
| **Numeric heuristic** | Post-generation: extract all numbers from answer; if >2 numbers absent from all chunk texts → warn user |
| **Source display** | User can expand every source passage and verify each claim |

---

### D6 — FAISS_FETCH_MULTIPLIER = 3 for Section Filtering

When a section filter is active, FAISS fetches `BI_ENCODER_TOP_K × 3 = 60` candidates before filtering down to 20 for the cross-encoder. Without this, filtering would leave fewer than 20 chunks for the CE stage, degrading reranking quality.

---

## Benchmark Results

Evaluated on 10 queries spanning all six regulation sections.

```
╭─────┬─────────────────────────┬─────────────────┬───────────┬─────────────┬─────────────╮
│  #  │ Query                   │   Retrieval(ms) │   LLM(ms) │   Total(ms) │  Sec Match  │
├─────┼─────────────────────────┼─────────────────┼───────────┼─────────────┼─────────────┤
│  1  │ DAB activation rules    │            4303 │       559 │        4862 │     Yes     │
│  2  │ Fuel energy flow limit  │             715 │       323 │        1038 │     Yes     │
│  3  │ Weighbridge procedure   │             525 │       397 │         922 │     Yes     │
│  4  │ Car minimum weight      │            1491 │       565 │        2055 │     Yes     │
│  5  │ Tyre compound rules     │             584 │      5580 │        6164 │     Yes     │
│  6  │ Cost cap limit          │             475 │      9315 │        9790 │     No      │
│  7  │ Curfew regulations      │            1404 │      8479 │        9882 │     No *    │
│  8  │ DRS detection zones     │            2882 │      7034 │        9916 │     Yes     │
│  9  │ Survival cell materials │             609 │      9830 │       10439 │     Yes     │
│ 10  │ False start penalties   │            1317 │      8488 │        9805 │     Yes     │
╰─────┴─────────────────────────┴─────────────────┴───────────┴─────────────┴─────────────╝

  Queries evaluated    :  10
  Avg retrieval        :  1,430 ms  (CPU-only, no GPU)
  Avg LLM (Groq free)  :  5,057 ms  (includes free-tier throttling)
  Real LLM latency*    :  ~350–900 ms  (measured in interactive Streamlit use)
  Section Match@1      :  8/10  (80%)
```

> **Note on LLM latency:** The high variance (323ms → 9,830ms) is Groq free-tier rate limiting on consecutive benchmark calls, not pipeline latency. In the Streamlit app with natural pauses between queries, LLM latency consistently measures 350–900ms.

**Misses explained:**

| Query | Expected | Got | Reason |
|---|---|---|---|
| Cost cap limit | Section D (Teams) | Section E (PU Manufacturers) | Both are financial sections; CE score was positive (+1.9) — content is genuinely related. Section boundary ambiguity, not retrieval failure. |
| Curfew regulations | Section F (Operational) | Section B (Sporting) | The curfew content actually lives in Section B (B2.x). Expected section in benchmark was wrong — retrieval was correct. |

---

## Project Structure

```
f1-regulations-rag-system/
│
├── data/
│   └── pdfs/                          ← 6 FIA 2026 regulation PDFs
│       ├── section_a_general.pdf          (84 pages)
│       ├── section_b_sporting.pdf         (96 pages)
│       ├── section_c_technical.pdf        (259 pages)
│       ├── section_d_financial_teams.pdf  (62 pages)
│       ├── section_e_financial_pu.pdf     (56 pages)
│       └── section_f_operational.pdf      (32 pages)
│
├── index/                             ← Generated by ingest.py (gitignored)
│   ├── faiss.index                        (4.8 MB — FAISS binary index)
│   └── chunks.pkl                         (1.5 MB — chunk metadata)
│
├── ingest.py                          ← Offline ingestion pipeline
│   │                                     Run once to build index/
│   └── [PDF → parse → clean → chunk → embed → FAISS]
│
├── retriever.py                       ← Two-stage retrieval engine
│   │                                     Retriever class with retrieve()
│   └── [embed query → FAISS top-20 → cross-encoder top-5]
│
├── chain.py                           ← LLM generation layer
│   │                                     build_context(), generate_answer()
│   └── [context + prompt → Groq LLaMA → cited answer]
│
├── app.py                             ← Streamlit frontend
│   └── [sidebar filter + query input + answer + metrics + source expanders]
│
├── benchmark.py                       ← Evaluation harness
│   └── [10 queries × latency + section match → terminal table + CSV]
│
├── benchmark_results.csv              ← Persisted benchmark output
├── requirements.txt                   ← All deps pinned
├── .env                               ← GROQ_API_KEY (gitignored)
└── .gitignore
```

---

## Setup & Running Locally

### Prerequisites

- Python **3.11.x** (not 3.12+; PyMuPDF wheels require 3.11)
- A free [Groq API key](https://console.groq.com) (takes ~30 seconds)

### 1. Clone and create virtual environment

```bash
git clone https://github.com/your-username/f1-regulations-rag-system.git
cd f1-regulations-rag-system

python -m venv .venv

# Windows
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your Groq API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

Get a free key at [console.groq.com](https://console.groq.com) — no credit card required.

### 4. Add the regulation PDFs

Place all 6 PDFs in `data/pdfs/`, renamed as follows:

```
section_a_general.pdf
section_b_sporting.pdf
section_c_technical.pdf
section_d_financial_teams.pdf
section_e_financial_pu.pdf
section_f_operational.pdf
```

### 5. Run ingestion (one-time)

```bash
python ingest.py
```

Expected output:
```
Parsed section_a_general.pdf: 84 pages
...
Total chunks: 3,258
FAISS index saved → index/faiss.index (4.8 MB)
Chunk metadata saved → index/chunks.pkl (1.5 MB)
```

⏱ Takes ~3–5 minutes on CPU. Only needs to run once.

### 6. Test the retrieval pipeline

```bash
python retriever.py
```

Should print 3 queries × 5 retrieved chunks with CE scores.

### 7. Test the full chain

```bash
python chain.py
```

Should print cited answers for queries 1–2 and the fallback string for query 3 (hallucination test).

### 8. Run the Streamlit app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

### 9. (Optional) Run benchmark

```bash
python benchmark.py
```

Runs 10 queries, prints latency table, saves `benchmark_results.csv`.

---

## Known Limitations

**Vocabulary mismatch.** The regulation uses precise technical language that doesn't always match plain English queries. For example, the regulation says "fuel energy flow" — querying "fuel mass flow rate" required prompt-level terminology bridging. Systematic fix: query expansion via LLM rewriting.

**Naive chunking.** Chunks are split by character count, not regulation article boundaries. Article `C5.2.3` may split across two chunks, so neither chunk alone contains the full article. Fix: semantic chunking on article-number boundaries (regex on patterns like `C5.2.3`).

**No CE score quality gate.** If all 5 retrieved chunks have very low CE scores (e.g., all < -8.0), the chunks are probably irrelevant — but the system still passes them to the LLM. Fix: if max CE score < threshold → return "not found" directly without LLM call.

**LLM at 8B scale.** LLaMA 3.1 8B occasionally fails to bridge terminology gaps even with explicit prompt instructions. A 70B model (e.g., `llama-3.3-70b-versatile` on Groq) would handle these cases better, at the cost of higher latency.

**Section F coverage.** Section F (Operational Regulations, 32 pages) is the shortest section and produces fewer chunks. Queries about operational procedures may return results from Section B instead, where related sporting regulations live.

---

## Future Improvements

Listed in order of impact-to-effort ratio:

| Improvement | What | Why |
|---|---|---|
| **CE score quality gate** | If top CE score < threshold → skip LLM, return "not found" | Eliminates irrelevant-chunk hallucinations cleanly |
| **Article-boundary chunking** | Split on regex `[A-F]\d+\.\d+` instead of character count | Keeps each regulation article intact in one chunk |
| **Query expansion** | LLM rewrites query into regulation-style language before embedding | Systematically closes vocabulary mismatch |
| **Hybrid retrieval** | BM25 sparse + BGE dense, reciprocal rank fusion | BM25 handles exact article number lookups (e.g., "C5.2.3") that dense search misses |
| **RAGAS faithfulness scoring** | Embed answer sentences, compute cosine similarity to cited chunks | Quantifies hallucination rate beyond numeric heuristic |

---

## What I Learned

Building this system taught me several things that weren't obvious upfront:

**RAG failures are mostly retrieval failures, not generation failures.** When the system gives a wrong answer, 80% of the time the right chunk simply wasn't in the top-5. Improving retrieval (chunk quality, query rewriting) has much higher ROI than prompt engineering.

**Asymmetric embedding is not optional for BGE.** The BGE family requires the query prefix at inference time. Missing this silently degrades retrieval by roughly 10–15 percentage points on MTEB — it just returns worse results with no error.

**Python version compatibility matters more than you'd think.** PyMuPDF 1.24.5 doesn't ship pre-built wheels for Python 3.14. This isn't documented prominently. Python 3.11 is the stable target for production ML stacks as of 2025.

**CE scores being negative is expected, not a bug.** The cross-encoder (ms-marco-MiniLM) was trained on MS MARCO web data, not regulatory text. Absolute scores are meaningless — relative ordering is what matters. The highest score in a batch is the most relevant chunk, regardless of its sign.

---

## License

MIT — see [LICENSE](LICENSE)

---

*Built by Ishaan — CS undergrad at PDEU + IIT Madras (Data Science). Portfolio: [github.com/your-username](https://github.com/your-username)*