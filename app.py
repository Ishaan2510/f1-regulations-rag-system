# app.py — Streamlit frontend for TechReg Analyst
# Run with: streamlit run app.py

import os
import re
import time
import streamlit as st

# Import our two pipeline modules
from retriever import Retriever
from chain import generate_answer

# ─────────────────────────────────────────────
# PAGE CONFIG — must be the FIRST Streamlit call
# Sets browser tab title, icon, and layout width
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="TechReg Analyst",
    page_icon="🏎️",
    layout="wide",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — F1-inspired dark theme
# Injected once at startup; applies globally.
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@300;400;600;700;900&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Titillium Web', sans-serif;
}

/* ── App background ── */
.stApp {
    background-color: #080808;
    color: #F0F0F0;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #0F0F0F;
    border-right: 1px solid #1E1E1E;
}
[data-testid="stSidebar"] * {
    font-family: 'Titillium Web', sans-serif;
}

/* ── Sidebar title ── */
[data-testid="stSidebar"] h1 {
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #E10600;
}

/* ── Main title ── */
h1 {
    font-size: 3rem !important;
    font-weight: 900 !important;
    letter-spacing: -0.02em;
    color: #FFFFFF !important;
}

/* ── Section headings ── */
h3 {
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    font-size: 0.85rem !important;
    color: #888888 !important;
    border-bottom: 1px solid #1E1E1E;
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
}

/* ── Text input ── */
[data-testid="stTextInput"] input {
    background-color: #111111 !important;
    border: 1px solid #2A2A2A !important;
    border-radius: 4px !important;
    color: #F0F0F0 !important;
    font-family: 'Titillium Web', sans-serif !important;
    font-size: 1rem !important;
    padding: 0.75rem 1rem !important;
    transition: border-color 0.2s;
}
[data-testid="stTextInput"] input:focus {
    border-color: #E10600 !important;
    box-shadow: 0 0 0 2px rgba(225,6,0,0.15) !important;
}
[data-testid="stTextInput"] input::placeholder {
    color: #444444 !important;
}

/* ── Primary button (Analyse) ── */
.stButton > button[kind="primary"] {
    background-color: #E10600 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'Titillium Web', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.5rem 1.5rem !important;
    transition: background-color 0.2s, transform 0.1s;
}
.stButton > button[kind="primary"]:hover {
    background-color: #FF1801 !important;
    transform: translateY(-1px);
}
.stButton > button[kind="primary"]:active {
    transform: translateY(0);
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background-color: #111111;
    border: 1px solid #1E1E1E;
    border-radius: 6px;
    padding: 1rem 1.25rem;
    border-left: 3px solid #E10600;
}
[data-testid="metric-container"] label {
    font-family: 'Titillium Web', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #666666 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.6rem !important;
    font-weight: 600 !important;
    color: #FFFFFF !important;
}

/* ── Expanders (source passages) ── */
[data-testid="stExpander"] {
    background-color: #0F0F0F !important;
    border: 1px solid #1E1E1E !important;
    border-radius: 4px !important;
    margin-bottom: 0.4rem;
}
[data-testid="stExpander"]:hover {
    border-color: #2A2A2A !important;
}
.streamlit-expanderHeader {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    color: #AAAAAA !important;
    font-weight: 400 !important;
}
.streamlit-expanderHeader:hover {
    color: #FFFFFF !important;
}

/* ── Answer text ── */
[data-testid="stMarkdownContainer"] p {
    font-size: 1.05rem;
    line-height: 1.75;
    color: #E8E8E8;
}
[data-testid="stMarkdownContainer"] li {
    font-size: 1.0rem;
    line-height: 1.8;
    color: #E8E8E8;
    margin-bottom: 0.25rem;
}

/* ── Warning banner ── */
[data-testid="stAlert"] {
    background-color: #1A0A00 !important;
    border: 1px solid #E10600 !important;
    border-radius: 4px !important;
    color: #FF6B6B !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background-color: #111111 !important;
    border: 1px solid #2A2A2A !important;
    border-radius: 4px !important;
    color: #F0F0F0 !important;
}

/* ── Horizontal rule ── */
hr {
    border-color: #1E1E1E !important;
}

/* ── Caption text ── */
[data-testid="stCaptionContainer"] {
    color: #555555 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.04em;
}

/* ── Spinner ── */
[data-testid="stSpinner"] {
    color: #E10600 !important;
}
</style>
""", unsafe_allow_html=True)



# ─────────────────────────────────────────────
# SECRETS HANDLING
# Locally:          python-dotenv loads .env → os.getenv() works
# Streamlit Cloud:  secrets set in dashboard → st.secrets works
# This try/except makes the same code work in both environments.
# Think of it like: "try the cloud config, fall back to local config"
# ─────────────────────────────────────────────
if not os.path.exists(".env"):
    try:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

# ─────────────────────────────────────────────
# SINGLETON RETRIEVER — @st.cache_resource
#
# Streamlit reruns the ENTIRE script on every interaction
# (button click, dropdown change, text input).
# Without caching, Retriever.__init__() would reload:
#   - FAISS index from disk
#   - bge-small-en-v1.5 bi-encoder (~130MB)
#   - cross-encoder/ms-marco-MiniLM-L-6-v2 (~80MB)
# ...on EVERY keypress. That's ~3 seconds of dead time.
#
# @st.cache_resource: "run this function ONCE, store the return
# value in process memory, return the same object every rerun."
# Like std::call_once + a static pointer in C++.
# ─────────────────────────────────────────────
@st.cache_resource
def load_retriever() -> Retriever:
    """Load and cache the Retriever singleton."""
    return Retriever()

# ─────────────────────────────────────────────
# HALLUCINATION HEURISTIC
#
# Strategy: extract all numeric values (integers + decimals)
# from the LLM answer. If more than 2 numbers appear in the
# answer that do NOT appear in any retrieved chunk, warn the user.
#
# This is a lightweight heuristic — not foolproof — but it catches
# the most common RAG failure mode: the LLM hallucinating specific
# figures (lap times, dimensions, article numbers) not in context.
# ─────────────────────────────────────────────
def check_hallucination(answer: str, chunks: list) -> bool:
    """
    Returns True if the answer likely contains hallucinated numbers.
    Extracts numeric tokens from the answer, checks each against
    all chunk texts. If >2 numbers are absent from all chunks → warn.
    """
    # Find all numbers in the answer (e.g. "1.5", "42", "100")
    answer_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', answer))
    if not answer_numbers:
        return False  # No numbers to check

    # Build a single string of all chunk text for fast membership test
    all_chunk_text = " ".join(c.text for c in chunks)

    # Count how many answer numbers are absent from retrieved chunks
    missing = sum(
        1 for num in answer_numbers
        if num not in all_chunk_text
    )

    # Threshold: more than 2 unverifiable numbers → flag it
    return missing > 2

# ─────────────────────────────────────────────
# SESSION STATE INITIALISATION
#
# st.session_state persists values across reruns —
# like a heap-allocated struct that survives the
# script's "stack frame" being torn down and rebuilt.
#
# Without this, the answer would vanish every time
# the user touches the sidebar dropdown.
# ─────────────────────────────────────────────
# LOAD RETRIEVER (triggers on first run only)
# ─────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
retriever = load_retriever()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Controls")
    st.markdown("---")

    # Section filter dropdown
    # "All Sections" → passes None to retrieve() → no filter applied
    # Any other option → passes the label string → filters FAISS results
    SECTION_OPTIONS = [
        "All Sections",
        "Section A – General Provisions",
        "Section B – Sporting Regulations",
        "Section C – Technical Regulations",
        "Section D – Financial (Teams)",
        "Section E – Financial (Power Unit Manufacturers)",
        "Section F – Operational Regulations",
    ]
    selected_section = st.selectbox(
        label="Filter by Section",
        options=SECTION_OPTIONS,
        index=0,
        help="Restrict retrieval to a specific regulation section."
    )

    # Convert "All Sections" → None for retrieve() call
    section_filter = None if selected_section == "All Sections" else selected_section

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown(
        "**TechReg Analyst** is a RAG system over the "
        "FIA Formula 1 2026 Regulation PDFs (Sections A–F).\n\n"
        "Ask any regulation question in plain English."
    )
    st.markdown("---")
    st.markdown("### 🔧 System")
    st.markdown(
        "- **Embeddings:** BGE-small-en-v1.5\n"
        "- **Reranker:** ms-marco-MiniLM-L-6\n"
        "- **LLM:** LLaMA 3.1 8B (Groq)\n"
        "- **Vector DB:** FAISS IndexFlatL2\n"
        "- **Chunks:** 3,258"
    )

# ─────────────────────────────────────────────
# MAIN AREA — HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="padding: 2rem 0 1.5rem 0;">
    <div style="display:inline-block; background:#E10600; width:3px; height:2.8rem;
                vertical-align:middle; margin-right:0.75rem; border-radius:2px;"></div>
    <span style="font-family:'Titillium Web',sans-serif; font-size:2.8rem;
                 font-weight:900; letter-spacing:-0.02em; vertical-align:middle;">
        TechReg Analyst
    </span>
    <div style="margin-top:0.6rem; color:#555555; font-size:0.95rem;
                letter-spacing:0.04em; font-weight:400;">
        FIA Formula 1 2026 Regulations — Retrieval-Augmented Q&amp;A
    </div>
</div>
<hr style="border-color:#1E1E1E; margin-bottom:1.5rem;">
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# QUERY INPUT
# ─────────────────────────────────────────────
query = st.text_input(
    label="Your question",
    placeholder="e.g. What are the rules for Driver Adjustable Bodywork activation?",
    label_visibility="collapsed",
)

analyse_clicked = st.button("Analyse", type="primary", use_container_width=False)

# ─────────────────────────────────────────────
# PIPELINE EXECUTION
#
# Runs when the button is clicked AND a non-empty query is present.
# The result dict is stored in session_state so it persists across
# subsequent reruns (e.g. when the user tweaks the sidebar).
# ─────────────────────────────────────────────
if analyse_clicked and query.strip():
    with st.spinner("Retrieving and generating answer..."):

        # ── Stage 1: Retrieval ──────────────────
        t_retrieval_start = time.perf_counter()
        chunks, retrieval_meta = retriever.retrieve(
            query=query,
            section_filter=section_filter,
        )
        t_retrieval_end = time.perf_counter()
        retrieval_ms = (t_retrieval_end - t_retrieval_start) * 1000

        # ── Stage 2: Generation ─────────────────
        gen_result = generate_answer(query=query, chunks=chunks)

        # ── Stage 3: Hallucination check ────────
        warn_hallucination = check_hallucination(gen_result["answer"], chunks)

        # ── Store in session_state ───────────────
        # Everything the display section needs, packed into one dict.
        st.session_state.result = {
            "query":             query,
            "answer":            gen_result["answer"],
            "chunks":            chunks,
            "retrieval_ms":      retrieval_ms,
            "llm_ms":            gen_result["llm_latency_ms"],
            "total_ms":          retrieval_ms + gen_result["llm_latency_ms"],
            "warn_hallucination": warn_hallucination,
            "section_filter":    selected_section,
        }
        st.session_state.last_query = query

elif analyse_clicked and not query.strip():
    st.warning("Please enter a question before clicking Analyse.")

# ─────────────────────────────────────────────
# RESULTS DISPLAY
#
# Reads from session_state — NOT from local variables.
# This means results survive sidebar interactions and reruns.
# ─────────────────────────────────────────────
if st.session_state.result is not None:
    r = st.session_state.result

    st.markdown("---")

    # ── Hallucination warning banner ───────────
    # Displayed ABOVE the answer so the user sees it first
    if r["warn_hallucination"]:
        st.warning(
            "⚠️ **Possible hallucination detected.** "
            "This answer contains numeric values that could not be "
            "verified in the retrieved passages. Cross-check with sources below."
        )

    # ── Answer ─────────────────────────────────
    st.markdown("### Answer")
    st.markdown(r["answer"])

    # ── Latency metrics — 3 columns ────────────
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="🔍 Retrieval",
            value=f"{r['retrieval_ms']:.0f} ms",
            help="Time for bi-encoder FAISS search + cross-encoder rerank"
        )
    with col2:
        st.metric(
            label="🤖 LLM Generation",
            value=f"{r['llm_ms']:.0f} ms",
            help="Time for Groq LLaMA 3.1 8B to generate the answer"
        )
    with col3:
        st.metric(
            label="⏱️ Total",
            value=f"{r['total_ms']:.0f} ms",
            help="End-to-end latency from query to answer"
        )

    # ── Source expanders ───────────────────────
    # One expander per retrieved chunk, labeled with citation number,
    # section, page, and cross-encoder score.
    # User can expand any source to read the exact passage the LLM used.
    st.markdown("---")
    st.markdown("### 📄 Sources")
    st.caption(
        f"Showing top {len(r['chunks'])} passages retrieved "
        f"(filtered to: {r['section_filter']})"
    )

    for i, chunk in enumerate(r["chunks"]):
        # Format the cross-encoder score to 3 decimal places
        ce_score = f"{chunk.cross_encoder_score:.3f}"

        # Expander label mirrors the citation number in the answer
        label = (
            f"[{i+1}] {chunk.section}  |  "
            f"Page {chunk.page}  |  "
            f"CE score: {ce_score}"
        )
        with st.expander(label, expanded=False):
            st.markdown(
                f"**Source file:** `{chunk.source}`  \n"
                f"**Chunk ID:** {chunk.chunk_id}"
            )
            st.markdown("---")
            # Display the raw chunk text in a code block for readability
            st.text(chunk.text)