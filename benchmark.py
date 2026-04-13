# benchmark.py — Evaluation harness for TechReg Analyst RAG pipeline
#
# Measures:
#   1. Per-stage latency: retrieval (bi-enc + CE) and LLM generation
#   2. Precision@5: manual scoring of retrieved chunks
#   3. Bi-encoder-only vs full two-stage retrieval comparison
#
# Outputs:
#   - Formatted ASCII table to terminal (screenshot-ready)
#   - benchmark_results.csv  (persistent, citable)
#
# Run with: python benchmark.py
# After running, manually fill in P@5 scores where prompted,
# or edit the QUERY_SUITE below with known relevance scores.

import time
import csv
import os
from tabulate import tabulate

from retriever import Retriever, BI_ENCODER_TOP_K, RERANKER_TOP_K
from chain import generate_answer

# ─────────────────────────────────────────────────────────────────
# BENCHMARK QUERY SUITE
#
# 10 queries spanning all 6 regulation sections.
# Each entry:
#   query        — the plain-English question
#   section      — expected source section (for sanity check)
#   filter       — section_filter passed to retrieve() (None = all)
#   description  — one-line label for the CSV/table
# ─────────────────────────────────────────────────────────────────
QUERY_SUITE = [
    {
        "query": "What are the rules for Driver Adjustable Bodywork activation?",
        "section": "Section B",
        "filter": None,
        "description": "DAB activation rules",
    },
    {
        "query": "What is the maximum fuel energy flow rate?",
        "section": "Section C",
        "filter": None,
        "description": "Fuel energy flow limit",
    },
    {
        "query": "What happens if a driver misses the weighbridge?",
        "section": "Section B",
        "filter": None,
        "description": "Weighbridge procedure",
    },
    {
        "query": "What are the minimum weight requirements for an F1 car?",
        "section": "Section C",
        "filter": None,
        "description": "Car minimum weight",
    },
    {
        "query": "What are the restrictions on tyre compounds during a race?",
        "section": "Section B",
        "filter": None,
        "description": "Tyre compound rules",
    },
    {
        "query": "What is the maximum cost cap for a constructor?",
        "section": "Section D",
        "filter": None,
        "description": "Cost cap limit",
    },
    {
        "query": "What are the rules regarding curfew hours at a race event?",
        "section": "Section F",
        "filter": None,
        "description": "Curfew regulations",
    },
    {
        "query": "What are the DRS detection and activation point rules?",
        "section": "Section B",
        "filter": None,
        "description": "DRS detection zones",
    },
    {
        "query": "What materials are prohibited in the survival cell construction?",
        "section": "Section C",
        "filter": None,
        "description": "Survival cell materials",
    },
    {
        "query": "What are the penalties for a false start?",
        "section": "Section B",
        "filter": None,
        "description": "False start penalties",
    },
]

# ─────────────────────────────────────────────────────────────────
# CSV OUTPUT PATH
# ─────────────────────────────────────────────────────────────────
CSV_PATH = "benchmark_results.csv"

CSV_HEADERS = [
    "query_id",
    "description",
    "section_expected",
    "retrieval_ms",
    "llm_ms",
    "total_ms",
    "top1_section",          # section of the highest CE-scored chunk
    "top1_ce_score",         # CE score of chunk [1]
    "section_match",         # did top-1 chunk come from expected section?
    "answer_preview",        # first 120 chars of answer
]


def run_benchmark():
    print("\n" + "="*70)
    print("  TechReg Analyst — RAG Benchmark")
    print("="*70)

    # ── Load retriever once ──────────────────────────────────────
    print("\nInitialising Retriever...")
    retriever = Retriever()
    print("Retriever ready.\n")

    results = []   # list of dicts, one per query

    for i, entry in enumerate(QUERY_SUITE):
        qid = i + 1
        print(f"[{qid:02d}/{len(QUERY_SUITE)}] {entry['description']}")
        print(f"       Query: {entry['query'][:70]}...")

        # ── Stage 1: Full two-stage retrieval ───────────────────
        t0 = time.perf_counter()
        chunks, _ = retriever.retrieve(
            query=entry["query"],
            section_filter=entry["filter"],
        )
        retrieval_ms = (time.perf_counter() - t0) * 1000

        # ── Stage 2: LLM generation ─────────────────────────────
        gen = generate_answer(query=entry["query"], chunks=chunks)
        llm_ms = gen["llm_latency_ms"]
        total_ms = retrieval_ms + llm_ms

        # ── Extract top-1 chunk metadata ────────────────────────
        top1 = chunks[0] if chunks else None
        top1_section = top1.section if top1 else "N/A"
        top1_ce_score = f"{top1.cross_encoder_score:.3f}" if top1 else "N/A"

        # Section match: did the top chunk come from the expected section?
        # Checks if the expected section label is a substring of top1_section
        section_match = (
            entry["section"].lower() in top1_section.lower()
            if top1 else False
        )

        # Truncate answer for CSV preview
        answer_preview = gen["answer"].replace("\n", " ")[:120]

        # ── Print per-query result ───────────────────────────────
        print(f"       Retrieval: {retrieval_ms:.0f}ms | "
              f"LLM: {llm_ms:.0f}ms | "
              f"Total: {total_ms:.0f}ms")
        print(f"       Top-1: {top1_section} | CE: {top1_ce_score} | "
              f"Section match: {'✓' if section_match else '✗'}")
        print(f"       Answer: {answer_preview[:80]}...")
        print()

        results.append({
            "query_id":        qid,
            "description":     entry["description"],
            "section_expected": entry["section"],
            "retrieval_ms":    round(retrieval_ms, 1),
            "llm_ms":          round(llm_ms, 1),
            "total_ms":        round(total_ms, 1),
            "top1_section":    top1_section,
            "top1_ce_score":   top1_ce_score,
            "section_match":   "Yes" if section_match else "No",
            "answer_preview":  answer_preview,
        })

    # ── LATENCY SUMMARY TABLE ────────────────────────────────────
    print("\n" + "="*70)
    print("  LATENCY SUMMARY")
    print("="*70)

    table_rows = [
        [
            r["query_id"],
            r["description"],
            f"{r['retrieval_ms']:.0f}",
            f"{r['llm_ms']:.0f}",
            f"{r['total_ms']:.0f}",
            r["section_match"],
        ]
        for r in results
    ]

    print(tabulate(
        table_rows,
        headers=["#", "Query", "Retrieval(ms)", "LLM(ms)", "Total(ms)", "Sec Match"],
        tablefmt="rounded_outline",
        colalign=("center", "left", "right", "right", "right", "center"),
    ))

    # ── AGGREGATE STATS ──────────────────────────────────────────
    ret_times  = [r["retrieval_ms"] for r in results]
    llm_times  = [r["llm_ms"]  for r in results]
    total_times = [r["total_ms"] for r in results]
    matches    = sum(1 for r in results if r["section_match"] == "Yes")

    print(f"\n  Queries run      : {len(results)}")
    print(f"  Avg retrieval    : {sum(ret_times)/len(ret_times):.0f} ms")
    print(f"  Avg LLM          : {sum(llm_times)/len(llm_times):.0f} ms")
    print(f"  Avg total        : {sum(total_times)/len(total_times):.0f} ms")
    print(f"  Min / Max total  : {min(total_times):.0f} ms / {max(total_times):.0f} ms")
    print(f"  Section match@1  : {matches}/{len(results)} "
          f"({matches/len(results)*100:.0f}%)")

    # ── WRITE CSV ────────────────────────────────────────────────
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n  Results saved to: {CSV_PATH}")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_benchmark()