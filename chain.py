"""
chain.py
--------
Grounded answer generation using LangChain + Groq LLaMA 3.

Takes: user query + list[RetrievedChunk] from retriever.py
Does:  formats context → fills prompt → calls Groq → returns answer + latency

Key concept — grounded generation:
  The LLM is constrained to answer ONLY from the provided context.
  This converts the LLM from a "knowledge source" (where it can hallucinate)
  into a "reading comprehension engine" (where it can only use what we give it).

In C++ terms: this file is a single stateless function with internal helpers.
  GenerationResult generate(string query, vector<RetrievedChunk> chunks)
"""

import os
import time

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()  # reads GROQ_API_KEY from .env into os.environ

# Configuration

GROQ_MODEL       = "llama-3.1-8b-instant"
GROQ_TEMPERATURE = 0.0   # 0 = fully deterministic, no creative deviation
GROQ_MAX_TOKENS  = 512   # enough for a detailed regulation answer


# Context builder

def build_context(chunks) -> str:
    """
    Format retrieved chunks into a numbered context string for the LLM.

    Why numbered? So the LLM can write "per [1]" or "according to [2]"
    and the user can trace the claim back to the exact source passage
    shown in the UI. This is what makes the system auditable.

    In C++ terms: string join(vector<RetrievedChunk> chunks, formatter)
    """
    parts = []
    for i, chunk in enumerate(chunks):
        header = (
            f"[{i+1}] Source: {chunk.section} | Page {chunk.page}"
        )
        parts.append(f"{header}\n{chunk.text}")

    # Separator "---" makes boundaries clear to the LLM
    return "\n\n---\n\n".join(parts)

# Prompt template

# ChatPromptTemplate takes a list of (role, message) tuples.
# "system" = sets the LLM's persona and constraints (not shown to user)
# "human"  = the actual question with context injected
#
# {context} and {query} are placeholder variables filled at invoke() time.
# In C++ terms: this is like a format string with named placeholders.
#
# Why temperature=0 + "ONLY the context below"?
# Two-layer hallucination mitigation:
#   Layer 1 (prompt):  instruction constraint — tells the model what to do
#   Layer 2 (sampling): temperature=0 — removes randomness from token selection
#   Together: the model produces the most likely tokens given the constraint.

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a technical assistant for FIA Formula 1 2026 Regulations. "
        "Your ONLY job is to answer questions from the context passages below.\n\n"
        "RULES — follow these exactly:\n"
        "1. Always answer from the context. The regulation text uses precise technical "
        "language that may differ from the question's wording — this is expected. "
        "For example: 'fuel energy flow' is the regulation term for fuel flow rate; "
        "'referred to the stewards' is the consequence of procedural violations; "
        "'FIA garage weighing' is the weighbridge procedure.\n"
        "2. If a passage is relevant to the question's intent, USE it — even if the "
        "exact words don't match. Paraphrase and explain the relevant regulation.\n"
        "3. Cite every claim with [1], [2], etc. referencing the passage number.\n"
        "4. Only output 'This information is not present in the provided regulation "
        "passages.' if NONE of the 5 passages have ANY connection to the question topic. "
        "This should be rare. If even one passage is partially relevant, answer from it. "
        "However, if the question is nonsensical (e.g. 'what is made up', 'invent something'), "
        "fictional, a test query, or entirely unrelated to F1 regulations, output the fallback "
        "immediately — do not analyse the passages at all, do not invent connections. "
        "A question must reference a real F1 regulation topic to deserve an answer."
    )),
    ("human", (
        "Context passages:\n\n{context}\n\n"
        "Question: {query}\n\n"
        "Answer (with citations):"
    ))
])

# LLM factory

def get_llm() -> ChatGroq:
    """
    Construct and return a Groq LLM client.
    Called fresh each time generate() runs — lightweight, no state.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not found.\n"
            "Create a .env file with: GROQ_API_KEY=your_key_here\n"
            "Get a free key at: https://console.groq.com"
        )
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=GROQ_TEMPERATURE,
        max_tokens=GROQ_MAX_TOKENS,
        groq_api_key=api_key
    )

# Main generation function

def generate_answer(query: str, chunks) -> dict:
    """
    Full RAG generation: context formatting → prompt filling → LLM call.

    Returns a dict with:
      "answer"         → str: the LLM's grounded response
      "llm_latency_ms" → float: Groq API call time in milliseconds
      "context_used"   → str: the formatted context sent to the LLM
                              (shown in UI so users can verify citations)

    try/except wraps the entire LLM call — if Groq is down or the key
    is wrong, the app shows a safe error message instead of crashing.
    In C++ terms: this is exception handling with a guaranteed return value.
    """
    context = build_context(chunks)

    try:
        llm = get_llm()

        # LangChain v0.2 Runnable pipe syntax:
        #   prompt | llm  is equivalent to: lambda x: llm(prompt.format(x))
        # chain.invoke(dict) fills the {context} and {query} placeholders,
        # sends the filled prompt to Groq, and returns an AIMessage object.
        chain = RAG_PROMPT | llm

        t0 = time.perf_counter()
        response = chain.invoke({
            "context": context,
            "query":   query
        })
        t1 = time.perf_counter()

        return {
            "answer":         response.content,  # AIMessage.content = the text
            "llm_latency_ms": (t1 - t0) * 1000,
            "context_used":   context
        }

    except Exception as e:
        return {
            "answer":         f"Generation failed: {str(e)}",
            "llm_latency_ms": 0.0,
            "context_used":   context
        }

# Standalone test

if __name__ == "__main__":
    from retriever import Retriever

    print("=== chain.py Standalone Test ===\n")

    retriever = Retriever()

    test_cases = [
        "What are the rules for Driver Adjustable Bodywork activation?",
        "What materials are permitted for the survival cell?",
        "What is completely made up and not in any regulation?",  # hallucination test
    ]

    for query in test_cases:
        print(f"Query: {query}")
        chunks, ret_lat = retriever.retrieve(query)
        result = generate_answer(query, chunks)

        print(f"Retrieval: {ret_lat['total_retrieval_ms']:.0f}ms | "
              f"LLM: {result['llm_latency_ms']:.0f}ms")
        print(f"Answer:\n{result['answer']}")
        print("\n" + "="*60 + "\n")