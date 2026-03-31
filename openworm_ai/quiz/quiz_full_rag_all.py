"""
Clean RAG Evaluation Script
============================
Tests three answering modes per question, per LLM:

  1. LLM Only     — no retrieval, pure model knowledge
  2. RAG Only     — retrieve context, answer from context only (no fallback)
  3. RAG + Fallback — retrieve if context quality >= threshold, else fall back
                      to LLM-only answer

This lets us directly compare:
  - Does RAG help on C. elegans / Corpus questions?
  - Does RAG + Fallback beat both RAG-only and LLM-only overall?

Usage:
    GEN_RAG_VS_CONFIG=./vector-stores.json python QuizEvalRAG_Clean.py

Prerequisites:
    - Vector store built:  python -m openworm_ai.rag_chroma
    - GEN_RAG_VS_CONFIG env var pointing to vector-stores.json
    - neuroml.ai RAG package installed
"""

import asyncio
import datetime
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Suppress NeuroML-AI debug logging IMMEDIATELY at import time.
# The gen_rag / neuroml_ai_utils packages set their loggers to DEBUG by
# default. If we don't suppress before importing them, the first import
# already registers handlers and we can't silence them cleanly later.
# Setting WARNING on root here covers everything downstream.
# ---------------------------------------------------------------------------
import logging as _logging_early
_logging_early.root.setLevel(_logging_early.WARNING)
for _ln in ("NeuroML-AI", "RAG", "Vector_Stores", "neuroml_ai_utils", "gen_rag"):
    _lg = _logging_early.getLogger(_ln)
    _lg.setLevel(_logging_early.WARNING)
    _lg.propagate = False

# ---------------------------------------------------------------------------
# Monkey-patch NML setup_llm to bypass strict health checks (same as before)
# ---------------------------------------------------------------------------
try:
    from openworm_ai.quiz.llm_setup_patched import setup_llm_patched, setup_embedding_patched
    import neuroml_ai_utils.llm
    neuroml_ai_utils.llm.setup_llm = setup_llm_patched
    neuroml_ai_utils.llm.setup_embedding = setup_embedding_patched
    print("✓ Patched NML setup_llm")
except ImportError:
    print("! Could not patch setup_llm — continuing without patch")

# ---------------------------------------------------------------------------
# RAG package
# ---------------------------------------------------------------------------
try:
    from gen_rag.rag import RAG
    from gen_rag.stores import Vector_Stores
except ImportError:
    print("ERROR: gen_rag package not found.")
    print("Install: pip install git+https://github.com/NeuroML/neuroml-ai.git#subdirectory=rag_pkg/gen_rag")
    sys.exit(1)

# ---------------------------------------------------------------------------
# LLM list
# ---------------------------------------------------------------------------
from openworm_ai.utils.llms import (
    LLM_GPT4o,
    LLM_CLAUDE37,
    LLM_GPT35,
    LLM_HF_QWEN25_72B,
    LLM_HF_QWEN25_32B,
    LLM_HF_QWEN25_14B,
    LLM_HF_LLAMA31_8B,
    LLM_HF_GEMMA_2_9B,
    LLM_HF_LLAMA32_1b,
    LLM_HF_COHERE_AYA_32B,
    LLM_HF_MISTRAL_7B,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

QUIZ_FILES = {
    "General Knowledge": "openworm_ai/quiz/samples/huggingface_Qwen_Qwen2.5-72B-Instruct_100questions_general_v2.json",
    "Science":           "openworm_ai/quiz/samples/huggingface_Qwen_Qwen2.5-72B-Instruct_100questions_science_v2.json",
    "C. elegans":        "openworm_ai/quiz/samples/huggingface_Qwen_Qwen2.5-72B-Instruct_100questions_celegans_v2.json",
    "C. elegans (Corpus)": "openworm_ai/quiz/samples/huggingface_Qwen_Qwen2.5-72B-Instruct_100questions_celegans_corpus.json",
}

LLMS = [
    LLM_GPT4o,
    LLM_CLAUDE37,
    LLM_GPT35,
    LLM_HF_LLAMA32_1b,
    LLM_HF_MISTRAL_7B,
    LLM_HF_LLAMA31_8B,
    LLM_HF_GEMMA_2_9B,
    LLM_HF_QWEN25_14B,
    LLM_HF_QWEN25_32B,
    LLM_HF_COHERE_AYA_32B,
    LLM_HF_QWEN25_72B,
]

INDEXING = ["A", "B", "C", "D"]

# Minimum retrieval quality score (0-1) to trust RAG context.
# If best chunk score is below this, we consider context "not good enough".
RAG_RETRIEVAL_THRESHOLD = 0.7

# Number of chunks to retrieve
NUM_CHUNKS = 5

OUTPUT_DIR = "openworm_ai/quiz/scores/rag_clean"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_questions(filepath: str) -> List[Dict]:
    """Load MCQ questions from JSON file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        questions = []
        for q in data.get("questions", []):
            if "question" in q and "answers" in q:
                answers = [
                    {"ans": a["ans"], "correct": a["correct"]}
                    for a in q["answers"]
                    if "ans" in a and "correct" in a
                ]
                if answers:
                    questions.append({"question": q["question"], "answers": answers})
        return questions
    except Exception as e:
        print(f"  ! Error loading {filepath}: {e}")
        return []


def shuffle_and_label(answers: List[Dict]) -> Tuple[Dict[str, str], str]:
    """
    Shuffle answers, assign A/B/C/D labels.
    Returns (presented_answers dict, correct_letter).
    """
    shuffled = answers.copy()
    random.shuffle(shuffled)
    presented = {}
    correct_letter = None
    for i, ans in enumerate(shuffled):
        label = INDEXING[i]
        presented[label] = ans["ans"]
        if ans["correct"]:
            correct_letter = label
    return presented, correct_letter


def extract_letter(text: str) -> Optional[str]:
    """Extract A/B/C/D letter from model response."""
    if not text:
        return None
    text = text.strip()

    # First character
    if text[0] in "ABCD":
        return text[0]

    # Patterns: "A.", "A:", "A)", "(A)", "[A]", "Answer: A"
    for pattern in [
        r"^([ABCD])[.:\)]",
        r"\(([ABCD])\)",
        r"\[([ABCD])\]",
        r"[Aa]nswer[:\s]+([ABCD])",
        r"\b([ABCD])\b",
    ]:
        m = re.search(pattern, text)
        if m:
            return m.group(1)
    return None


def format_mcq_prompt(question: str, options: Dict[str, str], context: str = "") -> str:
    """Build the prompt sent to the LLM."""
    options_text = "\n".join(f"{k}: {v}" for k, v in options.items())

    if context:
        return (
            f"Use the following context to answer the question.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION:\n{question}\n\n"
            f"OPTIONS:\n{options_text}\n\n"
            f"Respond with ONLY the letter (A, B, C, or D)."
        )
    else:
        return (
            f"QUESTION:\n{question}\n\n"
            f"OPTIONS:\n{options_text}\n\n"
            f"Respond with ONLY the letter (A, B, C, or D)."
        )


def llm_model_string(llm: str) -> str:
    """Normalise LLM string for the RAG package."""
    llm = llm.replace(":auto", "")
    if llm.count(":") == 1:          # huggingface:ModelName → add :auto
        llm = f"{llm}:auto"
    return llm


# ---------------------------------------------------------------------------
# Retrieval (direct, bypassing full LangGraph pipeline)
# ---------------------------------------------------------------------------

def retrieve_context(
    stores: Vector_Stores,
    domain: str,
    query: str,
    k: int = NUM_CHUNKS,
) -> Tuple[str, float]:
    """
    Retrieve top-k chunks for query from the vector store.

    Returns:
        context_text  — concatenated chunk text
        best_score    — highest relevance score (0-1); 0 if no results
    """
    try:
        results = stores.retrieve(domain_name=domain, query=query)
        if not results:
            return "", 0.0

        # results: list of (Document, score) tuples, higher = more relevant
        sorted_results = sorted(results, key=lambda t: t[1], reverse=True)
        top = sorted_results[:k]
        best_score = top[0][1]

        chunks = []
        for doc, score in top:
            chunks.append(doc.page_content)
        context_text = "\n\n---\n\n".join(chunks)
        return context_text, best_score

    except Exception as e:
        return "", 0.0


def classify_domain(stores: Vector_Stores, question: str, model=None) -> Optional[str]:
    """
    Classify which vector store domain is relevant for this question,
    using the same LLM-based logic as the RAG package but as a single
    cheap call with no retry loop.

    Falls back to keyword matching if model is None or the call fails.

    Returns the domain name string, or None if the question is out-of-domain
    (meaning: skip retrieval, go straight to LLM-only answer).
    """
    # --- LLM-based classification (preferred) ---
    if model is not None:
        try:
            from langchain_core.messages import HumanMessage
            from pydantic import create_model
            from typing_extensions import Literal

            domains = stores.domains
            domain_descriptions = "\n".join(f"- {d}" for d in domains)

            prompt = (
                f"Classify this question into one of these retrieval domains, "
                f"or 'none' if it does not relate to any of them.\n\n"
                f"Domains:\n{domain_descriptions}\n- none\n\n"
                f"Question: {question}\n\n"
                f"Reply with ONLY the domain name exactly as written above, nothing else."
            )
            output = model.invoke(
                [HumanMessage(content=prompt)],
                config={"configurable": {"temperature": 0.0}},
            )
            raw = output.content.strip() if hasattr(output, "content") else str(output).strip()

            # Match against known domains (case-insensitive, partial ok)
            raw_lower = raw.lower()
            for d in domains:
                if d.lower() in raw_lower or raw_lower in d.lower():
                    return d
            # If model said "none" or something unrecognised, return None
            return None

        except Exception:
            pass  # Fall through to keyword fallback

    # --- Keyword fallback (used if model unavailable or call failed) ---
    celegans_keywords = [
        "c. elegans", "caenorhabditis", "worm", "nematode",
        "neuron", "pharynx", "vulva", "hermaphrodite", "amphid",
        "neuroml", "wormatlas", "synapse", "connectome", "muscle cell",
        "daf-", "mec-", "glr-", "avl", "avm", "unc-", "elegans",
        "locomotion", "gait", "neural circuit", "chemosensory",
    ]
    q_lower = question.lower()
    if any(kw in q_lower for kw in celegans_keywords):
        for d in stores.domains:
            if any(t in d.lower() for t in ("corpus", "wormatlas", "elegans")):
                return d
        return stores.domains[0] if stores.domains else None
    return None


# ---------------------------------------------------------------------------
# LLM-only answering (direct invoke, no RAG graph)
# ---------------------------------------------------------------------------

async def answer_with_llm(model, question: str, options: Dict[str, str]) -> str:
    """Call the LLM directly with no context."""
    prompt_text = format_mcq_prompt(question, options)
    from langchain_core.messages import HumanMessage
    try:
        output = model.invoke(
            [HumanMessage(content=prompt_text)],
            config={"configurable": {"temperature": 0.0}},
        )
        return output.content if hasattr(output, "content") else str(output)
    except Exception as e:
        return ""


async def answer_with_context(model, question: str, options: Dict[str, str], context: str) -> str:
    """Call the LLM with retrieved context."""
    prompt_text = format_mcq_prompt(question, options, context=context)
    from langchain_core.messages import HumanMessage
    try:
        output = model.invoke(
            [HumanMessage(content=prompt_text)],
            config={"configurable": {"temperature": 0.0}},
        )
        return output.content if hasattr(output, "content") else str(output)
    except Exception as e:
        return ""


# ---------------------------------------------------------------------------
# Per-question evaluation
# ---------------------------------------------------------------------------

async def evaluate_question(
    model,
    stores: Vector_Stores,
    question_text: str,
    answers: List[Dict],
    precomputed_domain: Optional[str] = None,
) -> Dict:
    """
    Run one question through all three modes.
    If precomputed_domain is provided, skip the classify call.
    """
    presented, correct_letter = shuffle_and_label(answers)

    # --- Determine retrieval domain ---
    # Uses LLM-based classification so indirect/phrased questions aren't missed.
    # If domain was pre-classified (shared across LLMs), use that directly.
    if precomputed_domain is not None:
        domain = precomputed_domain
    else:
        domain = classify_domain(stores, question_text, model=model)

    # --- Retrieve context (always attempt if domain found) ---
    context = ""
    best_score = 0.0
    rag_used = False

    if domain:
        context, best_score = retrieve_context(stores, domain, question_text)
        rag_used = bool(context)

    context_above_threshold = best_score >= RAG_RETRIEVAL_THRESHOLD

    # --- Mode 1: LLM only ---
    llm_response = await answer_with_llm(model, question_text, presented)
    llm_guess = extract_letter(llm_response) or random.choice(INDEXING)

    # --- Mode 2: RAG only ---
    if rag_used:
        rag_response = await answer_with_context(model, question_text, presented, context)
        rag_guess = extract_letter(rag_response) or random.choice(INDEXING)
    else:
        # No context available — RAG can't answer, mark as None (unanswerable)
        rag_guess = None

    # --- Mode 3: RAG + Fallback ---
    if rag_used and context_above_threshold:
        # Good context — use RAG answer
        hybrid_guess = rag_guess
        hybrid_source = "rag"
    else:
        # Context missing or weak — fall back to LLM
        hybrid_guess = llm_guess
        hybrid_source = "llm_fallback"

    return {
        "correct_letter": correct_letter,
        "llm_guess": llm_guess,
        "rag_guess": rag_guess,
        "hybrid_guess": hybrid_guess,
        "hybrid_source": hybrid_source,
        "rag_used": rag_used,
        "context_above_threshold": context_above_threshold,
        "best_score": round(best_score, 4),
        "retrieval_domain": domain,
    }


# ---------------------------------------------------------------------------
# Per-LLM × per-quiz evaluation
# ---------------------------------------------------------------------------

async def evaluate_llm_on_quiz(
    llm_name: str,
    quiz_category: str,
    questions: List[Dict],
    stores: Vector_Stores,
    model,
    precomputed_domains: List[Optional[str]] = None,
) -> Dict:
    """Run all questions for one LLM on one quiz category.

    precomputed_domains: optional list of domain strings (one per question),
    pre-classified before the LLM loop to avoid redundant classify calls.
    """

    n = len(questions)
    counters = {
        "llm_correct": 0,
        "rag_correct": 0,
        "rag_answered": 0,       # questions where RAG had context
        "hybrid_correct": 0,
        "hybrid_rag_used": 0,
        "hybrid_llm_fallback": 0,
        "response_times": [],
    }

    print(f"\n  {'Q':>4}  {'Correct':>7}  {'LLM':>5}  {'RAG':>5}  {'Hybrid':>7}  Score")
    print(f"  {'─'*50}")

    for idx, q_data in enumerate(questions, 1):
        # Use precomputed domain if available, otherwise classify now
        precomputed = precomputed_domains[idx - 1] if precomputed_domains else None
        t0 = time.time()
        result = await evaluate_question(
            model, stores, q_data["question"], q_data["answers"],
            precomputed_domain=precomputed,
        )
        elapsed = time.time() - t0
        counters["response_times"].append(elapsed)

        correct = result["correct_letter"]

        llm_ok    = result["llm_guess"]    == correct
        rag_ok    = result["rag_guess"]    == correct if result["rag_guess"] else False
        hybrid_ok = result["hybrid_guess"] == correct

        if llm_ok:    counters["llm_correct"] += 1
        if rag_ok:    counters["rag_correct"] += 1
        if hybrid_ok: counters["hybrid_correct"] += 1

        if result["rag_used"]:
            counters["rag_answered"] += 1
        if result["hybrid_source"] == "rag":
            counters["hybrid_rag_used"] += 1
        else:
            counters["hybrid_llm_fallback"] += 1

        llm_sym    = "✓" if llm_ok    else "✗"
        rag_sym    = "✓" if rag_ok    else ("─" if result["rag_guess"] is None else "✗")
        hybrid_sym = "✓" if hybrid_ok else "✗"
        src_tag    = "[R]" if result["hybrid_source"] == "rag" else "[L]"

        print(
            f"  {idx:>4}  {correct:>7}  "
            f"{llm_sym}({result['llm_guess']:>1})  "
            f"{rag_sym}({(result['rag_guess'] or '─'):>1})  "
            f"{hybrid_sym}({result['hybrid_guess']:>1}){src_tag}  "
            f"{result['best_score']:.2f}"
        )

    avg_time = sum(counters["response_times"]) / n

    def pct(num, denom):
        return round(100 * num / denom, 2) if denom > 0 else 0.0

    summary = {
        "Quiz Category":           quiz_category,
        "LLM":                     llm_name,
        "Total Questions":         n,

        # LLM-only
        "LLM Correct":             counters["llm_correct"],
        "LLM Accuracy (%)":        pct(counters["llm_correct"], n),

        # RAG-only (only scored on questions where context was retrieved)
        "RAG Questions Answered":  counters["rag_answered"],
        "RAG Correct":             counters["rag_correct"],
        "RAG Accuracy on Retrieved (%)": pct(counters["rag_correct"], counters["rag_answered"]),
        "RAG Accuracy Overall (%)":      pct(counters["rag_correct"], n),

        # Hybrid
        "Hybrid Correct":          counters["hybrid_correct"],
        "Hybrid Accuracy (%)":     pct(counters["hybrid_correct"], n),
        "Hybrid Used RAG":         counters["hybrid_rag_used"],
        "Hybrid Used LLM Fallback": counters["hybrid_llm_fallback"],

        # Improvement deltas
        "Hybrid vs LLM Delta (pp)":    round(pct(counters["hybrid_correct"], n) - pct(counters["llm_correct"], n), 2),
        "Hybrid vs RAG-Overall Delta (pp)": round(pct(counters["hybrid_correct"], n) - pct(counters["rag_correct"], n), 2),

        "Avg Response Time (s)":   round(avg_time, 3),
    }

    print(f"\n  ── Results ──────────────────────────────")
    print(f"  LLM only:        {summary['LLM Accuracy (%)']:>6.1f}%  ({counters['llm_correct']}/{n})")
    print(f"  RAG only:        {summary['RAG Accuracy Overall (%)']:>6.1f}%  ({counters['rag_correct']}/{n})  "
          f"[retrieved context for {counters['rag_answered']}/{n} questions]")
    print(f"  RAG + Fallback:  {summary['Hybrid Accuracy (%)']:>6.1f}%  ({counters['hybrid_correct']}/{n})  "
          f"[RAG={counters['hybrid_rag_used']}, LLM fallback={counters['hybrid_llm_fallback']}]")
    print(f"  Δ Hybrid vs LLM: {summary['Hybrid vs LLM Delta (pp)']:>+.1f} pp")

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    vs_config = os.getenv("GEN_RAG_VS_CONFIG", "./vector-stores.json")

    if not Path(vs_config).exists():
        print(f"ERROR: Vector store config not found: {vs_config}")
        print("Run:  python -m openworm_ai.rag_chroma")
        print("Then: export GEN_RAG_VS_CONFIG=./vector-stores.json")
        sys.exit(1)

    # ---------------------------------------------------------------------------
    # Silence NeuroML-AI debug logging BEFORE any store or model is created.
    # The package uses a hardcoded logger named "NeuroML-AI" at DEBUG level.
    # Setting WARNING on root + the specific logger name covers both cases.
    # This must happen before stores.setup() or the first retrieve() call.
    # ---------------------------------------------------------------------------
    logging.root.setLevel(logging.WARNING)
    for _noisy in ("NeuroML-AI", "RAG", "Vector_Stores",
                   "neuroml_ai_utils", "gen_rag", ""):
        lg = logging.getLogger(_noisy)
        lg.setLevel(logging.WARNING)
        lg.propagate = False

    print("\n" + "=" * 70)
    print("  RAG EVALUATION  —  LLM / RAG / RAG+Fallback")
    print("=" * 70)
    print(f"  Config:     {vs_config}")
    print(f"  Threshold:  {RAG_RETRIEVAL_THRESHOLD}  (min chunk score to trust RAG)")
    print(f"  LLMs:       {len(LLMS)}")
    print(f"  Quizzes:    {list(QUIZ_FILES.keys())}")
    print("=" * 70)

    # ---------------------------------------------------------------------------
    # Initialise shared vector store ONCE, then patch out the reload.
    # Looking at the debug output, Vector_Stores.retrieve() calls load() on
    # each VectorStoreInfo object before every query — this reloads Chroma
    # from disk every single time. We call setup() once to get the stores
    # loaded, then replace the load() method on each VectorStoreInfo with a
    # no-op so the already-loaded Chroma objects stay in memory permanently.
    # ---------------------------------------------------------------------------
    print("\nLoading vector store...")
    stores = Vector_Stores(vs_config_file=vs_config)
    stores.setup()

    # Patch out per-query reloads by replacing load() on each store info object
    _patched = 0
    for _domain_cfg in stores.vs_config.domains.values():
        for _vs_info in _domain_cfg.vector_stores:
            if _vs_info.loaded_object is not None:
                _vs_info.load = lambda *a, **kw: None
                _patched += 1
    print(f"  Patched {_patched} store(s) to prevent per-query disk reloads")
    stores.setup = lambda: None  # also block any future setup() calls
    print(f"  Domains: {stores.domains}")

    # Verify store has content
    for domain in stores.domains:
        test = stores.retrieve(domain_name=domain, query="neuron")
        print(f"  [{domain}] test retrieval: {len(test)} chunks")
        if not test:
            print(f"  WARNING: domain '{domain}' returned no results — store may be empty!")

    # -------------------------------------------------------------------------
    # Load all models ONCE up front — health check runs once per model,
    # not once per model per quiz category (which would be 4× wasteful).
    # -------------------------------------------------------------------------
    from neuroml_ai_utils.llm import setup_llm as _setup_llm

    print("\nLoading models (health check runs once each)...")
    loaded_models: Dict[str, object] = {}
    for llm_name in LLMS:
        model_str = llm_model_string(llm_name)
        try:
            logger = logging.getLogger(f"model_{llm_name}")
            model = _setup_llm(model_str, logger)
            if model is None:
                raise ValueError("setup_llm returned None")
            loaded_models[llm_name] = model
            print(f"  ✓ {llm_name}")
        except Exception as e:
            print(f"  ✗ {llm_name}: {e} — will be skipped in all quizzes")

    print(f"  {len(loaded_models)}/{len(LLMS)} models ready\n")

    all_results = []
    timestamp = datetime.datetime.now().strftime("%d-%m-%y_%H%M")

    for quiz_category, quiz_file in QUIZ_FILES.items():
        if not Path(quiz_file).exists():
            print(f"\n! Quiz file not found: {quiz_file} — skipping")
            continue

        questions = load_questions(quiz_file)
        if not questions:
            print(f"\n! No questions loaded from {quiz_file} — skipping")
            continue

        print(f"\n{'='*70}")
        print(f"  QUIZ: {quiz_category}  ({len(questions)} questions)")
        print(f"{'='*70}")

        # ------------------------------------------------------------------
        # Pre-classify domains ONCE per quiz category.
        # Domain classification is independent of which LLM answers the
        # question, so doing it 11 times (once per LLM) would be wasteful.
        #
        # Strategy:
        #   - For C. elegans / Corpus categories: every question should
        #     attempt retrieval — we force the default domain.
        #   - For General / Science categories: keyword-based fast check;
        #     most questions will return None (no retrieval needed), but any
        #     question that does mention C. elegans terms will still retrieve.
        #
        # Why not use the LLM here? Because we don't have a model loaded yet
        # at this point, and keyword classification is reliable enough given
        # that the category label already gives us strong prior information.
        # The LLM-based fallback inside classify_domain fires per-question
        # only when precomputed_domain is None (i.e. General/Science questions
        # that slipped past keywords).
        # ------------------------------------------------------------------
        force_retrieval = any(t in quiz_category.lower() for t in ("elegans", "corpus"))
        default_domain = stores.domains[0] if stores.domains else None

        precomputed_domains: List[Optional[str]] = []
        for q in questions:
            if force_retrieval:
                precomputed_domains.append(default_domain)
            else:
                precomputed_domains.append(classify_domain(stores, q["question"], model=None))

        retrieval_count = sum(1 for d in precomputed_domains if d is not None)
        print(f"  Pre-classification: {retrieval_count}/{len(questions)} questions will attempt retrieval")
        print(f"  (force_retrieval={'yes' if force_retrieval else 'no'}, fallback to LLM classify for ambiguous General/Science Qs)\n")

        for llm_idx, llm_name in enumerate(LLMS, 1):
            print(f"\n[{llm_idx}/{len(LLMS)}] {llm_name}")
            print("-" * 60)

            if llm_name not in loaded_models:
                print(f"  — skipped (failed to load at startup)")
                continue

            model = loaded_models[llm_name]

            try:
                result = await evaluate_llm_on_quiz(
                    llm_name, quiz_category, questions, stores, model,
                    precomputed_domains=precomputed_domains,
                )
                all_results.append(result)
            except Exception as e:
                print(f"  ! Evaluation error: {e}")
                import traceback; traceback.print_exc()


    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = f"{OUTPUT_DIR}/eval_{timestamp}.json"

    output = {
        "Title": "RAG Evaluation — LLM / RAG-Only / RAG+Fallback",
        "Description": (
            "Compares three answering strategies: "
            "(1) LLM-only knowledge, "
            "(2) RAG-only (answer from retrieved context, skip if none), "
            "(3) RAG+Fallback (use RAG if score >= threshold, else LLM)."
        ),
        "RAG Retrieval Threshold": RAG_RETRIEVAL_THRESHOLD,
        "Num Chunks Retrieved": NUM_CHUNKS,
        "Date": datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        "Results": all_results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"\n{'='*70}")
    print(f"  ALL DONE — results saved to {output_path}")
    print(f"{'='*70}")

    # -------------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------------
    print(f"\n{'─'*70}")
    print(f"  {'Category':<25} {'LLM':>6}  {'LLM%':>6}  {'RAG%':>6}  {'Hybrid%':>8}  {'Δ':>6}")
    print(f"{'─'*70}")

    for cat in QUIZ_FILES.keys():
        cat_results = [r for r in all_results if r["Quiz Category"] == cat]
        if not cat_results:
            continue
        avg_llm    = sum(r["LLM Accuracy (%)"]     for r in cat_results) / len(cat_results)
        avg_rag    = sum(r["RAG Accuracy Overall (%)"] for r in cat_results) / len(cat_results)
        avg_hybrid = sum(r["Hybrid Accuracy (%)"]   for r in cat_results) / len(cat_results)
        delta      = avg_hybrid - avg_llm
        print(f"  {cat:<25} {len(cat_results):>6}  {avg_llm:>5.1f}%  {avg_rag:>5.1f}%  {avg_hybrid:>7.1f}%  {delta:>+5.1f}pp")

    print(f"{'─'*70}\n")


if __name__ == "__main__":
    asyncio.run(main())