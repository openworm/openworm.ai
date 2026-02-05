from __future__ import annotations

import os
import json
import random
import glob
from typing import List, Tuple

from llama_index.core import Document, VectorStoreIndex

from openworm_ai.quiz.QuizModel import MultipleChoiceQuiz, Question, Answer
from openworm_ai.utils.llms import (
    ask_question_get_response,
    LLM_GPT4o,
    LLM_HF_DEFAULT,
    LLM_OLLAMA_GEMMA2,
)

# Reuse validated pipeline helpers from QuizMaster.py
from openworm_ai.quiz.QuizMaster import (
    _is_valid_mcq_item,
    parse_free_text_mcqs_to_items,
    score_question_with_critic,
    deduplicate_questions_with_index,
    get_default_critic_llm_ver,
    get_embed_model_for_llm,
)

indexing = ["A", "B", "C", "D"]

STRICT_GENERATE_Q = """
🔹 TASK: Generate exactly <QUESTION_NUMBER> multiple-choice questions using ONLY the provided text.
- The questions must be highly specific and cannot come from general knowledge.
- If the topic is not in the provided text, DO NOT generate a question about it.
- Questions should challenge researchers and advanced students.
- DO NOT include sources in the question text (no "according to [source]").

FORMAT (repeat per question):
QUESTION: <Insert question>
CORRECT ANSWER: <Correct answer>
WRONG ANSWER: <Wrong answer 1>
WRONG ANSWER: <Wrong answer 2>
WRONG ANSWER: <Wrong answer 3>

IMPORTANT: If the text does not have enough content for <QUESTION_NUMBER> questions, generate as many as possible.
""".strip()


def load_corpus_sections(
    papers_glob: str = "processed/json/papers/*.json",
) -> List[dict]:
    """
    Load sections from all processed paper JSONs, skipping obvious non-body-text
    sections like References, Bibliography, etc.

    Returns:
      {"text": "...", "source": "Paper.json: [Title, Section X](url)"}
    """
    json_inputs = glob.glob(papers_glob)
    sections: List[dict] = []

    if not json_inputs:
        print(f"! Warning: no JSON papers found under {papers_glob}")
        return sections

    for json_file in json_inputs:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"! Error reading {json_file}: {e}")
            continue

        for title, doc_contents in data.items():
            src_page = doc_contents.get("source", json_file)
            for section_name, details in doc_contents.get("sections", {}).items():
                sec_name_lower = section_name.lower()

                if any(
                    key in sec_name_lower
                    for key in [
                        "reference",
                        "bibliograph",
                        "supplementary",
                        "acknowledg",
                        "funding",
                        "author contributions",
                        "materials and methods",
                    ]
                ):
                    continue

                paragraphs = details.get("paragraphs", [])
                text = " ".join(p.get("contents", "") for p in paragraphs).strip()

                if len(text.split()) < 30:
                    continue

                lower_text = text.lower()
                if "doi.org" in lower_text or "doi:" in lower_text:
                    continue

                src_info = (
                    f"{os.path.basename(json_file)}: "
                    f"[{title}, Section {section_name}]({src_page})"
                )
                sections.append({"text": text, "source": src_info})

    print(f"Loaded {len(sections)} sections from corpus papers (after filtering)")
    return sections


def build_corpus_index_for_mcq(
    llm_ver: str, papers_glob: str = "processed/json/papers/*.json"
) -> Tuple[VectorStoreIndex, List[Document]]:
    """
    Build a VectorStoreIndex over the corpus sections, to use for RAG-style retrieval.
    Returns (index, docs).
    """
    sections = load_corpus_sections(papers_glob=papers_glob)
    if not sections:
        raise ValueError("No sections found for corpus index.")

    docs: List[Document] = []
    for sid, sec in enumerate(sections):
        docs.append(
            Document(
                text=sec["text"],
                metadata={"source": sec["source"], "sid": sid},
            )
        )

    embed_model = get_embed_model_for_llm(llm_ver)
    print("[RAG] Building VectorStoreIndex for corpus MCQ generation...")
    index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
    print(f"[RAG] Built index over {len(docs)} documents")
    return index, docs


def save_quiz_v2(
    num_questions: int = 100,
    llm_ver: str = LLM_GPT4o,
    temperature: float = 0.2,
    questions_per_context: int = 1,
    similarity_top_k: int = 3,
    overgen_factor: int = 3,
    similarity_threshold_dedup: float = 0.75,
    papers_glob: str = "processed/json/papers/*.json",
):
    """
    Corpus+RAG MCQ generator.
    Uses free-text generation (small-LLM-friendly), then parses to structured MCQ items,
    critic-scores, embedding-dedups, and saves a MultipleChoiceQuiz JSON.

    Additionally: preserves where question stems from in the saved quiz JSON via Question.metadata["source"].
    """
    try:
        index, docs = build_corpus_index_for_mcq(llm_ver, papers_glob=papers_glob)
    except Exception as e:
        print(f"! Error building corpus index: {e}")
        return

    if not docs:
        print("! Error: No documents in corpus index.")
        return

    retriever = index.as_retriever(similarity_top_k=similarity_top_k)

    raw_target = num_questions * max(1, overgen_factor)
    all_items: List[dict] = []

    max_attempts = raw_target * 6
    attempts = 0

    while len(all_items) < raw_target and attempts < max_attempts:
        attempts += 1

        seed_doc = random.choice(docs)
        seed_text = seed_doc.text

        try:
            results = retriever.retrieve(seed_text)
        except Exception as e:
            print(f"! RAG retrieval failed on attempt {attempts}: {e}")
            continue

        ctx_texts: List[str] = []
        sources = set()

        for r in results:
            try:
                ctx_texts.append(r.get_content())
                md = getattr(r, "metadata", None) or {}
                src = md.get("source", "")
            except Exception:
                node = getattr(r, "node", None)
                if node is None:
                    continue
                ctx_texts.append(getattr(node, "text", "") or "")
                md = getattr(node, "metadata", None) or {}
                src = md.get("source", "")

            if src:
                sources.add(src)

        context = "\n\n".join(t for t in ctx_texts if t.strip())
        if not context.strip():
            continue

        source_str = "; ".join(sorted(sources)) if sources else ""

        prompt = (
            STRICT_GENERATE_Q.replace("<QUESTION_NUMBER>", str(questions_per_context))
            + "\n\nTEXT (use ONLY this):\n"
            + '"""\n'
            + context
            + '\n"""'
        ).strip()

        raw = ask_question_get_response(prompt, llm_ver, temperature)

        items = parse_free_text_mcqs_to_items(raw)
        valid_items = [it for it in items if _is_valid_mcq_item(it)]
        if not valid_items:
            continue

        for it in valid_items:
            if source_str:
                it["_source"] = source_str

        all_items.extend(valid_items)

        if attempts % 10 == 0:
            print(
                f"[Corpus+RAG] attempts={attempts}, valid_items={len(all_items)}/{raw_target}"
            )

    if not all_items:
        print("! Error: No valid MCQs generated from RAG-based corpus passages.")
        return

    print(f"[Corpus+RAG] Generated {len(all_items)} valid MCQs before critic/dedup")

    critic_llm_ver = get_default_critic_llm_ver()
    print(
        f"[Corpus+RAG] Using critic model {critic_llm_ver} to score {len(all_items)} questions"
    )

    for idx, item in enumerate(all_items):
        score, _ = score_question_with_critic(item, llm_ver_critic=critic_llm_ver)
        item["_critic_score"] = score
        if idx < 10 or (idx + 1) % 25 == 0:
            print(f"  [Critic] Q{idx}: score={score:.1f}")

    all_items.sort(key=lambda x: x.get("_critic_score", 0.0), reverse=True)

    try:
        selected_items = deduplicate_questions_with_index(
            all_items,
            llm_ver=llm_ver,
            similarity_threshold=similarity_threshold_dedup,
            max_items=num_questions,
        )
        print(
            f"[Corpus+RAG] Selected {len(selected_items)} after dedup (target={num_questions})"
        )
    except Exception as e:
        print(f"! [Corpus+RAG] Dedup failed, falling back to top-{num_questions}: {e}")
        selected_items = all_items[:num_questions]

    quiz = MultipleChoiceQuiz(
        title=f"{llm_ver.replace(':', '_')}_{num_questions}questions_celegans_corpus_rag_v2",
        source=(
            f"Corpus-based (RAG) quiz generated from processed papers by {llm_ver}, "
            f"temperature={temperature}, free-text->parse->critic->dedup"
        ),
    )

    # -----------------------------
    # NEW: Preserve per-question source area
    # -----------------------------
    for item in selected_items:
        stem = item["question"].strip()
        q_obj = Question(question=stem)

        # attach metadata
        src = item.get("_source")
        if src:
            try:
                # If Question supports metadata in constructor or as an attribute
                if hasattr(q_obj, "metadata"):
                    q_obj.metadata = {"source": src}
                else:
                    # Fallback: store as a custom attribute
                    setattr(q_obj, "metadata", {"source": src})
            except Exception:
                # Last resort
                pass

        for i, opt in enumerate(item["options"]):
            text = opt["text"].strip()
            is_correct = opt["label"] == item["correct_label"]
            q_obj.answers.append(Answer(str(i + 1), text, is_correct))

        quiz.questions.append(q_obj)

    print("===============================\nGenerated corpus+RAG quiz:\n")
    print(quiz.to_yaml())

    out_path = (
        "openworm_ai/quiz/samples/"
        f"{llm_ver.replace(':', '_')}_{num_questions}questions_celegans_corpus_rag_v2.json"
    )
    quiz.to_json_file(out_path)
    print(f" Saved corpus+RAG quiz to {out_path}")


if __name__ == "__main__":
    import sys

    if os.getenv("OPENAI_API_KEY"):
        llm_ver = LLM_GPT4o
    elif os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        llm_ver = LLM_HF_DEFAULT
    else:
        llm_ver = LLM_OLLAMA_GEMMA2

    print(f"Selected LLM: {llm_ver}")

    if "-ask" in sys.argv:
        num_questions = 4
        quiz_json = (
            "openworm_ai/quiz/samples/"
            f"{llm_ver.replace(':', '_')}_{num_questions}questions_celegans_corpus_rag_v2.json"
        )

        print(f"Loading quiz from: {quiz_json}")
        quiz = MultipleChoiceQuiz.from_file(quiz_json)

        total_qs = 0
        total_correct = 0
        wrong_answers = "Incorrect answers:\n"

        from openworm_ai.quiz.Templates import ASK_Q

        for qi, question in enumerate(quiz.questions):
            q = question["question"]

            answers = ""
            random.shuffle(question["answers"])

            presented_answers = {}
            correct_answer = None
            correct_text = None

            for index, answer in enumerate(question["answers"]):
                ref = indexing[index]
                present = f"{ref}: {answer['ans']}"
                if answer["correct"]:
                    correct_answer = ref
                    correct_text = present
                presented_answers[ref] = present
                answers += f"{present}\n"

            full_question = ASK_Q.replace("<QUESTION>", q).replace("<ANSWERS>", answers)

            resp = ask_question_get_response(
                full_question, llm_ver, print_question=False
            ).strip()

            total_qs += 1
            correct_guess = resp == correct_answer
            if correct_guess:
                total_correct += 1
            else:
                wrong_answers += f"  {q}; Wrong: {resp}; Correct: {correct_answer} ({correct_text})\n"

            print(
                f" >> {qi}) {q} -> Guess: {resp}, Correct: {correct_answer} -> {correct_guess}"
            )

        print(wrong_answers)
        print(f"\nTotal correct: {total_correct} / {total_qs}")

    else:
        num = 4
        for a in sys.argv:
            if a.isnumeric():
                num = int(a)

        save_quiz_v2(
            num_questions=num,
            llm_ver=llm_ver,
            temperature=0.2,
            questions_per_context=1,
            similarity_top_k=3,
            overgen_factor=3,
            papers_glob="processed/json/papers/*.json",
        )
