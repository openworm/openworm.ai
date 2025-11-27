import os
import json
import random
from typing import List
import glob

from openworm_ai.quiz.QuizModel import MultipleChoiceQuiz, Question, Answer
from openworm_ai.utils.llms import (
    ask_question_get_response,
    LLM_GPT4o,
    LLM_OLLAMA_GEMMA2,
)

from openworm_ai.quiz.QuizMaster import (
    _is_valid_mcq_item,
    score_question_with_critic,
    deduplicate_questions_with_index,
    get_default_critic_llm_ver,
    get_embed_model_for_llm,
)

from llama_index.core import Document, VectorStoreIndex


indexing = ["A", "B", "C", "D"]
TOKEN_LIMIT = 30_000  # 🔹 Keeps request within OpenAI's limits

# **STRICT Prompt to prevent external knowledge**
STRICT_GENERATE_Q = """
🔹 **TASK:** Generate exactly <QUESTION_NUMBER> multiple-choice questions using **only** the provided text.  
- The questions must be **highly specific** and **cannot** come from general knowledge.  
- If the topic is not in the provided text, **DO NOT** generate a question about it.  
- Questions should challenge **researchers** and **advanced students**.  
- DO NOT include the sources in the questions (...according to [source])
🔹 **FORMAT:**  
QUESTION: <Insert question>  
CORRECT ANSWER: <Correct answer>  
WRONG ANSWER: <Wrong answer 1>  
WRONG ANSWER: <Wrong answer 2>  
WRONG ANSWER: <Wrong answer 3>  

📌 **IMPORTANT:** If the text does not have enough content for <QUESTION_NUMBER> questions, generate as many as possible.  
"""


def load_corpus_sections(
    papers_glob: str = "processed/json/papers/*.json",
) -> List[dict]:
    """
    Load sections from all processed paper JSONs, skipping obvious non-body-text
    sections like References, Bibliography, etc.

    Returns a list of dicts:
      {
        "text": "...section text...",
        "source": "PaperFile.json: [Title, Section X](url)"
      }
    """
    json_inputs = glob.glob(papers_glob)
    sections: List[dict] = []

    if not json_inputs:
        print(f"⚠ Warning: no JSON papers found under {papers_glob}")
        return sections

    for json_file in json_inputs:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"⚠ Error reading {json_file}: {e}")
            continue

        for title, doc_contents in data.items():
            src_page = doc_contents.get("source", json_file)
            for section_name, details in doc_contents.get("sections", {}).items():
                sec_name_lower = section_name.lower()

                # 🔹 Skip obvious reference-like / non-content sections
                if any(
                    key in sec_name_lower
                    for key in [
                        "reference",
                        "bibliograph",
                        "supplementary",
                        "acknowledg",
                        "funding",
                        "author contributions",
                        "materials and methods",  # optional, remove if you want methods Qs
                    ]
                ):
                    continue

                paragraphs = details.get("paragraphs", [])
                text = " ".join(p.get("contents", "") for p in paragraphs).strip()

                # Skip ultra-short or weird sections (tables, axes, etc.)
                if len(text.split()) < 30:
                    continue

                # Skip sections that look like pure citation/DOI blobs
                lower_text = text.lower()
                if "doi.org" in lower_text or "doi:" in lower_text:
                    continue

                src_info = (
                    f"{os.path.basename(json_file)}: "
                    f"[{title}, Section {section_name}]({src_page})"
                )
                sections.append(
                    {
                        "text": text,
                        "source": src_info,
                    }
                )

    print(f" Loaded {len(sections)} sections from corpus papers (after filtering)")
    return sections


def build_corpus_index_for_mcq(
    llm_ver: str, papers_glob: str = "processed/json/papers/*.json"
) -> tuple[VectorStoreIndex, List[Document]]:
    """
    Build a VectorStoreIndex over the corpus sections, to use for RAG-style
    context selection when generating MCQs.

    Returns:
      (index, docs)
      - index: VectorStoreIndex over all sections
      - docs: list of Documents with .text and metadata["source"]
    """
    # Reuse your existing loader (with filtering)
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
    questions_per_section: int = 3,
):
    """
    Generate and save a corpus-based quiz where each MCQ is grounded
    in the processed paper JSONs, using RAG-style context selection.

    Pipeline:
      - Build a VectorStoreIndex over processed/json/papers/*.json
      - Repeatedly pick a random seed doc and retrieve top-k similar docs
      - Use concatenated retrieved text as context for MCQ generation
      - Validate items (_is_valid_mcq_item)
      - Critic-score, rank, and deduplicate (using the same pipeline as QuizMaster)
      - Build a MultipleChoiceQuiz and save as JSON
    """
    # 🔹 Build RAG index over corpus
    try:
        index, docs = build_corpus_index_for_mcq(llm_ver)
    except Exception as e:
        print(f"⚠ Error building corpus index: {e}")
        return

    if not docs:
        print("⚠ Error: No documents in corpus index.")
        return

    retriever = index.as_retriever(similarity_top_k=3)

    # Over-generate so critic + dedup have room
    raw_target = num_questions * 3
    all_items: List[dict] = []

    # We'll cap the number of retrieval/generation attempts so we don't loop forever
    max_attempts = raw_target * 2
    attempts = 0

    while len(all_items) < raw_target and attempts < max_attempts:
        attempts += 1

        # 🔹 Pick a random seed document and use it as a "query"
        seed_doc = random.choice(docs)
        seed_text = seed_doc.text

        try:
            results = retriever.retrieve(seed_text)
        except Exception as e:
            print(f"⚠ RAG retrieval failed on attempt {attempts}: {e}")
            continue

        # Build context from top-k retrieved docs
        ctx_texts = []
        sources = set()
        for r in results:
            try:
                # NodeWithScore in LlamaIndex allows get_content()
                ctx_texts.append(r.get_content())
                src = r.metadata.get("source", "")
            except AttributeError:
                # Fallback if API differs
                node = getattr(r, "node", None)
                if node is not None:
                    ctx_texts.append(getattr(node, "text", ""))
                    src = node.metadata.get("source", "")
                else:
                    continue
            if src:
                sources.add(src)

        context = "\n\n".join(t for t in ctx_texts if t.strip())
        if not context.strip():
            continue

        source = "; ".join(sorted(sources))

        prompt = f"""
You are generating multiple-choice questions based on scientific papers
about C. elegans.

You are given the following reference material, composed of several
semantically related passages:

\"\"\"{context}\"\"\"

Use ONLY this material (no external knowledge) to generate
{questions_per_section} multiple-choice questions.

Each question must:
- Be clearly answerable from the provided material.
- Have exactly one correct answer and 3 plausible incorrect answers.
- Be specific and technically accurate, suitable for advanced students or researchers.
- NOT reference the text explicitly (no "according to the text" phrasing).

Return your output as a JSON array. Each element must have the form:
{{
  "question": "...",
  "options": [
    {{"label": "A", "text": "..."}},
    {{"label": "B", "text": "..."}},
    {{"label": "C", "text": "..."}},
    {{"label": "D", "text": "..."}}
  ],
  "correct_label": "A"
}}

Do not include any extra keys, commentary, or code fences.
""".strip()

        raw = ask_question_get_response(prompt, llm_ver, temperature)

        try:
            from openworm_ai.quiz.QuizMaster import _extract_json_array  # reuse helper

            json_str = _extract_json_array(raw)
            items = json.loads(json_str)
        except Exception:
            print(
                "⚠ Failed to parse JSON from RAG-based corpus generation. Skipping this batch."
            )
            continue

        valid_items = [it for it in items if _is_valid_mcq_item(it)]
        if not valid_items:
            continue

        # Attach source metadata
        for it in valid_items:
            it["_source"] = source

        all_items.extend(valid_items)

    if not all_items:
        print("⚠ Error: No valid MCQs generated from RAG-based corpus passages.")
        return

    print(
        f"📊 Corpus+RAG generation produced {len(all_items)} valid MCQs before critic/dedup"
    )

    # Critic scoring (same as before)
    critic_llm_ver = get_default_critic_llm_ver()
    print(
        f"[Corpus+RAG] Using critic model {critic_llm_ver} to score {len(all_items)} questions"
    )

    for idx, item in enumerate(all_items):
        score, _ = score_question_with_critic(item, llm_ver_critic=critic_llm_ver)
        item["_critic_score"] = score
        print(f"  [Corpus+RAG Critic] Q{idx}: score={score:.1f}")

    all_items.sort(key=lambda x: x.get("_critic_score", 0.0), reverse=True)

    # Dedup with same VectorStore-based logic
    try:
        selected_items = deduplicate_questions_with_index(
            all_items,
            llm_ver=llm_ver,
            similarity_threshold=0.9,
            max_items=num_questions,
        )
        print(
            f"[Corpus+RAG Step 7] Selected {len(selected_items)} corpus-based questions "
            f"after dedup (target={num_questions})"
        )
    except Exception as e:
        print(
            f"⚠ [Corpus+RAG Step 7] VectorStore-based dedup failed, "
            f"falling back to top-{num_questions}: {e}"
        )
        selected_items = all_items[:num_questions]

    # 🔹 Build MultipleChoiceQuiz
    quiz = MultipleChoiceQuiz(
        title=f"{llm_ver.replace(':', '_')}_{num_questions}questions_celegans_corpus_rag_v2",
        source=f"Corpus-based (RAG) quiz generated from processed papers by {llm_ver}, "
        f"temperature: {temperature}",
    )

    for item in selected_items:
        stem = item["question"].strip()
        q_obj = Question(question=stem)

        for i, opt in enumerate(item["options"]):
            text = opt["text"].strip()
            is_correct = opt["label"] == item["correct_label"]
            q_obj.answers.append(Answer(str(i + 1), text, is_correct))

        quiz.questions.append(q_obj)

    print("===============================\n  Generated corpus+RAG quiz:\n")
    print(quiz.to_yaml())

    out_path = (
        f"openworm_ai/quiz/samples/"
        f"{llm_ver.replace(':', '_')}_{num_questions}questions_celegans_corpus_rag_v2.json"
    )
    quiz.to_json_file(out_path)
    print(f"💾 Saved corpus+RAG JSON-v2 quiz to {out_path}")


if __name__ == "__main__":
    import sys
    import os

    if os.getenv("OPENAI_API_KEY"):
        llm_ver = LLM_GPT4o
    else:
        llm_ver = LLM_OLLAMA_GEMMA2

    print(f"Selected LLM: {llm_ver}")

    if "-ask" in sys.argv:
        # Match the new v2 filename pattern
        num_questions = 4
        quiz_json = (
            f"openworm_ai/quiz/samples/"
            f"{llm_ver.replace(':', '_')}_{num_questions}questions_celegans_corpus_v2.json"
        )

        print(f"Loading quiz from: {quiz_json}")
        quiz = MultipleChoiceQuiz.from_file(quiz_json)

        total_qs = 0
        total_correct = 0
        wrong_answers = "Incorrect answers:\n"

        for qi, question in enumerate(quiz.questions):
            q = question["question"]

            from openworm_ai.quiz.Templates import ASK_Q

            answers = ""
            random.shuffle(question["answers"])

            presented_answers = {}
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

            print(
                f" >> {qi}) {q} → Guess: {resp}, Correct: {correct_answer} → {correct_guess}"
            )

        print(f"\nTotal correct: {total_correct} / {total_qs}")

    else:
        # Use the new v2 generator
        save_quiz_v2(
            num_questions=4,
            llm_ver=llm_ver,
            temperature=0.2,
            questions_per_section=1,
        )
