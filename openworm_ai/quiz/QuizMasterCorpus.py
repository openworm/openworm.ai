from __future__ import annotations

import os
import json
import random
import glob
from typing import List, Optional, Tuple

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding

from openworm_ai.quiz.QuizModel import Answer, MultipleChoiceQuiz, Question
from openworm_ai.utils.llms import (
    LLM_CLAUDE37,
    LLM_HF_QWEN25_72B,
    ask_question_get_response,
    get_anthropic_key,
    get_llm,
    get_llm_from_argv,
)

from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# CLI constants
# -----------------------------

LABEL_SETS = [
    ["A", "B", "C", "D"],
    ["E", "F", "G", "H"],
    ["J", "K", "L", "M"],
    ["P", "Q", "R", "S"],
]

RANDOMIZE_ASK_LABELS = True


def get_default_critic_llm_ver():
    """
    Choose the default critic model:
    - If an Anthropic key is available -> Claude 3.7 Sonnet
    - Otherwise -> fall back to HF model (QWEN)
    """
    try:
        key = get_anthropic_key()
    except Exception:
        key = None
    return LLM_CLAUDE37 if key else LLM_HF_QWEN25_72B


# -----------------------------
# Critic: simplified (no anchors)
# -----------------------------


def score_question_with_critic(
    item: dict,
    llm_ver_critic: Optional[str] = None,
    temperature: float = 0.0,
) -> Tuple[float, bool]:
    """
    Score a single MCQ item using a critic LLM.

    Returns: (score: float, reject: bool)
    """
    if llm_ver_critic is None:
        llm_ver_critic = get_default_critic_llm_ver()

    mcq_json_str = json.dumps(item, ensure_ascii=False, indent=2)

    critic_prompt = """
You are a STRICT evaluator of multiple-choice questions (MCQs).

You will be given ONE MCQ in JSON format:
- "question": the question text
- "options": array of 4 answers (A–D)
- "correct_label": the intended correct option.

First, silently check:
- Is the correct answer unambiguous and factually stable?
- Could any other option be arguably correct?
- Are distractors plausible (not silly / irrelevant)?

REJECT must be true if ANY:
- The question depends on rankings, "most", "first", "largest", or "as of <year>" style facts without a stable anchor.
- More than one option could be correct under any reasonable interpretation.
- The correct option depends on definition/region/timeframe.
- Any option is nonsense / fabricated.
- Distractors are obviously non-competitive or unrelated.

SCORING (0–100) — use the FULL SCALE (do NOT cluster around 80–90):
- 95–100: Excellent. Crystal clear, one correct, strong distractors, interesting, not trivial.
- 85–94: Very good. Minor weakness (slightly easy OR one distractor a bit weak) but still clean.
- 70–84: OK. Noticeably easy, or distractors weak/revealing, or wording slightly clunky.
- 50–69: Poor. Ambiguity risk, shaky fact stability, or multiple distractors weak.
- 0–49: Broken. Likely wrong/ambiguous OR should be rejected.

IMPORTANT: If you are tempted to give 85, ask yourself: is it actually "very good", or merely "OK" (70–84)?
Also: only give 95+ if the distractors are genuinely competitive.

Return ONLY valid JSON with EXACT keys "score" and "reject":
{{"score": <integer 0-100>, "reject": <true/false>}}

MCQ:
{mcq_json}
""".strip()

    prompt = PromptTemplate(template=critic_prompt, input_variables=["mcq_json"])

    try:
        llm = get_llm(llm_ver_critic, temperature)
        chain = prompt | llm | StrOutputParser()
        resp = chain.invoke({"mcq_json": mcq_json_str}).strip()
    except Exception as e:
        print(f"! Critic LLM call failed ({llm_ver_critic}): {e}")
        return 50.0, True  # fail-safe: reject

    # Robust JSON extraction & brace normalization
    try:
        start = resp.find("{")
        end = resp.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found.")

        candidate = resp[start : end + 1].strip()
        candidate = candidate.replace("{{", "{").replace("}}", "}")

        obj = json.loads(candidate)
        score = float(obj.get("score", 50.0))
        reject = bool(obj.get("reject", False))
        return score, reject
    except Exception:
        print("! Failed to parse critic response as JSON:")
        print(resp)
        return 50.0, True


# -----------------------------
# Free-text -> structured MCQ items
# -----------------------------


def parse_free_text_mcqs_to_items(text: str) -> List[dict]:
    """
    Tolerant streaming parser for small LLMs.

    Accepts:
    - QUESTION: ...   OR any line ending with '?'
    - CORRECT ANSWER: ...  (also accepts CORRECT:, ANSWER:)
    - WRONG ANSWER: ...    (also accepts WRONG:, INCORRECT:)

    Produces items in the canonical internal format:
      {
        "question": "...",
        "options": [{"label":"A","text":"..."}, ...],
        "correct_label":"A"
      }
    """

    if not isinstance(text, str) or not text.strip():
        return []

    i = text.upper().find("QUESTION:")
    if i != -1:
        text = text[i:]

    items: List[dict] = []
    cur_q: Optional[str] = None
    correct: Optional[str] = None
    wrongs: List[str] = []

    def flush():
        nonlocal cur_q, correct, wrongs
        if cur_q and correct and len(wrongs) >= 3:
            options_text = [correct] + wrongs[:3]
            options = []
            for j, opt_text in enumerate(options_text):
                options.append(
                    {"label": ["A", "B", "C", "D"][j], "text": opt_text.strip()}
                )
            items.append(
                {"question": cur_q.strip(), "options": options, "correct_label": "A"}
            )
        cur_q, correct, wrongs = None, None, []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        u = line.upper()

        if u.startswith("QUESTION:") or line.endswith("?"):
            flush()
            cur_q = line.split(":", 1)[1].strip() if ":" in line else line
            continue

        if (
            u.startswith("CORRECT ANSWER:")
            or u.startswith("CORRECT:")
            or (u.startswith("ANSWER:") and correct is None)
        ):
            correct = line.split(":", 1)[1].strip() if ":" in line else line
            continue

        if (
            u.startswith("WRONG ANSWER:")
            or u.startswith("WRONG:")
            or u.startswith("INCORRECT:")
            or u.startswith("INCORRECT ANSWER:")
        ):
            wrongs.append(line.split(":", 1)[1].strip() if ":" in line else line)
            continue

    flush()
    return items


def _is_valid_mcq_item(item: dict) -> bool:
    try:
        q = item.get("question", "")
        if not isinstance(q, str) or not q.strip():
            return False

        options = item.get("options")
        if not isinstance(options, list) or len(options) != 4:
            return False

        labels = set()
        for opt in options:
            if not isinstance(opt, dict):
                return False
            label = opt.get("label")
            text = opt.get("text")
            if label not in ["A", "B", "C", "D"]:
                return False
            if not isinstance(text, str) or not text.strip():
                return False
            labels.add(label)

        correct = item.get("correct_label")
        if correct not in labels:
            return False

        return True
    except Exception:
        return False


# -----------------------------
# Embedding dedup
# -----------------------------


def question_to_text(item: dict) -> str:
    # NOTE: keep this simple; the stem carries most "dup" signal
    return item.get("question", "").strip()


def get_embed_model_for_llm(llm_ver: str):
    openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")

    if llm_ver.startswith("huggingface:") or not openai_key:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        return HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")

    return OpenAIEmbedding()


def build_question_index(questions: List[dict], llm_ver: str) -> VectorStoreIndex:
    docs: List[Document] = []
    for idx, q in enumerate(questions):
        docs.append(Document(text=question_to_text(q), metadata={"qid": idx}))
    embed_model = get_embed_model_for_llm(llm_ver)
    return VectorStoreIndex.from_documents(docs, embed_model=embed_model)


def deduplicate_questions_with_index(
    questions: List[dict],
    llm_ver: str,
    similarity_threshold: float = 0.85,  # <-- IMPORTANT: don't set this too low
    max_items: Optional[int] = None,
) -> List[dict]:
    if not questions:
        return []

    import nest_asyncio

    nest_asyncio.apply()

    index = build_question_index(questions, llm_ver)
    retriever = index.as_retriever(similarity_top_k=5)

    kept: List[int] = []
    for idx, q in enumerate(questions):
        if max_items is not None and len(kept) >= max_items:
            break

        results = retriever.retrieve(question_to_text(q))

        dup = False
        for node in results:
            other_id = (node.metadata or {}).get("qid")
            score = node.score
            if other_id == idx:
                continue
            if other_id in kept and score is not None and score >= similarity_threshold:
                dup = True
                break

        if not dup:
            kept.append(idx)

    return [questions[i] for i in kept]


# -----------------------------
# Corpus loading and indexing
# -----------------------------


def load_corpus_sections(
    papers_glob: str = "processed/json/papers/*.json",
) -> List[dict]:
    """
    Load sections from all processed paper JSONs, skipping obvious non-body-text
    sections like References, Bibliography, etc.

    Returns:
      [{"text": "...", "source": "Paper.json: [Title, Section X](url)"}, ...]
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

                # Skip non-content sections
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

                # Skip very short sections
                if len(text.split()) < 30:
                    continue

                # Skip sections that are mostly DOIs/references
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


def build_corpus_index(
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


# -----------------------------
# Generation prompt template
# -----------------------------

CORPUS_GENERATE_Q = """
Generate <QUESTION_NUMBER> multiple-choice questions based EXCLUSIVELY on the provided text below.

CRITICAL RULES:
- Questions MUST be answerable ONLY from the provided text
- DO NOT use general knowledge - if the answer isn't in the text, don't ask it
- Questions should be specific, challenging, and test deep understanding
- DO NOT include source citations in questions (no "according to the paper...")
- All 4 answer options must be plausible based on the text

The questions should be answerable by researchers and advanced students familiar with C. elegans.

There should be 4 possible answers, only one of which is unambiguously correct. All answers should be brief and plausible.

Each of the <QUESTION_NUMBER> question/answer sets should be presented in the following format:

QUESTION: <Insert question>  
CORRECT ANSWER: <Correct answer>  
WRONG ANSWER: <Wrong answer 1>  
WRONG ANSWER: <Wrong answer 2>  
WRONG ANSWER: <Wrong answer 3>
""".strip()


# -----------------------------
# Main generation entrypoint
# -----------------------------


def save_quiz_corpus(
    num_questions: int,
    llm_ver: str,
    temperature: float = 0.7,
    papers_glob: str = "processed/json/papers/*.json",
):
    """
    Corpus-based quiz generation with RAG retrieval, aligned with working QuizMaster.

    Key design (from working QuizMaster):
    - BATCH_SIZE=10 questions per LLM call
    - accepted_descriptions fed back to prevent repeats
    - Top-up rounds with incremental dedup
    - similarity_threshold=0.85
    """

    # Build corpus index
    try:
        index, docs = build_corpus_index(llm_ver, papers_glob=papers_glob)
    except Exception as e:
        print(f"! Error building corpus index: {e}")
        return

    if not docs:
        print("! Error: No documents in corpus index.")
        return

    retriever = index.as_retriever(similarity_top_k=3)

    BATCH_SIZE = 10
    target_pool = num_questions * 2
    max_calls = max(10, (target_pool // BATCH_SIZE) * 3)

    pool: List[dict] = []
    pool_texts: set = set()
    accepted_descriptions: List[str] = []
    calls = 0
    critic_llm_ver = get_default_critic_llm_ver()

    def make_prompt(batch_num: int) -> str:
        exclusion_context = ""
        if accepted_descriptions:
            recent = accepted_descriptions[-30:]
            exclusion_context += (
                "\n\nQuestions already accepted - DO NOT repeat or closely resemble any of these:\n"
                + "\n".join(f"- {d}" for d in recent)
                + "\n"
            )

        instruction = (
            f"\n\nGenerate batch #{batch_num} of {BATCH_SIZE} unique questions "
            f"based ONLY on the provided text.\n\n"
        )

        # Sample random seed and retrieve context
        seed_doc = random.choice(docs)
        seed_text = seed_doc.text

        try:
            results = retriever.retrieve(seed_text)
        except Exception as e:
            print(f"! RAG retrieval failed: {e}")
            return None, None

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
            return None, None

        source_str = "; ".join(sorted(sources)) if sources else ""

        prompt = (
            CORPUS_GENERATE_Q.replace("<QUESTION_NUMBER>", str(BATCH_SIZE))
            + exclusion_context
            + instruction
            + 'TEXT (use ONLY this):\n"""\n'
            + context
            + '\n"""'
        ).strip()

        return prompt, source_str

    def score_and_add_to_pool(candidates: List[dict], source_str: str) -> int:
        """Score with critic and add passing items to pool."""
        added = 0
        for item in candidates:
            if not _is_valid_mcq_item(item):
                continue
            key = item["question"].lower().strip()
            if key in pool_texts:
                continue

            # Attach source metadata
            if source_str:
                item["_source"] = source_str

            s, reject = score_question_with_critic(
                item, llm_ver_critic=critic_llm_ver, temperature=0.2
            )
            item["_critic_score"] = s
            item["_critic_reject"] = reject
            if not reject:
                pool.append(item)
                pool_texts.add(key)
                accepted_descriptions.append(item["question"].strip()[:100])
                added += 1
        return added

    # Initial generation
    print(f"Using critic model {critic_llm_ver}")
    while calls < max_calls:
        calls += 1

        prompt, source_str = make_prompt(calls)
        if prompt is None:
            continue

        raw = ask_question_get_response(prompt, llm_ver, temperature)
        parsed = parse_free_text_mcqs_to_items(raw)
        added = score_and_add_to_pool(parsed, source_str)

        print(
            f"[Gen] call={calls}/{max_calls} | batch_parsed={len(parsed)} | "
            f"added={added} | pool={len(pool)} (target={target_pool})"
        )
        if len(pool) >= target_pool:
            break

    if not pool:
        raise ValueError(f"No valid questions after {calls} generation calls.")

    # Sort pool by critic score, then semantic-dedup to get initial selection
    pool.sort(key=lambda x: float(x.get("_critic_score", 0.0)), reverse=True)
    selected = deduplicate_questions_with_index(
        pool, llm_ver=llm_ver, similarity_threshold=0.85, max_items=num_questions
    )
    print(f"[Init] After semantic dedup: {len(selected)}/{num_questions} questions")

    # Top-up rounds (incremental dedup against fixed `selected`)
    generation_round = 1
    max_rounds = 5

    while len(selected) < num_questions and generation_round <= max_rounds:
        shortage = num_questions - len(selected)
        batches_needed = max(2, (shortage * 2 + BATCH_SIZE - 1) // BATCH_SIZE)
        print(
            f"\n[ROUND {generation_round + 1}] Need {shortage} more. "
            f"Generating {batches_needed} batches (~{batches_needed * BATCH_SIZE} questions)..."
        )

        new_valid: List[dict] = []
        for _ in range(batches_needed):
            calls += 1

            prompt, source_str = make_prompt(calls)
            if prompt is None:
                continue

            raw = ask_question_get_response(prompt, llm_ver, temperature)
            parsed = parse_free_text_mcqs_to_items(raw)
            for item in parsed:
                if not _is_valid_mcq_item(item):
                    continue
                key = item["question"].lower().strip()
                if key in pool_texts:
                    continue

                if source_str:
                    item["_source"] = source_str

                s, reject = score_question_with_critic(
                    item, llm_ver_critic=critic_llm_ver, temperature=0.2
                )
                item["_critic_score"] = s
                item["_critic_reject"] = reject
                if not reject:
                    new_valid.append(item)
                    pool_texts.add(key)
                    accepted_descriptions.append(item["question"].strip()[:100])

        print(f"  {len(new_valid)} new candidates passed critic + exact-dedup")

        if new_valid:
            # Incremental dedup: pass [selected + new_valid_sorted] so that selected
            # items are processed first and always kept
            new_valid.sort(
                key=lambda x: float(x.get("_critic_score", 0.0)), reverse=True
            )
            combined = selected + new_valid
            selected = deduplicate_questions_with_index(
                combined,
                llm_ver=llm_ver,
                similarity_threshold=0.85,
                max_items=num_questions,
            )
            for item in new_valid:
                pool.append(item)

        print(f"  After incremental dedup: {len(selected)}/{num_questions} questions")
        generation_round += 1

    if len(selected) < num_questions:
        print(
            f"[WARNING] Only produced {len(selected)} questions (target={num_questions})"
        )

    # Build quiz
    quiz = MultipleChoiceQuiz(
        title=f"{llm_ver.replace(':', '_')}_{len(selected)}questions_celegans_corpus",
        source=f"Corpus-based (RAG) from processed papers by {llm_ver}, temperature={temperature}, free-text->parse(tolerant)->critic->pool->dedup(0.85)",
    )

    indexing_local = ["1", "2", "3", "4"]
    for item in selected:
        q_obj = Question(question=item["question"].strip())

        # Attach source metadata if available
        src = item.get("_source")
        if src:
            try:
                if hasattr(q_obj, "metadata"):
                    q_obj.metadata = {"source": src}
                else:
                    setattr(q_obj, "metadata", {"source": src})
            except Exception:
                pass

        for i, opt in enumerate(item["options"]):
            is_correct = opt["label"] == item["correct_label"]
            q_obj.answers.append(
                Answer(indexing_local[i], opt["text"].strip(), is_correct)
            )
        quiz.questions.append(q_obj)

    print("===============================\n  Generated quiz:\n")
    print(quiz.to_yaml())

    quiz.to_json_file(
        "openworm_ai/quiz/samples/%s_%iquestions_celegans_corpus.json"
        % (llm_ver.replace(":", "_").replace("/", "_"), len(selected))
    )


# -----------------------------
# CLI runner
# -----------------------------

if __name__ == "__main__":
    import sys

    llm_ver = get_llm_from_argv(sys.argv)
    print(f"Selected LLM: {llm_ver}")

    if "-ask" in sys.argv:
        # ASK MODE
        quiz_json = None
        for i, arg in enumerate(sys.argv):
            if arg == "--quiz-file" and i + 1 < len(sys.argv):
                quiz_json = sys.argv[i + 1]
                break

        if not quiz_json:
            safe_model_name = llm_ver.replace(":", "_").replace("/", "_")

            num = 5
            for a in sys.argv:
                if a.isnumeric():
                    num = int(a)

            quiz_json = f"openworm_ai/quiz/samples/{safe_model_name}_{num}questions_celegans_corpus.json"

        if not os.path.exists(quiz_json):
            print(f"! Quiz file not found: {quiz_json}")
            print("Available quiz files:")
            for f in glob.glob("openworm_ai/quiz/samples/*.json"):
                print(f"  {f}")
            sys.exit(1)

        print(f"Using quiz file: {quiz_json}")
        quiz = MultipleChoiceQuiz.from_file(quiz_json)

        print(
            f"Asking LLM {llm_ver} {len(quiz.questions)} questions from file: {quiz_json}"
        )

        total_qs = 0
        total_correct = 0
        wrong_answers = "Incorrect answers:\n"

        for qi, question in enumerate(quiz.questions):
            q = question["question"]

            from openworm_ai.quiz.Templates import ASK_Q

            answers = ""
            random.shuffle(question["answers"])

            labels = (
                random.choice(LABEL_SETS)
                if RANDOMIZE_ASK_LABELS
                else ["A", "B", "C", "D"]
            )

            presented_answers = {}
            correct_answer = None
            correct_text = None

            for index, answer in enumerate(question["answers"]):
                ref = labels[index]
                present = f"{ref}: {answer['ans']}"
                if answer["correct"]:
                    correct_answer = ref
                    correct_text = present
                presented_answers[ref] = present
                answers += f"{present}\n"

            full_question = ASK_Q.replace("<QUESTION>", q).replace("<ANSWERS>", answers)

            orig_resp = ask_question_get_response(
                full_question, llm_ver, print_question=False
            ).strip()
            resp = orig_resp

            if "<think>" in resp:
                try:
                    before = resp[: resp.index("<think>")]
                    after = resp[resp.index("</think>") + len("</think>") :]
                    resp = (before + "\n" + after).strip()
                except ValueError:
                    resp = orig_resp

            first_line = resp.splitlines()[0].strip() if resp else ""

            guess = None
            for ch in first_line:
                if ch in labels:
                    guess = ch
                    break

            if guess is None:
                candidate = first_line.split(":")[0].strip()
                guess = candidate[0] if candidate else "Z"

            total_qs += 1
            correct_guess = guess == correct_answer

            if guess in presented_answers:
                g = presented_answers[guess]
            else:
                g = "[%s] [[%s]] (this cannot be interpreted!)" % (guess, orig_resp)

            print(
                f" >> {qi}) Is their guess of ({g}) for ({q}) correct (right answer: {correct_text})? {correct_guess}"
            )

            if correct_guess:
                total_correct += 1
            else:
                wrong_answers += (
                    f"  {q};\tWrong answer: {g};\tCorrect: {correct_text}\n"
                )

        print(wrong_answers)
        print(
            f"\n  The LLM {llm_ver} got {total_correct} out of {total_qs} questions correct "
            f"({'%.2f %%' % (100 * total_correct / total_qs)})!\n"
        )

    else:
        num = 100
        for a in sys.argv:
            if a.isnumeric():
                num = int(a)

        print(f"Using LLM {llm_ver} for saving corpus-based quiz with {num} questions")
        save_quiz_corpus(
            num_questions=num,
            llm_ver=llm_ver,
            temperature=0.7,
            papers_glob="processed/json/papers/*.json",
        )
