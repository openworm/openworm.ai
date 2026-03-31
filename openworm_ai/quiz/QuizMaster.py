from __future__ import annotations

import os
import json
import random
from enum import Enum
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
# CLI enums / constants
# -----------------------------

QuizScope = Enum(
    "QuizScope", [("GeneralKnowledge", 1), ("Science", 2), ("CElegans", 3)]
)

# Used ONLY for the -ask mode UI.
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
# Free-text -> structured MCQ items (tolerant, old-style)
# -----------------------------


def parse_free_text_mcqs_to_items(text: str) -> List[dict]:
    """
    Tolerant streaming parser for small LLMs (llama3.2, etc).

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
# Embedding dedup (RAG-style)
# -----------------------------


def question_to_text(item: dict) -> str:
    # NOTE: keep this simple; the stem carries most “dup” signal
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
# Main generation entrypoint
# -----------------------------


def save_quiz(num_questions, llm_ver, quiz_scope, temperature):
    """
    Batch generation (BATCH_SIZE q/call) with context-aware dedup.

    Key design:
    - BATCH_SIZE=10 questions per LLM call: the model sees the full batch so natural
      within-batch diversity; far fewer total API calls.
    - accepted_descriptions fed back into every prompt so the LLM avoids repeating
      questions that already passed the critic (fixes the "stateless call" problem).
    - Top-up rounds do incremental dedup: `selected` is treated as fixed and new
      candidates are checked *against* it rather than re-deduping from scratch
      (prevents previously accepted questions being evicted on re-sort).
    """
    if quiz_scope == QuizScope.GeneralKnowledge:
        from openworm_ai.quiz.Templates import GENERATE_Q, TEXT_ANSWER_EXAMPLE

        suffix = "_general_v2"
        use_topic_exclusion = True
        topic_length = "8-12 word HIGHLY SPECIFIC"
    elif quiz_scope == QuizScope.Science:
        from openworm_ai.quiz.TemplatesScience import GENERATE_Q, TEXT_ANSWER_EXAMPLE

        suffix = "_science_v2"
        use_topic_exclusion = True
        topic_length = "8-12 word HIGHLY SPECIFIC"
    elif quiz_scope == QuizScope.CElegans:
        from openworm_ai.quiz.TemplatesCelegans import GENERATE_Q, TEXT_ANSWER_EXAMPLE

        suffix = "_celegans_v2"
        use_topic_exclusion = True
        topic_length = "8-12 word HIGHLY SPECIFIC"
    else:
        raise ValueError(f"Unsupported quiz scope: {quiz_scope}")

    BATCH_SIZE = 10
    # Aim for 2x target in the pool before semantic dedup (leaves room for critic rejects)
    target_pool = num_questions * 2
    max_calls = max(10, (target_pool // BATCH_SIZE) * 3)

    pool: List[dict] = []  # critic-passed, exact-deduped questions
    pool_texts: set = set()  # exact-text keys for O(1) dedup
    used_topics: List[str] = []
    # Short question texts fed back to the LLM to prevent it re-generating similar questions
    accepted_descriptions: List[str] = []
    calls = 0
    critic_llm_ver = get_default_critic_llm_ver()

    def extract_topics_from_raw(raw: str) -> None:
        """Extract every 'Topic: ...' line from a batch response."""
        for line in raw.splitlines():
            stripped = line.strip()
            if stripped.startswith("Topic:"):
                topic = stripped.split("Topic:", 1)[1].strip()
                if topic:
                    used_topics.append(topic)

    def make_prompt(batch_num: int) -> str:
        exclusion_context = ""
        if use_topic_exclusion and used_topics:
            exclusion_context += (
                f"\n\nTopics already used (choose something DIFFERENT): "
                f"{', '.join(used_topics[-40:])}\n"
            )
        if accepted_descriptions:
            recent = accepted_descriptions[-30:]
            exclusion_context += (
                "\n\nQuestions already accepted - DO NOT repeat or closely resemble any of these:\n"
                + "\n".join(f"- {d}" for d in recent)
                + "\n"
            )

        if use_topic_exclusion:
            topic_instruction = (
                f"\n\nGenerate batch #{batch_num} of {BATCH_SIZE} unique questions. "
                f"For each question, first write a {topic_length} topic tag "
                f"(e.g., 'Topic: Wright Brothers First Powered Flight at Kitty Hawk 1903'), "
                f"then the full question in the format below.\n\n"
            )
        else:
            topic_instruction = (
                f"\n\nGenerate batch #{batch_num} of {BATCH_SIZE} unique questions.\n\n"
            )

        return (
            GENERATE_Q.replace("<QUESTION_NUMBER>", str(BATCH_SIZE)).replace(
                "<ANSWER_NUMBER>", "4"
            )
            + exclusion_context
            + topic_instruction
            + TEXT_ANSWER_EXAMPLE
        )

    def score_and_add_to_pool(candidates: List[dict]) -> int:
        """
        Score each candidate with the critic and add passing, non-duplicate items to pool.
        Returns the number added.
        """
        added = 0
        for item in candidates:
            if not _is_valid_mcq_item(item):
                continue
            key = item["question"].lower().strip()
            if key in pool_texts:
                continue
            s, reject = score_question_with_critic(
                item, llm_ver_critic=critic_llm_ver, temperature=0.2
            )
            item["_critic_score"] = s
            item["_critic_reject"] = reject
            if not reject:
                pool.append(item)
                pool_texts.add(key)
                # Feed a short version back to the LLM in future prompts
                accepted_descriptions.append(item["question"].strip()[:100])
                added += 1
        return added

    # ---------- initial generation ----------
    print(f"Using critic model {critic_llm_ver}")
    while calls < max_calls:
        calls += 1
        raw = ask_question_get_response(make_prompt(calls), llm_ver, temperature)
        extract_topics_from_raw(raw)
        parsed = parse_free_text_mcqs_to_items(raw)
        added = score_and_add_to_pool(parsed)
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

    # ---------- top-up rounds (incremental dedup against fixed `selected`) ----------
    generation_round = 1
    max_rounds = 5

    while len(selected) < num_questions and generation_round <= max_rounds:
        shortage = num_questions - len(selected)
        # Generate enough batches to get ~2x the shortage after critic rejection
        batches_needed = max(2, (shortage * 2 + BATCH_SIZE - 1) // BATCH_SIZE)
        print(
            f"\n[ROUND {generation_round + 1}] Need {shortage} more. "
            f"Generating {batches_needed} batches (~{batches_needed * BATCH_SIZE} questions)..."
        )

        new_valid: List[dict] = []
        for _ in range(batches_needed):
            calls += 1
            raw = ask_question_get_response(make_prompt(calls), llm_ver, temperature)
            extract_topics_from_raw(raw)
            parsed = parse_free_text_mcqs_to_items(raw)
            for item in parsed:
                if not _is_valid_mcq_item(item):
                    continue
                key = item["question"].lower().strip()
                if key in pool_texts:
                    continue
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
            # items (indices 0..len(selected)-1) are processed first and always kept,
            # then new items are appended only if they don't dup anything already kept.
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
            # Persist new_valid into pool for any further rounds
            for item in new_valid:
                pool.append(item)

        print(f"  After incremental dedup: {len(selected)}/{num_questions} questions")
        generation_round += 1

    if len(selected) < num_questions:
        print(
            f"[WARNING] Only produced {len(selected)} questions (target={num_questions})"
        )

    # ---------- Build quiz ----------
    quiz = MultipleChoiceQuiz(
        title=f"{llm_ver.replace(':', '_')}_{len(selected)}questions{suffix}",
        source=f"Generated by {llm_ver}, temperature={temperature}, free-text->parse(tolerant)->critic->pool->dedup(0.75)",
    )

    indexing_local = ["1", "2", "3", "4"]
    for item in selected:
        q_obj = Question(question=item["question"].strip())
        for i, opt in enumerate(item["options"]):
            is_correct = opt["label"] == item["correct_label"]
            q_obj.answers.append(
                Answer(indexing_local[i], opt["text"].strip(), is_correct)
            )
        quiz.questions.append(q_obj)

    print("===============================\n  Generated quiz:\n")
    print(quiz.to_yaml())

    quiz.to_json_file(
        "openworm_ai/quiz/samples/%s_%iquestions%s.json"
        % (llm_ver.replace(":", "_").replace("/", "_"), len(selected), suffix)
    )


# -----------------------------
# CLI runner
# -----------------------------

if __name__ == "__main__":
    import sys

    llm_ver = get_llm_from_argv(sys.argv)
    print(f"Selected LLM: {llm_ver}")

    if "-ask" in sys.argv:
        # ASK MODE - unchanged
        quiz_json = None
        for i, arg in enumerate(sys.argv):
            if arg == "--quiz-file" and i + 1 < len(sys.argv):
                quiz_json = sys.argv[i + 1]
                break

        if not quiz_json:
            safe_model_name = llm_ver.replace(":", "_").replace("/", "_")

            suffix = "_general_v2"
            if "--celegans" in sys.argv:
                suffix = "_celegans_v2"
            elif "--science" in sys.argv:
                suffix = "_science_v2"

            num = 5
            for a in sys.argv:
                if a.isnumeric():
                    num = int(a)

            quiz_json = f"openworm_ai/quiz/samples/{safe_model_name}_{num}questions{suffix}.json"

        if not os.path.exists(quiz_json):
            print(f"! Quiz file not found: {quiz_json}")
            print("Available quiz files:")
            import glob

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
                f" >> {qi}) Is their guess of ({g}) for ({q}) correct "
                f"(right answer: {correct_text})? {correct_guess}"
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

        quiz_scope = QuizScope.GeneralKnowledge
        if "--celegans" in sys.argv:
            quiz_scope = QuizScope.CElegans
        elif "--science" in sys.argv:
            quiz_scope = QuizScope.Science

        print(
            f"Using LLM {llm_ver} for saving quiz with {num} questions (scope={quiz_scope.name})"
        )
        save_quiz(num, llm_ver, quiz_scope=quiz_scope, temperature=0.7)
