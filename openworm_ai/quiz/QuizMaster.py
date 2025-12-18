from __future__ import annotations

import json
import random
from enum import Enum
from typing import List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

from openworm_ai.quiz.QuizModel import Answer, MultipleChoiceQuiz, Question
from openworm_ai.utils.llms import (
    LLM_CLAUDE37,
    LLM_OLLAMA_GEMMA2,
    ask_question_get_response,
    get_anthropic_key,
    get_llm,
    get_llm_from_argv,
)

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
    - Otherwise -> fall back to local Ollama model (gemma2)
    """
    try:
        key = get_anthropic_key()
    except Exception:
        key = None
    return LLM_CLAUDE37 if key else LLM_OLLAMA_GEMMA2


def score_question_with_critic(
    item: dict, llm_ver_critic: Optional[str] = None, temperature: float = 0.0
):
    """
    Score a single MCQ item using a critic LLM.

    Returns: (score: float, comment: None)
    """
    if llm_ver_critic is None:
        llm_ver_critic = get_default_critic_llm_ver()

    mcq_json_str = json.dumps(item, ensure_ascii=False, indent=2)

    critic_prompt = """
You are an expert evaluator of multiple-choice questions.

You will be given ONE MCQ in JSON format:
- "question": the question text
- "options": array of answers (A–D)
- "correct_label": the intended correct option.

Evaluate QUALITY on:
1) Clarity
2) Unambiguity (exactly one correct)
3) Factual correctness
4) Distractor quality
5) Appropriateness

Return ONLY valid JSON: {{"score": <integer 0-100>}}
Do not include any other keys or any extra text.

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
        return 50.0, None

    try:
        start = resp.find("{")
        end = resp.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in critic response.")
        obj = json.loads(resp[start : end + 1])
        return float(obj.get("score", 50.0)), None
    except Exception as e:
        print("! Failed to parse critic response as JSON:")
        print(resp)
        print(e)
        return 50.0, None


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

    Ignores:
    - explanations
    - prompt echoes
    - random extra lines

    Produces items in the canonical internal format:
      {
        "question": "...",
        "options": [{"label":"A","text":"..."}, ...],
        "correct_label":"A"
      }
    """

    if not isinstance(text, str) or not text.strip():
        return []

    # If the model echoed the whole prompt, try to start at the first QUESTION:
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

        # QUESTION line
        if u.startswith("QUESTION:") or line.endswith("?"):
            flush()
            cur_q = line.split(":", 1)[1].strip() if ":" in line else line
            continue

        # CORRECT line
        if (
            u.startswith("CORRECT ANSWER:")
            or u.startswith("CORRECT:")
            or (u.startswith("ANSWER:") and correct is None)
        ):
            correct = line.split(":", 1)[1].strip() if ":" in line else line
            continue

        # WRONG line
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
    stem = item.get("question", "").strip()
    opts = item.get("options", [])
    parts = [stem] + [f"{o.get('label')}. {o.get('text')}" for o in opts]
    return " ".join([p for p in parts if p]).strip()


def get_embed_model_for_llm(llm_ver: str):
    if llm_ver.startswith("Ollama:"):
        return OllamaEmbedding(model_name=llm_ver.replace("Ollama:", ""))
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
    similarity_threshold: float = 0.75,
    max_items: Optional[int] = None,
) -> List[dict]:
    if not questions:
        return []
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
# Main generation entrypoint (keeps old name)
# -----------------------------


def save_quiz(num_questions, num_answers, llm_ver, quiz_scope, temperature=0):
    """
    Same external behavior as the old version, but internally:
    free-text generation -> tolerant parse -> critic -> embed-dedup -> save quiz json.
    """
    if quiz_scope == QuizScope.GeneralKnowledge:
        from openworm_ai.quiz.Templates import GENERATE_Q, TEXT_ANSWER_EXAMPLE

        suffix = "_general_v2"
    elif quiz_scope == QuizScope.Science:
        from openworm_ai.quiz.TemplatesScience import GENERATE_Q, TEXT_ANSWER_EXAMPLE

        suffix = "_science_v2"
    elif quiz_scope == QuizScope.CElegans:
        from openworm_ai.quiz.TemplatesCelegans import GENERATE_Q, TEXT_ANSWER_EXAMPLE

        suffix = "_celegans_v2"
    else:
        raise ValueError(f"Unsupported quiz scope: {quiz_scope}")

    # Over-generate so critic+dedup have room to work
    OVERGEN = 2
    target_valid = num_questions * OVERGEN

    # Retry budget (small models fail format often)
    # This keeps the pipeline robust instead of crashing early.
    max_calls = max(20, target_valid * 3)

    items: List[dict] = []
    calls = 0

    while calls < max_calls:
        calls += 1

        prompt = (
            GENERATE_Q.replace("<QUESTION_NUMBER>", "1") + "\n\n" + TEXT_ANSWER_EXAMPLE
        )
        raw = ask_question_get_response(prompt, llm_ver, temperature)

        parsed = parse_free_text_mcqs_to_items(raw)
        if parsed:
            items.extend(parsed)

        data = [d for d in items if _is_valid_mcq_item(d)]
        if len(data) >= target_valid:
            break

        if calls % 5 == 0:
            print(
                f"[Gen] calls={calls}/{max_calls} | parsed_items={len(items)} | valid_items={len(data)} (target={target_valid})"
            )

    data = [d for d in items if _is_valid_mcq_item(d)]
    if not data:
        raise ValueError(
            f"No valid questions parsed after {calls} generation calls. "
            f"Try a different local model (e.g. -ge2 or -o-m) or loosen the GENERATE_Q prompt."
        )

    # Critic scoring
    critic_llm_ver = get_default_critic_llm_ver()
    print(f"Using critic model {critic_llm_ver} to score {len(data)} questions")

    scored = []
    for idx, item in enumerate(data):
        s, _ = score_question_with_critic(item, llm_ver_critic=critic_llm_ver)
        item["_critic_score"] = s
        if idx < 10 or (idx + 1) % 25 == 0:
            print(f"  [Critic] Q{idx}: score={s:.1f}")
        scored.append(item)

    scored.sort(key=lambda x: x.get("_critic_score", 0.0), reverse=True)

    # Dedup (embedding-based)
    try:
        selected = deduplicate_questions_with_index(
            scored, llm_ver=llm_ver, similarity_threshold=0.9, max_items=num_questions
        )
        print(
            f"Selected {len(selected)} questions after embedding dedup (target={num_questions})"
        )
    except Exception as e:
        print(f"! Embedding dedup failed, falling back to top-{num_questions}: {e}")
        selected = scored[:num_questions]

    # Build MultipleChoiceQuiz
    quiz = MultipleChoiceQuiz(
        title=f"{llm_ver.replace(':', '_')}_{num_questions}questions{suffix}",
        source=f"Generated by {llm_ver}, temperature={temperature}, free-text->parse(tolerant)->critic->dedup",
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
        % (llm_ver.replace(":", "_"), num_questions, suffix)
    )


# -----------------------------
# CLI runner (keeps old behavior)
# -----------------------------

if __name__ == "__main__":
    import sys

    llm_ver = get_llm_from_argv(sys.argv)
    print(f"Selected LLM: {llm_ver}")

    if "-ask" in sys.argv:
        quiz_json = (
            "openworm_ai/quiz/samples/Ollama_llama3.2_3questions_celegans_v2.json"
        )
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

        quiz_scope = QuizScope.CElegans
        if "--general" in sys.argv:
            quiz_scope = QuizScope.GeneralKnowledge
        elif "--science" in sys.argv:
            quiz_scope = QuizScope.Science

        print(
            f"Using LLM {llm_ver} for saving quiz with {num} questions (scope={quiz_scope.name})"
        )
        save_quiz(num, 4, llm_ver, quiz_scope=quiz_scope, temperature=0.2)
