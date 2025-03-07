from openworm_ai.quiz.QuizModel import MultipleChoiceQuiz, Question, Answer
#from openworm_ai.quiz.TemplatesCelegans import GENERATE_Q, TEXT_ANSWER_EXAMPLE
from openworm_ai.quiz.Templates import GENERATE_Q, TEXT_ANSWER_EXAMPLE

from openworm_ai.utils.llms import ask_question_get_response
from openworm_ai.utils.llms import get_llm_from_argv

import random

indexing = ["A", "B", "C", "D"]


def save_quiz(num_questions, num_answers, llm_ver, temperature=0):
    question = (
        GENERATE_Q.replace("<QUESTION_NUMBER>", str(num_questions)).replace(
            "<ANSWER_NUMBER>", str(num_answers)
        )
        + TEXT_ANSWER_EXAMPLE
    )

    response = ask_question_get_response(question, llm_ver, temperature)

    quiz = MultipleChoiceQuiz(
        title="GPT4o_%iquestions_celegans" % num_questions,
        source="Generated by %s, temperature: %s" % (llm_ver, temperature),
    )

    last_question = None

    indexing = ["1", "2", "3", "4"]
    for line in response.split("\n"):
        if len(line.strip()) > 0:
            if "QUESTION" in line or line[-1] == "?":
                question = line.split(":")[-1].strip()
                print("Question: <%s>" % question)
                last_question = Question(question=question)
                quiz.questions.append(last_question)
            elif "CORRECT ANSWER" in line:
                ans = line.split(":")[-1].strip()
                print("CORRECT ANSWER: <%s>" % ans)
                i = len(last_question.answers)
                last_question.answers.append(Answer(indexing[i], ans, True))
            elif "WRONG ANSWER" in line:
                ans = line.split(":")[-1].strip()
                print("WRONG ANSWER: <%s>" % ans)
                i = len(last_question.answers)
                last_question.answers.append(Answer(indexing[i], ans, False))

    print("===============================\n  Generated quiz:\n")
    print(quiz.to_yaml())

    quiz.to_json_file(
        "openworm_ai/quiz/samples/%s_%iquestions_celegans.json"
        % (llm_ver.replace(":", "_"), num_questions)
    )


if __name__ == "__main__":
    import sys

    llm_ver = get_llm_from_argv(sys.argv)

    print(f"Selected LLM: {llm_ver}")

    if "-ask" in sys.argv:
        # quiz_json = "openworm_ai/quiz/samples/GPT4o_50questions.json"
        # quiz_json = "openworm_ai/quiz/samples/GPT4o_10questions.json"

        quiz_json = "openworm_ai/quiz/samples/GPT4o_100questions.json"


        quiz = MultipleChoiceQuiz.from_file(quiz_json)

        print(f"Asking LLM {llm_ver} {len(quiz.questions)} questions from file: {quiz_json}")

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

            from openworm_ai.utils.llms import ask_question_get_response

            resp = ask_question_get_response(
                full_question, llm_ver, print_question=False
            ).strip()

            if "<think>" in resp:  # Give deepseek a fighting chance...
                resp = (
                    resp[0 : resp.index("<think>")] + resp[resp.index("</think>") + 8 :]
                )
                resp = resp.replace("\n", " ").strip()
                guess = resp[-1]
            else:
                guess = resp.split(":")[0].strip()
                if " " in guess:
                    guess = guess[0]

            total_qs += 1
            correct_guess = guess == correct_answer

            if guess in presented_answers:
                g = presented_answers[guess]
            else:
                g = "%s (cannot be interpreted!)" % guess
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
            f"\n  The LLM {llm_ver} got {total_correct} out of {total_qs} questions correct ({'%.2f %%' % (100 * total_correct / total_qs)})!\n"
        )
    # make this into a method which returns a dictionary of all the "stats" that lists the llm, correct/incorrect answers
    # this can be used to plot comparison of variety of llms on general knowledge
    else:
        num = 100
        for a in sys.argv:
            if a.isnumeric():
                num = int(a)
        print(f"Using LLM {llm_ver} for saving quiz with {num} questions")
        save_quiz(num, 4, llm_ver, temperature=0.2)

