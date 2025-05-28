import json
import time
import random
import datetime

from openworm_ai.utils.llms import (
    LLM_OLLAMA_LLAMA32_1B,
    LLM_OLLAMA_LLAMA32_3B,
    # LLM_GPT4o,
    # LLM_GEMINI,
    # LLM_CLAUDE37,
    # LLM_GPT35,
    # LLM_OLLAMA_PHI4,
    # LLM_OLLAMA_GEMMA2,
    # LLM_OLLAMA_GEMMA,
    # LLM_OLLAMA_QWEN,
    # LLM_OLLAMA_TINYLLAMA,
    ask_question_get_response,
)


from openworm_ai.quiz.Templates import (
    ASK_Q,
)  # Ensure this matches the correct import path

field = "celegans"  # general/science/celegans
iteration_per_day = 3
current_date = datetime.datetime.now().strftime("%d-%m-%y")
SOURCE_QUESTIONS_FILE = "openworm_ai/quiz/samples/GPT4o_100questions_celegans.json"
OUTPUT_FILENAME = f"llm_scores_{field}_{current_date}_{iteration_per_day}.json"
SAVE_DIRECTORY = f"openworm_ai/quiz/scores/{field}"
TITLE = "Performance of LLMs in specific C. elegans knowledge Quiz"


indexing = ["A", "B", "C", "D"]  # Answer labels


def load_llms():
    """Loads only the selected LLMs: Ollama Llama3 and GPT-3.5."""
    llms = [
        LLM_OLLAMA_LLAMA32_1B,
        LLM_OLLAMA_LLAMA32_3B,
        # LLM_GPT4o,
        #####LLM_GEMINI,
        ####LLM_CLAUDE37,
        ###LLM_GPT35,
        ##LLM_OLLAMA_PHI4,
        # LLM_OLLAMA_GEMMA2,
        # LLM_OLLAMA_DEEPSEEK - unable to answer A-D(too few params?),
        # LLM_OLLAMA_GEMMA,
        # LLM_OLLAMA_QWEN,
        # LLM_OLLAMA_TINYLLAMA,
        # LLM_OLLAMA_FALCON2 - 'only an assistant with no acess to external resources',
        # LLM_OLLAMA_CODELLAMA - understands only a fraction of questions, doesnt understand prompts
    ]  # Defined constants
    return llms


def load_questions_from_json(filename):
    """Loads a structured quiz JSON file and extracts questions and answers."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "questions" not in data or not isinstance(data["questions"], list):
            raise ValueError(
                "Invalid JSON format: Missing or malformed 'questions' list."
            )

        questions = []
        for q in data["questions"]:
            if "question" in q and isinstance(q["question"], str) and "answers" in q:
                formatted_answers = [
                    {"ans": ans["ans"], "correct": ans["correct"]}
                    for ans in q["answers"]
                    if "ans" in ans and "correct" in ans
                ]
                if formatted_answers:
                    questions.append(
                        {"question": q["question"], "answers": formatted_answers}
                    )
        if len(questions) == 0:
            raise ValueError("Error: No valid questions found in the JSON file.")

        return questions

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return []
    except json.JSONDecodeError:
        print(
            f"Error: Failed to decode JSON. Check that '{filename}' is properly formatted."
        )
        return []
    except ValueError as e:
        print(f"Error: {e}")
        return []


def evaluate_llm(llm, questions, temperature=0):
    """Iterates over all questions, asks the LLM, and evaluates the answers."""
    results = {
        "LLM": llm,
        "Total Questions": len(questions),
        "Correct Answers": 0,
        "Response Times": [],
    }

    for question_data in questions:
        question_text = question_data["question"]
        answers = question_data["answers"]

        # Shuffle answers for randomness
        random.shuffle(answers)

        # Assign answer labels (A, B, C, D)
        presented_answers = {}
        correct_answer = None
        # correct_text = None

        for index, answer in enumerate(answers):
            ref = indexing[index]
            formatted_answer = f"{ref}: {answer['ans']}"
            presented_answers[ref] = formatted_answer
            if answer["correct"]:
                correct_answer = ref
                # correct_text = formatted_answer # unused?

        # Format the question
        full_question = ASK_Q.replace("<QUESTION>", question_text).replace(
            "<ANSWERS>", "\n".join(presented_answers.values())
        )

        # Ask the LLM
        start_time = time.time()
        response = ask_question_get_response(
            full_question, llm, temperature, print_question=False
        ).strip()
        response_time = time.time() - start_time

        # Process the LLM's response
        guess = response.split(":")[0].strip()
        if " " in guess:
            guess = guess[0]  # Ensure we get only the letter

        correct_guess = guess == correct_answer
        if correct_guess:
            results["Correct Answers"] += 1

        results["Response Times"].append(response_time)

        print(
            f" >> LLM ({llm}) - Question: {question_text} | Guess: {guess} | Correct: {correct_answer} | Correct? {correct_guess}"
        )

    # Compute final stats
    results["Accuracy (%)"] = round(
        100 * results["Correct Answers"] / results["Total Questions"], 2
    )
    results["Avg Response Time (s)"] = round(
        sum(results["Response Times"]) / results["Total Questions"], 3
    )
    del results["Response Times"]  # Remove detailed response times before saving

    return results


def iterate_over_llms(questions, temperature=0):
    """Iterates over all selected LLMs and collects results."""
    llms = load_llms()
    evaluation_results = []

    for llm in llms:
        llm_results = evaluate_llm(llm, questions, temperature)
        evaluation_results.append(llm_results)

    return evaluation_results


def save_results_to_json(
    results, filename=OUTPUT_FILENAME, save_path=SAVE_DIRECTORY, title=TITLE
):
    """Saves the collected scores as a structured JSON file without using os.

    Args:
        results (list): The results data to save.
        filename (str): The name of the JSON file (default: "llm_scores_celegans.json").
        save_path (str, optional): The directory to save the file in. If None, saves in the default folder.
        title (str, optional): The title to be included in the JSON file.
    """

    if save_path:
        file_path = f"{save_path}/{filename}"  # Manually construct path
    else:
        file_path = f"openworm_ai/quiz/scores/{filename}"  # Default path

    # Get the current date and time
    current_datetime = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    # Wrap results with a title and date in a dictionary
    output_data = {
        "Title": title,
        "Date of Testing": current_datetime,  # Add the date of testing
        "Results": results,
    }

    try:
        with open(file_path, "w") as f:
            json.dump(output_data, f, indent=4)
        print(f"Results saved to: {file_path}")
    except FileNotFoundError:
        print(
            f"Error: Directory '{save_path or 'openworm_ai/quiz/scores'}' does not exist."
        )
    except Exception as e:
        print(f"Error saving JSON file: {e}")


def main():
    """Main execution function."""

    questions = load_questions_from_json(SOURCE_QUESTIONS_FILE)

    if not questions:
        print("No valid questions to process. Exiting...")
        return

    results = iterate_over_llms(questions)
    save_results_to_json(results, OUTPUT_FILENAME, SAVE_DIRECTORY, TITLE)
    print(f"Results saved to llm_scores_celegans.json: {results}")


if __name__ == "__main__":
    main()
