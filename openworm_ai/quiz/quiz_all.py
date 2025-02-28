import json
import time
import random  # Used for dummy scoring function
from openworm_ai.utils.llms import PREF_ORDER_LLMS, ask_question_get_response


def load_llms():
    """Loads the list of available LLMs from llms.py."""
    return list(PREF_ORDER_LLMS)


def score_response(response):
    """Evaluates an LLM response and returns a score.

    Replace this function with a real evaluation method.
    """
    # Dummy scoring: Assigns a random score between 0 and 100
    return random.randint(0, 100)


def ask_llm(llm, question, temperature=0):
    """Calls the ask function from llms.py for a given LLM and question,
    then evaluates and stores the score instead of the raw response.
    """
    start_time = time.time()
    response = ask_question_get_response(
        question, llm, temperature, print_question=False
    )
    response_time = time.time() - start_time

    # Score the response
    score = score_response(response)

    return {"llm": llm, "score": score, "time": response_time}


def iterate_over_llms(questions, temperature=0):
    """Iterates over all LLMs, collecting scores for each question."""
    llms = load_llms()
    results = {}
    for question in questions:
        results[question] = []
        for llm in llms:
            result = ask_llm(llm, question, temperature)
            results[question].append(result)
    return results


def save_results_to_json(results, filename="llm_scores.json"):
    """Saves the collected scores as a JSON file."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


def load_results_from_json(filename="llm_scores.json"):
    """Reads the JSON file to prepare for figure generation."""
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def load_questions_from_json(filename):
    """Loads and extracts questions from a structured quiz JSON file."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Ensure the JSON structure contains a "questions" key
        if "questions" not in data or not isinstance(data["questions"], list):
            raise ValueError(
                "Invalid JSON format: Missing or malformed 'questions' list."
            )

        # Extract the 'question' field from each question object
        questions = [
            q["question"]
            for q in data["questions"]
            if "question" in q and isinstance(q["question"], str)
        ]

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


def main():
    """Main execution function."""
    questions_file = "openworm_ai/quiz/samples/quiz_questions.json"
    questions = load_questions_from_json(questions_file)

    if not questions:  # Stop execution if no valid questions are loaded
        print("No valid questions to process. Exiting...")
        return

    results = iterate_over_llms(questions)
    save_results_to_json(results)
    print("Scores saved to llm_scores.json")


if __name__ == "__main__":
    main()
