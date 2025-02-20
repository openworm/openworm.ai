import json
import time
import random
import datetime
from openworm_ai.utils.llms import LLM_OLLAMA_LLAMA32, LLM_GPT4o, LLM_GEMINI, LLM_CLAUDE35,LLM_GPT35,LLM_OLLAMA_PHI4,LLM_OLLAMA_GEMMA2,LLM_OLLAMA_DEEPSEEK,LLM_OLLAMA_GEMMA, LLM_OLLAMA_QWEN, ask_question_get_response
from openworm_ai.quiz.QuizModel import MultipleChoiceQuiz  # Ensure this matches the correct import path
from openworm_ai.quiz.Templates import ASK_Q  # Ensure this matches the correct import path

indexing = ["A", "B", "C", "D"]  # Answer labels

def load_llms():
    """Loads only the selected LLMs: Ollama Llama3 and GPT-3.5."""
    llms = [LLM_OLLAMA_LLAMA32, 
            LLM_GPT4o, 
            LLM_GEMINI, 
            LLM_CLAUDE35,
            LLM_GPT35,
            LLM_OLLAMA_PHI4,
            LLM_OLLAMA_GEMMA2,
            #LLM_OLLAMA_DEEPSEEK,
            LLM_OLLAMA_GEMMA,
            LLM_OLLAMA_QWEN
            ]  # Defined constants
    print(f"Debug: Loaded LLMs -> {llms}")
    return llms

def load_questions_from_json(filename):
    """Loads a structured quiz JSON file and extracts questions and answers."""
    try:
        with open(filename, "r") as f:
            data = json.load(f)

        if "questions" not in data or not isinstance(data["questions"], list):
            raise ValueError("Invalid JSON format: Missing or malformed 'questions' list.")

        questions = []
        for q in data["questions"]:
            if "question" in q and isinstance(q["question"], str) and "answers" in q:
                formatted_answers = [
                    {"ans": ans["ans"], "correct": ans["correct"]}
                    for ans in q["answers"]
                    if "ans" in ans and "correct" in ans
                ]
                if formatted_answers:
                    questions.append({"question": q["question"], "answers": formatted_answers})

        if len(questions) == 0:
            raise ValueError("Error: No valid questions found in the JSON file.")

        return questions

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON. Check that '{filename}' is properly formatted.")
        return []
    except ValueError as e:
        print(f"Error: {e}")
        return []

def evaluate_llm(llm, questions, temperature=0):
    """Iterates over all questions, asks the LLM, and evaluates the answers."""
    results = {"LLM": llm, "Total Questions": len(questions), "Correct Answers": 0, "Response Times": []}

    for question_data in questions:
        question_text = question_data["question"]
        answers = question_data["answers"]

        # Shuffle answers for randomness
        random.shuffle(answers)

        # Assign answer labels (A, B, C, D)
        presented_answers = {}
        correct_answer = None
        correct_text = None

        for index, answer in enumerate(answers):
            ref = indexing[index]
            formatted_answer = f"{ref}: {answer['ans']}"
            presented_answers[ref] = formatted_answer
            if answer["correct"]:
                correct_answer = ref
                correct_text = formatted_answer

        # Format the question
        full_question = ASK_Q.replace("<QUESTION>", question_text).replace(
            "<ANSWERS>", "\n".join(presented_answers.values())
        )

        # Ask the LLM
        start_time = time.time()
        response = ask_question_get_response(full_question, llm, temperature, print_question=False).strip()
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
    results["Accuracy (%)"] = round(100 * results["Correct Answers"] / results["Total Questions"], 2)
    results["Avg Response Time (s)"] = round(sum(results["Response Times"]) / results["Total Questions"], 3)
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

def save_results_to_json(results, filename="llm_scores_celegans.json", save_path=None, title="Performance of LLMs in Science Quiz"):
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
        "Results": results
    }

    try:
        with open(file_path, "w") as f:
            json.dump(output_data, f, indent=4)
        print(f"Results saved to: {file_path}")
    except FileNotFoundError:
        print(f"Error: Directory '{save_path or 'openworm_ai/quiz/scores'}' does not exist.")
    except Exception as e:
        print(f"Error saving JSON file: {e}")


def main():
    """Main execution function."""
    questions_file = "openworm_ai/quiz/samples/GPT4o_100questions_Celegans.json"
    questions = load_questions_from_json(questions_file)  

    if not questions:
        print("No valid questions to process. Exiting...")
        return

    results = iterate_over_llms(questions)
    save_results_to_json(results)
    print(f"Results saved to llm_scores_celegans.json: {results}")

if __name__ == "__main__":
    main()
