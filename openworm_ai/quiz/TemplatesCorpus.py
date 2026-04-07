import os
import ollama


def load_documents(directory="."):
    """Load text content from all documents in the directory."""
    documents = []
    for file in os.listdir(directory):
        if file.endswith((".txt", ".md", ".pdf")):  # Extend as needed
            with open(os.path.join(directory, file), "r", encoding="utf-8") as f:
                documents.append(f.read())
    return "\n\n".join(documents)


# === QUESTION GENERATION TEMPLATE === #
GENERATE_Q = """
Generate <QUESTION_NUMBER> multiple choice question **strictly based on the provided text/corpus**.  

**Rules:**  
- Questions MUST be answerable from the information provided in the text
- Focus on specific details, facts, or data mentioned in the corpus
- Avoid general knowledge questions - prioritize document-specific information
- Questions should test deep understanding of the material, suitable for researchers or advanced students
- If the text covers multiple topics, ensure variety across different sections

**Format Requirements:**
- Each question should have <ANSWER_NUMBER> possible answers
- Only one answer should be unambiguously correct
- All answers should be kept brief
- Questions should be presented in the following format (focus on the format not the actual question or answers specifically):

"""

TEXT_ANSWER_EXAMPLE = """
QUESTION: <Insert question>  
CORRECT ANSWER: <Correct answer>  
WRONG ANSWER: <Wrong answer 1>  
WRONG ANSWER: <Wrong answer 2>  
WRONG ANSWER: <Wrong answer 3>

"""

# === LLM RESPONSE FORMAT FOR QUIZ === #
ASK_Q = """You are to select the correct answer for a multiple choice question. 
A number of answers will be presented and you should respond with only the letter corresponding to the correct answer.
For example if the question is: 

What are the dimensions of the C. elegans pharynx?

and the potential answers are:

E: 80 µm long and 15 µm in diameter
F: 100 µm long and 20 µm in diameter
G: 150 µm long and 25 µm in diameter
H: 200 µm long and 35 µm in diameter

you should only answer: 

F

This is your question:

<QUESTION>

These are the potential answers:

<ANSWERS>
"""

if __name__ == "__main__":
    document_text = load_documents()

    # If no documents are found, rely entirely on model knowledge
    if not document_text.strip():
        print("! No valid documents found. Using model's knowledge instead.")
        document_text = "**No external documents available. Use your own knowledge.**"

    # Generate questions prompt
    question_prompt = (
        GENERATE_Q.replace("<QUESTION_NUMBER>", "100")
        + TEXT_ANSWER_EXAMPLE
        + "\n\n🔹 **Document Content (if available):**\n"
        + document_text
    )

    print("--------------------------------------------------------")
    print(f"Asking Phi-4:\n{question_prompt}")
    print("--------------------------------------------------------")

    response = ollama.chat(
        model="phi4",
        messages=[{"role": "user", "content": question_prompt}],
        temperature=0,
    )

    print("--------------------------------------------------------")
    print(f"Response:\n{response['message']['content']}")
    print("--------------------------------------------------------")
