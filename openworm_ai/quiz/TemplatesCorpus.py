import os
import ollama


def load_documents(directory="."):
    """Load text content from all documents in the directory."""
    documents = []
    for file in os.listdir(directory):
        if (
            file.endswith(".txt") or file.endswith(".md") or file.endswith(".pdf")
        ):  # Extend as needed
            with open(os.path.join(directory, file), "r", encoding="utf-8") as f:
                documents.append(f.read())
    return "\n\n".join(documents)


GENERATE_Q = """
You must generate exactly <QUESTION_NUMBER> multiple-choice questions **strictly based on the provided text**.  
Do **not** generate any questions outside of the document content.  

🔹 **Rules:**  
- The questions must be **highly specific** to details found **only** in the provided text.  
- **Do not generalize** beyond the document's explicit information.  
- If a topic is not in the text, **do not generate a question about it.**  
- The questions should challenge a researcher or advanced student.  

🔹 **Format:**  
QUESTION: <Insert question>  
CORRECT ANSWER: <Correct answer>  
WRONG ANSWER: <Wrong answer 1>  
WRONG ANSWER: <Wrong answer 2>  
WRONG ANSWER: <Wrong answer 3>  

📌 **Reminder: If the provided text does not contain enough information for <QUESTION_NUMBER> questions, stop at the highest possible valid number.**  
"""


TEXT_ANSWER_EXAMPLE = """
QUESTION: What are the dimensions of the C. elegans pharynx?
CORRECT ANSWER: 100 µm long and 20 µm in diameter
WRONG ANSWER: 80 µm long and 15 µm in diameter
WRONG ANSWER: 150 µm long and 25 µm in diameter
WRONG ANSWER: 200 µm long and 35 µm in diameter
"""

ASK_Q = """
QUESTION:What are the dimensions of the C. elegans pharynx?
CORRECT ANSWER: 100 µm long and 20 µm in diameter
WRONG ANSWER: 80 µm long and 15 µm in diameter
WRONG ANSWER: 150 µm long and 25 µm in diameter
WRONG ANSWER: 200 µm long and 35 µm in diameter

"""

ASK_Q = """You are to select the correct answer for a multiple choice question. 
A number of answers will be presented and you should respond with only the letter corresponding to the correct answer.
For example if the question is: 

What are the dimensions of the C. elegans pharynx

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

    if not document_text.strip():
        print("Error: No valid documents found.")
        exit()

    question_prompt = (
        GENERATE_Q.replace("<QUESTION_NUMBER>", "5").replace("<ANSWER_NUMBER>", "4")
        + TEXT_ANSWER_EXAMPLE
        + "\n\nDocuments:\n"
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
