from openworm_ai.utils.llms import get_llm_from_argv
from openworm_ai.utils.llms import generate_response

# Modify the prompt to focus on C. elegans
GENERATE_Q_C = """
Generate a list of 50 multiple-choice questions specifically about general knowledge of *Caenorhabditis elegans* (C. elegans). 
The questions should cover topics such as genetics, neurobiology, behavior, development, physiology, and research significance. 
Each question should have 4 possible answers (A, B, C, D), with only one correct answer.
Format the questions as follows:

"""
# alter it so it the num of questions can be changed

TEXT_ANSWER_EXAMPLE_C = """
QUESTION: What is the primary food source for C. elegans in lab conditions?
CORRECT ANSWER: E. coli
WRONG ANSWER: Algae
WRONG ANSWER: Fungi
WRONG ANSWER: Bacteria mix

"""

ASK_Q_C = """You are to select the correct answer for a multiple-choice question. 
A number of answers will be presented, and you should respond with only the letter corresponding to the correct answer.
For example, if the question is: 

What is the primary food source for C. elegans?

and the potential answers are:

A: Algae
B: E. coli
C: Fungi
D: Bacteria mix

You should only answer: 

B

This is your question:

<QUESTION>

These are the potential answers:

<ANSWERS>

"""

if __name__ == "__main__":
    import sys

    # Ensure the question request is for C. elegans and 50 questions
    question = GENERATE_Q_C + TEXT_ANSWER_EXAMPLE_C

    llm_ver = get_llm_from_argv(sys.argv)

    print("--------------------------------------------------------")
    print("Asking question:\n   %s" % question)
    print("--------------------------------------------------------")

    print(" ... Connecting to: %s" % llm_ver)

    # Set `only_celegans=True` to make sure it focuses on C. elegans topics
    response = generate_response(question, llm_ver, temperature=0, only_celegans=True)

    print("--------------------------------------------------------")
    print("Answer:\n   %s" % response)
    print("--------------------------------------------------------")
    print()
