from openworm_ai.utils.llms import get_llm_from_argv
from openworm_ai.utils.llms import generate_response


GENERATE_Q = """
Generate <QUESTION_NUMBER> multiple choice question to test someone's general scientific knowledge.

The question should be answerable by a reasonably intelligent adult and should cover diverse topics such as physics, chemistry, biology, astronomy, earth science, mathematics, or scientific methods.

IMPORTANT:
- Avoid repeating common trivia patterns.
- Avoid basic textbook fact recall such as "speed of light", atomic numbers, or simple formula definitions unless used in a more applied or conceptual way.
- Prefer conceptual understanding, real-world applications, or scientific reasoning.
- Ensure the question is clear, unambiguous, and factually stable.

There should be <ANSWER_NUMBER> possible answers, only one of which is unambiguously correct. All answers should be brief and plausible.

Each of the <QUESTION_NUMBER> question/answer sets should be presented in the following format (focus on the format, not the specific content):
"""

TEXT_ANSWER_EXAMPLE = """
QUESTION: <Insert question>  
CORRECT ANSWER: <Correct answer>  
WRONG ANSWER: <Wrong answer 1>  
WRONG ANSWER: <Wrong answer 2>  
WRONG ANSWER: <Wrong answer 3>

"""

ASK_Q = """You are to select the correct answer for a multiple choice question. 
A number of answers will be presented and you should respond with only the letter corresponding to the correct answer.
For example if the question is: 

What is the primary food source for C. elegans in lab conditions?

and the potential answers are:

E: Algae
F: E. coli
G: Fungi
H: Bacteria mix

you should only answer: 

F

This is your question:

<QUESTION>

These are the potential answers:

<ANSWERS>

"""

if __name__ == "__main__":
    import sys

    question = (
        GENERATE_Q.replace("<QUESTION_NUMBER>", "5").replace("<ANSWER_NUMBER>", "4")
        + TEXT_ANSWER_EXAMPLE
    )

    llm_ver = get_llm_from_argv(sys.argv)

    print("--------------------------------------------------------")
    print("Asking question:\n   %s" % question)
    print("--------------------------------------------------------")

    print(" ... Connecting to: %s" % llm_ver)

    response = generate_response(question, llm_ver, temperature=0, only_celegans=False)

    print("--------------------------------------------------------")
    print("Answer:\n   %s" % response)
    print("--------------------------------------------------------")
    print()
