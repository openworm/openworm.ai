from openworm_ai.utils.llms import get_llm_from_argv
from openworm_ai.utils.llms import generate_response


GENERATE_Q = """
Generate a list of <QUESTION_NUMBER> multiple choice questions to test someone's general knowledge.
The questions should be answerable by a reasonably intelligent adult, and should be on a wide range of subjects.
There should be <ANSWER_NUMBER> possible answers, only one of which is unambiguously correct, and all of the answers should be kept brief.
Each of the <QUESTION_NUMBER> question/answer sets should be presented in the following format:

"""

TEXT_ANSWER_EXAMPLE = """
QUESTION: What is the capital of France?
CORRECT ANSWER: Paris
WRONG ANSWER: Madrid
WRONG ANSWER: Rome
WRONG ANSWER: Dublin

"""

ASK_Q = """You are to select the correct answer for a multiple choice question. 
A number of answers will be presented and you should respond with only the letter corresponding to the correct answer.
For example if the question is: 

What is the capital of France?

and the potential answers are:

E: Madrid
F: Paris
G: Rome
H: Dublin

you should only answer: 

F

This is your question:

<QUESTION>

These are the potential answers:

<ANSWERS>

Remember: only respond with the letter of the correct answer!
"""

if __name__ == "__main__":
    import sys

    question = (
        GENERATE_Q.replace("<QUESTION_NUMBER>", "100").replace("<ANSWER_NUMBER>", "4")
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
