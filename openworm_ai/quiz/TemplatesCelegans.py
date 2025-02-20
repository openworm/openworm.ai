from openworm_ai.utils.llms import get_llm_from_argv
from openworm_ai.utils.llms import generate_response

GENERATE_Q = """

Generate a list of <QUESTION_NUMBER> unique, non-repeated multiple choice questions to test someone's general knowledge about C. elegans.
The questions should be answerable by an intelligent adult knowledgable about anatomy and functioning, and should be on a wide range of C. elegans related subjects.
There should be <ANSWER_NUMBER> possible answers, only one of which is unambiguously correct, and all of the answers should be kept brief.
Each of the <QUESTION_NUMBER> question/answer sets should be presented in the following format:

"""

TEXT_ANSWER_EXAMPLE = """
QUESTION: What is the primary food source for C. elegans in lab conditions?
CORRECT ANSWER: E. coli
WRONG ANSWER: Algae
WRONG ANSWER: Fungi
WRONG ANSWER: Bacteria mix

"""

ASK_Q = """You are to select the correct answer for a multiple-choice question. 
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