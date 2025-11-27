from openworm_ai.utils.llms import get_llm_from_argv
from openworm_ai.utils.llms import generate_response


GENERATE_Q = """
Generate a list of <QUESTION_NUMBER> multiple choice questions to test someone's general knowledge of Caenorhabditis elegans (C. elegans).
The questions should be answerable by an intelligent adult, and should cover topics such as genetics, neurobiology, behavior, development, physiology, and research significance.
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

# New JSON-based MCQ generation template (v2)
GENERATE_Q_JSON = """
You are an expert on *Caenorhabditis elegans* (C. elegans) biology and neuroscience.

Generate <QUESTION_NUMBER> high-quality multiple-choice questions about C. elegans.
Cover a range of topics (anatomy, nervous system, behaviour, genetics, development, physiology, lab techniques, and research significance).
Questions should be answerable by a scientifically literate, intelligent adult without needing to be a specialist in C. elegans.

Each question MUST:
- Be specific to C. elegans (not generic animal biology).
- Be clearly and precisely worded.
- Have exactly ONE correct answer and three incorrect but plausible answers.
- Be answerable in a way that two well-informed experts on C. elegans would agree on the same option.

STRICTLY AVOID AMBIGUITY:
- Do NOT use vague terms like "main", "best", "most important", or "most likely"
  unless the question explicitly defines them clearly enough that only one option fits.
- Do NOT ask questions where more than one option could reasonably be argued correct.
- Avoid vague pronouns ("this", "it", "they") if it might be unclear what they refer to.
- If a question could be interpreted in multiple ways, REWRITE it until the meaning is unique.

For the incorrect options:
- They must be factually wrong for C. elegans.
- They must still sound plausible to someone with partial understanding of C. elegans.
- Avoid obviously silly or irrelevant answers.
- Do NOT use "All of the above" or "None of the above".

Return ONLY valid JSON, with no extra commentary. The JSON must be an array:

[
  {
    "question": "string",
    "options": [
      {"label": "A", "text": "string"},
      {"label": "B", "text": "string"},
      {"label": "C", "text": "string"},
      {"label": "D", "text": "string"}
    ],
    "correct_label": "A"
  },
  ...
]

Do not include fewer or more than <QUESTION_NUMBER> objects in the array.
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
