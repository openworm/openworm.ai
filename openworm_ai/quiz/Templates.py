from openworm_ai.utils.llms import get_llm_from_argv
from openworm_ai.utils.llms import generate_response


GENERATE_Q = """
Generate a list of <QUESTION_NUMBER> multiple choice questions to test someone's general knowledge.
The questions should be answerable by an intelligent adult and should cover a wide range of topics,
such as history, geography, science, culture, technology, medicine, society, and everyday facts.
There should be <ANSWER_NUMBER> possible answers, only one of which is unambiguously correct, and all of the answers should be kept brief.
Each of the <QUESTION_NUMBER> question/answer sets should be presented in the following format:

"""

TEXT_ANSWER_EXAMPLE = """
QUESTION: What is the capital city of Japan?
CORRECT ANSWER: Tokyo
WRONG ANSWER: Osaka
WRONG ANSWER: Kyoto
WRONG ANSWER: Nagoya

"""

# New JSON-based MCQ generation template (v2)
GENERATE_Q_JSON = """
You are an expert academic question writer.

Generate <QUESTION_NUMBER> high-quality general-knowledge multiple-choice questions.
Cover a wide range of topics such as history, geography, science, culture, medicine,
society, world affairs, technology, and everyday factual knowledge.

Each question MUST:
- Be clearly and precisely worded.
- Be answerable by an intelligent adult without needing specialist knowledge.
- Have exactly ONE correct answer and three incorrect but plausible answers.
- Be unambiguous so that two well-informed people would independently choose the same correct option.

STRICTLY AVOID AMBIGUITY:
- Do NOT use vague terms like "main", "best", "most important", or "most likely"
  unless the question defines them clearly.
- Do NOT write questions where more than one answer could be argued correct.
- Avoid vague pronouns ("this", "it", "they") if unclear what they refer to.
- Avoid questions whose answer might depend on opinion or interpretation.

For the incorrect options:
- They must be factually wrong.
- They must still sound plausible to someone with partial knowledge.
- Avoid joke answers or irrelevant answers.
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

What is the capital city of Japan?

and the potential answers are:

E: Osaka
F: Tokyo
G: Kyoto
H: Nagoya

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
