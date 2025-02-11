import os
import time

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

LLM_GPT35 = "GPT3.5"
LLM_GPT4 = "GPT4"
LLM_GPT4o = "GPT4o"
LLM_LLAMA2 = "LLAMA2"
LLM_GEMINI = "Gemini"
LLM_AI21 = "AI21"
LLM_CLAUDE2 = "Claude2.1"
LLM_COHERE = "Cohere"
LLM_OLLAMA_LLAMA32 = "Ollama:Llama3.2"
LLM_OLLAMA_MISTRAL = "Ollama:mistral"
LLM_OLLAMA_TINYLLAMA = "Ollama:tinyllama"

OPENAI_LLMS = [LLM_GPT35, LLM_GPT4, LLM_GPT4o]

PREF_ORDER_LLMS = (
    LLM_GEMINI,
    LLM_LLAMA2,
    LLM_GPT35,
    LLM_GPT4,
    LLM_GPT4o,
    LLM_AI21,
    LLM_CLAUDE2,
    LLM_COHERE,
    LLM_OLLAMA_LLAMA32,
    LLM_OLLAMA_MISTRAL,
    LLM_OLLAMA_TINYLLAMA,
)


def requires_openai_key(llm_ver):
    return llm_ver in OPENAI_LLMS


def get_openai_api_key():
    # if openai_api_key_sb == None or len(openai_api_key_sb)==0:
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key is None:
        openai_api_key = str(open("../oaik", "r").readline())
    # else:
    #   openai_api_key = openai_api_key_sb
    return openai_api_key


def get_llamaapi_key():
    llamaapi_key = os.environ.get("LLAMAAPI_KEY")

    return llamaapi_key


def get_gemini_api_key():
    gemini_api_key = os.environ.get("GEMINIAPI_KEY")

    return gemini_api_key


def get_ai21_api_key():
    ai21_api_key = os.environ.get["AI21_API_KEY"]

    return ai21_api_key


def get_anthropic_key():
    anthropic_api_key = os.environ.get["ANTHROPIC_API_KEY"]

    return anthropic_api_key


def get_cohere_key():
    cohere_api_key = os.environ.get["COHERE_API_KEY"]

    return cohere_api_key


def get_llm(llm_ver, temperature):
    if llm_ver == LLM_GPT35:
        from langchain_openai import OpenAI

        llm = OpenAI(temperature=temperature, openai_api_key=get_openai_api_key())

    elif llm_ver == LLM_GPT4:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model_name="gpt-4",
            openai_api_key=get_openai_api_key(),
            temperature=temperature,
        )
    elif llm_ver == LLM_GPT4o:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model_name="gpt-4o",
            openai_api_key=get_openai_api_key(),
            temperature=temperature,
        )

    elif llm_ver == LLM_LLAMA2:
        from llamaapi import LlamaAPI
        import asyncio

        # Create a new event loop
        loop = asyncio.new_event_loop()

        # Set the event loop as the current event loop
        asyncio.set_event_loop(loop)

        llama = LlamaAPI(get_llamaapi_key())

        from langchain_experimental.llms import ChatLlamaAPI

        llm = ChatLlamaAPI(client=llama)

    elif llm_ver == LLM_GEMINI:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model="gemini-pro", google_api_key=get_gemini_api_key()
        )

    elif llm_ver == LLM_AI21:
        from langchain_ai21 import AI21LLM

        llm = AI21LLM(model="j2-ultra")

    elif llm_ver == LLM_CLAUDE2:
        from langchain_anthropic import AnthropicLLM

        llm = AnthropicLLM(model="claude-2.1")

    elif llm_ver == LLM_COHERE:
        from langchain_cohere import ChatCohere

        llm = ChatCohere()

    elif llm_ver == LLM_OLLAMA_LLAMA32:
        from langchain_ollama.llms import OllamaLLM

        llm = OllamaLLM(model="llama3.2:1b")

    elif llm_ver == LLM_OLLAMA_MISTRAL:
        from langchain_ollama.llms import OllamaLLM

        llm = OllamaLLM(model="mistral")

    elif llm_ver == LLM_OLLAMA_TINYLLAMA:
        from langchain_ollama.llms import OllamaLLM

        llm = OllamaLLM(model="tinyllama")

    return llm


GENERAL_QUERY_PROMPT_TEMPLATE = """Answer the following question. Provide succinct, yet scientifically accurate
    answers. Question: {question}

    Answer: """

GENERAL_QUERY_LIMITED_PROMPT_TEMPLATE = """You are a neuroscientist who is answering questions about the worm C. elegans. Provide succinct, yet scientifically accurate
    answers. If the question is not related to biology, physics or chemistry, then don't answer the question, but instead explain that you 
    can currently only answer questions related to C. elegans. Question: {question}

    Answer: """


def generate_response(input_text, llm_ver, temperature, only_celegans):
    template = (
        GENERAL_QUERY_LIMITED_PROMPT_TEMPLATE
        if only_celegans
        else GENERAL_QUERY_PROMPT_TEMPLATE
    )
    prompt = PromptTemplate(template=template, input_variables=["question"])

    try:
        llm = get_llm(llm_ver, temperature)

        llm_chain = prompt | llm | StrOutputParser()

        response = llm_chain.invoke(input_text)
    except Exception as e:
        return "Error when processing that request:\n\n%s" % (e)

    return response


def generate_panel_response(input_text, llm_panelists, llm_panel_chair, temperature):
    responses = {}

    for llm_ver in llm_panelists:
        prompt = PromptTemplate(
            template=GENERAL_QUERY_PROMPT_TEMPLATE, input_variables=["question"]
        )

        llm = get_llm(llm_ver, temperature)

        llm_chain = prompt | llm | StrOutputParser()

        responses[llm_ver] = llm_chain.invoke(input_text)

    panel_chair_prompt = """You are a neuroscientist chairing a panel discussion on the nematode C. elegans. A researcher has asked the following question:
    {question}
    and %i experts on the panel have give their answers. 
    """ % (len(llm_panelists))

    for llm_ver in llm_panelists:
        panel_chair_prompt += """
The panelist named Dr. %s has provided the answer: %s
""" % (
            llm_ver,
            responses[llm_ver],
        )

    panel_chair_prompt += (
        """
Please generate a brief answer to the researcher's question based on their responses, pointing out where there is any inconsistency"""
        + """ in their answers, and using your own knowledge of C. elegans to try to resolve it."""
    )

    print(panel_chair_prompt)

    prompt = PromptTemplate(template=panel_chair_prompt, input_variables=["question"])

    llm = get_llm(llm_panel_chair, temperature)

    llm_chain = prompt | llm | StrOutputParser()

    response_chair = llm_chain.invoke(input_text)

    response = """**%s**: %s""" % (llm_panel_chair, response_chair)

    response += """

-----------------------------------
_Individual responses:_

"""
    for llm_ver in responses:
        response += """
_**%s**:_ _%s_
""" % (
            llm_ver,
            responses[llm_ver].strip().replace("\n", " "),
        )

    return response


def get_llm_from_argv(argv):
    llm_ver = LLM_GPT35

    if "-g" in argv:
        llm_ver = LLM_GEMINI

    if "-l" in argv:
        llm_ver = LLM_LLAMA2

    if "-a" in argv:
        llm_ver = LLM_AI21

    if "-cl" in argv:
        llm_ver = LLM_CLAUDE2

    if "-co" in argv:
        llm_ver = LLM_COHERE

    if "-o-l32" in argv:
        llm_ver = LLM_OLLAMA_LLAMA32

    if "-o-m" in argv:
        llm_ver = LLM_OLLAMA_MISTRAL

    if "-o-t" in argv:
        llm_ver = LLM_OLLAMA_TINYLLAMA

    return llm_ver


def ask_question_get_response(
    question, llm_ver, temperature=0, only_celegans=False, print_question=True
):
    print("--------------------------------------------------------")
    if print_question:
        print("Asking question:\n   %s" % question)
        print("--------------------------------------------------------")

    print(" ... Connecting to LLM: %s" % llm_ver)

    start = time.time()
    response = generate_response(
        question, llm_ver=llm_ver, temperature=temperature, only_celegans=only_celegans
    )
    print(" ... Processed in %.3f sec" % (time.time() - start))

    print("--------------------------------------------------------")
    print("Answer:\n   %s" % response)
    print("--------------------------------------------------------")
    print()

    return response


if __name__ == "__main__":
    import sys

    question = "What is the most common type of neuron in the brain?"

    llm_ver = get_llm_from_argv(sys.argv)

    ask_question_get_response(question, llm_ver)
