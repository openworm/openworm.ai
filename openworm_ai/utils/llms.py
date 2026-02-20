import time
import os

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

LLM_CMD_LINE_ARGS = {}

LLM_GPT35 = "gpt-3.5-turbo-instruct"
LLM_CMD_LINE_ARGS["-gpt35"] = LLM_GPT35
LLM_GPT4 = "gpt-4"
LLM_GPT4o = "gpt-4o"

LLM_LLAMA2 = "LLAMA2"
LLM_CMD_LINE_ARGS["-l"] = LLM_LLAMA2

LLM_GEMINI_2F = "gemini-2.0-flash"
LLM_CMD_LINE_ARGS["-g"] = LLM_GEMINI_2F
LLM_GEMINI_25F = "gemini-2.5-flash"
LLM_CMD_LINE_ARGS["-g25"] = LLM_GEMINI_25F

LLMS_GEMINI = [LLM_GEMINI_2F, LLM_GEMINI_25F]

LLM_CLAUDE37 = "claude-3-7-sonnet-20250219"
LLM_CMD_LINE_ARGS["-c"] = LLM_CLAUDE37

LLM_COHERE = "Cohere"
LLM_CMD_LINE_ARGS["-co"] = LLM_COHERE

# ----------------------------
# Hugging Face Inference Provider default (NEW)
LLM_HF_QWEN25_7B = "huggingface:Qwen/Qwen2.5-7B-Instruct"
LLM_CMD_LINE_ARGS["-hf"] = LLM_HF_QWEN25_7B
# ----------------------------

LLM_OLLAMA_LLAMA32 = "ollama:llama3.2"
LLM_CMD_LINE_ARGS["-o-l32"] = LLM_OLLAMA_LLAMA32
LLM_OLLAMA_LLAMA32_1B = "ollama:llama3.2:1b"
LLM_CMD_LINE_ARGS["-o-l321b"] = LLM_OLLAMA_LLAMA32_1B

LLM_OLLAMA_LLAMA32_3B = "ollama:llama3.2:3b"
LLM_CMD_LINE_ARGS["-o-l323b"] = LLM_OLLAMA_LLAMA32_3B

LLM_OLLAMA_MISTRAL = "ollama:mistral"
LLM_CMD_LINE_ARGS["-o-m"] = LLM_OLLAMA_MISTRAL
LLM_OLLAMA_TINYLLAMA = "ollama:tinyllama"
LLM_CMD_LINE_ARGS["-o-t"] = LLM_OLLAMA_TINYLLAMA
LLM_OLLAMA_PHI3 = "ollama:phi3:latest"
LLM_CMD_LINE_ARGS["-o-phi3"] = LLM_OLLAMA_PHI3
LLM_OLLAMA_PHI4 = "ollama:phi4:latest"
LLM_CMD_LINE_ARGS["-o-phi4"] = LLM_OLLAMA_PHI4
LLM_OLLAMA_GEMMA = "ollama:gemma:7b"
LLM_CMD_LINE_ARGS["-ge"] = LLM_OLLAMA_GEMMA
LLM_OLLAMA_GEMMA2 = "ollama:gemma2:latest"
LLM_CMD_LINE_ARGS["-ge2"] = LLM_OLLAMA_GEMMA2
LLM_OLLAMA_GEMMA3 = "ollama:gemma3:4b"
LLM_CMD_LINE_ARGS["-ge3"] = LLM_OLLAMA_GEMMA3
LLM_OLLAMA_DEEPSEEK = "ollama:deepseek-r1:7b"
LLM_CMD_LINE_ARGS["-o-dsr1"] = LLM_OLLAMA_DEEPSEEK

LLM_OLLAMA_QWEN = "ollama:qwen3:1.7b"
LLM_CMD_LINE_ARGS["-o-qw"] = LLM_OLLAMA_QWEN
LLM_OLLAMA_QWEN3_4B = "ollama:qwen3:4b"
LLM_CMD_LINE_ARGS["-o-qw34b"] = LLM_OLLAMA_QWEN3_4B

LLM_OLLAMA_CODELLAMA = "ollama:codellama:latest"
LLM_OLLAMA_FALCON2 = "ollama:falcon2:latest"

LLM_OLLAMA_OLMO2_7B = "ollama:olmo2:7b"
LLM_CMD_LINE_ARGS["-o-olmo27b"] = LLM_OLLAMA_OLMO2_7B

OPENAI_LLMS = [LLM_GPT4, LLM_GPT4o]

PREF_ORDER_LLMS = LLMS_GEMINI + [
    LLM_LLAMA2,
    LLM_GPT35,
    LLM_GPT4,
    LLM_GPT4o,
    LLM_CLAUDE37,
    LLM_COHERE,
    # (HF isn't in PREF_ORDER_LLMS because it’s provider-routed; still supported via init_chat_model)
    LLM_OLLAMA_LLAMA32,
    LLM_OLLAMA_LLAMA32_1B,
    LLM_OLLAMA_MISTRAL,
    LLM_OLLAMA_TINYLLAMA,
    LLM_OLLAMA_PHI3,
    LLM_OLLAMA_PHI4,
    LLM_OLLAMA_GEMMA2,
    LLM_OLLAMA_GEMMA3,
    LLM_OLLAMA_DEEPSEEK,
    LLM_OLLAMA_GEMMA,
    LLM_OLLAMA_QWEN,
    LLM_OLLAMA_CODELLAMA,
    LLM_OLLAMA_FALCON2,
    LLM_OLLAMA_OLMO2_7B,
]


def requires_openai_key(llm_ver):
    return llm_ver in OPENAI_LLMS


def get_openai_api_key():
    """
    Returns the OpenAI API key from:
    1. Environment variables (preferred)
    2. A file '../oaik' (legacy OpenWorm option), IF it exists
    """
    key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
    if key:
        return key.strip()

    oaik_path = "../oaik"
    if os.path.exists(oaik_path):
        with open(oaik_path, "r") as f:
            return f.read().strip()

    raise RuntimeError(
        "OpenAI API key not found.\n"
        "Set environment variable OPENAI_API_KEY or place a key in '../oaik'."
    )


def get_llamaapi_key():
    llamaapi_key = os.environ.get("LLAMAAPI_KEY")
    return llamaapi_key


def get_gemini_api_key():
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    return gemini_api_key


def get_anthropic_key():
    anthropic_api_key = os.environ.get("CLAUDE_API_KEY")
    return anthropic_api_key


def get_cohere_key():
    cohere_api_key = os.environ.get("COHERE_API_KEY")
    return cohere_api_key


# ----------------------------
# HF token detector (NEW)
def has_hf_token():
    # HF hub + inference providers commonly use one of these
    return bool(
        (os.environ.get("HF_TOKEN") or "").strip()
        or (os.environ.get("HUGGINGFACEHUB_API_TOKEN") or "").strip()
    )


# ----------------------------


GENERAL_QUERY_PROMPT_TEMPLATE = """Answer the following question. Provide succinct, yet scientifically accurate
    answers. Question: {question}

    Answer: """

GENERAL_QUERY_LIMITED_PROMPT_TEMPLATE = """You are a neuroscientist who is answering questions about the worm C. elegans. Provide succinct, yet scientifically accurate
    answers. If the question is not related to biology, physics or chemistry, then don't answer the question, but instead explain that you 
    can currently only answer questions related to C. elegans. Question: {question}

    Answer: """


def get_llm(llm_ver, temperature, limit_to_openwormai_llms=False):
    if llm_ver in LLMS_GEMINI:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=llm_ver,
            google_api_key=get_gemini_api_key(),
            temperature=temperature,
        )

    elif llm_ver == LLM_COHERE:
        from langchain_cohere import ChatCohere

        llm = ChatCohere()
        return llm

    elif limit_to_openwormai_llms and llm_ver not in PREF_ORDER_LLMS:
        raise ValueError(
            "LLM version %s not recognized by openworm.ai. Try the direct route (call init_chat_model in langchain.chat_models)..."
            % llm_ver
        )

    print(" ... Initializing chat model for LLM: %s using langchain" % llm_ver)
    llm = init_chat_model(llm_ver, temperature=temperature)
    return llm


def generate_response(input_text, llm_ver, temperature, only_celegans):
    template = (
        GENERAL_QUERY_LIMITED_PROMPT_TEMPLATE
        if only_celegans
        else GENERAL_QUERY_PROMPT_TEMPLATE
    )
    prompt = PromptTemplate(template=template, input_variables=["question"])

    try:
        llm = get_llm(llm_ver, temperature=temperature)

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

        llm = get_llm(llm_ver, temperature=temperature)

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

    llm = get_llm(llm_panel_chair, temperature=temperature)

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
    # Default remains GPT-4o
    llm_ver = LLM_GPT4o

    # Allow command-line flags to override
    for arg, model_name in LLM_CMD_LINE_ARGS.items():
        if arg in argv:
            return model_name

    # Allow explicit model names as positional args
    for a in argv[1:]:
        if a.startswith("Ollama:"):
            return a
        if a in PREF_ORDER_LLMS:
            return a
        if a.upper() in ("GPT4O", "GPT-4O"):
            return LLM_GPT4o
        # NEW: allow passing huggingface:* directly
        if a.startswith("huggingface:"):
            return a

    # --- FINAL FAILSAFE ---
    # Prefer: OpenAI -> HuggingFace -> Ollama
    try:
        if requires_openai_key(llm_ver):
            _ = get_openai_api_key()
            return llm_ver
    except Exception:
        pass

    if has_hf_token():
        print("! No OpenAI key found -> using Hugging Face inference model instead.")
        return LLM_HF_QWEN25_7B

    print("! No OpenAI key found and no HF token -> using local Ollama model instead.")
    return LLM_OLLAMA_LLAMA32


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
    question = "Why is the worm C. elegans important to scientists?"
    question = "Tell me briefly about the neuronal control of C. elegans locomotion and the influence of monoamines."

    question = (
        "I know you are a Large Language Model. Tell me your name, version and maker."
    )

    llm_ver = get_llm_from_argv(sys.argv)

    ask_question_get_response(question, llm_ver)
