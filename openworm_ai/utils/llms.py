import os
import time

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()  # Load .env file (HF_TOKEN, etc.)

LLM_CMD_LINE_ARGS = {}

# OpenAI models
LLM_GPT35 = "gpt-3.5-turbo-instruct"
LLM_CMD_LINE_ARGS["-gpt35"] = LLM_GPT35
LLM_GPT4 = "gpt-4"
LLM_GPT4o = "gpt-4o"

LLM_LLAMA2 = "LLAMA2"
LLM_CMD_LINE_ARGS["-l"] = LLM_LLAMA2

# Gemini models
LLM_GEMINI_2F = "gemini-2.0-flash"
LLM_CMD_LINE_ARGS["-g"] = LLM_GEMINI_2F
LLM_GEMINI_25F = "gemini-2.5-flash"
LLM_CMD_LINE_ARGS["-g25"] = LLM_GEMINI_25F

LLMS_GEMINI = [LLM_GEMINI_2F, LLM_GEMINI_25F]

# Anthropic models
LLM_CLAUDE37 = "claude-3-7-sonnet-20250219"
LLM_CMD_LINE_ARGS["-c"] = LLM_CLAUDE37

# Cohere
LLM_COHERE = "Cohere"
LLM_CMD_LINE_ARGS["-co"] = LLM_COHERE

# ============================================================================
# HuggingFace models
# Format: huggingface:org/model
# These use HF's free inference API - just need HF_TOKEN
# ============================================================================
LLM_HF_MISTRAL_7B = "huggingface:mistralai/Mistral-7B-Instruct-v0.3"
LLM_CMD_LINE_ARGS["-hf-mistral"] = LLM_HF_MISTRAL_7B

LLM_HF_LLAMA32_3B = "huggingface:meta-llama/Llama-3.2-3B-Instruct"
LLM_CMD_LINE_ARGS["-hf-llama32"] = LLM_HF_LLAMA32_3B

LLM_HF_LLAMA31_8B = "huggingface:meta-llama/Llama-3.1-8B-Instruct"
LLM_CMD_LINE_ARGS["-hf-llama31"] = LLM_HF_LLAMA31_8B

LLM_HF_QWEN25_7B = "huggingface:Qwen/Qwen2.5-7B-Instruct"
LLM_CMD_LINE_ARGS["-hf-qwen"] = LLM_HF_QWEN25_7B

LLM_HF_PHI3_MINI = "huggingface:microsoft/Phi-3-mini-4k-instruct"
LLM_CMD_LINE_ARGS["-hf-phi3"] = LLM_HF_PHI3_MINI

LLM_HF_GEMMA2_9B = "huggingface:google/gemma-2-9b-it"
LLM_CMD_LINE_ARGS["-hf-gemma2"] = LLM_HF_GEMMA2_9B

LLM_HF_ZEPHYR_7B = "huggingface:HuggingFaceH4/zephyr-7b-beta"
LLM_CMD_LINE_ARGS["-hf-zephyr"] = LLM_HF_ZEPHYR_7B

LLMS_HUGGINGFACE = [
    LLM_HF_MISTRAL_7B,
    LLM_HF_LLAMA32_3B,
    LLM_HF_LLAMA31_8B,
    LLM_HF_QWEN25_7B,
    LLM_HF_PHI3_MINI,
    LLM_HF_GEMMA2_9B,
    LLM_HF_ZEPHYR_7B,
]

# Default HuggingFace model (good balance of quality and speed)
LLM_HF_DEFAULT = LLM_HF_MISTRAL_7B

# ============================================================================
# Ollama models (local inference)
# ============================================================================
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

PREF_ORDER_LLMS = (
    LLMS_HUGGINGFACE
    + LLMS_GEMINI
    + [
        LLM_LLAMA2,
        LLM_GPT35,
        LLM_GPT4,
        LLM_GPT4o,
        LLM_CLAUDE37,
        LLM_COHERE,
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
)


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
    return os.environ.get("LLAMAAPI_KEY")


def get_gemini_api_key():
    return os.environ.get("GEMINI_API_KEY")


def get_anthropic_key():
    return os.environ.get("CLAUDE_API_KEY")


def get_cohere_key():
    return os.environ.get("COHERE_API_KEY")


def get_hf_token() -> str:
    """Get HuggingFace API token from environment."""
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    return (token or "").strip()


def has_hf_token() -> bool:
    """Check if HuggingFace token is available."""
    return bool(get_hf_token())


def is_huggingface_model(model_name: str) -> bool:
    return isinstance(model_name, str) and model_name.lower().startswith("huggingface:")


def is_ollama_model(model_name: str) -> bool:
    return isinstance(model_name, str) and model_name.lower().startswith("ollama:")


def strip_huggingface_prefix(model_name_full: str) -> str:
    """
    Accepts:
      huggingface:ORG/MODEL
      huggingface:ORG/MODEL:cheapest   (we just ignore ':cheapest' here)
    Returns:
      ORG/MODEL
    """
    s = model_name_full.strip()
    s = s[len("huggingface:") :]
    if s.endswith(":cheapest"):
        s = s.replace(":cheapest", "")
    return s.strip()


GENERAL_QUERY_PROMPT_TEMPLATE = """Answer the following question. Provide succinct, yet scientifically accurate
    answers. Question: {question}

    Answer: """

GENERAL_QUERY_LIMITED_PROMPT_TEMPLATE = """You are a neuroscientist who is answering questions about the worm C. elegans. Provide succinct, yet scientifically accurate
    answers. If the question is not related to biology, physics or chemistry, then don't answer the question, but instead explain that you 
    can currently only answer questions related to C. elegans. Question: {question}

    Answer: """


def get_llm(llm_ver, temperature, limit_to_openwormai_llms=False):
    # --- Hugging Face (matches NeuroML style: huggingface:MODEL[:cheapest]) ---
    if is_huggingface_model(llm_ver):
        hf_token = get_hf_token()
        if not hf_token:
            raise RuntimeError("HF_TOKEN (or HUGGINGFACEHUB_API_TOKEN) is not set")

        model_name = strip_huggingface_prefix(llm_ver)

        # Uses langchain-huggingface just like the NeuroML codebase does
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        endpoint = HuggingFaceEndpoint(
            repo_id=model_name,
            huggingfacehub_api_token=hf_token,
            temperature=temperature,
        )
        return ChatHuggingFace(llm=endpoint)

    # --- Gemini ---
    if llm_ver in LLMS_GEMINI:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=llm_ver,
            google_api_key=get_gemini_api_key(),
            temperature=temperature,
        )

    # --- Cohere ---
    if llm_ver == LLM_COHERE:
        from langchain_cohere import ChatCohere

        return ChatCohere()

    # --- Guardrail for unknown names ---
    if (
        limit_to_openwormai_llms
        and llm_ver not in PREF_ORDER_LLMS
        and not is_huggingface_model(llm_ver)
    ):
        raise ValueError(
            "LLM version %s not recognized by openworm.ai. Try the direct route (call init_chat_model in langchain.chat_models)..."
            % llm_ver
        )

    # --- Default LangChain init path (OpenAI/Ollama/Anthropic/etc) ---
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
    """
    Determine which LLM to use based on command-line args and environment.

    Priority order:
    1. Environment variable: OPENWORM_AI_CHAT_MODEL or NML_AI_CHAT_MODEL
    2. Command-line flags (e.g., -hf-mistral, -o-l32)
    3. Explicit model names in argv
    4. Default: HuggingFace (if HF_TOKEN set) -> Ollama (if available) -> GPT-4o
    """
    # 1) Environment variable override
    env_model = os.getenv("OPENWORM_AI_CHAT_MODEL") or os.getenv("NML_AI_CHAT_MODEL")
    if env_model and env_model.strip():
        return env_model.strip()

    # 2) Command-line flags
    for arg, model_name in LLM_CMD_LINE_ARGS.items():
        if arg in argv:
            return model_name

    # 3) Explicit model names as positional args
    for a in argv[1:]:
        if is_huggingface_model(a):
            return a
        if is_ollama_model(a):
            return a
        if a in PREF_ORDER_LLMS:
            return a
        if a.upper() in ("GPT4O", "GPT-4O"):
            return LLM_GPT4o

    # 4) Smart default: HuggingFace > Ollama > GPT-4o
    # Prefer HuggingFace API (free, no local setup needed)
    if has_hf_token():
        print("Using HuggingFace API (HF_TOKEN found)")
        return LLM_HF_DEFAULT

    # Fallback to Ollama if no HF token
    print("! No HF_TOKEN found -> trying local Ollama")
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

    question = "Tell me briefly about the neuronal control of C. elegans locomotion and the influence of monoamines."

    question = (
        "I know you are a Large Language Model. Tell me your name, version and maker."
    )

    llm_ver = get_llm_from_argv(sys.argv)
    ask_question_get_response(question, llm_ver)
