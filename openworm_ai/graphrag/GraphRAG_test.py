# Based on https://docs.llamaindex.ai/en/stable/examples/cookbooks/GraphRAG_v1/

from openworm_ai import print_
from openworm_ai.utils.llms import get_llm_from_argv

from llama_index.core import Document
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import load_index_from_storage
from llama_index.core import PromptTemplate
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings

# HF embeddings fallback
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# LLMs - Native LlamaIndex (no LangChain dependency)
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

import glob
import sys
import json
import os


STORE_DIR = "store"
SOURCE_DOCUMENT = "source document"

Settings.chunk_size = 3000
Settings.chunk_overlap = 50


def _has_openai_key() -> bool:
    key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
    return bool(key and key.strip())


def _select_embed_model():
    """
    Prefer OpenAI embeddings if a key is present, otherwise fall back to HF BGE small.
    Also fall back to HF if OpenAI embedding init fails for any reason.
    """
    if _has_openai_key():
        try:
            # Trigger Settings.embed_model resolution (OpenAI default) if not set
            _ = Settings.embed_model
            print_("Embedding model: OpenAI (default)")
            return Settings.embed_model
        except Exception as e:
            print_(
                f"! OpenAI embeddings unavailable ({type(e).__name__}: {e}) -> falling back to HF BGE-small."
            )

    # Fallback: HF BGE small (fast + good)
    hf = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    print_("Embedding model: HuggingFace BAAI/bge-small-en-v1.5")
    return hf


# Choose once, then use consistently for create + query + reload
EMBED_MODEL = _select_embed_model()
Settings.embed_model = EMBED_MODEL


def _get_embedding_folder_name():
    """
    Returns a stable folder name based on the embedding model being used.
    """
    if EMBED_MODEL.__class__.__name__.lower().startswith("openai"):
        return "embed_openai"

    if hasattr(EMBED_MODEL, "model_name"):
        name = EMBED_MODEL.model_name
        return "embed_" + name.replace("/", "_").replace(":", "_")

    return "embed_unknown"


def _make_llamaindex_llm(model: str):
    """
    Create a LlamaIndex LLM with smart fallbacks:
    1. Try OpenAI if key exists (for your prof)
    2. Try HuggingFace if token exists (for you)
    3. Fall back to Ollama (local, always works)
    """
    # Ollama models - explicit choice
    if isinstance(model, str) and model.startswith("ollama:"):
        local_name = model.split(":", 1)[1]
        print_(f"Using Ollama: {local_name}")
        return Ollama(model=local_name, request_timeout=60.0)

    # HuggingFace models - explicit choice
    if model.startswith("huggingface:"):
        hf_model = model.split(":", 1)[1]
        print_(f"Using HuggingFace Inference: {hf_model}")
        return HuggingFaceInferenceAPI(model_name=hf_model)

    # For anything else (gpt-4o, gemini, etc), try OpenAI first
    if _has_openai_key():
        print_(f"Using OpenAI: {model}")
        return OpenAI(model=model)

    # No OpenAI key, fall back to HuggingFace
    if os.getenv("HF_TOKEN"):
        print_(f"No OpenAI key, using HuggingFace default model instead of {model}")
        return HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-7B-Instruct")

    # Last resort: Ollama local
    print_(f"No OpenAI/HF keys, using Ollama llama3.2 instead of {model}")
    return Ollama(model="llama3.2", request_timeout=60.0)


def create_store(model):
    json_inputs = glob.glob("processed/json/*/*.json")

    documents = []
    for json_file in json_inputs:
        print_("Adding file to document store: %s" % json_file)

        with open(json_file, encoding="utf-8") as f:
            doc_model = json.load(f)

        for title in doc_model:
            print_("  Processing document: %s" % title)
            doc_contents = doc_model[title]
            src_page = doc_contents["source"]
            for section in doc_contents["sections"]:
                all_text = ""
                if "paragraphs" in doc_contents["sections"][section]:
                    print_(
                        "    Processing section: %s\t(%i paragraphs)"
                        % (
                            section,
                            len(doc_contents["sections"][section]["paragraphs"]),
                        )
                    )
                    for p in doc_contents["sections"][section]["paragraphs"]:
                        all_text += p["contents"] + "\n\n"
                if len(all_text) == 0:
                    all_text = " "
                src_type = "Publication"
                if "wormatlas" in json_file:
                    src_type = "WormAtlas Handbook"
                src_info = f"{src_type}: [{title}, Section {section}]({src_page})"
                doc = Document(text=all_text, metadata={SOURCE_DOCUMENT: src_info})
                documents.append(doc)

    print_("Creating a vector store index for %s" % model)

    # Folder depends on embedding model (not chat model)
    STORE_SUBFOLDER = "/" + _get_embedding_folder_name()

    index = VectorStoreIndex.from_documents(
        documents, embed_model=EMBED_MODEL, show_progress=True
    )

    print_("Persisting vector store index")
    index.storage_context.persist(persist_dir=STORE_DIR + STORE_SUBFOLDER)


def load_index(model):
    print_("Creating a storage context for %s" % model)

    STORE_SUBFOLDER = "/" + _get_embedding_folder_name()

    storage_context = StorageContext.from_defaults(
        docstore=SimpleDocumentStore.from_persist_dir(
            persist_dir=STORE_DIR + STORE_SUBFOLDER
        ),
        vector_store=SimpleVectorStore.from_persist_dir(
            persist_dir=STORE_DIR + STORE_SUBFOLDER
        ),
        index_store=SimpleIndexStore.from_persist_dir(
            persist_dir=STORE_DIR + STORE_SUBFOLDER
        ),
    )

    print_("Reloading index for %s" % model)
    index_reloaded = load_index_from_storage(storage_context)
    return index_reloaded


def get_query_engine(index_reloaded, model, similarity_top_k=4):
    print_("Creating query engine for %s" % model)

    text_qa_template_str = (
        "Context information is"
        " below.\n---------------------\n{context_str}\n---------------------\nUsing"
        " both the context information and also using your own knowledge, answer"
        " the question: {query_str}\nIf the context isn't helpful, you can also"
        " answer the question on your own.\n"
    )
    text_qa_template = PromptTemplate(text_qa_template_str)

    refine_template_str = (
        "The original question is as follows: {query_str}\nWe have provided an"
        " existing answer: {existing_answer}\nWe have the opportunity to refine"
        " the existing answer (only if needed) with some more context"
        " below.\n------------\n{context_msg}\n------------\nUsing both the new"
        " context and your own knowledge, update or repeat the existing answer.\n"
    )
    refine_template = PromptTemplate(refine_template_str)

    llm = _make_llamaindex_llm(model)

    query_engine = index_reloaded.as_query_engine(
        llm=llm,
        text_qa_template=text_qa_template,
        refine_template=refine_template,
        embed_model=EMBED_MODEL,  # keep embedding space consistent
    )

    # set retrieval depth
    query_engine.retriever.similarity_top_k = similarity_top_k
    return query_engine


def process_query(query, model, query_engine, verbose=False):
    print_("Processing query: %s" % query)
    response = query_engine.query(query)

    response_text = str(response)

    if "<think>" in response_text:
        response_text = (
            response_text[0 : response_text.index("<think>")]
            + response_text[response_text.index("</think>") + 8 :]
        )

    cutoff = 0.2
    files_used = []
    for sn in response.source_nodes:
        if verbose:
            print_("===================================")
            print_(sn.metadata["source document"])
            print_("-------")
            print_("Length of selection below: %i" % len(sn.text))
            print_(sn.text)

        sd = sn.metadata["source document"]
        if sd not in files_used:
            if len(files_used) == 0 or sn.score >= cutoff:
                files_used.append(f"{sd} (score: {sn.score})")

    file_info = ",\n   ".join(files_used)
    print_(f"""
===============================================================================
QUERY: {query}
MODEL: {model}
-------------------------------------------------------------------------------
RESPONSE: {response_text}
SOURCES: 
   {file_info}
===============================================================================
""")

    return response_text, response.metadata


if __name__ == "__main__":
    llm_ver = get_llm_from_argv(sys.argv)

    if "-test" not in sys.argv:
        if "-q" not in sys.argv:
            create_store(llm_ver)

        index_reloaded = load_index(llm_ver)
        query_engine = get_query_engine(index_reloaded, llm_ver)

        queries = [
            "What are the main differences between NeuroML versions 1 and 2?",
            "What are the main types of cells in the C. elegans pharynx?",
            "Give me 3 facts about the coelomocyte system in C. elegans",
            "Tell me about the neurotransmitter betaine in C. elegans",
            "Tell me about the different locomotory gaits of C. elegans",
        ]

        print_("Processing %i queries" % len(queries))

        for query in queries:
            process_query(query, llm_ver, query_engine)
