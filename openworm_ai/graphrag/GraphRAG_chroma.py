# Based on https://docs.llamaindex.ai/en/stable/examples/cookbooks/GraphRAG_v1/
# Modified to use Chroma vector store instead of LlamaIndex SimpleVectorStore

import glob
import json
import os
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document as LangChainDocument
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.core import Settings
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI

from openworm_ai import print_
from openworm_ai.utils.llms import get_llm_from_argv

load_dotenv()  # Load .env file


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

    CHANGED: Returns LangChain-compatible embeddings for Chroma
    """
    if _has_openai_key():
        try:
            from langchain_openai import OpenAIEmbeddings

            print_("Embedding model: OpenAI (default)")
            return OpenAIEmbeddings()
        except Exception as e:
            print_(
                f"! OpenAI embeddings unavailable ({type(e).__name__}: {e}) -> falling back to HF BGE-small."
            )

    # Fallback: HF BGE small (fast + good) - LangChain version for Chroma
    hf = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    print_("Embedding model: HuggingFace BAAI/bge-small-en-v1.5")
    return hf


# Choose once, then use consistently for create + query + reload
EMBED_MODEL = _select_embed_model()
# Don't set Settings.embed_model since we're using LangChain embeddings


def _get_embedding_folder_name():
    """
    Returns a stable folder name based on the embedding model being used.

    CHANGED: Works with LangChain embedding models
    """
    if "openai" in EMBED_MODEL.__class__.__name__.lower():
        return "embed_openai"

    if hasattr(EMBED_MODEL, "model_name"):
        name = EMBED_MODEL.model_name
        return "embed_" + name.replace("/", "_").replace(":", "_")

    return "embed_" + EMBED_MODEL.__class__.__name__.lower()


def _make_llamaindex_llm(model: str):
    """
    Create a LlamaIndex LLM with smart fallbacks:
    1. Try OpenAI if key exists
    2. Try HuggingFace if token exists
    3. Fall back to Ollama (local, always works)
    """
    if isinstance(model, str) and model.startswith("ollama:"):
        local_name = model.split(":", 1)[1]
        print_(f"Using Ollama: {local_name}")
        return Ollama(model=local_name, request_timeout=60.0)

    if model.startswith("huggingface:"):
        hf_model = model.split(":", 1)[1]
        print_(f"Using HuggingFace Inference API: {hf_model}")
        return HuggingFaceInferenceAPI(model_name=hf_model, token=os.getenv("HF_TOKEN"))

    if _has_openai_key():
        print_(f"Using OpenAI: {model}")
        return OpenAI(model=model)

    if os.getenv("HF_TOKEN"):
        print_(f"No OpenAI key, using HuggingFace Inference API instead of {model}")
        return HuggingFaceInferenceAPI(
            model_name="Qwen/Qwen2.5-7B-Instruct", token=os.getenv("HF_TOKEN")
        )

    print_(f"No OpenAI/HF keys, using Ollama llama3.2 instead of {model}")
    return Ollama(model="llama3.2", request_timeout=60.0)


def create_store(model):
    """
    Create Chroma vector store from processed JSON files.

    CHANGED: One LangChain Document per paragraph instead of concatenating
    all paragraphs into a section blob. Uses paragraph-level provenance
    metadata when present (added by parse_pdfs.py), falls back gracefully
    to section-level info for older JSON files without metadata.
    """
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

            src_type = "Publication"
            if "wormatlas" in json_file:
                src_type = "WormAtlas Handbook"

            for section in doc_contents["sections"]:
                if "paragraphs" not in doc_contents["sections"][section]:
                    continue

                paragraphs = doc_contents["sections"][section]["paragraphs"]
                print_(
                    "    Processing section: %s\t(%i paragraphs)"
                    % (section, len(paragraphs))
                )

                # One Document per paragraph — fine-grained chunks for retrieval.
                # Uses paragraph-level metadata when present, falls back to
                # section info for older JSON files without metadata.
                for para_idx, p in enumerate(paragraphs):
                    text = p["contents"].strip()
                    if not text:
                        continue

                    page_num = p.get("page_number")
                    para_index = p.get("paragraph_index", para_idx)

                    if page_num is not None:
                        src_info = (
                            f"{src_type}: [{title}, p.{page_num}, "
                            f"para {para_index}]({src_page})"
                        )
                    else:
                        src_info = (
                            f"{src_type}: [{title}, Section {section}]({src_page})"
                        )

                    doc = LangChainDocument(
                        page_content=text, metadata={SOURCE_DOCUMENT: src_info}
                    )
                    documents.append(doc)

    print_("Creating Chroma vector store for %s" % model)

    STORE_SUBFOLDER = "/" + _get_embedding_folder_name()
    chroma_dir = Path(STORE_DIR + STORE_SUBFOLDER)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    chroma_settings = chromadb.config.Settings(
        is_persistent=True,
        persist_directory=str(chroma_dir.absolute()),
        anonymized_telemetry=False,
    )

    vectorstore = Chroma(
        collection_name="openworm-corpus",
        embedding_function=EMBED_MODEL,
        client_settings=chroma_settings,
    )

    print_("Adding documents to Chroma vector store...")
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        print_(f"  Adding documents {i + 1} to {i + len(batch)}...")
        vectorstore.add_documents(batch)

    print_("Persisting Chroma vector store to %s" % chroma_dir)


def load_index(model):
    """
    Load Chroma vector store.

    CHANGED: Loads Chroma instead of LlamaIndex storage
    """
    print_("Loading Chroma vector store for %s" % model)

    STORE_SUBFOLDER = "/" + _get_embedding_folder_name()
    chroma_dir = Path(STORE_DIR + STORE_SUBFOLDER)

    chroma_settings = chromadb.config.Settings(
        is_persistent=True,
        persist_directory=str(chroma_dir.absolute()),
        anonymized_telemetry=False,
    )

    vectorstore = Chroma(
        collection_name="openworm-corpus",
        embedding_function=EMBED_MODEL,
        client_settings=chroma_settings,
    )

    print_("Reloaded Chroma vector store for %s" % model)
    return vectorstore


def get_retriever(vectorstore, similarity_top_k=4):
    """
    Get a retriever from the Chroma vector store.

    CHANGED: Use similarity search to get real scores and better diversity
    """
    print_("Creating retriever with similarity_top_k=%d" % similarity_top_k)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": similarity_top_k * 2,
        },
    )

    return retriever


def process_query_simple(query, model, vectorstore, verbose=False):
    """
    Simple query processing for standalone testing.

    NOTE: For neuroml.ai RAG integration, you don't need this function.
    The neuroml.ai package handles LLM querying. This is just for testing
    the Chroma retriever works correctly.
    """
    print_("Processing query: %s" % query)

    docs_with_scores = vectorstore.similarity_search_with_score(query, k=8)

    print_(f"Found {len(docs_with_scores)} relevant documents")

    # Group by source document, keep best (lowest distance) chunk per source
    source_groups = {}

    for doc, distance in docs_with_scores:
        source = doc.metadata.get(SOURCE_DOCUMENT, "Unknown source")

        if source not in source_groups:
            source_groups[source] = (doc, distance)
        elif distance < source_groups[source][1]:
            source_groups[source] = (doc, distance)

    # Sort by distance (best matches first), limit to top 4 unique sources
    unique_docs = sorted(source_groups.values(), key=lambda x: x[1])[:4]

    files_used = []

    for doc, distance in unique_docs:
        # Convert L2 distance to approximate similarity score (0-1)
        similarity_score = max(0, 1 - (distance / 2))

        if verbose:
            print_("===================================")
            print_(doc.metadata.get(SOURCE_DOCUMENT, "Unknown source"))
            print_("-------")
            print_(f"Distance: {distance:.3f}, Similarity: {similarity_score:.3f}")
            print_("Length of selection below: %i" % len(doc.page_content))
            print_(doc.page_content[:300] + "...")

        sd = doc.metadata.get(SOURCE_DOCUMENT, "Unknown source")
        files_used.append(f"{sd} (sim: {similarity_score:.3f})")

    file_info = ",\n   ".join(files_used)

    # Build context for LLM from unique docs
    context = "\n\n".join([doc.page_content for doc, _ in unique_docs])

    llm = _make_llamaindex_llm(model)

    prompt = f"""Context information:
{context}

Question: {query}

Answer the question based on the context above. If the context doesn't contain enough information, say so.
"""

    from llama_index.core.llms import ChatMessage

    messages = [ChatMessage(role="user", content=prompt)]
    response = llm.chat(messages)
    response_text = str(response.message.content)

    if "<think>" in response_text:
        response_text = (
            response_text[0 : response_text.index("<think>")]
            + response_text[response_text.index("</think>") + 8 :]
        )

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

    return response_text


if __name__ == "__main__":
    llm_ver = get_llm_from_argv(sys.argv)

    if "-test" not in sys.argv:
        if "-q" not in sys.argv:
            create_store(llm_ver)

        vectorstore = load_index(llm_ver)

        queries = [
            "What are the main differences between NeuroML versions 1 and 2?",
            "What are the main types of cells in the C. elegans pharynx?",
            "Give me 3 facts about the coelomocyte system in C. elegans",
            "Tell me about the neurotransmitter betaine in C. elegans",
            "Tell me about the different locomotory gaits of C. elegans",
        ]

        print_("Processing %i queries" % len(queries))

        for query in queries:
            process_query_simple(query, llm_ver, vectorstore)
