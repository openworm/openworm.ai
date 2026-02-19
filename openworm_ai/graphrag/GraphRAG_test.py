# Based on https://docs.llamaindex.ai/en/stable/examples/cookbooks/GraphRAG_v1/

# Load environment variables FIRST

import os
import glob
import time
import sys
import json

from pathlib import Path
import hashlib

from openworm_ai import print_
from openworm_ai.utils.llms import (
    get_llm_from_argv,
    is_huggingface_model,
    is_ollama_model,
    strip_huggingface_prefix,
    get_hf_token,
    LLM_GPT4o,
)

from llama_index.core import Document
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import load_index_from_storage
from llama_index.core import PromptTemplate
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import Settings

from dotenv import load_dotenv

load_dotenv()

_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
print(f"[DEBUG] HF Token loaded: {_token[:15] if _token else 'NONE'}...", flush=True)
print(f"[DEBUG] Embedding model: {os.getenv('NML_AI_EMBEDDING_MODEL')}", flush=True)

print("=" * 60, flush=True)
print("GraphRAG starting... (imports may take 30-60 seconds)", flush=True)
print("=" * 60, flush=True)

print("Imports complete. Loading configuration...", flush=True)

STORE_DIR = "store"
SOURCE_DOCUMENT = "source document"

Settings.chunk_size = 3000
Settings.chunk_overlap = 50

SOURCE_REGISTRY_PATH = Path("corpus/papers/source_registry.json")


def load_source_registry(path: Path):
    if not path.exists():
        return {"papers": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def make_chunk_id(
    paper_ref: str, section_title: str, para_index: int, text: str
) -> str:
    base = f"{paper_ref}|{section_title}|{para_index}|{text[:80]}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]


def normalize_ollama_model_name(model: str) -> str:
    s = (model or "").strip()
    for prefix in ("Ollama:", "ollama:", "llama:"):
        if s.startswith(prefix):
            s = s[len(prefix) :].strip()
    return s


def get_embedding_model(model: str):
    """
    Get embedding model. Priority: HuggingFace API > Ollama > OpenAI default
    """
    # Check for embedding model override in env
    embed_model_env = os.getenv("NML_AI_EMBEDDING_MODEL") or os.getenv(
        "OPENWORM_AI_EMBEDDING_MODEL"
    )

    if embed_model_env and embed_model_env.startswith("huggingface:"):
        hf_model = strip_huggingface_prefix(embed_model_env)
        print_(f"Using HuggingFace API embedding: {hf_model}")
        from llama_index.embeddings.huggingface_api import (
            HuggingFaceInferenceAPIEmbedding,
        )

        return HuggingFaceInferenceAPIEmbedding(
            model_name=hf_model, token=get_hf_token()
        )

    if is_huggingface_model(model):
        hf_embed_model = "BAAI/bge-small-en-v1.5"
        print_(f"Using HuggingFace API embedding: {hf_embed_model}")
        from llama_index.embeddings.huggingface_api import (
            HuggingFaceInferenceAPIEmbedding,
        )

        return HuggingFaceInferenceAPIEmbedding(
            model_name=hf_embed_model, token=get_hf_token()
        )

    if is_ollama_model(model):
        ollama_model = normalize_ollama_model_name(model)
        print_(f"Using Ollama embedding: {ollama_model}")
        from llama_index.embeddings.ollama import OllamaEmbedding

        return OllamaEmbedding(model_name=ollama_model)

    print_("Using default embedding model")
    return None


def get_store_subfolder(model: str) -> str:
    if is_huggingface_model(model):
        hf_model = strip_huggingface_prefix(model)
        return "/" + hf_model.replace("/", "_").replace(":", "_")
    if is_ollama_model(model):
        ollama_model = normalize_ollama_model_name(model)
        return "/" + ollama_model.replace(":", "_")
    return ""


def create_store(model):
    None if model == LLM_GPT4o else normalize_ollama_model_name(model)

    start_time = time.time()

    json_inputs = glob.glob("processed/json/*/*.json")
    print_(f"Found {len(json_inputs)} JSON files to process")

    source_registry = load_source_registry(SOURCE_REGISTRY_PATH)
    papers_meta = source_registry.get("papers", {})

    documents = []
    for json_file in json_inputs:
        print_("Adding file to document store: %s" % json_file)

        with open(json_file, encoding="utf-8") as f:
            doc_model = json.load(f)

        for title in doc_model:
            print_("  Processing document: %s" % title)
            doc_contents = doc_model[title]

            src_page = doc_contents.get("source", "") or ""
            paper_ref = title
            paper_meta = papers_meta.get(paper_ref, {})

            source_url = paper_meta.get("source_url", src_page) or ""
            doi = paper_meta.get("doi", "") or ""
            s2_paper_id = paper_meta.get("s2_paper_id", "") or ""
            citation_short = paper_meta.get("citation_short", paper_ref) or paper_ref

            src_type = "Publication"
            if "wormatlas" in json_file:
                src_type = "WormAtlas Handbook"

            sections_dict = doc_contents.get("sections", {})
            if not isinstance(sections_dict, dict):
                continue

            for section_title, section_obj in sections_dict.items():
                if not isinstance(section_obj, dict):
                    continue

                paras = section_obj.get("paragraphs", [])
                if not isinstance(paras, list) or not paras:
                    continue

                for para_index, p in enumerate(paras):
                    if not isinstance(p, dict):
                        continue

                    text = (p.get("contents") or "").strip()
                    if not text:
                        continue

                    chunk_id = make_chunk_id(
                        paper_ref, str(section_title), para_index, text
                    )

                    meta = {
                        "paper_ref": paper_ref,
                        "citation_short": citation_short,
                        "source_url": source_url,
                        "doi": doi,
                        "s2_paper_id": s2_paper_id,
                        "source_type": src_type,
                        "section_title": str(section_title),
                        "para_index": para_index,
                        "chunk_id": chunk_id,
                    }

                    documents.append(Document(text=text, metadata=meta))

    elapsed = time.time() - start_time
    print_(
        f"\n[{elapsed:.1f}s] Loaded {len(documents)} document chunks from {len(json_inputs)} files"
    )
    print_(f"[{elapsed:.1f}s] Creating vector store index (this may take a while)...")
    print_(f"[{elapsed:.1f}s] Using model: {model}")

    embed_model = get_embedding_model(model)
    store_subfolder = get_store_subfolder(model)

    if embed_model is not None:
        index = VectorStoreIndex.from_documents(
            documents, embed_model=embed_model, show_progress=True
        )
    else:
        index = VectorStoreIndex.from_documents(documents, show_progress=True)

    elapsed = time.time() - start_time
    print_(f"\n[{elapsed:.1f}s] Indexing complete! Persisting to disk...")
    index.storage_context.persist(persist_dir=STORE_DIR + store_subfolder)
    total_time = time.time() - start_time
    print_(f"[{total_time:.1f}s] Vector store created and saved!")
    print_(f"{'=' * 60}\n")


def load_index(model):
    print_(f"Creating storage context for {model}")

    store_subfolder = get_store_subfolder(model)

    storage_context = StorageContext.from_defaults(
        docstore=SimpleDocumentStore.from_persist_dir(
            persist_dir=STORE_DIR + store_subfolder
        ),
        vector_store=SimpleVectorStore.from_persist_dir(
            persist_dir=STORE_DIR + store_subfolder
        ),
        index_store=SimpleIndexStore.from_persist_dir(
            persist_dir=STORE_DIR + store_subfolder
        ),
    )
    print_(f"Reloading index for {model}")

    embed_model = get_embedding_model(model)
    if embed_model is not None:
        Settings.embed_model = embed_model

    index_reloaded = load_index_from_storage(storage_context)
    return index_reloaded


def get_query_engine(index_reloaded, model, similarity_top_k=4):
    embed_model = get_embedding_model(model)
    if embed_model is not None:
        Settings.embed_model = embed_model

    print_(f"Creating query engine for {model}")

    text_qa_template_str = (
        "Context information is below.\n---------------------\n{context_str}\n"
        "---------------------\nUsing both the context information and your own knowledge, "
        "answer the question: {query_str}\nIf the context isn't helpful, you can also answer on your own.\n"
    )
    text_qa_template = PromptTemplate(text_qa_template_str)

    refine_template_str = (
        "The original question is: {query_str}\nWe have an existing answer: {existing_answer}\n"
        "We can refine it with more context below.\n------------\n{context_msg}\n------------\n"
        "Using both the new context and your knowledge, update or repeat the existing answer.\n"
    )
    refine_template = PromptTemplate(refine_template_str)

    if is_ollama_model(model):
        from llama_index.llms.ollama import Ollama
        from llama_index.embeddings.ollama import OllamaEmbedding

        ollama_model = normalize_ollama_model_name(model)
        llm = Ollama(model=ollama_model, request_timeout=120.0)
        ollama_embedding = OllamaEmbedding(model_name=ollama_model)

        query_engine = index_reloaded.as_query_engine(
            llm=llm,
            text_qa_template=text_qa_template,
            refine_template=refine_template,
            embed_model=ollama_embedding,
        )
        query_engine.retriever.similarity_top_k = similarity_top_k
    else:
        # HuggingFace and other models: use retriever + response synthesizer
        retriever = VectorIndexRetriever(
            index=index_reloaded,
            similarity_top_k=similarity_top_k,
        )

        response_synthesizer = get_response_synthesizer(
            response_mode="refine",
            text_qa_template=text_qa_template,
            refine_template=refine_template,
        )

        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )

    return query_engine


def process_query(query, model, verbose=False):
    query_start = time.time()
    print_(f"\n[Query] {query[:80]}...")
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
            print_(f"SCORE: {sn.score}")
            print_("METADATA:")
            print_(sn.metadata)
            print_("-------")
            print_(f"Length of selection: {len(sn.text)}")
            print_(sn.text)

        md = sn.metadata or {}

        citation = md.get("citation_short", md.get("paper_ref", "unknown"))
        paper_ref = md.get("paper_ref", "unknown")
        section_title = md.get("section_title", "unknown")
        para_index = md.get("para_index", "?")
        source_url = md.get("source_url", "")
        chunk_id = md.get("chunk_id", "")

        label = f"{citation} — {paper_ref} — {section_title} ¶{para_index}"
        if source_url:
            label += f" ({source_url})"
        if chunk_id:
            label += f" [chunk:{chunk_id}]"

        if label not in files_used:
            if len(files_used) == 0 or (sn.score is not None and sn.score >= cutoff):
                files_used.append(f"{label} (score: {sn.score})")

    file_info = ",\n   ".join(files_used)
    print_(
        f"""
===============================================================================
QUERY: {query}
MODEL: {model}
-------------------------------------------------------------------------------
RESPONSE: {response_text}
SOURCES:
   {file_info}
===============================================================================
"""
    )

    query_time = time.time() - query_start
    print_(f"[{query_time:.1f}s] Query completed")
    return response_text, response.metadata


if __name__ == "__main__":
    llm_ver = get_llm_from_argv(sys.argv)

    if "-test" not in sys.argv:
        if "-q" not in sys.argv:
            create_store(llm_ver)

        index_reloaded = load_index(llm_ver)
        query_engine = get_query_engine(index_reloaded, llm_ver)

        queries = [
            "What are the main types of cells in the C. elegans pharynx?",
            "Tell me about the egg laying apparatus in C. elegans",
            "Tell me briefly about the neuronal control of C. elegans locomotion and the influence of monoamines.",
        ]

        print_(f"Processing {len(queries)} queries")

        for query in queries:
            process_query(query, llm_ver)
