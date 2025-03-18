# Based on https://docs.llamaindex.ai/en/stable/examples/cookbooks/GraphRAG_v1/

from openworm_ai import print_
from openworm_ai.utils.llms import get_llm_from_argv
from openworm_ai.utils.llms import LLM_GPT4o

from llama_index.core import Document
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import load_index_from_storage
from llama_index.embeddings.ollama import OllamaEmbedding


# one extra dep
from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama
import glob
import sys
import json
import os
import time

# from modelspec.utils import load_json

STORE_DIR = "store"
SOURCE_DOCUMENT = "source document"


def create_store(model):
    OLLAMA_MODEL = model.replace("Ollama:", "") if model is not LLM_GPT4o else None

    json_inputs = [
        file
        for file in glob.glob("processed/json/*/*.json")
        if os.path.normpath(file)
        != os.path.normpath("processed/json/papers/Corsi_et_al_2015.json")
    ]

    documents = []
    for json_file in json_inputs:
        print_("Adding %s" % json_file)

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
                # print_(f'---------------------\n{all_text}\n---------------------')
                src_info = (
                    f"WormAtlas Handbook: [{title}, Section {section}]({src_page})"
                )
                doc = Document(text=all_text, metadata={SOURCE_DOCUMENT: src_info})
                documents.append(doc)

    if "-test" in sys.argv:
        print_("Finishing before section requiring OPENAI_API_KEY...")

    else:
        print_("Creating a vector store index for %s" % model)

        STORE_SUBFOLDER = ""

        if OLLAMA_MODEL is not None:
            ollama_embedding = OllamaEmbedding(
                model_name=OLLAMA_MODEL,
            )
            STORE_SUBFOLDER = "/%s" % OLLAMA_MODEL.replace(":", "_")

            # create an index from the parsed markdown
            index = VectorStoreIndex.from_documents(
                documents, embed_model=ollama_embedding
            )
        else:
            index = VectorStoreIndex.from_documents(documents)

        print_("Persisting vector store index")

        index.storage_context.persist(persist_dir=STORE_DIR + STORE_SUBFOLDER)


def load_index(model):
    OLLAMA_MODEL = model.replace("Ollama:", "") if model is not LLM_GPT4o else None

    print_("Creating a storage context for %s" % model)

    STORE_SUBFOLDER = (
        "" if OLLAMA_MODEL is None else "/%s" % OLLAMA_MODEL.replace(":", "_")
    )

    # index_reloaded =SimpleIndexStore.from_persist_dir(persist_dir=INDEX_STORE_DIR)
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


def get_query_engine(index_reloaded, model):
    OLLAMA_MODEL = model.replace("Ollama:", "") if model is not LLM_GPT4o else None

    print_("Creating query engine for %s" % model)

    # create a query engine for the index
    if OLLAMA_MODEL is not None:
        llm = Ollama(model=OLLAMA_MODEL)
        ollama_embedding = OllamaEmbedding(
            model_name=OLLAMA_MODEL,
        )
        query_engine = index_reloaded.as_query_engine(
            llm=llm, embed_model=ollama_embedding
        )
    else:
        query_engine = index_reloaded.as_query_engine()

    return query_engine


def process_query(query, model):
    """Processes a single query, logs the exact prompt used, and prints the retrieved context."""

    print_(f"\n🔹 TESTING QUERY: {query}")
    print_("-----------------------------------------------------------")

    # Measure retrieval performance
    start_time = time.time()

    # Run retrieval
    retrieval_results = query_engine.query(query)
    retrieval_texts = [str(doc) for doc in retrieval_results[:5]]  # Limit to top 2 docs

    retrieval_time = time.time() - start_time
    print_(f"\n🔹 Retrieval Time: {retrieval_time:.2f}s")

    # Log retrieved context
    if retrieval_texts:
        print_("\n🔹 Retrieved Context:")
        for idx, text in enumerate(retrieval_texts, start=1):
            print_(f"  [{idx}] {text[:1000]}...")  # Print first 500 characters
    else:
        print_(
            "⚠ No relevant documents retrieved. The model may rely on pre-trained knowledge."
        )

    # Prepare formatted query with retrieval results
    formatted_query = (
        "Use the retrieved context below to generate the best answer.\n\n"
        + f" Query: {query}\n\n"
        + " **Retrieved Context:**\n"
        + "\n\n".join(retrieval_texts)
    )

    # Log the exact query being sent to the model
    print_("\n Final Prompt Sent to LLM:")
    print_(formatted_query)

    # Run LLM query
    start_time = time.time()
    response = query_engine.query(formatted_query)
    response_time = time.time() - start_time

    # Capture response
    response_text = str(response)

    print_(" Model Response:")
    print_(response_text)

    print_(f" Response Time: {response_time:.2f}s")
    print_("-----------------------------------------------------------")

    return response_text


if __name__ == "__main__":
    import sys

    llm_ver = get_llm_from_argv(sys.argv)

    if "-q" not in sys.argv:
        create_store(llm_ver)

    if "-test" not in sys.argv:
        index_reloaded = load_index(llm_ver)
        query_engine = get_query_engine(index_reloaded, llm_ver)

        # query the engine
        query = "What can you tell me about the neurons of the pharynx of C. elegans?"
        query = "Write 100 words on how C. elegans eats"
        query = "How does the pharyngeal epithelium of C. elegans maintain its shape?"

        queries = [
            "How many neurons are present in the adult hermaphrodite C. elegans?"
        ]

        for query in queries:
            process_query(query, llm_ver)
