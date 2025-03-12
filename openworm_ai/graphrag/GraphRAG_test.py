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

# from modelspec.utils import load_json

STORE_DIR = "store"
SOURCE_DOCUMENT = "source document"


def create_store(model):
    OLLAMA_MODEL = model.replace("Ollama:", "") if model is not LLM_GPT4o else None

    json_inputs = glob.glob("processed/json/*/*.json")
    # print_(json_inputs)

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


def process_query(response, model):
    response = query_engine.query(query)
    response_text = str(response)
    metadata = response.metadata
    files_used = []
    for k in metadata:
        v = metadata[k]
        if SOURCE_DOCUMENT in v:
            if v[SOURCE_DOCUMENT] not in files_used:
                files_used.append(v[SOURCE_DOCUMENT])

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

    return response_text, metadata


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
            "What can you tell me about the properties of electrical connectivity between the muscles of C. elegans?",
            "What are the dimensions of the C. elegans pharynx?",
            "What color is C. elegans?",
            "What is the main function of cell AVBR?",
            "Give me 3 facts about the coelomocyte system in C. elegens",
            "Give me 3 facts about the control of motor programs in c. elegans by monoamines",
            "The NeuroPAL transgene is amazing. Give me some examples of fluorophores in it.",
            "When was the first metazoan genome sequenced? Answer only with the year.",
            "In what year was William Shakespeare born? If the answer is not in the provided context information, please answer using your own knowledge.",
        ]

        for query in queries:
            process_query(query, llm_ver)
