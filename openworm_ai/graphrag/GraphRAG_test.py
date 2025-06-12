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
from llama_index.core import PromptTemplate
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import Settings


# one extra dep
from llama_index.llms.ollama import Ollama
import glob
import sys
import json


STORE_DIR = "store"
SOURCE_DOCUMENT = "source document"

Settings.chunk_size = 3000
Settings.chunk_overlap = 50


def create_store(model):
    OLLAMA_MODEL = model.replace("Ollama:", "") if model is not LLM_GPT4o else None

    json_inputs = glob.glob("processed/json/papers/*.json")

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
                # print_(f'---------------------\n{all_text}\n---------------------')
                src_type = "Publication"
                if "wormatlas" in json_file:
                    src_type = "WormAtlas Handbook"
                src_info = f"{src_type}: [{title}, Section {section}]({src_page})"
                doc = Document(text=all_text, metadata={SOURCE_DOCUMENT: src_info})
                documents.append(doc)

    print_("Creating a vector store index for %s" % model)

    STORE_SUBFOLDER = ""

    if OLLAMA_MODEL is not None:
        ollama_embedding = OllamaEmbedding(
            model_name=OLLAMA_MODEL,
        )
        STORE_SUBFOLDER = "/%s" % OLLAMA_MODEL.replace(":", "_")

        # create an index from the parsed markdown
        index = VectorStoreIndex.from_documents(
            documents, embed_model=ollama_embedding, show_progress=True
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


def get_query_engine(index_reloaded, model, similarity_top_k=4):
    OLLAMA_MODEL = model.replace("Ollama:", "") if model is not LLM_GPT4o else None

    print_("Creating query engine for %s" % model)

    # Based on: https://docs.llamaindex.ai/en/stable/examples/customization/prompts/completion_prompts/

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

    # create a query engine for the index
    if OLLAMA_MODEL is not None:
        llm = Ollama(model=OLLAMA_MODEL, request_timeout=60.0)

        ollama_embedding = OllamaEmbedding(
            model_name=OLLAMA_MODEL,
        )

        query_engine = index_reloaded.as_query_engine(
            llm=llm,
            text_qa_template=text_qa_template,
            refine_template=refine_template,
            embed_model=ollama_embedding,
        )
        # print(dir(query_engine.retriever))

        query_engine.retriever.similarity_top_k = similarity_top_k

    else:  # use OpenAI...
        # configure retriever
        retriever = VectorIndexRetriever(
            index=index_reloaded,
            similarity_top_k=similarity_top_k,
        )

        # configure response synthesizer
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
    print_("Processing query: %s" % query)
    response = query_engine.query(query)

    response_text = str(response)

    if "<think>" in response_text:  # Give deepseek a fighting chance...
        response_text = (
            response_text[0 : response_text.index("<think>")]
            + response_text[response_text.index("</think>") + 8 :]
        )

    metadata = response.metadata
    cutoff = 0.2
    files_used = []
    for sn in response.source_nodes:
        if verbose:
            print_("===================================")
            # print(dir(sn))
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

    return response_text, metadata


if __name__ == "__main__":
    import sys

    llm_ver = get_llm_from_argv(sys.argv)

    if "-test" not in sys.argv:
        if "-q" not in sys.argv:
            create_store(llm_ver)

        index_reloaded = load_index(llm_ver)
        query_engine = get_query_engine(index_reloaded, llm_ver)

        # query the engine
        query = "What can you tell me about the neurons of the pharynx of C. elegans?"
        query = "Write 100 words on how C. elegans eats"
        query = "How does the pharyngeal epithelium of C. elegans maintain its shape?"

        """
            "What can you tell me about the properties of electrical connectivity between the muscles of C. elegans?",
            "What are the dimensions of the C. elegans pharynx?",
            "What color is C. elegans?",
            "Give me 3 facts about the coelomocyte system in C. elegans",
            "Give me 3 facts about the control of motor programs in c. elegans by monoamines",
            "When was the first metazoan genome sequenced? Answer only with the year.","""

        queries = [
            "What is the main function of cell pair AVB?",
            "In what year was William Shakespeare born? ",
            "Tell me about the egg laying apparatus in C. elegans",
            "Tell me briefly about the neuronal control of C. elegans locomotion and the influence of monoamines.",
            "What can you tell me about Alan Coulson?",
            "The NeuroPAL transgene is amazing. Give me some examples of fluorophores in it.",
        ]
        queries = [
            "What are the main differences between NeuroML versions 1 and 2?",
        ]

        print_("Processing %i queries" % len(queries))

        for query in queries:
            process_query(query, llm_ver)
