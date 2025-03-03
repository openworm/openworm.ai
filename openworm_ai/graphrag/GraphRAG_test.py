# Based on https://docs.llamaindex.ai/en/stable/examples/cookbooks/GraphRAG_v1/

from llama_index.core import Document
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import load_index_from_storage

# one extra dep
from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama
import glob
import sys
import json

# from modelspec.utils import load_json

STORE_DIR = "store"
SOURCE_DOCUMENT = "source document"

json_inputs = glob.glob("processed/json/*/*.json")
print(json_inputs)

documents = []
for json_file in json_inputs:
    print("Adding %s" % json_file)

    with open(json_file, encoding="utf-8") as f:
        doc_model = json.load(f)

    for title in doc_model:
        print("  Processing document: %s" % title)
        doc_contents = doc_model[title]
        src_page = doc_contents["source"]
        for section in doc_contents["sections"]:
            all_text = ""
            if "paragraphs" in doc_contents["sections"][section]:
                print(
                    "    Processing section: %s\t(%i paragraphs)"
                    % (section, len(doc_contents["sections"][section]["paragraphs"]))
                )
                for p in doc_contents["sections"][section]["paragraphs"]:
                    all_text += p["contents"] + "\n\n"
            if len(all_text) == 0:
                all_text = " "
            # print(f'---------------------\n{all_text}\n---------------------')
            src_info = f"WormAtlas Handbook: [{title}, Section {section}]({src_page})"
            doc = Document(text=all_text, metadata={SOURCE_DOCUMENT: src_info})
            documents.append(doc)


if "-test" in sys.argv:
    print("Finishing before section requiring OPENAI_API_KEY...")

else:
    # create an index from the parsed markdown
    index = VectorStoreIndex.from_documents(documents)

    index.storage_context.persist(persist_dir=STORE_DIR)

    # index_reloaded =SimpleIndexStore.from_persist_dir(persist_dir=INDEX_STORE_DIR)
    storage_context = StorageContext.from_defaults(
        docstore=SimpleDocumentStore.from_persist_dir(persist_dir=STORE_DIR),
        vector_store=SimpleVectorStore.from_persist_dir(persist_dir=STORE_DIR),
        index_store=SimpleIndexStore.from_persist_dir(persist_dir=STORE_DIR),
    )
    index_reloaded = load_index_from_storage(storage_context)

    # create a query engine for the index
    llm = Ollama(model="llama3.2:1b")
    query_engine = index_reloaded.as_query_engine(llm=llm)

    def process_query(response):
        response = query_engine.query(query)
        files_used = []
        for k in response.metadata:
            v = response.metadata[k]
            if SOURCE_DOCUMENT in v:
                if v[SOURCE_DOCUMENT] not in files_used:
                    files_used.append(v[SOURCE_DOCUMENT])

        file_info = ",\n  ".join(files_used)
        print(f"""
===============================================================================
QUERY: {query}
-------------------------------------------------------------------------------
RESPONSE: {response}
SOURCES: 
  {file_info}
===============================================================================
""")

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
    ]

    for query in queries:
        process_query(query)
