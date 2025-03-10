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

    def is_response_relevant(query, response_text):
        """
        Basic check: If the response does not mention anything related to the query,
        assume irrelevance.
        """
        query_keywords = set(query.lower().split())
        response_keywords = set(response_text.lower().split())

        common_words = query_keywords.intersection(response_keywords)

        # If too few words overlap, assume irrelevance (adjust threshold if needed)
        return len(common_words) > 2

    def process_query(query):
        """
        Process the query with the RAG pipeline. If no relevant sources are found,
        fallback to the LLM's pre-trained knowledge.
        """

        print(f"\nProcessing query: {query}\n")

        response = query_engine.query(query)
        files_used = []

        # Extract metadata from response
        if hasattr(response, "metadata") and isinstance(response.metadata, dict):
            for k, v in response.metadata.items():
                if isinstance(v, dict) and SOURCE_DOCUMENT in v:
                    if v[SOURCE_DOCUMENT] not in files_used:
                        files_used.append(v[SOURCE_DOCUMENT])

        response_text = str(response)  # Ensure response is a string
        is_relevant = is_response_relevant(query, response_text)

        # If no relevant documents OR response is not relevant, use fallback
        if not files_used or not is_relevant:
            fallback_prompt = f"""
            The query could not be answered based on the available documents.
            Instead, use your own knowledge to provide the best possible answer.
            Do not falsely attribute information to the documents.
            If the documents do not contain the answer, rely on your own knowledge without citing sources.
            If you used your own pre-trained knowledge, do not cite any sources.

            Query: {query}
            """
            response = llm.complete(fallback_prompt)
            source_message = (
                "No relevant documents were found. Answering from general knowledge."
            )
            files_used = []
        else:
            source_message = ",\n  ".join(files_used)

        # Print response properly formatted
        print(f"""
    ===============================================================================
    QUERY: {query}
    -------------------------------------------------------------------------------
    RESPONSE: {response}
    """)

        # Only print sources if they exist
        if files_used:
            print(f"SOURCES:\n  {source_message}")
        else:
            print("No sources were found.")
        print(
            "===============================================================================\n"
        )

    # Query list
    queries = [
        "When did World War 2 start?",
        "What can you tell me about the properties of electrical connectivity between the muscles of C. elegans?",
        "What are the dimensions of the C. elegans pharynx?",
        "What color is C. elegans?",
        "What is the main function of cell AVBR?",
        "Give me 3 facts about the coelomocyte system in C. elegans",
        "Give me 3 facts about the control of motor programs in C. elegans by monoamines",
        "The NeuroPAL transgene is amazing. Give me some examples of fluorophores in it.",
        "When was the first metazoan genome sequenced? Answer only with the year.",
    ]

    # Run queries
    for query in queries:
        process_query(query)
