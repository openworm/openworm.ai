import os
import sys
import json
import time
import random
import glob
from openworm_ai import print_
from openworm_ai.utils.llms import get_llm_from_argv, LLM_GPT4o
from llama_index.core import Document, load_index_from_storage
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama

# === Directory Paths === #
STORE_DIR = "store"
QUIZ_SCORE_DIR = "openworm_ai/quiz/scores"
QUIZ_FILE = "openworm_ai/quiz/samples/GPT4o_100questions_celegans_batched.json"
SOURCE_DOCUMENT = "source document"


# === Create Vector Store Index === #
def create_store(model):
    """Creates a vector store index from JSON documents."""
    OLLAMA_MODEL = model.replace("Ollama:", "") if model is not LLM_GPT4o else None

    json_inputs = [
        file
        for file in glob.glob("processed/json/*/*.json")
        if os.path.normpath(file)
        != os.path.normpath("processed/json/papers/Corsi_et_al_2015.json")
    ]

    documents = []
    for json_file in json_inputs:
        print_(f"Adding {json_file}")

        with open(json_file, encoding="utf-8") as f:
            doc_model = json.load(f)

        for title, doc_contents in doc_model.items():
            print_(f"  Processing document: {title}")
            src_page = doc_contents["source"]
            for section, details in doc_contents.get("sections", {}).items():
                all_text = (
                    "\n\n".join([p["contents"] for p in details.get("paragraphs", [])])
                    or " "
                )
                src_info = (
                    f"WormAtlas Handbook: [{title}, Section {section}]({src_page})"
                )
                documents.append(
                    Document(text=all_text, metadata={SOURCE_DOCUMENT: src_info})
                )

    if "-test" in sys.argv:
        print_("Finishing before section requiring OPENAI_API_KEY...")
        return

    print_(f"Creating a vector store index for {model}")
    STORE_SUBFOLDER = f"/{OLLAMA_MODEL.replace(':', '_')}" if OLLAMA_MODEL else ""

    # Create index with embeddings if needed
    ollama_embedding = (
        OllamaEmbedding(model_name=OLLAMA_MODEL) if OLLAMA_MODEL else None
    )
    index = (
        VectorStoreIndex.from_documents(documents, embed_model=ollama_embedding)
        if ollama_embedding
        else VectorStoreIndex.from_documents(documents)
    )

    print_("Persisting vector store index")
    index.storage_context.persist(persist_dir=STORE_DIR + STORE_SUBFOLDER)


# === Load Index === #
def load_index(model):
    """Loads the stored index for the given model."""
    OLLAMA_MODEL = model.replace("Ollama:", "") if model is not LLM_GPT4o else None
    STORE_SUBFOLDER = f"/{OLLAMA_MODEL.replace(':', '_')}" if OLLAMA_MODEL else ""

    storage_context = StorageContext.from_defaults(
        docstore=SimpleDocumentStore.from_persist_dir(STORE_DIR + STORE_SUBFOLDER),
        vector_store=SimpleVectorStore.from_persist_dir(STORE_DIR + STORE_SUBFOLDER),
        index_store=SimpleIndexStore.from_persist_dir(STORE_DIR + STORE_SUBFOLDER),
    )

    print_(f"Reloading index for {model}")
    return load_index_from_storage(storage_context)


# === Query Engine === #
def get_query_engine(index_reloaded, model):
    """Creates a query engine for the model."""
    OLLAMA_MODEL = model.replace("Ollama:", "") if model is not LLM_GPT4o else None

    if OLLAMA_MODEL:
        llm = Ollama(model=OLLAMA_MODEL)
        ollama_embedding = OllamaEmbedding(model_name=OLLAMA_MODEL)
        return index_reloaded.as_query_engine(llm=llm, embed_model=ollama_embedding)

    return index_reloaded.as_query_engine()


# === Load Questions from JSON File === #
def load_queries():
    """Loads multiple-choice questions from a JSON file and shuffles answer choices."""
    if not os.path.exists(QUIZ_FILE):
        print_(f" Error: Query file {QUIZ_FILE} not found.")
        return []

    with open(QUIZ_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f).get("questions", [])

    # Shuffle the answers for each question
    for question_data in questions:
        random.shuffle(question_data["answers"])  # Shuffle answer order

    return questions


# === Process Queries and Evaluate === #
def process_queries(model, query_engine):
    """Processes multiple-choice queries, evaluates responses, and saves results."""

    queries = load_queries()
    if not queries:
        print_(" No queries found.")
        return

    total_time, correct_count, total_queries = 0, 0, len(queries)
    results = []

    for i, question_data in enumerate(queries):
        query = question_data["question"]
        print_(f"\n🔹 Query {i + 1}/{total_queries}: {query}")

        # Start time measurement
        start_time = time.time()

        # Retrieve relevant documents
        retrieval_results = query_engine.retrieve(query)
        retrieval_texts = [str(doc) for doc in retrieval_results[:2]]  # Get top 2

        # Track whether retrieval is relevant
        files_used = [
            doc.metadata.get("source", "Unknown") for doc in retrieval_results
        ]
        is_relevant = retrieval_texts and len(" ".join(retrieval_texts).split()) > 30

        # Log retrieved context
        print_(
            f"\n🔹 Retrieved Context for Query {i + 1}:\n"
            + "\n\n".join(retrieval_texts)
            if retrieval_texts
            else "⚠ No relevant context retrieved."
        )

        # **FALLBACK: If retrieval fails or is irrelevant, use model knowledge**
        if not files_used or not is_relevant:
            fallback_prompt = f"""
            The query could not be answered based on the available documents.
            Instead, use your own knowledge to provide the best possible answer.
            Do not falsely attribute information to the documents.
            If the documents do not contain the answer, rely on your own knowledge without citing sources.
            If you used your own pre-trained knowledge, do not cite any sources.

            Query: {query}
            """
            response = query_engine.query(
                fallback_prompt
            )  # Force the model to use its knowledge
            source_message = (
                "No relevant documents were found. Answering from general knowledge."
            )
            files_used = []
        else:
            # Use retrieved context normally
            source_message = ",\n  ".join(files_used)
            formatted_query = (
                "Use your own expert knowledge first. If needed, consider the retrieved context below.\n\n"
                + f"🔹 Query: {query}\n\n"
                + "🔹 **Retrieved Context:**\n"
                + "\n\n".join(retrieval_texts)
            )
            response = query_engine.query(formatted_query)

        # Stop time measurement
        elapsed_time = time.time() - start_time
        total_time += elapsed_time

        # Capture response
        response_text = str(response).strip().upper()

        # Validate response format (ensure A, B, C, or D)
        valid_choices = [ans["ref"] for ans in question_data["answers"]]
        response_text = (
            response_text
            if response_text in valid_choices
            else random.choice(valid_choices)
        )

        # Check correctness
        is_correct = any(
            ans["ref"] == response_text and ans["correct"]
            for ans in question_data["answers"]
        )
        correct_count += int(is_correct)

        print_(
            f" Response Time: {elapsed_time:.2f}s | Correct: {is_correct} | Answer: {response_text} | Source: {source_message}"
        )

        results.append(
            {
                "query": query,
                "response": response_text,
                "correct": is_correct,
                "response_time": elapsed_time,
                "source": source_message,
            }
        )

    # Compute accuracy and average response time
    accuracy = (correct_count / total_queries) * 100 if total_queries else 0
    avg_response_time = total_time / total_queries if total_queries else 0

    # Save results
    score_data = {
        "title": f"{model.replace(':', '_')}_score",
        "score": correct_count,
        "avg_response_time": avg_response_time,
    }

    os.makedirs(QUIZ_SCORE_DIR, exist_ok=True)
    score_file = os.path.join(
        QUIZ_SCORE_DIR, f"{model.replace(':', '_')}_score_rag.json"
    )

    with open(score_file, "w", encoding="utf-8") as f:
        json.dump(score_data, f, indent=4)

    print_(f"\nResults saved to: {score_file}")
    print_(f"Accuracy: {accuracy:.2f}% | ⏱ Avg Response Time: {avg_response_time:.2f}s")


# === Main Execution === #
if __name__ == "__main__":
    llm_ver = get_llm_from_argv(sys.argv)

    if "-q" not in sys.argv:
        create_store(llm_ver)

    if "-test" not in sys.argv:
        index_reloaded = load_index(llm_ver)
        query_engine = get_query_engine(index_reloaded, llm_ver)

        print_("\nProcessing Queries from JSON...")
        process_queries(llm_ver, query_engine)
