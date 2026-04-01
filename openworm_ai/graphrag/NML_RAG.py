import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Import the neuroml.ai RAG package
try:
    from gen_rag.rag import RAG
except ImportError:
    print("ERROR: neuroml.ai RAG package not installed.")
    print("Install with:")
    print(
        "  pip install git+https://github.com/NeuroML/neuroml-ai.git#subdirectory=utils_pkg"
    )
    print(
        "  pip install git+https://github.com/NeuroML/neuroml-ai.git#subdirectory=rag_pkg/gen_rag"
    )
    sys.exit(1)


async def main():
    # Check environment variables
    vs_config = os.getenv("GEN_RAG_VS_CONFIG")
    chat_model = os.getenv("GEN_RAG_CHAT_MODEL")

    if not vs_config:
        print("ERROR: GEN_RAG_VS_CONFIG not set")
        print("Set it to the path of your vector-stores.json file:")
        print("  export GEN_RAG_VS_CONFIG=./vector-stores.json")
        sys.exit(1)

    if not chat_model:
        print("ERROR: GEN_RAG_CHAT_MODEL not set")
        print("Set it to your preferred model:")
        print("  export GEN_RAG_CHAT_MODEL=ollama:llama3.2")
        print("  export GEN_RAG_CHAT_MODEL=huggingface:Qwen/Qwen2.5-32B-Instruct")
        sys.exit(1)

    if not Path(vs_config).exists():
        print(f"ERROR: Vector store config not found at: {vs_config}")
        print("Run rag_chroma.py first to generate it")
        sys.exit(1)

    print("Initializing RAG with:")
    print(f"  Config: {vs_config}")
    print(f"  Model: {chat_model}")
    print()

    # Initialize the RAG system
    rag = RAG(
        vs_config_file=vs_config,
        chat_model=chat_model,
        memory=True,  # Enable conversation memory
    )

    await rag.setup()
    print("RAG system initialized successfully")
    print()

    # Test queries
    queries = [
        "What are the main differences between NeuroML versions 1 and 2?",
        "What are the main types of cells in the C. elegans pharynx?",
        "Tell me about the neurotransmitter betaine in C. elegans",
    ]

    thread_id = "test_thread"

    for query in queries:
        print("=" * 80)
        print(f"QUERY: {query}")
        print("-" * 80)

        # Use invoke for simple request-response
        response = await rag.run_graph_invoke(query, thread_id=thread_id)

        print(f"RESPONSE:\n{response}")
        print()

    print("=" * 80)
    print("Test complete!")


if __name__ == "__main__":
    asyncio.run(main())
