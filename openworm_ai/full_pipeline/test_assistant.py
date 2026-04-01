# test_assistant.py - End-to-end test for the NML_Assistant pipeline
# Tests: question (RAG only), database (RAG + WormBase), simulation (RAG + HH)
import asyncio
import os
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


async def test():
    from neuroml_ai.assistant import NML_Assistant

    # Try openworm_ai LLM constants, fall back to string
    try:
        from openworm_ai.utils.llms import LLM_HF_QWEN25_14B
    except ImportError:
        LLM_HF_QWEN25_14B = "huggingface:Qwen/Qwen2.5-14B-Instruct"

    # Determine vector store config path
    vs_config = os.getenv("GEN_RAG_VS_CONFIG")
    if not vs_config:
        repo_root = Path(__file__).resolve().parent.parent.parent
        vs_config = str(repo_root / "vector-stores.json")

    if not Path(vs_config).exists():
        print(f"ERROR: vector-stores.json not found at {vs_config}")
        print("Set GEN_RAG_VS_CONFIG env var or place vector-stores.json in repo root")
        sys.exit(1)

    chat_model = os.getenv("GEN_RAG_CHAT_MODEL", LLM_HF_QWEN25_14B)

    print(f"Using model: {chat_model}")
    print(f"Using vector store config: {vs_config}")
    print()

    assistant = NML_Assistant(
        vs_config_file=vs_config,
        chat_model=chat_model,
    )
    await assistant.setup()

    tests = [
        ("Question (RAG only)", "How many neurons does C. elegans have?"),
        ("Database (RAG + WormBase)", "Look up eat-4 on WormBase"),
        (
            "Simulation (RAG + HH)",
            "Run a Hodgkin-Huxley simulation with 0.1nA current injection",
        ),
    ]

    for label, query in tests:
        print(f"\n{'=' * 70}")
        print(f"=== TEST: {label} ===")
        print(f"QUERY: {query}")
        print(f"{'=' * 70}")
        try:
            result = await assistant.run_graph_invoke(query)
            print(f"\nRESULT:\n{result}")
        except Exception as e:
            print(f"\nERROR: {e}")
            traceback.print_exc()

    print(f"\n{'=' * 70}")
    print("All tests complete.")


if __name__ == "__main__":
    asyncio.run(test())
