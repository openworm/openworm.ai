"""Quick test for OpenWormAssistant — runs a few queries to verify
classifier routing and tool execution."""

import asyncio
import os
import traceback
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")


async def test():
    from openworm_ai.assistant import OpenWormAssistant

    repo_root = Path(__file__).resolve().parent.parent.parent
    vs_config = os.getenv("GEN_RAG_VS_CONFIG", str(repo_root / "vector-stores.json"))
    chat_model = os.getenv("GEN_RAG_CHAT_MODEL", "huggingface:Qwen/Qwen2.5-7B-Instruct")

    print(f"Model:  {chat_model}")
    print(f"VS config: {vs_config}")
    print()

    assistant = OpenWormAssistant(
        vs_config_file=vs_config,
        chat_model=chat_model,
    )
    await assistant.setup()

    tests = [
        ("Question → RAG", "How many neurons does C. elegans have?"),
        ("Task → HH tool", "Run an HH simulation with 10 uA/cm2 current for 100ms"),
        ("Task → WormBase", "Look up the gene eat-4 on WormBase"),
    ]

    for label, query in tests:
        print(f"\n{'=' * 70}")
        print(f"TEST: {label}")
        print(f"QUERY: {query}")
        print(f"{'=' * 70}")
        try:
            state = await assistant.run_graph_invoke_state({"query": query})
            print(f"\nClassified as: {state.get('query_type', '?')}")
            answer = state.get("message_for_user", "No answer")
            print(f"\nANSWER:\n{answer[:500]}")
            refs = state.get("reference_material", {})
            if refs:
                print(f"\nReferences: {len(refs)} retrieval queries")
        except Exception as e:
            print(f"\nERROR: {e}")
            traceback.print_exc()

    print(f"\n{'=' * 70}")
    print("All tests complete.")


if __name__ == "__main__":
    asyncio.run(test())
