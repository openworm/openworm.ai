from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")


def get_llama_api_key() -> str:
    key = os.environ.get("LLAMA_CLOUD_API_KEY")
    if not key:
        raise RuntimeError(
            "LLAMA_CLOUD_API_KEY is not set. "
            "Export it in your shell or define it in a .env file."
        )
    return key


async def _parse_async(pdf_path: Path, json_output_path: Path) -> None:
    """
    Core async logic: upload PDF via SDK, parse it, save raw JSON output.
    Uses the llama_cloud>=1.0 SDK (AsyncLlamaCloud) instead of raw requests.
    """
    try:
        from llama_cloud import AsyncLlamaCloud  # pip install llama_cloud>=1.0
    except ImportError as exc:
        raise ImportError(
            "llama_cloud>=1.0 is required to call LlamaParse. "
            "Run: pip install 'llama_cloud>=1.0'"
        ) from exc

    client = AsyncLlamaCloud(api_key=get_llama_api_key())

    tier = os.environ.get("LLAMAPARSE_TIER", "cost_effective")

    print(f"[llamaparse] uploading: {pdf_path.name} (tier={tier})", flush=True)

    # Step 1: upload the file
    with open(pdf_path, "rb") as f:
        file_obj = await client.files.create(file=f, purpose="parse")

    print(f"[llamaparse] uploaded, file_id={file_obj.id}", flush=True)

    # Step 2: parse with our preferred options
    result = await client.parsing.parse(
        file_id=file_obj.id,
        tier=tier,
        version="latest",
        output_options={
            "markdown": {
                "tables": {
                    "compact_markdown_tables": True,
                    "output_tables_as_markdown": True,
                },
                "annotate_links": False,
                "inline_images": False,
            }
        },
        processing_options={
            "ignore": {
                # Reduces noise from two-column PDF layouts
                "ignore_diagonal_text": True,
            }
        },
        # CHANGED: added "items" so per-paragraph structure is preserved.
        # items gives us individual text blocks per page rather than one
        # large page blob, which produces much better RAG chunk granularity.
        # text is kept as a fallback only.
        expand=["text", "markdown", "items"],
    )

    print("[llamaparse] parse complete, building output", flush=True)

    # Step 3: serialise into the raw JSON shape that convert_to_json expects.
    # The old item-level pipeline reads items[*]["md"] as individual paragraphs.
    output = {}

    if result.markdown:
        output["markdown"] = {
            "pages": [
                {"page_number": p.page_number, "markdown": p.markdown}
                for p in result.markdown.pages
            ]
        }

    if result.text:
        output["text"] = {
            "pages": [
                {"page_number": p.page_number, "text": p.text}
                for p in result.text.pages
            ]
        }

    # ADDED: serialise items into the same per-page structure.
    # Each page's items list contains individual text blocks (paragraphs,
    # headings, table rows etc.) which convert_to_json reads as separate
    # Paragraph objects — giving us the fine-grained chunking we want.
    if result.items:
        output["items"] = {
            "pages": [
                {
                    "page_number": p.page_number,
                    "items": [
                        # Each item has type, md, text fields.
                        # We preserve all fields so convert_to_json can
                        # choose between md and text as needed.
                        item.model_dump() if hasattr(item, "model_dump") else dict(item)
                        for item in p.items
                    ],
                }
                for p in result.items.pages
            ]
        }

    if not output:
        debug_path = json_output_path.with_suffix(".debug.json")
        debug_path.write_text(
            json.dumps({"error": "no content in result"}, indent=2),
            encoding="utf-8",
        )
        raise RuntimeError(
            f"No content returned from LlamaParse. Debug info saved to {debug_path}"
        )

    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    json_output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[llamaparse] saved to {json_output_path}", flush=True)


def generate_raw_json(pdf_path: str | Path, json_output_path: str | Path) -> None:
    """
    Public entry point (synchronous) — matches the signature expected by
    the rest of the pipeline (parse_pdfs.py).
    """
    asyncio.run(_parse_async(Path(pdf_path), Path(json_output_path)))
