from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable, List

from llama_parse import LlamaParse


def get_llama_api_key() -> str:
    """
    Read the API key from the environment and fail fast if it is missing.
    """
    key = os.environ.get("LLAMA_CLOUD_API_KEY")
    if not key:
        raise RuntimeError(
            "LLAMA_CLOUD_API_KEY is not set. "
            "Export it in your shell or define it in a .env file."
        )
    return key


def generate_raw_json(pdf_path: str | Path, json_output_path: str | Path) -> None:
    """
    Use the llama-parse CLI to generate the same raw JSON output that the
    LlamaParse web UI would produce.

    This calls:
        llama-parse <pdf_path> --output-raw-json --output-file <json_output_path>

    The JSON structure is what ParseLlamaIndexJson.py already expects.
    """
    pdf_path = Path(pdf_path)
    json_output_path = Path(json_output_path)

    cmd = [
        "llama-parse",
        str(pdf_path),
        "--output-raw-json",
        "--output-file",
        str(json_output_path),
    ]

    # This will raise CalledProcessError if something goes wrong,
    # which is usually what we want when building the corpus.
    subprocess.run(cmd, check=True)



