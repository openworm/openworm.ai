from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests
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


def generate_raw_json(pdf_path: str | Path, json_output_path: str | Path) -> None:
    """
    Use the llama-parse v2 API to generate raw JSON output.
    """
    pdf_path = Path(pdf_path)
    json_output_path = Path(json_output_path)
    json_output_path.parent.mkdir(parents=True, exist_ok=True)

    api_key = get_llama_api_key()
    headers = {"Authorization": f"Bearer {api_key}", "accept": "application/json"}

    upload_url = "https://api.cloud.llamaindex.ai/api/v2/parse/upload"

    configuration = {
        "tier": os.environ.get("LLAMAPARSE_TIER", "cost_effective"),
        "version": "latest",
        "output_options": {
            "markdown": {
                "tables": {
                    "compact_markdown_tables": True,
                    "output_tables_as_markdown": True,
                },
                "annotate_links": False,
                "inline_images": False,
            }
        },
    }

    print(f"[llamaparse] uploading: {pdf_path.name}", flush=True)

    with open(pdf_path, "rb") as f:
        files = {
            "file": (pdf_path.name, f, "application/pdf"),
            "configuration": (None, json.dumps(configuration)),
        }
        r = requests.post(upload_url, headers=headers, files=files, timeout=180)
    r.raise_for_status()
    resp = r.json()

    job_id = resp.get("id") or resp.get("job_id")
    if not job_id:
        raise RuntimeError(f"Unexpected response (no job id): {resp}")

    result_url = f"https://api.cloud.llamaindex.ai/api/v2/parse/{job_id}"

    deadline = time.time() + 1200  # 20 minutes
    poll_sleep_s = 3

    print(f"[llamaparse] polling job_id={job_id}", flush=True)

    # Use VALID expand parameters to get the content inlined
    # These are the valid ones from the error message you got earlier
    expand = "text,items,markdown"

    # Poll for completion
    response = None
    while time.time() < deadline:
        try:
            s = requests.get(
                result_url, headers=headers, params={"expand": expand}, timeout=60
            )
            if s.status_code >= 400:
                raise RuntimeError(f"Polling failed ({s.status_code}): {s.text}")

            response = s.json()

        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
        ) as e:
            print(f"[llamaparse] transient connection error: {e}", flush=True)
            time.sleep(5)
            continue

        job = response.get("job") or {}
        status = (job.get("status") or "").lower()
        print(f"[llamaparse] job.status={status}", flush=True)

        if status in {"completed", "complete", "done"}:
            break
        if status in {"error", "failed"}:
            raise RuntimeError(
                f"LlamaParse failed: {job.get('error_message') or response}"
            )

        time.sleep(poll_sleep_s)
    else:
        raise RuntimeError("Timed out waiting for parse")

    if not response:
        raise RuntimeError("No response received from API")

    # Now the content should be inlined
    print(f"[llamaparse] extracting content from response", flush=True)

    output = {}

    # Extract what we got
    if response.get("text"):
        output["text"] = response["text"]
        print(f"[llamaparse] extracted text", flush=True)

    if response.get("items"):
        output["items"] = response["items"]
        print(f"[llamaparse] extracted items", flush=True)

    if response.get("markdown"):
        output["markdown"] = response["markdown"]
        print(f"[llamaparse] extracted markdown", flush=True)

    if not output:
        # Save debug info
        debug_path = json_output_path.with_suffix(".debug.json")
        debug_path.write_text(
            json.dumps(response, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise RuntimeError(
            f"No content found after expanding. "
            f"Response keys: {list(response.keys())}. "
            f"Full response saved to {debug_path}"
        )

    # Save the output
    json_output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[llamaparse] saved to {json_output_path}", flush=True)
