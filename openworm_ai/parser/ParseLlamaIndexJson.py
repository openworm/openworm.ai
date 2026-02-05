from openworm_ai import print_
from openworm_ai.parser.DocumentModels import Document, Section, Paragraph
from openworm_ai.parser.llamaparse_backend import generate_raw_json

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


json_output_dir = "processed/json/papers"
markdown_output_dir = "processed/markdown/papers"
plaintext_output_dir = "processed/plaintext/papers"

PDF_FOLDER = Path("corpus/papers/test/pdfs")

MANIFEST_PATH = Path("processed") / "manifest.json"

RAW_JSON_DIR = Path("corpus/papers/test/raw_json")
RAW_JSON_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_REGISTRY_PATH = Path("corpus/papers/source_registry.json")


def parse_args():
    parser = argparse.ArgumentParser(description="Parse PDFs with LlamaParse")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--skip", action="store_true", help="Don't call LlamaParse")
    mode.add_argument(
        "--reparse-all", action="store_true", help="Ignore manifest - reparse all PDFs"
    )

    parser.add_argument(
        "--max-age-days",
        type=int,
        default=30,
        help="Refresh PDFs that haven't been parsed for more than N days (default 30).",
    )

    return parser.parse_args()


def load_source_registry(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"papers": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(doc_model, file_name, json_output_dir):
    Path(json_output_dir).mkdir(parents=True, exist_ok=True)
    Path(markdown_output_dir).mkdir(parents=True, exist_ok=True)
    Path(plaintext_output_dir).mkdir(parents=True, exist_ok=True)

    file_path = Path(f"{json_output_dir}/{file_name}")
    doc_model.to_json_file(file_path)
    print_(f"  JSON file saved at: {file_path}")

    md_file_path = Path(f"{markdown_output_dir}/{file_name.replace('.json', '.md')}")
    doc_model.to_markdown(md_file_path)
    print_(f"  Markdown file saved at: {md_file_path}")

    text_file_path = Path(
        f"{plaintext_output_dir}/{file_name.replace('.json', '.txt')}"
    )
    doc_model.to_plaintext(text_file_path)
    print_(f"  Plaintext file saved at: {text_file_path}")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"version": 1, "created_at": utc_now_iso(), "entries": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def check_llamaparse_success(raw_json_path: Path) -> None:
    """
    Minimal guard: some failures come back as {"detail": "..."} or [{"detail": "..."}].
    """
    data = json.loads(raw_json_path.read_text(encoding="utf-8"))

    if isinstance(data, dict) and "detail" in data:
        raise RuntimeError(f"LlamaParse failed for {raw_json_path}: {data['detail']}")

    if (
        isinstance(data, list)
        and len(data) > 0
        and isinstance(data[0], dict)
        and "detail" in data[0]
    ):
        raise RuntimeError(
            f"LlamaParse failed for {raw_json_path}: {data[0]['detail']}"
        )


def should_parse_pdf(
    pdf_path: Path, manifest: Dict[str, Any], max_age_days: int | None = None
) -> Optional[str]:
    pdf_key = pdf_path.as_posix()
    entries = manifest.get("entries", {})

    if pdf_key not in entries:
        return "new"

    entry = entries[pdf_key]
    prev_pdf = entry.get("pdf", {})

    st = pdf_path.stat()
    curr_mtime = st.st_mtime
    curr_size = st.st_size

    prev_mtime = prev_pdf.get("mtime")
    prev_size = prev_pdf.get("size")

    if prev_mtime != curr_mtime or prev_size != curr_size:
        return "changed"

    if max_age_days is not None:
        parsed_at_str = entry.get("parsed_at")
        if parsed_at_str:
            parsed_at = datetime.fromisoformat(parsed_at_str)
            age = datetime.now(timezone.utc) - parsed_at
            if age.days > max_age_days:
                return "stale"

    return None


def _extract_pages(payload: Dict[str, Any]) -> list:
    """
    Extract page data from LlamaParse output.

    The SDK/API returns v2 format:
    {
        "text": {"pages": [{"page_number": 1, "text": "raw OCR..."}, ...]},
        "markdown": {"pages": [{"page_number": 1, "markdown": "# Clean..."}, ...]},
        "items": [...]
    }

    We PRIORITIZE markdown.pages[].markdown (clean) over text.pages[].text (raw OCR).
    This is critical for RAG - clean text produces better embeddings.
    """
    # Handle list wrapper
    if isinstance(payload, list) and len(payload) > 0:
        payload = payload[0] if isinstance(payload[0], dict) else {}

    if not isinstance(payload, dict):
        return []

    pages = []

    # Get markdown pages (clean, well-formatted) - PREFERRED for RAG
    markdown_pages = {}
    if isinstance(payload.get("markdown"), dict):
        for p in payload["markdown"].get("pages", []):
            if isinstance(p, dict):
                page_num = p.get("page_number")
                markdown_pages[page_num] = p.get("markdown", "")

    # Get text pages (raw OCR with layout artifacts) - FALLBACK only
    text_pages = {}
    if isinstance(payload.get("text"), dict):
        for p in payload["text"].get("pages", []):
            if isinstance(p, dict):
                page_num = p.get("page_number")
                text_pages[page_num] = p.get("text", "")

    # Merge: prefer markdown, fall back to text
    all_page_nums = set(markdown_pages.keys()) | set(text_pages.keys())

    for page_num in sorted(all_page_nums):
        # Prioritize markdown (clean) over text (raw OCR)
        content = markdown_pages.get(page_num) or text_pages.get(page_num) or ""
        pages.append(
            {
                "page": page_num,
                "md": content,  # Store in 'md' field for consistency
            }
        )

    # Fallback: old CLI format with top-level pages array
    if not pages and isinstance(payload.get("pages"), list):
        for p in payload["pages"]:
            if isinstance(p, dict):
                pages.append(
                    {
                        "page": p.get("page"),
                        "md": p.get("md") or p.get("text") or "",
                    }
                )

    return pages


def convert_to_json(paper_ref, paper_info, output_dir):
    """
    Convert raw LlamaParse JSON to our internal Document model.

    Uses clean markdown for RAG-friendly output.
    """
    loc = Path(paper_info[0])
    print_(f"Converting: {loc}")

    with open(loc, "r", encoding="utf-8") as f:
        root = json.load(f)

    pages = _extract_pages(root)

    if not pages:
        print_(f"  WARNING: No pages found in {loc}")

    doc_model = Document(
        id=paper_ref,
        title=paper_ref.replace("_", " "),
        source=paper_info[1],
    )

    for page in pages:
        if not isinstance(page, dict):
            continue

        page_number = page.get("page")
        section_title = f"Page {page_number}" if page_number is not None else "Page"
        current_section = Section(section_title)

        # Get the markdown content (already prioritized in _extract_pages)
        page_content = (page.get("md") or "").strip()

        if page_content:
            current_section.paragraphs.append(Paragraph(page_content))

        # Only add sections with content
        if current_section.paragraphs:
            doc_model.sections.append(current_section)

    save_json(doc_model, f"{paper_ref}.json", output_dir)


def convert_pdf_via_api(paper_ref: str, pdf_path: str, source_url: str, manifest: Dict[str, Any]) -> None:
    pdf_loc = Path(pdf_path)
    raw_json_path = RAW_JSON_DIR / f"{paper_ref}.llamaparse_raw.json"

    print_(f"Generating raw JSON via API: {pdf_loc} -> {raw_json_path}")
    generate_raw_json(pdf_loc, raw_json_path)

    check_llamaparse_success(raw_json_path)

    paper_info = [str(raw_json_path), source_url]
    convert_to_json(paper_ref, paper_info, json_output_dir)

    pdf_key = Path(pdf_path).as_posix()
    pdf_stat = Path(pdf_path).stat()

    outputs = [
        Path(f"{json_output_dir}/{paper_ref}.json").as_posix(),
        Path(f"{markdown_output_dir}/{paper_ref}.md").as_posix(),
        Path(f"{plaintext_output_dir}/{paper_ref}.txt").as_posix(),
        raw_json_path.as_posix(),
    ]

    manifest["entries"][pdf_key] = {
        "paper_ref": paper_ref,
        "source_url": source_url,
        "parsed_at": utc_now_iso(),
        "pdf": {"mtime": pdf_stat.st_mtime, "size": pdf_stat.st_size},
        "outputs": outputs,
    }

    save_manifest(MANIFEST_PATH, manifest)
    print_(f"Manifest updated: {MANIFEST_PATH}")


def convert_existing_raw_json(
    paper_ref: str, raw_json_path: Path, source_url: str
) -> None:
    """
    Convert an existing raw JSON file without re-parsing.
    Useful for regenerating processed output from cached raw JSON.
    """
    print_(f"Converting existing raw JSON: {raw_json_path}")
    check_llamaparse_success(raw_json_path)
    paper_info = [str(raw_json_path), source_url]
    convert_to_json(paper_ref, paper_info, json_output_dir)


if __name__ == "__main__":
    args = parse_args()

    source_registry = load_source_registry(SOURCE_REGISTRY_PATH)

    papers_api = {}

    for pdf_path in PDF_FOLDER.glob("*.pdf"):
        paper_ref = pdf_path.stem
        paper_meta = source_registry.get("papers", {}).get(paper_ref, {})
        source_url = paper_meta.get("source_url", "") or ""
        papers_api[paper_ref] = [str(pdf_path), source_url]

    if not papers_api:
        print(f"WARNING: No PDFs found in {PDF_FOLDER.resolve()}")

    manifest = load_manifest(MANIFEST_PATH)

    for paper_ref, (pdf_path, source_url) in papers_api.items():
        pdf_loc = Path(pdf_path)

        if args.skip:
            # Skip parsing but still convert existing raw JSON
            raw_json_path = RAW_JSON_DIR / f"{paper_ref}.llamaparse_raw.json"
            if raw_json_path.exists():
                print(f"Converting existing (--skip): {raw_json_path}")
                convert_existing_raw_json(paper_ref, raw_json_path, source_url)
            else:
                print(f"Skipping (no raw JSON): {pdf_loc}")
            continue

        try:
            if args.reparse_all:
                print(f"Parsing (forced): {pdf_loc}")
                convert_pdf_via_api(paper_ref, pdf_path, source_url, manifest)
                continue

            reason = should_parse_pdf(pdf_loc, manifest, max_age_days=args.max_age_days)

            if reason is None:
                print(f"Skipping (fresh + unchanged): {pdf_loc}")
                continue

            print(f"Parsing ({reason}): {pdf_loc}")
            convert_pdf_via_api(paper_ref, pdf_path, source_url, manifest)

        except Exception as e:
            print_(f"!! Failed parsing {pdf_loc}: {e}")
            import traceback

            traceback.print_exc()
            continue

    print(f"PDF_FOLDER resolved: {PDF_FOLDER.resolve()}")
    print(f"PDF count: {len(list(PDF_FOLDER.glob('*.pdf')))}")
