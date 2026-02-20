from openworm_ai import print_
from openworm_ai.parser.DocumentModels import Document, Section, Paragraph
from openworm_ai.parser.llamaparse_backend import generate_raw_json

import argparse
import json
import re
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


def _clean_markdown(text: str) -> str:
    """
    Sanitize LaTeX artifacts from markdown output while preserving content.
    Strips wrappers (e.g. $...$, \\text{}) but keeps the text inside so
    chemical/ionic notation like Ca^{2+} remains searchable in RAG.
    """
    # Remove display math delimiters, keep inner content
    text = re.sub(r"\$\$(.+?)\$\$", r"\1", text, flags=re.DOTALL)
    # Remove inline math delimiters, keep inner content
    text = re.sub(r"\$(.+?)\$", r"\1", text)

    # Strip common LaTeX text/formatting wrappers, keep content
    text = re.sub(r"\\text\{(.+?)\}", r"\1", text)
    text = re.sub(r"\\mathrm\{(.+?)\}", r"\1", text)
    text = re.sub(r"\\mathbf\{(.+?)\}", r"\1", text)
    text = re.sub(r"\\boldsymbol\{(.+?)\}", r"\1", text)

    # Remove bracket/paren sizing commands (purely visual, no content)
    text = re.sub(r"\\left[\[\(]|\\right[\]\)]", "", text)

    # Simplify brace-wrapped super/subscripts to readable form
    # e.g. ^{2+} -> ^2+,  _{i} -> _i
    text = re.sub(r"\^\{(.+?)\}", r"^\1", text)
    text = re.sub(r"_\{(.+?)\}", r"_\1", text)

    # Remove stray backslashes before punctuation characters
    text = re.sub(r"\\([,;:!%])", r"\1", text)

    # Convert HTML superscripts to plain readable form e.g. <sup>2+</sup> -> ^2+
    text = re.sub(r"<sup>(.*?)</sup>", r"^\1", text)
    # Strip any other residual HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Strip mermaid diagram blocks — not useful for RAG
    text = re.sub(r"```mermaid.*?```", "", text, flags=re.DOTALL)

    # Collapse runs of 3+ blank lines introduced by the removals above
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _is_noise_paragraph(text: str) -> bool:
    s = text.strip()
    if not s:
        return True

    if re.match(r"^\d+\s+of\s+\d+$", s):
        return True

    if re.match(r"^[\w\s,\.]+eLife\s+\d{4}.*DOI:\s*$", s):
        return True

    if re.match(r"^\((?:PG|RAS|[A-Z]{1,4})\);?\\?$", s):
        return True

    # CHANGED: require sentence-ending punctuation at the END of the string,
    # not just anywhere — "Boyle et al." has a dot but isn't a sentence
    if (
        len(s) <= 60
        and not s.startswith("#")
        and not s.startswith("|")
        and not s.startswith(">")
        and not re.search(r"[.?!:,]\s*$", s)  # must END with punctuation
    ):
        return True

    return False


def _extract_pages(payload: Dict[str, Any]) -> list:
    """
    Extract page data from LlamaParse output, preferring item-level granularity.

    Priority order:
      1. items — individual text blocks per page (best RAG chunk granularity)
      2. markdown — full page blob, sanitized (fallback when no items)
      3. text — raw OCR (last resort)
    """
    # Handle list wrapper
    if isinstance(payload, list) and len(payload) > 0:
        payload = payload[0] if isinstance(payload[0], dict) else {}

    if not isinstance(payload, dict):
        return []

    pages = []

    # Build lookup dicts keyed by page_number
    markdown_pages = {}
    if isinstance(payload.get("markdown"), dict):
        for p in payload["markdown"].get("pages", []):
            if isinstance(p, dict):
                markdown_pages[p.get("page_number")] = p.get("markdown", "")

    text_pages = {}
    if isinstance(payload.get("text"), dict):
        for p in payload["text"].get("pages", []):
            if isinstance(p, dict):
                text_pages[p.get("page_number")] = p.get("text", "")

    # items give per-paragraph granularity — one chunk per text block
    # rather than one chunk per page, much better for RAG retrieval
    items_pages = {}
    if isinstance(payload.get("items"), dict):
        for p in payload["items"].get("pages", []):
            if isinstance(p, dict):
                items_pages[p.get("page_number")] = p.get("items", [])

    all_page_nums = (
        set(markdown_pages.keys()) | set(text_pages.keys()) | set(items_pages.keys())
    )

    for page_num in sorted(all_page_nums):
        items = items_pages.get(page_num, [])

        if items:
            # Item-level path — one entry per text block, cleaned individually
            page_items = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                text = item.get("md") or item.get("text") or ""
                text = _clean_markdown(text.strip())
                if text and "CURRENT_PAGE_RAW_OCR_TEXT" not in text:
                    page_items.append(text)
            if page_items:
                pages.append({"page": page_num, "items": page_items})
                continue

        # Fallback to page-level markdown blob when items are absent
        md_content = markdown_pages.get(page_num, "")
        txt_content = text_pages.get(page_num, "")
        content = _clean_markdown(md_content) if md_content else txt_content
        if content:
            pages.append({"page": page_num, "md": content})

    # Fallback: old CLI format with top-level pages array
    if not pages and isinstance(payload.get("pages"), list):
        for p in payload["pages"]:
            if isinstance(p, dict):
                pages.append(
                    {
                        "page": p.get("page"),
                        "md": _clean_markdown(p.get("md") or p.get("text") or ""),
                    }
                )

    return pages


def convert_to_json(paper_ref, paper_info, output_dir):
    """
    Convert raw LlamaParse JSON to our internal Document model.

    Uses item-level content when available (one Paragraph per text block),
    falling back to full page markdown blob when items are absent.
    Each Paragraph carries lightweight provenance metadata (section_id,
    paragraph_index, page_number) for RAG chunk referencing. doc_id and
    source are omitted from Paragraph as they are already on the Document.
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

        if "items" in page:
            # Item-level path: one Paragraph per text block
            for para_idx, item_text in enumerate(page["items"]):
                if not _is_noise_paragraph(item_text):
                    current_section.paragraphs.append(
                        Paragraph(
                            contents=item_text,
                            paragraph_index=para_idx,
                            page_number=page_number,
                        )
                    )
        else:
            # Page-level fallback: single Paragraph for the whole page blob
            page_content = (page.get("md") or "").strip()
            if page_content and not _is_noise_paragraph(page_content):
                current_section.paragraphs.append(
                    Paragraph(
                        contents=page_content,
                        paragraph_index=0,
                        page_number=page_number,
                    )
                )

        if current_section.paragraphs:
            doc_model.sections.append(current_section)

    save_json(doc_model, f"{paper_ref}.json", output_dir)


def convert_pdf_via_api(
    paper_ref: str, pdf_path: str, source_url: str, manifest: Dict[str, Any]
) -> None:
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
