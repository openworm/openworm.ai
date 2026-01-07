from openworm_ai import print_
from openworm_ai.parser.DocumentModels import Document, Section, Paragraph
from openworm_ai.parser.llamaparse_backend import generate_raw_json

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict
import argparse


json_output_dir = "processed/json/papers"
markdown_output_dir = "processed/markdown/papers"
plaintext_output_dir = "processed/plaintext/papers"

PDF_FOLDER = Path("corpus/papers/test/pdfs")

MANIFEST_PATH = Path("processed") / "manifest.json"

RAW_JSON_DIR = Path("corpus/papers/test/raw_json")
RAW_JSON_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Parse PDFs with Llamaparse - updates")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--skip", action="store_true", help="Don't call Llamaparse (no extra parsing)"
    )
    mode.add_argument(
        "--reparse-all", action="store_true", help="Ignore manifest - reparse all pdfs!"
    )

    parser.add_argument(
        "--max-age-days",
        type=int,
        default=30,
        help="Refresh PDFs that haven't been parsed for more than N days (default set to 30 days) - keep updated with latest Llamaparse updates",
    )

    return parser.parse_args()

    # Function to save JSON content


def save_json(doc_model, file_name, json_output_dir):
    # Full path to the file
    file_path = Path(f"{json_output_dir}/{file_name}")

    # Write content to the final json file
    # with open(file_path, "w", encoding="utf-8") as json_file:
    # json.dump(content, json_file, indent=4, ensure_ascii=False)
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


def convert_to_json(paper_ref, paper_info, output_dir):
    """
    Take a raw LlamaParse JSON file (from either the old UI export or the new CLI/API),
    normalise its structure, and convert it into our internal Document model.

    Strategy (to match the old behaviour more closely):
    - Try to use page["items"][*]["md"] / ["text"] as individual paragraphs/headings.
    - If there are no usable items, fall back to page["md"] or page["text"].
    """

    loc = Path(paper_info[0])
    print_(f"Converting: {loc}")

    # Load the input JSON file
    with open(loc, "r", encoding="utf-8") as JSON:
        root = json.load(JSON)

    # ---- Normalise to a list of page dicts ----
    # Case 1: old UI format -> {"pages": [ ... ]}
    if isinstance(root, dict) and "pages" in root:
        pages = root["pages"]

    # Case 2: new CLI/API format -> [ { "pages": [ ... ], ... } ]
    elif isinstance(root, list):
        if len(root) == 0:
            pages = []
        elif isinstance(root[0], dict) and "pages" in root[0]:
            pages = root[0]["pages"]
        else:
            # Fallback: assume the list itself is already a list of page dicts
            pages = root
    else:
        print_(
            f"  WARNING: Unexpected JSON structure in {loc}: "
            f"top-level type={type(root)}"
        )
        pages = []

    # ---- Build Document model ----
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

        paragraphs_added = 0

        # 1) Preferred path: use item-level content (like the old pipeline)
        items = page.get("items", [])
        for item in items:
            if not isinstance(item, dict):
                continue

            text = item.get("md") or item.get("text")
            if not text:
                continue

            text = text.strip()
            if not text:
                continue

            # Skip noisy sentinel strings
            if "CURRENT_PAGE_RAW_OCR_TEXT" in text:
                continue

            current_section.paragraphs.append(Paragraph(text))
            paragraphs_added += 1

        # 2) Fallback: if no good items, use page-level md/text once
        if paragraphs_added == 0:
            page_md = page.get("md")
            page_text = page.get("text")

            fallback = (page_md or page_text or "").strip()
            if fallback:
                # Also strip the sentinel if it appears at page level
                fallback = fallback.replace("CURRENT_PAGE_RAW_OCR_TEXT", "").strip()
                if fallback:
                    current_section.paragraphs.append(Paragraph(fallback))

        # Only add non-empty sections
        if current_section.paragraphs:
            doc_model.sections.append(current_section)

    # Save the final JSON + markdown + plaintext outputs
    save_json(doc_model, f"{paper_ref}.json", output_dir)


# Block to create pdf manifest - details of if or when a pdf has already been parsed - for when we want to reparse all, only parse new ones, or parse no new pdfs in a run


# function to give us the timestamp for a pdf that we can use in our manifest
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# loading manifest (as a dict) - if it exists, or if not, then creates blank manifest that parsed pdf can then fill
# return json etc - loads file as str (path.read_text), then converts str JSON into python dict - dict can then be modified - e.g when reparsing
def load_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"version": 1, "created_at": utc_now_iso(), "entries": {}}
    return json.loads(path.read_text(encoding="utf-8"))


# saving updated manifest [entries] once pdf is parsed function
# json.dumps converts the updated dict back to str JSON, path.write_text writes the updated manifest to memory
def save_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


# dict allows quick [entries] lookup to check if pdf is parsed, compared to list which would need to be scanned through to check - saves time as dict


def check_llamaparse_success(raw_json_path: Path) -> None:
    """
    If LlamaParse failed, it often writes a JSON object like:
      {"detail": "Invalid API Key ..."}
    This helper detects that and fails fast so we don't convert junk into a Document.
    """
    data = json.loads(raw_json_path.read_text(encoding="utf-8"))

    # Common error shape
    if isinstance(data, dict) and "detail" in data:
        raise RuntimeError(f"LlamaParse failed for {raw_json_path}: {data['detail']}")

    # Defensive: sometimes list-wrapped
    if (
        isinstance(data, list)
        and len(data) > 0
        and isinstance(data[0], dict)
        and "detail" in data[0]
    ):
        raise RuntimeError(
            f"LlamaParse failed for {raw_json_path}: {data[0]['detail']}"
        )


# Function to determine whether a pdf should be parsed on any run - based on if and when it has already been parsed
def should_parse_pdf(
    pdf_path: Path, manifest: Dict[str, Any], max_age_days: int | None = None
) -> bool:
    pdf_key = pdf_path.as_posix()
    entries = manifest.get("entries", {})

    # New PDF
    if pdf_key not in entries:
        return "new"

    entry = entries[pdf_key]

    # Checking if PDF has changed - reparse
    prev_pdf = entry.get("pdf", {})

    st = pdf_path.stat()
    curr_mtime = st.st_mtime
    curr_size = st.st_size

    prev_mtime = prev_pdf.get("mtime")
    prev_size = prev_pdf.get("size")

    if prev_mtime != curr_mtime or prev_size != curr_size:
        return "changed"

    # Reparse PDFs if last parse was 30 days or more old
    if max_age_days is not None:
        parsed_at_str = entry.get("parsed_at")

        if parsed_at_str:
            parsed_at = datetime.fromisoformat(parsed_at_str)
            age = datetime.now(timezone.utc) - parsed_at

            if age.days > max_age_days:
                return "stale"

    return None


def convert_pdf_via_api(
    paper_ref: str, pdf_path: str, source_url: str, manifest: Dict[str, Any]
) -> None:
    """
    1. Use the llama-parse CLI (via generate_raw_json) to generate a raw JSON file
       from the input PDF. This raw JSON has the same structure as the JSON that
       would be downloaded from the LlamaParse web UI before its additional post-processing.
    2. Try to use a function to follow the same processing to split up json into markdown elements
    3. Feed that raw JSON now processed into the existing convert_to_json() function, which
       builds a Document model and saves JSON/markdown/plaintext outputs.
    """

    pdf_loc = Path(pdf_path)

    raw_json_path = RAW_JSON_DIR / f"{paper_ref}.llamaparse_raw.json"

    print_(f"Generating raw JSON via API: {pdf_loc} -> {raw_json_path}")
    generate_raw_json(pdf_loc, raw_json_path)

    # Stop if API error
    check_llamaparse_success(raw_json_path)

    # Reuse the existing convert_to_json workflow:
    # paper_info[0] = path to raw JSON, paper_info[1] = source URL
    paper_info = [str(raw_json_path), source_url]
    convert_to_json(paper_ref, paper_info, json_output_dir)

    pdf_key = Path(pdf_path).as_posix()

    # 2) record what files we produced
    outputs = [
        Path(f"{json_output_dir}/{paper_ref}.json").as_posix(),
        Path(f"{markdown_output_dir}/{paper_ref}.md").as_posix(),
        Path(f"{plaintext_output_dir}/{paper_ref}.txt").as_posix(),
        raw_json_path.as_posix(),
    ]

    pdf_stat = Path(pdf_path).stat()

    # 3) create/update the entry in the manifest
    manifest["entries"][pdf_key] = {
        "paper_ref": paper_ref,
        "source_url": source_url,
        "parsed_at": utc_now_iso(),
        "pdf": {
            "mtime": pdf_stat.st_mtime,
            "size": pdf_stat.st_size,
        },
        "outputs": outputs,
    }

    # 4) save to disk
    save_manifest(MANIFEST_PATH, manifest)
    print_(f"Manifest updated: {MANIFEST_PATH}")


# Main execution block
if __name__ == "__main__":
    # Legacy path (using pre-downloaded UI JSON) – kept for reference:
    # papers_ui = {
    #     "Donnelly_et_al_2013": [
    #         "corpus/papers/test/Donnelly2013_Llamaparse_Accurate.pdf.json",
    #         "https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1001529",
    #     ],
    #     ...
    # }
    # for paper_ref, paper_info in papers_ui.items():
    #     convert_to_json(paper_ref, paper_info, json_output_dir)

    # New API-based path: start from PDFs instead of UI JSON

    args = parse_args()

    papers_api = {}

    for pdf_path in PDF_FOLDER.glob("*.pdf"):
        # Convert file name to a clean reference ID
        # Example: "Donnelly2013.pdf" -> "Donnelly2013"
        paper_ref = pdf_path.stem

        # No source URL available unless we add metadata later
        source_url = ""

        papers_api[paper_ref] = [str(pdf_path), source_url]

    if not papers_api:
        print(f"WARNING: No PDFs found in {PDF_FOLDER.resolve()}")

    manifest = load_manifest(MANIFEST_PATH)

    MAX_AGE_DAYS = 30  # monthly refresh

    # Loop through papers and process via the API-backed pipeline
    for paper_ref, (pdf_path, source_url) in papers_api.items():
        pdf_loc = Path(pdf_path)

        # 1: skip everything
        if args.skip:
            print(f"Skipping parse (--skip): {pdf_loc}")
            continue

        # 2: force reparse

        try:
            if args.reparse_all:
                print(f"Parsing (forced): {pdf_loc}")
                convert_pdf_via_api(paper_ref, pdf_path, source_url, manifest)
                continue

            # If the PDF has been parsed less than 30 days ago, skip it - only incremental parsing + monthly refresh
            reason = should_parse_pdf(pdf_loc, manifest, max_age_days=MAX_AGE_DAYS)

            if reason is None:
                print(f"Skipping (fresh + unchanged): {pdf_loc}")
                continue

            print(f"Parsing ({reason}): {pdf_loc}")
            convert_pdf_via_api(paper_ref, pdf_path, source_url, manifest)

        except Exception as e:
            print_(f"!! Failed parsing {pdf_loc}: {e}")
            # IMPORTANT: do not update manifest on failure (your code already avoids this)
            continue

    print(f"PDF_FOLDER resolved: {PDF_FOLDER.resolve()}")
    print(f"PDF count: {len(list(PDF_FOLDER.glob('*.pdf')))}")


# If we dont want to write out the papers individually.
# Found a glob.glob technique but I remember you using something else.

# if __name__ == "__main__":
# Dynamically load all JSON files from the folder
# input_dir = "openworm.ai/processed/markdown/wormatlas"
# papers = {Path(file).stem: file for file in glob.glob(f"{input_dir}/*.json")}

# Loop through papers and process markdown sections
# for paper_ref, paper_location in papers.items():
# convert_to_json(paper_ref, paper_location, output_dir)
