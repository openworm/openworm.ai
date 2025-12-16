from openworm_ai import print_
from openworm_ai.parser.DocumentModels import Document, Section, Paragraph
from openworm_ai.parser.llamaparse_backend import generate_raw_json

import json
from pathlib import Path

json_output_dir = "processed/json/papers"
markdown_output_dir = "processed/markdown/papers"
plaintext_output_dir = "processed/plaintext/papers"

PDF_FOLDER = Path("corpus/papers/tests")

# Function to save JSON content
def save_json(doc_model, file_name, json_output_dir):
    # Full path to the file
    file_path = Path(f"{json_output_dir}/{file_name}")

    # Write content to the the final json file
    # with open(file_path, "w", encoding="utf-8") as json_file:
    #    json.dump(content, json_file, indent=4, ensure_ascii=False)
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


# Function to process JSON and extract markdown content
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

def convert_pdf_via_api(paper_ref: str, pdf_path: str, source_url: str) -> None:
    """
    1. Use the llama-parse CLI (via generate_raw_json) to generate a raw JSON file
       from the input PDF. This raw JSON has the same structure as the JSON that
       would be downloaded from the LlamaParse web UI before its additional post-processing.
    2. Try to use a function to follow the same processing to split up json into markdown elements
    3. Feed that raw JSON now processed into the existing convert_to_json() function, which
       builds a Document model and saves JSON/markdown/plaintext outputs.
    """
    pdf_loc = Path(pdf_path)

    # Put the raw JSON next to the PDF, with a clear suffix.
    # e.g. Donnelly2013.pdf -> Donnelly2013.llamaparse_raw.json
    raw_json_path = pdf_loc.with_suffix(".llamaparse_raw.json")

    print_(f"Generating raw JSON via API: {pdf_loc} -> {raw_json_path}")
    generate_raw_json(pdf_loc, raw_json_path)

    # Reuse the existing convert_to_json workflow:
    # paper_info[0] = path to raw JSON, paper_info[1] = source URL
    paper_info = [str(raw_json_path), source_url]
    convert_to_json(paper_ref, paper_info, json_output_dir)


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
    
    """""
    papers_api = {
        "Donnelly_et_al_2013": [
            "corpus/papers/test/Donnelly2013.pdf",
            "https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1001529",
        ],
        "Randi_et_al_2023": [
            "corpus/papers/test/Randi2023.pdf",
            "https://www.nature.com/articles/s41586-023-06683-4",
        ],
        "Corsi_et_al_2015": [
            "corpus/papers/test/PrimerOnCElegans.pdf",
            "https://academic.oup.com/genetics/article/200/2/387/5936175",
        ],
        "Sinha_et_al_2025": [
            "corpus/papers/test/SinhaEtAl2025.pdf",
            "https://elifesciences.org/articles/95135",
        ],
        "Wang_et_al_2024": [
            "corpus/papers/test/Wang2024_NeurotransmitterAtlas.pdf",
            "https://elifesciences.org/articles/95402" ,
        ],
    }
    """"

    papers_api = {}

    for pdf_path in PDF_FOLDER.glob("*.pdf"):
        # Convert file name to a clean reference ID
        # Example: "Donnelly2013.pdf" → "Donnelly2013"
        paper_ref = pdf_path.stem

        # No source URL available unless we add metadata later
        source_url = ""

        papers_api[paper_ref] = [str(pdf_path), source_url]

    # Loop through papers and process via the API-backed pipeline
    for paper_ref, (pdf_path, source_url) in papers_api.items():
        convert_pdf_via_api(paper_ref, pdf_path, source_url)


# If we dont want to write out the papers individually.
# Found a glob.glob technique but I remember you using something else.

# if __name__ == "__main__":
# Dynamically load all JSON files from the folder
# input_dir = "openworm.ai/processed/markdown/wormatlas"
# papers = {Path(file).stem: file for file in glob.glob(f"{input_dir}/*.json")}

# Loop through papers and process markdown sections
# for paper_ref, paper_location in papers.items():
# convert_to_json(paper_ref, paper_location, output_dir)
