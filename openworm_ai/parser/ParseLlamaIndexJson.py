from openworm_ai import print_
from openworm_ai.parser.DocumentModels import Document, Section, Paragraph

import json
from pathlib import Path

# This has to be altered accordingly
json_output_dir = "processed/json/papers"
markdown_output_dir = "processed/markdown/papers"
plaintext_output_dir = "processed/plaintext/papers"


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
    loc = Path(paper_info[0])

    print_(f"Converting: {loc}")

    # Load the input JSON file
    with open(loc, "r", encoding="utf-8") as JSON:
        json_dict = json.load(JSON)

    doc_model = Document(
        id=paper_ref, title=paper_ref.replace("_", " "), source=paper_info[1]
    )

    # Process each page and its items
    for page in json_dict["pages"]:
        page_sections = []
        current_section = Section(f"Page {page['page']}")
        for item in page.get("items", []):
            # Only extract 'md' sections (this can be altered depending on the desired section we want to include)
            if "md" in item and item["md"].strip():
                page_sections.append({"contents": item["md"]})
                current_section.paragraphs.append(Paragraph(item["md"]))

        # Save sections by page (if there are any markdown sections)
        if page_sections:
            doc_model.sections.append(current_section)

    # Save the final JSON output
    save_json(doc_model, f"{paper_ref}.json", output_dir)


# Main execution block
if __name__ == "__main__":
    papers = {
        "Donnelly_et_al_2013": [
            "corpus/papers/test/Donnelly2013_Llamaparse_Accurate.pdf.json",
            "https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1001529",
        ],
        "Randi_et_al_2023": [
            "corpus/papers/test/Randi2023_Llamaparse_Accurate.pdf.json",
            "https://www.nature.com/articles/s41586-023-06683-4",
        ],
        "Corsi_et_al_2015": [
            "corpus/papers/test/PrimerOnCElegans.pdf.json",
            "https://academic.oup.com/genetics/article/200/2/387/5936175",
        ],
        "Sinha_et_al_2025": [
            "corpus/papers/test/SinhaEtAl2025.pdf.json",
            "https://elifesciences.org/articles/95135",
        ],
    }

    # Loop through papers and process markdown sections
    for paper in papers:
        convert_to_json(paper, papers[paper], json_output_dir)

# If we dont want to write out the papers individually.
# Found a glob.glob technique but I remember you using something else.

# if __name__ == "__main__":
# Dynamically load all JSON files from the folder
# input_dir = "openworm.ai/processed/markdown/wormatlas"
# papers = {Path(file).stem: file for file in glob.glob(f"{input_dir}/*.json")}

# Loop through papers and process markdown sections
# for paper_ref, paper_location in papers.items():
# convert_to_json(paper_ref, paper_location, output_dir)
