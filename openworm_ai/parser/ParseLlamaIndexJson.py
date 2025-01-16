import json
from pathlib import Path

# This has to be altered accordingly
output_dir = "processed/final_json"


# Function to save JSON content
def save_json(content, file_name, output_dir):
    # Full path to the file
    file_path = Path(f"{output_dir}/{file_name}")

    # Write content to the the final json file
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(content, json_file, indent=4, ensure_ascii=False)

    print(f"  JSON file saved at: {file_path}")


# Function to process JSON and extract markdown content
def convert_to_json(paper_ref, paper_location, output_dir):
    loc = Path(paper_location)

    print(f"Converting: {loc}")

    # Load the input JSON file
    with open(loc, "r", encoding="utf-8") as JSON:
        json_dict = json.load(JSON)

    # Final JSON structure
    final_json = {
        f"{paper_ref}_page": {
            "title": f"{paper_ref}_page",
            "source": str(loc),
            "sections": {},
        }
    }

    # Process each page and its items
    for page in json_dict["pages"]:
        page_sections = []
        for item in page.get("items", []):
            # Only extract 'md' sections (this can be altered depending on the desired section we want to include)
            if "md" in item and item["md"].strip():
                page_sections.append({"contents": item["md"]})

        # Save sections by page (if there are any markdown sections)
        if page_sections:
            final_json[f"{paper_ref}_page"]["sections"][f"Page {page['page']}"] = {
                "paragraphs": page_sections
            }

    # Save the final JSON output
    save_json(final_json, f"{paper_ref}_final.json", output_dir)


# Main execution block
if __name__ == "__main__":
    papers = {
        "Donnelly_et_al_2013": "corpus/papers/test/Donnelly2013_Llamaparse_Accurate.pdf.json"
    }

    # Loop through papers and process markdown sections
    for paper in papers:
        convert_to_json(paper, papers[paper], output_dir)

# If we dont want to write out the papers individually.
# Found a glob.glob technique but I remember you using something else.

# if __name__ == "__main__":
# Dynamically load all JSON files from the folder
# input_dir = "openworm.ai/processed/markdown/wormatlas"
# papers = {Path(file).stem: file for file in glob.glob(f"{input_dir}/*.json")}

# Loop through papers and process markdown sections
# for paper_ref, paper_location in papers.items():
# convert_to_json(paper_ref, paper_location, output_dir)
