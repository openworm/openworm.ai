import json
from pathlib import Path
import os

# Define the output directory for saving markdown files
output_dir = r"C:\Users\janku\OneDrive\Documentos\GitHub\openworm.ai\processed\final_md"

# Function to save markdown content
def save_markdown(content, file_name, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the full path for the file
    file_path = os.path.join(output_dir, file_name)
    
    # Write content to the file
    with open(file_path, "w", encoding="utf-8") as md_file:
        md_file.write(content)
    
    print(f"Markdown file saved at: {file_path}")

# Function to process JSON and extract markdown content
def convert_to_model(paper_ref, paper_location):
    loc = Path(paper_location)

    print(f"Converting: {loc}")

    # Load the JSON file
    with open(loc, "r", encoding="utf-8") as JSON:
        json_dict = json.load(JSON)

    # Initialize a counter for markdown sections
    md_count = 0

    # Check for 'md' sections both at the page level and within 'items'
    for page in json_dict["pages"]:
        # Save 'md' content at the page level
        if "md" in page and page["md"].strip():
            md_count += 1
            save_markdown(page["md"], f"{paper_ref}_page_{page['page']}_level.md", output_dir)

        # Save 'md' content within the 'items' list
        md_items = [item for item in page.get('items', []) if 'md' in item and item['md'].strip()]
        
        # Save each item individually
        for idx, item in enumerate(md_items):
            md_count += 1
            save_markdown(item['md'], f"{paper_ref}_page_{page['page']}_item_{idx+1}.md", output_dir)

    print(f"Total markdown sections saved: {md_count}")

# Main execution block
if __name__ == "__main__":
    papers = {
        "Donnelly_et_al_2013": "corpus/papers/test/Donnelly2013_Llamaparse_Accurate.pdf.json"
    }

    # Loop through papers and process markdown sections
    for paper in papers:
        convert_to_model(paper, papers[paper])
