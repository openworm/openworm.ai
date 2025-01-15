# from openworm_ai.parser.DocumentModels import Document, Section, Paragraph

import json
from pathlib import Path

# MARKDOWN_DIR = "processed/markdown/wormatlas"
# PLAINTEXT_DIR = "processed/plaintext/wormatlas"

JSON_DIR = "processed/markdown/papers"

CORPUS_LOCATION = "corpus"


def convert_to_model(paper_ref, paper_location):
    loc = Path(paper_location)

    print("Converting: %s" % loc)

    with open(loc, "r", encoding="utf-8") as JSON:
        json_dict = json.load(JSON)

    for page in json_dict["pages"]:
        print(f"Page {page['page']} has {len(page['items'])} items")


if __name__ == "__main__":
    papers = {}

    papers["Donnelly et al. 2013"] = (
        "corpus/papers/test/Donnelly2013_Llamaparse_Accurate.pdf.json"
    )

    for paper in papers:
        convert_to_model(paper, papers[paper])
