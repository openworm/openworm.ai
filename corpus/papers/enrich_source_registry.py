import os
import json
import requests

# paths
REGISTRY_PATH = "corpus/papers/source_registry.json"

# env var for Semantic Scholar key (optional but recommended)
S2_API_KEY = os.getenv("S2_API_KEY") or os.getenv("SEMANTIC_SCHOLAR_API_KEY")

# Semantic Scholar endpoint: paper lookup by DOI
# Use: /paper/DOI:{doi}
BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/DOI:{}"
FIELDS = "paperId,title,year,authors,venue,url"


# load your existing registry JSON
with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
    registry = json.load(f)

papers = registry.get("papers", {})

headers = {"accept": "application/json"}
if S2_API_KEY:
    headers["x-api-key"] = S2_API_KEY


for paper_ref, meta in papers.items():
    doi = (meta.get("doi") or "").strip()

    if not doi:
        print(f"Skipping {paper_ref} (no doi)")
        continue

    url = BASE_URL.format(doi)
    params = {"fields": FIELDS}

    print(f"Fetching {paper_ref} DOI={doi}")
    r = requests.get(url, params=params, headers=headers, timeout=60)

    if r.status_code >= 400:
        print(f"  FAILED {paper_ref}: {r.status_code} {r.text}")
        continue

    data = r.json()

    # extract fields
    paper_id = data.get("paperId", "")
    title = data.get("title", "")
    year = data.get("year", "")
    venue = data.get("venue", "")
    authors = data.get("authors", []) or []
    author_names = [a.get("name", "") for a in authors if isinstance(a, dict)]
    s2_url = data.get("url", "")

    # authors -> last name(s) only
    if len(author_names) == 0:
        author_str = "Unknown"
    elif len(author_names) == 1:
        author_str = author_names[0].split()[-1]
    else:
        author_str = f"{author_names[0].split()[-1]} et al."

    citation = f"{author_str}, {year}"


    # write back into your registry structure
    meta["semantic_paperId"] = paper_id
    meta["title"] = title
    meta["year"] = year
    meta["authors"] = author_names
    meta["venue"] = venue
    meta["semantic_url"] = s2_url
    meta["citation"] = citation


# dump updated registry back to disk
with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
    json.dump(registry, f, ensure_ascii=False, indent=2)

print(f"\nDone. Updated: {REGISTRY_PATH}")
