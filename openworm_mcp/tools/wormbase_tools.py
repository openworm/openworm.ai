#!/usr/bin/env python3
"""
Tools for querying the WormBase REST API for C. elegans biology data.

Note that docstrings here should be written for the LLM to read.

Originally from neuroml-ai: openworm_mcp/tools/wormbase_tools.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from typing import Any, Dict

import requests

WORMBASE_REST_BASE = "https://rest.wormbase.org/rest"
WORMBASE_PARASITE_BASE = "https://parasite.wormbase.org/rest"
ALLIANCE_BASE = "https://www.alliancegenome.org/api"
HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}

logger = logging.getLogger(__name__)


def _get(url: str) -> Dict[str, Any]:
    """Internal helper to make a GET request and return parsed JSON or error dict."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else None
        return {"error": f"HTTP {status_code} error: {e}", "url": url, "status_code": status_code}
    except requests.exceptions.ConnectionError:
        return {"error": "Could not connect to WormBase API. Check network connection.", "url": url}
    except requests.exceptions.Timeout:
        return {"error": "WormBase API request timed out.", "url": url}
    except Exception as e:
        return {"error": f"Unexpected error: {e}", "url": url}


def _resolve_gene_id(gene_id: str) -> str:
    """Resolve a common gene name to a WBGene ID.

    If gene_id already looks like a WBGene ID (starts with 'WBGene'), returns
    it unchanged. Otherwise tries ParaSite first, then falls back to the
    Alliance of Genome Resources API.

    Returns the WBGene ID if found, or the original gene_id if resolution fails.
    """
    if gene_id.startswith("WBGene"):
        return gene_id

    # Try ParaSite first (original approach)
    try:
        url = f"{WORMBASE_PARASITE_BASE}/lookup/symbol/caenorhabditis_elegans/{gene_id}"
        r = requests.get(url, headers=HEADERS, timeout=10, allow_redirects=True)
        r.raise_for_status()
        data = r.json()
        wb_id = data.get("id")
        if wb_id and wb_id.startswith("WBGene"):
            logger.info(f"Resolved gene name '{gene_id}' → {wb_id} (via ParaSite)")
            return wb_id
    except Exception as e:
        logger.warning(f"ParaSite lookup failed for '{gene_id}': {e}")

    # Fallback: Alliance of Genome Resources API
    try:
        url = f"{ALLIANCE_BASE}/search_autocomplete"
        params = {"q": gene_id, "category": "gene", "taxonId": "NCBITaxon:6239", "limit": 1}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        results = r.json().get("results", [])
        if results:
            primary_key = results[0].get("primaryKey", "")
            # Alliance returns "WB:WBGene00001135" format — strip the "WB:" prefix
            wb_id = primary_key.replace("WB:", "") if primary_key.startswith("WB:") else primary_key
            if wb_id.startswith("WBGene"):
                logger.info(f"Resolved gene name '{gene_id}' → {wb_id} (via Alliance)")
                return wb_id
    except Exception as e:
        logger.warning(f"Alliance lookup also failed for '{gene_id}': {e}")

    return gene_id


async def query_wormbase_gene_tool(
    gene_id: str,
    field: str = "overview",
) -> Dict[str, Any]:
    """Query the WormBase REST API for information about a C. elegans gene.

    Use this tool to retrieve biological data about a specific C. elegans gene
    from the WormBase database. WormBase is the primary curated resource for
    C. elegans genetics, genomics, and biology.

    Gene IDs use the format WBGene00000000 (e.g. WBGene00000179 for unc-10).
    Common gene names (e.g. eat-4, unc-17, glr-1) can also be used as gene_id
    but WBGene IDs are more reliable.

    Inputs:

    - gene_id (str): WormBase gene identifier (e.g. "WBGene00001250") or
      common gene name (e.g. "eat-4"). Required.
    - field (str, default "overview"): which data field to retrieve.

      Available fields:
      - "overview": general gene info, name, sequence, description
      - "phenotype": phenotypes associated with loss-of-function mutants
      - "expression": tissue and cell expression data
      - "function": gene ontology (GO) terms, molecular function
      - "genetics": alleles, polymorphisms
      - "homology": orthologues and paralogues in other species
      - "interactions": genetic and physical interactions with other genes
      - "references": published papers about this gene

    Output:

    Dictionary containing the requested field data from WormBase.
    The structure varies by field but always includes either:
    - The requested biological data as nested dicts/lists
    - An "error" key with an error message if the query failed

    Examples:

    - Get overview of eat-4 gene: query_wormbase_gene_tool(gene_id="eat-4")
    - Get phenotypes for unc-17: query_wormbase_gene_tool(gene_id="unc-17", field="phenotype")
    - Get expression pattern of glr-1: query_wormbase_gene_tool(gene_id="glr-1", field="expression")
    - Get gene interactions: query_wormbase_gene_tool(gene_id="WBGene00001250", field="interactions")
    - Get published papers: query_wormbase_gene_tool(gene_id="eat-4", field="references")
    """
    # Resolve common gene names (eat-4, unc-17) to WBGene IDs
    resolved_id = _resolve_gene_id(gene_id)

    # Use widget endpoint for overview (field endpoint doesn't serve it),
    # use field endpoint for specific fields like concise_description, name, etc.
    if field == "overview":
        url = f"{WORMBASE_REST_BASE}/widget/gene/{resolved_id}/{field}"
    else:
        url = f"{WORMBASE_REST_BASE}/field/gene/{resolved_id}/{field}"

    result = _get(url)

    # If query failed with resolved ID, include the resolution info for debugging
    if "error" in result and resolved_id != gene_id:
        result["resolved_id"] = resolved_id
        result["original_id"] = gene_id

    return result


async def query_wormbase_neuron_tool(
    neuron_name: str,
    field: str = "overview",
) -> Dict[str, Any]:
    """Query the WormBase REST API for information about a specific C. elegans neuron.

    Use this tool to retrieve anatomical and functional data about a specific
    neuron or neuron class in C. elegans from WormBase. C. elegans has exactly
    302 neurons with well-characterised identities.

    Neuron names follow standard C. elegans nomenclature, e.g.:
    - AWC, AWA, AWB (chemosensory neurons)
    - AIY, AIZ, AIA (interneurons)
    - DB1-DB7, VB1-VB11 (motor neurons)
    - AVAL, AVAR, AVBL, AVBR (command interneurons)

    Inputs:

    - neuron_name (str): standard C. elegans neuron or anatomy term name.
      Examples: "ADAL", "AWC", "AIY", "DB1", "RIA". Required.
    - field (str, default "overview"): which data field to retrieve.

      Available fields:
      - "overview": general description, lineage, position
      - "anatomy_function": known functions of this neuron
      - "expressed_in": genes expressed in this neuron
      - "innervates": synaptic partners and connectivity
      - "references": published papers about this neuron

    Output:

    Dictionary containing the requested field data from WormBase.
    Always includes either the requested data or an "error" key.

    Examples:

    - Get overview of AWC neuron: query_wormbase_neuron_tool(neuron_name="AWCON")
    - Get function of AIY: query_wormbase_neuron_tool(neuron_name="AIYL", field="anatomy_function")
    - Get genes expressed in RIA: query_wormbase_neuron_tool(neuron_name="RIAL", field="expressed_in")
    """
    if field == "overview":
        url = f"{WORMBASE_REST_BASE}/widget/anatomy_term/{neuron_name}/{field}"
    else:
        url = f"{WORMBASE_REST_BASE}/field/anatomy_term/{neuron_name}/{field}"

    result = _get(url)
    return result


async def query_wormbase_phenotype_tool(
    phenotype_id: str,
) -> Dict[str, Any]:
    """Query the WormBase REST API for information about a C. elegans phenotype.

    Use this tool to look up a specific phenotype in WormBase and retrieve
    which genes are associated with it, along with descriptions.

    Phenotype IDs use the format WBPhenotype:0000000.
    Common phenotype terms include things like:
    - "uncoordinated" (Unc phenotype)
    - "paralysed"
    - "slow growth"
    - "embryonic lethal"

    Inputs:

    - phenotype_id (str): WormBase phenotype identifier in format
      "WBPhenotype:0000000" or a descriptive term like "uncoordinated". Required.

    Output:

    Dictionary with phenotype overview data including:
    - Associated genes
    - Phenotype description
    - Related phenotypes
    Or an "error" key if the query failed.

    Examples:

    - Look up uncoordinated phenotype: query_wormbase_phenotype_tool(phenotype_id="WBPhenotype:0000643")
    - Look up paralysed phenotype: query_wormbase_phenotype_tool(phenotype_id="WBPhenotype:0000475")
    """
    url = f"{WORMBASE_REST_BASE}/widget/phenotype/{phenotype_id}/overview"
    result = _get(url)
    return result


async def search_wormbase_tool(
    query: str,
    entity_type: str = "gene",
    limit: int = 5,
) -> Dict[str, Any]:
    """Search WormBase for C. elegans genes, neurons, or other biological entities by name or keyword.

    Use this tool when you have a gene name, partial name, or keyword and need
    to find the corresponding WormBase identifier or confirm an entity exists.
    This is useful before calling query_wormbase_gene_tool or query_wormbase_neuron_tool
    when you are unsure of the exact identifier.

    Inputs:

    - query (str): search term, e.g. a gene name, keyword, or partial name. Required.
      Examples: "eat-4", "glutamate", "acetylcholine", "chemosensory"
    - entity_type (str, default "gene"): type of entity to search for.
      Options: "gene", "anatomy", "phenotype", "protein", "variation"
    - limit (int, default 5): maximum number of results to return (1-20).

    Output:

    Dictionary with keys:
    - hits (list): list of matching entities, each with name and WormBase ID
    - total (int): total number of matches found
    - error (str): error message if query failed

    Examples:

    - Search for eat-4 gene: search_wormbase_tool(query="eat-4")
    - Search for glutamate genes: search_wormbase_tool(query="glutamate", entity_type="gene")
    - Search for chemosensory neurons: search_wormbase_tool(query="chemosensory", entity_type="anatomy")
    - Find uncoordinated phenotypes: search_wormbase_tool(query="uncoordinated", entity_type="phenotype")
    """
    # Use ParaSite lookup for gene name resolution (more reliable than WormBase search)
    if entity_type == "gene":
        resolved_id = _resolve_gene_id(query)
        if resolved_id.startswith("WBGene"):
            # Successfully resolved — return structured result
            url = f"{WORMBASE_REST_BASE}/field/gene/{resolved_id}/name"
            name_result = _get(url)
            if "error" not in name_result:
                name_data = name_result.get("name", {}).get("data", {})
                return {
                    "hits": [{
                        "name": name_data.get("label", query),
                        "id": resolved_id,
                        "class": "gene",
                    }],
                    "total": 1,
                }

    # Fallback: try WormBase REST search (may return HTML on some endpoints)
    try:
        url = f"{WORMBASE_REST_BASE}/search/{entity_type}/{requests.utils.quote(query)}"
        params = {"limit": limit}
        r = requests.get(url, headers=HEADERS, params=params, timeout=10)
        r.raise_for_status()
        content_type = r.headers.get("content-type", "")
        if "json" not in content_type:
            return {"error": "WormBase search returned non-JSON response. Try using a WBGene ID directly.", "hits": [], "total": 0}
        return r.json()
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP error: {e}", "hits": [], "total": 0}
    except requests.exceptions.ConnectionError:
        return {"error": "Could not connect to WormBase API.", "hits": [], "total": 0}
    except requests.exceptions.Timeout:
        return {"error": "WormBase API request timed out.", "hits": [], "total": 0}
    except Exception as e:
        return {"error": f"Unexpected error: {e}", "hits": [], "total": 0}
