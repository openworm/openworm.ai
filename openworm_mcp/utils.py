#!/usr/bin/env python3
"""
MCP utils — auto-registers tool functions from modules.

Originally from neuroml-ai: neuroml_mcp/utils.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import inspect


def register_tools(mcp, modules: list):
    """Register tools from a given module

    :param modules: list of modules with tool function definitions

    """
    for module in modules:
        for fname, fn in inspect.getmembers(module, inspect.isfunction):
            if fname.endswith("_tool"):
                mcp.tool(fn)
