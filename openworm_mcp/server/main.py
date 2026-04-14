#!/usr/bin/env python3
"""
MCP server for OpenWorm tools (HH simulation, WormBase queries).

Run with: python -m openworm_mcp.server.main
"""

import asyncio
from textwrap import dedent

from fastmcp import FastMCP

from openworm_mcp.tools import hh_tools, wormbase_tools
from openworm_mcp.utils import register_tools


async def create_server():
    """Create and configure the MCP server."""
    usage = dedent(
        """
        OpenWorm assistant server.

        Provides tools for:
        - Hodgkin-Huxley neuron simulations (based on openworm/hodgkin_huxley_tutorial)
        - WormBase REST API queries for C. elegans biology data
        """
    )
    mcp = FastMCP("openworm_MCP", instructions=usage)
    register_tools(mcp, [hh_tools, wormbase_tools])
    return mcp


def main():
    """Main runner method."""
    mcp = asyncio.run(create_server())
    mcp.run(transport="streamable-http", port=8543)


if __name__ == "__main__":
    main()
