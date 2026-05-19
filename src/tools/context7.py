from __future__ import annotations

import asyncio

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

CONTEXT7_RESOLVE_LIBRARY_ID_TOOL_NAME = "resolve-library-id"
CONTEXT7_QUERY_DOCS_TOOL_NAME = "query-docs"
CONTEXT7_TOOL_NAMES = {
    CONTEXT7_RESOLVE_LIBRARY_ID_TOOL_NAME,
    CONTEXT7_QUERY_DOCS_TOOL_NAME,
}


def default_context7_server_name(client: MultiServerMCPClient) -> str:
    if not client.connections:
        raise ValueError("Context7 MCP client has no configured connections.")
    return next(iter(client.connections))


def load_context7_tools(
    client: MultiServerMCPClient,
    *,
    server_name: str | None = None,
) -> dict[str, BaseTool]:
    resolved_server_name = server_name or default_context7_server_name(client)
    tools = asyncio.run(client.get_tools(server_name=resolved_server_name))
    return {tool.name: tool for tool in tools}
