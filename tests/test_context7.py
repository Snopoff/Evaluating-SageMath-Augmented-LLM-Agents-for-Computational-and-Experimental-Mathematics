import unittest
from unittest.mock import AsyncMock, patch

import hydra.utils as hu
import rootutils
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from omegaconf import OmegaConf

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)


from src.tools.context7 import default_context7_server_name, load_context7_tools  # noqa: E402


class Context7LoaderTests(unittest.TestCase):
    def test_hydra_instantiates_multiserver_mcp_client(self) -> None:
        cfg = OmegaConf.create(
            {
                "_target_": "langchain_mcp_adapters.client.MultiServerMCPClient",
                "tool_name_prefix": False,
                "connections": {
                    "context7": {
                        "transport": "http",
                        "url": "https://mcp.context7.com/mcp",
                        "headers": {"Authorization": "Bearer ctx7"},
                    }
                },
            }
        )

        client = hu.instantiate(cfg)

        self.assertIsInstance(client, MultiServerMCPClient)
        self.assertEqual(client.connections["context7"]["transport"], "http")
        self.assertEqual(client.connections["context7"]["url"], "https://mcp.context7.com/mcp")
        self.assertEqual(client.connections["context7"]["headers"]["Authorization"], "Bearer ctx7")

    def test_default_context7_server_name_uses_first_connection(self) -> None:
        client = MultiServerMCPClient(
            {
                "context7": {
                    "transport": "http",
                    "url": "https://mcp.context7.com/mcp",
                }
            }
        )

        self.assertEqual(default_context7_server_name(client), "context7")

    def test_load_context7_tools_returns_name_map(self) -> None:
        @tool("query-docs")
        def query_docs(libraryId: str, query: str) -> str:  # noqa: N803
            """Query docs."""

            return f"{libraryId}:{query}"

        @tool("resolve-library-id")
        def resolve_library_id(query: str) -> str:
            """Resolve library id."""

            return query

        client = MultiServerMCPClient(
            {
                "context7": {
                    "transport": "http",
                    "url": "https://mcp.context7.com/mcp",
                }
            }
        )
        with patch.object(client, "get_tools", AsyncMock(return_value=[query_docs, resolve_library_id])) as mocked_get_tools:
            loaded = load_context7_tools(client)

        self.assertEqual(set(loaded), {"query-docs", "resolve-library-id"})
        self.assertIs(loaded["query-docs"], query_docs)
        self.assertIs(loaded["resolve-library-id"], resolve_library_id)
        mocked_get_tools.assert_awaited_once_with(server_name="context7")


if __name__ == "__main__":
    unittest.main()
