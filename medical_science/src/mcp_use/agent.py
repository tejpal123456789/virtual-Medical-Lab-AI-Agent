from typing import Dict, Any
from .client import MCPClient


class MCPAgent:
    def __init__(self, client: MCPClient | None = None) -> None:
        self.client = client or MCPClient()

    def tools(self) -> list[str]:
        return self.client.list_tools()

    def call(self, tool: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return self.client.invoke(tool, payload) 