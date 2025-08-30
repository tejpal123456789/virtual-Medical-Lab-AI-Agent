from typing import Dict, Any, Callable
from medical_science.src.mcp.tools import load_tools


class MCPClient:
    def __init__(self) -> None:
        self._tools: Dict[str, Callable] = load_tools()

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def invoke(self, name: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
        tool = self._tools.get(name)
        if not tool:
            return {"error": f"tool '{name}' not found"}
        return tool(payload) 