from typing import Dict, Callable


class AgentRegistry:
    def __init__(self) -> None:
        self._registry: Dict[str, Callable] = {}

    def register(self, name: str, fn: Callable) -> None:
        self._registry[name] = fn

    def get(self, name: str) -> Callable | None:
        return self._registry.get(name) 