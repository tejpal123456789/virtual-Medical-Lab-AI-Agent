from typing import Protocol, Any


class Tool(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ... 