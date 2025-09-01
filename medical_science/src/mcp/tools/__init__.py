from typing import Dict, Callable
from .ping_tool import run as ping


def load_tools() -> Dict[str, Callable]:
    return {
        "ping": ping,
    }

__all__ = ["load_tools"] 