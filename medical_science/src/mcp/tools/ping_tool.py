from typing import Any, Dict


def run(payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {"status": "ok", "echo": payload or {}} 