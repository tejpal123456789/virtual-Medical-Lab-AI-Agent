from typing import Optional
from medical_science.config import load_all_configs


_PROMPTS = None


def _ensure_loaded() -> None:
    global _PROMPTS
    if _PROMPTS is None:
        _PROMPTS = load_all_configs().get("prompts", {})


def get_template(name: str) -> Optional[str]:
    _ensure_loaded()
    return _PROMPTS.get(name) 