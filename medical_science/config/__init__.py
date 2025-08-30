import os
import yaml
from pathlib import Path
from typing import Any, Dict

CONFIG_DIR = Path(__file__).parent


def _load_yaml(file_name: str) -> Dict[str, Any]:
    file_path = CONFIG_DIR / file_name
    if not file_path.exists():
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_all_configs() -> Dict[str, Any]:
    return {
        "model": _load_yaml("model_config.yaml"),
        "prompts": _load_yaml("prompt_templates.yaml"),
        "logging": _load_yaml("logging_config.yaml"),
    }


__all__ = ["load_all_configs"] 