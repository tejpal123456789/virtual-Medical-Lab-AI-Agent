import logging
import logging.config
from pathlib import Path
import yaml


def get_logger(name: str = "medical_science") -> logging.Logger:
    config_path = Path(__file__).parents[2] / "config" / "logging_config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        logging.config.dictConfig(cfg)
    return logging.getLogger(name) 