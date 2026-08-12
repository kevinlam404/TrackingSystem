from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml

REQUIRED_SECTIONS = (
    "camera",
    "detector",
    "behavior",
    "control",
    "gimbal",
    "interface",
)


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError("The configuration must contain a YAML mapping")

    missing_sections = [
        section
        for section in REQUIRED_SECTIONS
        if section not in config
    ]

    if missing_sections:
        missing = ", ".join(missing_sections)
        raise ValueError(
            f"Missing configuration sections: {missing}"
        )

    return config