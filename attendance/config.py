from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")
    root = config_path.parent
    for key, value in config.get("paths", {}).items():
        candidate = Path(value)
        config["paths"][key] = candidate if candidate.is_absolute() else root / candidate
    return config, root
