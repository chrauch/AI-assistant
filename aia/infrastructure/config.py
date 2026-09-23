# This file is part of AIA. Copyright (C) 2026 Christian Rauch.
# Distributed under terms of the GPL3 license.

import json
from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
CONFIG_FILE = PROJECT_DIR / "config.json"


@dataclass(frozen=True)
class Config:
    model_id: str
    editor: str
    file_explorer: str
    models_dir: Path
    commands_dir: Path
    context_dir: Path
    safe_context: bool
    allow_external_files: bool
    show_progress: bool
    log_dir: Path


def load_config() -> Config:
    try:
        with CONFIG_FILE.open(encoding="utf-8") as config_file:
            values = json.load(config_file)
    except FileNotFoundError:
        values = {}
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read {CONFIG_FILE}: {error}") from error
    if not isinstance(values, dict):
        raise ValueError(f"Configuration must be a JSON object: {CONFIG_FILE}")

    return Config(
        model_id=_string_value(values, "model_id", "local-model"),
        editor=_string_value(values, "editor", "nano"),
        file_explorer=_string_value(values, "file_explorer", "mc"),
        models_dir=_path_value(values, "models_dir", "models"),
        commands_dir=_path_value(values, "commands_dir", "commands"),
        context_dir=_path_value(values, "context_dir", "contexts"),
        safe_context=_bool_value(values, "safe_context", True),
        allow_external_files=_bool_value(
            values,
            "allow_external_files",
            False,
        ),
        show_progress=_bool_value(values, "show_progress", False),
        log_dir=_path_value(values, "log_dir", "logs"),
    )


def _string_value(values: dict[str, object], key: str, default: str) -> str:
    value = values.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Configuration value '{key}' must be a non-empty string")
    return value


def _path_value(values: dict[str, object], key: str, default: str) -> Path:
    value = _string_value(values, key, default)
    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


def _bool_value(values: dict[str, object], key: str, default: bool) -> bool:
    value = values.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"Configuration value '{key}' must be a boolean")
    return value