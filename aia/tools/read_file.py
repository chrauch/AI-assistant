# This file is part of AIA. Copyright (C) 2026 Christian Rauch.
# Distributed under terms of the GPL3 license.

from pathlib import Path
from typing import Any

from .base import Tool


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Read UTF-8 text from a file. Without external access, paths must stay "
        "inside data/. With external access, relative paths resolve from the "
        "launch directory and absolute paths are allowed."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to a UTF-8 text file.",
            }
        },
        "required": ["file_path"],
        "additionalProperties": False,
    }

    data_dir = Path(__file__).parent.parent / "data"

    def run(
        self,
        file_path: str,
        allow_external_files: bool = False,
        **_: Any,
    ) -> str:
        data_dir = self.data_dir.resolve()
        requested_path = Path(file_path).expanduser()
        if not requested_path.is_absolute():
            base_directory = Path.cwd() if allow_external_files else data_dir
            requested_path = base_directory / requested_path
        requested_path = requested_path.resolve()
        if not allow_external_files:
            try:
                requested_path.relative_to(data_dir)
            except ValueError as error:
                raise ValueError(
                    "File path must remain inside the data directory"
                ) from error
        if not requested_path.is_file():
            raise ValueError(f"File not found: {file_path}")
        try:
            return requested_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("File is not valid UTF-8 text") from error
