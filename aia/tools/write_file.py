# This file is part of AIA. Copyright (C) 2026 Christian Rauch.
# Distributed under terms of the GPL3 license.

from pathlib import Path
import time
from typing import Any

from .base import Tool


class WriteFileTool(Tool):
    name = "write_file"
    description = (
        "Write UTF-8 text to a new timestamped AI-generated file. Without "
        "external access, paths must stay inside data/. With external access, "
        "relative paths resolve from the launch directory and absolute paths "
        "are allowed."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Requested filename for the generated file.",
            },
            "content": {
                "type": "string",
                "description": "UTF-8 text to write to the generated file.",
            },
        },
        "required": ["file_path", "content"],
        "additionalProperties": False,
    }

    data_dir = Path(__file__).parent.parent / "data"

    def run(
        self,
        file_path: str,
        content: str,
        allow_external_files: bool = False,
        **_: Any,
    ) -> str:
        data_dir = self.data_dir.resolve()
        requested_path = Path(file_path).expanduser()
        if not requested_path.is_absolute():
            base_directory = Path.cwd() if allow_external_files else data_dir
            requested_path = base_directory / requested_path
        parent_directory = requested_path.parent.resolve()
        if not allow_external_files:
            try:
                parent_directory.relative_to(data_dir)
            except ValueError as error:
                raise ValueError(
                    "File path must remain inside the data directory"
                ) from error
        if not parent_directory.is_dir():
            raise ValueError(f"Directory not found: {requested_path.parent}")
        if not requested_path.name or requested_path.name in {".", ".."}:
            raise ValueError("File path must name a file")

        timestamp = time.strftime("%Y%m%dT%H%M%S", time.localtime())
        generated_name = (
            f"{requested_path.name}.AI-gen_{timestamp}_{time.time_ns()}"
            f"{requested_path.suffix}"
        )
        generated_path = parent_directory / generated_name
        try:
            with generated_path.open("x", encoding="utf-8") as file:
                file.write(content)
        except OSError as error:
            raise ValueError(f"Could not write generated file: {error}") from error
        try:
            return generated_path.relative_to(data_dir).as_posix()
        except ValueError:
            return str(generated_path)
