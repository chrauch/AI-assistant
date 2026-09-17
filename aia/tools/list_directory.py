# This file is part of AIA. Copyright (C) 2026 Christian Rauch.
# Distributed under terms of the GPL3 license.

from pathlib import Path
from typing import Any

from .base import Tool


class ListDirectoryTool(Tool):
    name = "list_directory"
    description = (
        "List files and directories. Without external access, paths must stay "
        "inside data/. With external access, relative paths resolve from the "
        "launch directory and absolute paths are allowed. "
        "Directories end with '/'. By default, only direct entries are returned."
    )
    parameters = {
        "type": "object",
        "properties": {
            "directory": {
                "type": "string",
                "description": "Directory path to list.",
            },
            "recursive": {
                "type": "boolean",
                "description": "Whether to include entries in nested directories.",
                "default": False,
            },
        },
        "required": ["directory"],
        "additionalProperties": False,
    }

    data_dir = Path(__file__).parent.parent / "data"

    def run(
        self,
        directory: str,
        recursive: bool = False,
        allow_external_files: bool = False,
        **_: Any,
    ) -> str:
        data_dir = self.data_dir.resolve()
        requested_directory = Path(directory).expanduser()
        if not requested_directory.is_absolute():
            base_directory = Path.cwd() if allow_external_files else data_dir
            requested_directory = base_directory / requested_directory
        requested_directory = requested_directory.resolve()
        if not allow_external_files:
            try:
                requested_directory.relative_to(data_dir)
            except ValueError as error:
                raise ValueError(
                    "Directory path must remain inside the data directory"
                ) from error
        if not requested_directory.is_dir():
            raise ValueError(f"Directory not found: {directory}")

        paths = (
            requested_directory.rglob("*")
            if recursive
            else requested_directory.iterdir()
        )
        entries = sorted(
            self._display_path(path, data_dir)
            for path in paths
            if path.is_file() or path.is_dir()
            if allow_external_files or self._is_inside_data_dir(path, data_dir)
        )
        return "\n".join(entries)

    @staticmethod
    def _display_path(path: Path, data_dir: Path) -> str:
        relative_path = path.relative_to(data_dir).as_posix()
        return f"{relative_path}/" if path.is_dir() else relative_path

    @staticmethod
    def _is_inside_data_dir(path: Path, data_dir: Path) -> bool:
        try:
            path.resolve().relative_to(data_dir)
        except ValueError:
            return False
        return True