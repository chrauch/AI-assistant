# This file is part of AIA. Copyright (C) 2026 Christian Rauch.
# Distributed under terms of the GPL3 license.

import json
import shutil
import time
from pathlib import Path
from typing import Any

from aia.domain.context import Context
from aia.infrastructure.output import out


class ContextStore:
    SETTINGS_FILE = "settings.json"

    def __init__(self, context_dir: Path, safe: bool) -> None:
        self.context_dir = context_dir
        self.safe = safe
        self.context_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, Context]:
        contexts: dict[str, Context] = {}
        for context_path in self.context_dir.iterdir():
            if not context_path.is_dir():
                continue
            context_files = self.active_files(context_path.name)
            if not context_files:
                context_files = sorted(
                    (
                        file_path
                        for file_path in context_path.glob("*.json")
                        if not file_path.name.endswith(".backup.json")
                        and file_path.name != self.SETTINGS_FILE
                    ),
                    reverse=True,
                )
            if not context_files:
                continue
            try:
                with context_files[0].open() as file:
                    contexts[context_path.name] = Context.from_dict(json.load(file))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                if self.safe:
                    raise
        return contexts

    def latest_file(self, context_id: str) -> Path | None:
        context_path = self.context_dir / context_id
        context_files = [
            file_path
            for file_path in context_path.glob("*.json")
            if (
                file_path.is_file()
                and not file_path.name.endswith(".backup.json")
                and file_path.name != self.SETTINGS_FILE
            )
        ]
        return max(context_files, default=None, key=lambda path: path.name)

    def active_files(self, context_id: str) -> list[Path]:
        context_path = self.context_dir / context_id
        return sorted(
            (
                file_path
                for file_path in context_path.glob("*.json")
                if not file_path.name.endswith(".backup.json")
                and not file_path.name.endswith(".compacted.json")
                and file_path.name != self.SETTINGS_FILE
            ),
            reverse=True,
        )

    def save(
        self,
        contexts: dict[str, Context],
        context_id: str,
        compacted: bool = False,
    ) -> Path | None:
        context_path = self.context_dir / context_id
        try:
            context_path.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%dT%H%M%S", time.localtime())
            context_file = context_path / (
                f"context_{timestamp}_{time.time_ns()}.json"
            )
            temporary_file = context_path / ".context.json.tmp"
            with temporary_file.open("w") as file:
                json.dump(contexts[context_id].to_dict(), file, indent=2)
            temporary_file.replace(context_file)
            if not compacted:
                for old_file in self.active_files(context_id):
                    if old_file != context_file:
                        old_file.unlink()
                return context_file

            timestamp = time.strftime("%Y%m%dT%H%M%S", time.localtime())
            compacted_file = context_path / (
                f"context_{timestamp}_{time.time_ns()}.compacted.json"
            )
            shutil.copy2(context_file, compacted_file)
            for old_file in self.active_files(context_id):
                old_file.unlink()
            return compacted_file
        except OSError:
            if self.safe:
                raise
            out(
                "SYS",
                f"Warning: could not persist context {context_id}.",
                context_id=context_id,
            )
            return None

    def load_settings(self, context_id: str) -> dict[str, Any]:
        settings_file = self.context_dir / context_id / self.SETTINGS_FILE
        try:
            with settings_file.open() as file:
                settings = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            settings = {}
        if not isinstance(settings, dict):
            return {}
        return settings

    def save_settings(self, context_id: str, settings: dict[str, Any]) -> None:
        context_path = self.context_dir / context_id
        context_path.mkdir(parents=True, exist_ok=True)
        settings_file = context_path / self.SETTINGS_FILE
        temporary_file = context_path / ".settings.json.tmp"
        with temporary_file.open("w") as file:
            json.dump(settings, file, indent=2)
        temporary_file.replace(settings_file)

    def memory_dir(self, context_id: str) -> Path:
        memory_path = self.context_dir / context_id / "memory"
        memory_path.mkdir(parents=True, exist_ok=True)
        return memory_path

    def backup(self, context_id: str) -> Path | None:
        context_path = self.context_dir / context_id
        try:
            context_files = self.active_files(context_id)
            if not context_files:
                return None
            timestamp = time.strftime("%Y%m%dT%H%M%S", time.localtime())
            backup_file = context_path / (
                f"context_{timestamp}_{time.time_ns()}.backup.json"
            )
            shutil.copy2(context_files[0], backup_file)
            return backup_file
        except OSError:
            if self.safe:
                raise
            out(
                "SYS",
                f"Warning: could not back up context {context_id}.",
                context_id=context_id,
            )
            return None

    def delete(self, contexts: dict[str, Context], context_id: str) -> None:
        contexts.pop(context_id, None)
        context_path = self.context_dir / context_id
        if context_path.is_dir():
            for file_path in context_path.iterdir():
                if file_path.is_file():
                    file_path.unlink()
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
            context_path.rmdir()

    def delete_all(self, contexts: dict[str, Context]) -> None:
        contexts.clear()
        for context_path in self.context_dir.iterdir():
            if context_path.is_dir():
                for file_path in context_path.iterdir():
                    if file_path.is_file():
                        file_path.unlink()
                    elif file_path.is_dir():
                        shutil.rmtree(file_path)
                context_path.rmdir()

    def truncate(self, contexts: dict[str, Context], context_id: str) -> int:
        if context_id not in contexts:
            raise ValueError(f"Unknown context ID: {context_id}")
        context_path = self.context_dir / context_id
        latest_file = self.latest_file(context_id)
        if latest_file is None:
            return 0
        deleted_count = 0
        for file_path in context_path.glob("*.json"):
            if (
                file_path != latest_file
                and file_path.name != self.SETTINGS_FILE
                and file_path.is_file()
            ):
                file_path.unlink()
                deleted_count += 1
        return deleted_count

    def timestamp(self, context_id: str) -> str:
        context_file = self.latest_file(context_id)
        if context_file is None:
            return ""
        filename_parts = context_file.name.split("_")
        if filename_parts[0] == "context" and len(filename_parts) > 1:
            return filename_parts[1]
        return filename_parts[0]

    def last_response(self, context_id: str) -> str:
        context_file = self.latest_file(context_id)
        if context_file is None:
            return ""
        try:
            with context_file.open() as file:
                messages: list[dict[str, Any]] = json.load(file).get("messages", [])
        except (OSError, json.JSONDecodeError, TypeError, AttributeError):
            return ""
        responses = [
            message.get("content", "")
            for message in messages
            if message.get("role") == "assistant"
        ]
        if not responses:
            return ""
        response = " ".join(str(responses[-1]).split())
        return f"{response[:30]}..." if len(response) > 30 else response
