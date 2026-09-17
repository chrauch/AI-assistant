# This file is part of AIA. Copyright (C) 2026 Christian Rauch.
# Distributed under terms of the GPL3 license.

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from secrets import token_hex
from typing import Any

from .base import Tool


class MemoryStore:
    def __init__(self, context_dir: Path | None, context_id: str | None) -> None:
        if context_dir is None or not context_id:
            raise ValueError("Memory tools require an active conversation context")
        root = Path(context_dir).resolve()
        context_path = (root / context_id).resolve()
        try:
            context_path.relative_to(root)
        except ValueError as error:
            raise ValueError("Invalid conversation context") from error
        if not context_path.is_dir():
            raise ValueError(f"Conversation context not found: {context_id}")
        memory_dir = (context_path / "memory").resolve()
        try:
            memory_dir.relative_to(context_path)
        except ValueError as error:
            raise ValueError("Memory directory must remain inside the context") from error
        self.memory_dir = memory_dir
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        key: str,
        content: str,
        summary: str | None,
    ) -> str:
        key = key.strip()
        if not key:
            raise ValueError("Memory key cannot be empty")
        if not content.strip():
            raise ValueError("Memory content cannot be empty")
        memory_file, existing = self._find_memory(key)
        now = datetime.now(timezone.utc).isoformat()
        memory = {
            "key": key,
            "summary": self._concise_summary(summary),
            "content": content,
            "created_at": existing.get("created_at", now) if existing else now,
            "updated_at": now,
        }
        if existing and memory["summary"] == "":
            memory["summary"] = existing.get("summary", "")
        self._write(memory_file, memory)
        return json.dumps(memory, indent=2)

    def load(self, key: str) -> str:
        memory_file, memory = self._find_memory(key.strip())
        if memory is None:
            raise ValueError(self._missing_memory_message(key))
        return json.dumps(memory, indent=2)

    def append(
        self,
        key: str,
        addition: str,
        summary: str | None,
    ) -> str:
        key = key.strip()
        if not addition.strip():
            raise ValueError("Memory addition cannot be empty")
        memory_file, existing = self._find_memory(key)
        if existing is None:
            raise ValueError(self._missing_memory_message(key))
        now = datetime.now(timezone.utc).isoformat()
        existing["content"] = (
            f"{existing.get('content', '').rstrip()}\n\n{addition.strip()}"
        )
        if isinstance(summary, str) and summary.strip():
            existing["summary"] = self._concise_summary(summary)
        existing["updated_at"] = now
        self._write(memory_file, existing)
        return json.dumps(existing, indent=2)

    def delete(self, key: str) -> str:
        memory_file, memory = self._find_memory(key.strip())
        if memory is None:
            raise ValueError(self._missing_memory_message(key))
        try:
            memory_file.unlink()
        except OSError as error:
            raise ValueError(f"Could not delete memory: {error}") from error
        return f"Deleted memory: {key.strip()}"

    def overview(self) -> str:
        memories: list[dict[str, str]] = []
        for memory_file in sorted(self.memory_dir.glob("*.json")):
            memory = self._read(memory_file)
            if memory is None:
                continue
            memories.append(
                {
                    "key": str(memory.get("key", "")),
                    "created_at": str(memory.get("created_at", "")),
                    "updated_at": str(memory.get("updated_at", "")),
                    "summary": str(memory.get("summary", "")),
                }
            )
        return json.dumps(memories, indent=2)

    def _find_memory(
        self,
        key: str,
    ) -> tuple[Path, dict[str, Any] | None]:
        key = key.strip()
        if not key:
            raise ValueError("Memory key cannot be empty")
        for memory_file in self.memory_dir.glob("*.json"):
            memory = self._read(memory_file)
            if memory is not None and memory.get("key") == key:
                return memory_file, memory
        return self._path_for_key(key), None

    def _path_for_key(self, key: str) -> Path:
        if not key:
            raise ValueError("Memory key cannot be empty")
        filename = re.sub(r"[\s<>:\"/\\|?*\x00-\x1f]+", "_", key).strip("._")
        if not filename:
            raise ValueError("Memory key does not produce a valid filename")
        path = (self.memory_dir / f"{filename}_{token_hex(4)}.json").resolve()
        try:
            path.relative_to(self.memory_dir.resolve())
        except ValueError as error:
            raise ValueError("Invalid memory key") from error
        return path

    def _missing_memory_message(self, key: str) -> str:
        return (
            f"Memory not found: {key}\n"
            "Available memories (metadata only):\n"
            f"{self.overview()}"
        )

    @staticmethod
    def _concise_summary(summary: str | None) -> str:
        if not isinstance(summary, str):
            return ""
        summary = " ".join(summary.split())
        if len(summary) <= 160:
            return summary
        return f"{summary[:157].rstrip()}..."

    @staticmethod
    def _read(memory_file: Path) -> dict[str, Any] | None:
        try:
            with memory_file.open(encoding="utf-8") as file:
                memory = json.load(file)
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None
        return memory if isinstance(memory, dict) else None

    @staticmethod
    def _write(memory_file: Path, memory: dict[str, Any]) -> None:
        temporary_file = memory_file.with_name(f".{memory_file.name}.tmp")
        try:
            with temporary_file.open("w", encoding="utf-8") as file:
                json.dump(memory, file, indent=2)
            temporary_file.replace(memory_file)
        except OSError as error:
            raise ValueError(f"Could not save memory: {error}") from error


class SaveMemoryTool(Tool):
    name = "save_memory"
    description = (
        "Save or update a durable memory for the current conversation. Use this "
        "for important decisions, constraints, facts, preferences, or goals "
        "that should survive context compaction. Do not save temporary details. "
        "When providing a summary, write one concise sentence of no more than "
        "160 characters."
    )
    parameters = {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "Stable name identifying the memory.",
            },
            "summary": {
                "type": "string",
                "description": (
                    "One concise sentence, no more than 160 characters, "
                    "shown in the memory index."
                ),
            },
            "content": {
                "type": "string",
                "description": "Complete durable information to remember.",
            },
        },
        "required": ["key", "content"],
        "additionalProperties": False,
    }

    def run(
        self,
        key: str,
        content: str,
        summary: str | None = None,
        context_dir: Path | None = None,
        context_id: str | None = None,
        **_: Any,
    ) -> str:
        return MemoryStore(context_dir, context_id).save(key, content, summary)


class LoadMemoryTool(Tool):
    name = "load_memory"
    description = "Load the complete content of one durable memory by its exact key."
    parameters = {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "Exact key of the memory to load.",
            }
        },
        "required": ["key"],
        "additionalProperties": False,
    }

    def run(
        self,
        key: str,
        context_dir: Path | None = None,
        context_id: str | None = None,
        **_: Any,
    ) -> str:
        return MemoryStore(context_dir, context_id).load(key)


class AppendMemoryTool(Tool):
    name = "append_memory"
    description = (
        "Append durable information to an existing memory by exact key. "
        "Use this when new information extends a known memory. Do not use it "
        "to create a new memory; if the key is missing, the result includes "
        "the available memory overview. An optional summary must be one concise "
        "sentence of no more than 160 characters."
    )
    parameters = {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "Exact key of the existing memory.",
            },
            "addition": {
                "type": "string",
                "description": "Durable information to append to the memory.",
            },
            "summary": {
                "type": "string",
                "description": (
                    "Optional concise replacement summary, no more than 160 "
                    "characters."
                ),
            },
        },
        "required": ["key", "addition"],
        "additionalProperties": False,
    }

    def run(
        self,
        key: str,
        addition: str,
        summary: str | None = None,
        context_dir: Path | None = None,
        context_id: str | None = None,
        **_: Any,
    ) -> str:
        return MemoryStore(context_dir, context_id).append(key, addition, summary)


class DeleteMemoryTool(Tool):
    name = "delete_memory"
    description = (
        "Delete one durable memory by its exact key. Use this when a memory is "
        "obsolete, incorrect, or explicitly no longer wanted. If the key is "
        "missing, the result includes the available memory overview."
    )
    parameters = {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "Exact key of the memory to delete.",
            }
        },
        "required": ["key"],
        "additionalProperties": False,
    }

    def run(
        self,
        key: str,
        context_dir: Path | None = None,
        context_id: str | None = None,
        **_: Any,
    ) -> str:
        return MemoryStore(context_dir, context_id).delete(key)


class ListMemoriesTool(Tool):
    name = "list_memories"
    description = "List memory keys, dates, and summaries without loading full content."
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    def run(
        self,
        context_dir: Path | None = None,
        context_id: str | None = None,
        **_: Any,
    ) -> str:
        return MemoryStore(context_dir, context_id).overview()
