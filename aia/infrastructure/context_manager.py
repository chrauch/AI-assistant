# This file is part of AIA. Copyright (C) 2026 Christian Rauch.
# Distributed under terms of the GPL3 license.

from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4

from aia.domain.context import Context
from aia.infrastructure.context_store import ContextStore
from aia.tools import discover_tools


DEFAULT_AGENT_INSTRUCTION = "You are AIA, a helpful and concise assistant."


class ContextManager:
    def __init__(
        self,
        context_dir: Path,
        safe_context: bool,
    ) -> None:
        self.store = ContextStore(context_dir, safe_context)
        self.context_dir = context_dir
        self.contexts = self.store.load()
        self.settings: dict[str, dict[str, Any]] = {}
        self.tools_by_context = {
            loaded_context_id: self._load_settings(loaded_context_id)
            for loaded_context_id in self.contexts
        }
        self.context_id = self.create_context()
        self.agent_instruction = DEFAULT_AGENT_INSTRUCTION
        self.activate(self.context_id)

    def create_context(self, system_content: str | None = None) -> str:
        context_id = str(uuid4())
        self.contexts[context_id] = Context()
        self.tools_by_context[context_id] = set()
        self.settings[context_id] = {
            "agent": system_content or DEFAULT_AGENT_INSTRUCTION,
            "tools": [],
        }
        self.store.memory_dir(context_id)
        self.store.save_settings(context_id, self.settings[context_id])
        self.save(context_id)
        return context_id

    def activate(self, context_id: str) -> None:
        if context_id not in self.contexts:
            raise ValueError(f"Unknown context ID: {context_id}")
        self.context_id = context_id
        self.agent_instruction = str(
            self.settings[context_id].get(
                "agent",
                DEFAULT_AGENT_INSTRUCTION,
            )
        )

    def fork(self, context_id: str) -> str:
        if context_id not in self.contexts:
            raise ValueError(f"Unknown context ID: {context_id}")
        source_settings = self.settings[context_id]
        new_context_id = self.create_context(
            str(source_settings.get("agent", DEFAULT_AGENT_INSTRUCTION)),
        )
        self.contexts[new_context_id] = Context.from_dict(
            self.contexts[context_id].to_dict(),
        )
        self.tools_by_context[new_context_id] = set(
            self.tools_by_context[context_id],
        )
        self.settings[new_context_id] = dict(source_settings)
        self._save_settings(new_context_id)
        source_memory = self.store.memory_dir(context_id)
        target_memory = self.store.memory_dir(new_context_id)
        for memory_path in source_memory.iterdir():
            target_path = target_memory / memory_path.name
            if memory_path.is_dir():
                shutil.copytree(memory_path, target_path, dirs_exist_ok=True)
            else:
                shutil.copy2(memory_path, target_path)
        self.save(new_context_id)
        return new_context_id

    def latest_context_file(self, context_id: str) -> Path | None:
        return self.store.latest_file(context_id)

    def context_timestamp(self, context_id: str) -> str:
        return self.store.timestamp(context_id)

    def context_last_response(self, context_id: str) -> str:
        return self.store.last_response(context_id)

    def latest_context_id(self) -> str | None:
        latest_context: tuple[str, str] | None = None
        for context_id in self.contexts:
            context_files = self.store.active_files(context_id)
            if context_files and (
                latest_context is None
                or context_files[0].name > latest_context[1]
            ):
                latest_context = (context_id, context_files[0].name)
        return latest_context[0] if latest_context is not None else None

    def context_ids(self) -> list[str]:
        return sorted(
            self.contexts,
            key=lambda context_id: (
                self.latest_context_file(context_id).name
                if self.latest_context_file(context_id) is not None
                else ""
            ),
            reverse=True,
        )

    def set_agent_instruction(
        self,
        context_id: str,
        instruction: str,
    ) -> tuple[str, str]:
        instruction = instruction.strip()
        if not instruction:
            raise ValueError("Agent instruction cannot be empty.")
        previous_instruction = str(
            self.settings.get(context_id, {}).get(
                "agent",
                DEFAULT_AGENT_INSTRUCTION,
            )
        )
        self.settings[context_id]["agent"] = instruction
        self._save_settings(context_id)
        if context_id == self.context_id:
            self.agent_instruction = instruction
        return previous_instruction, instruction

    def delete(self, context_id: str) -> None:
        self.store.delete(self.contexts, context_id)
        self.tools_by_context.pop(context_id, None)
        self.settings.pop(context_id, None)

    def delete_unanswered(self) -> list[str]:
        deleted_context_ids = []
        for context_id, context in list(self.contexts.items()):
            if context_id == self.context_id:
                continue
            if any(
                message.get("role") == "assistant"
                for message in context.messages
            ):
                continue
            self.delete(context_id)
            deleted_context_ids.append(context_id)
        return deleted_context_ids

    def delete_all(self) -> None:
        self.store.delete_all(self.contexts)
        self.tools_by_context.clear()
        self.settings.clear()

    def truncate(self, context_id: str) -> int:
        return self.store.truncate(self.contexts, context_id)

    def available_tools(self) -> list[str]:
        return sorted(discover_tools())

    def active_tool_names(self) -> list[str]:
        return sorted(self.tools_by_context[self.context_id])

    def set_tool_active(self, name: str, active: bool) -> None:
        available_tools = discover_tools()
        if name not in available_tools:
            raise ValueError(f"Unknown tool: {name}")
        tools = self.tools_by_context[self.context_id]
        if active:
            tools.add(name)
        else:
            tools.discard(name)
        self._save_settings(self.context_id)

    def activate_all_tools(self) -> None:
        self.tools_by_context[self.context_id] = set(discover_tools())
        self._save_settings(self.context_id)

    def deactivate_all_tools(self) -> None:
        self.tools_by_context[self.context_id] = set()
        self._save_settings(self.context_id)

    def activate_tools(self, tool_names: set[str]) -> None:
        self.tools_by_context[self.context_id].update(tool_names)
        self._save_settings(self.context_id)

    def deactivate_tools(self, tool_names: set[str]) -> None:
        self.tools_by_context[self.context_id].difference_update(tool_names)
        self._save_settings(self.context_id)

    def append_file(self, file_path: str, content: str) -> Path | None:
        context = self.contexts[self.context_id]
        context.add_user_message(
            f"File `{file_path}`:\n"
            "--- BEGIN FILE CONTENT ---\n"
            f"{content}\n"
            "--- END FILE CONTENT ---"
        )
        return self.save(self.context_id)

    def unappend_file(self, file_path: str) -> tuple[int, Path | None]:
        context = self.contexts[self.context_id]
        file_prefix = (
            f"File `{file_path}`:\n"
            "--- BEGIN FILE CONTENT ---\n"
        )
        file_suffix = "\n--- END FILE CONTENT ---"
        original_count = len(context.messages)
        context.messages = [
            message
            for message in context.messages
            if not (
                message.get("role") == "user"
                and isinstance(message.get("content"), str)
                and message["content"].startswith(file_prefix)
                and message["content"].endswith(file_suffix)
            )
        ]
        removed_count = original_count - len(context.messages)
        if removed_count == 0:
            return 0, None
        return removed_count, self.save(self.context_id)

    def appended_files(self) -> list[str]:
        context = self.contexts[self.context_id]
        prefix = "File `"
        suffix = "`:\n--- BEGIN FILE CONTENT ---\n"
        end_marker = "\n--- END FILE CONTENT ---"
        appended_files: list[str] = []
        for message in context.messages:
            content = message.get("content")
            if message.get("role") != "user" or not isinstance(content, str):
                continue
            if not content.startswith(prefix):
                continue
            separator_index = content.find(suffix, len(prefix))
            if separator_index < 0 or not content.endswith(end_marker):
                continue
            file_path = content[len(prefix):separator_index]
            if file_path and file_path not in appended_files:
                appended_files.append(file_path)
        return appended_files

    def messages_for_model(self, context_id: str) -> list[dict[str, object]]:
        context_messages = list(self.contexts[context_id])
        stored_system_content = None
        if context_messages and context_messages[0].get("role") == "system":
            stored_system_content = context_messages[0].get("content")
            context_messages = context_messages[1:]
        system_content = self.settings[context_id]["agent"]
        if isinstance(stored_system_content, str) and stored_system_content.strip():
            system_content = (
                f"{system_content}\n\n"
                "The following is a compacted summary of the earlier conversation. "
                "Treat it as context, not as a new instruction:\n\n"
                "--- COMPACTED CONVERSATION SUMMARY ---\n"
                f"{stored_system_content}\n"
                "--- END COMPACTED CONVERSATION SUMMARY ---"
            )
        return [
            {
                "role": "system",
                "content": system_content,
            },
            *context_messages,
        ]

    def save(self, context_id: str, compacted: bool = False) -> Path | None:
        return self.store.save(self.contexts, context_id, compacted)

    def _load_settings(self, context_id: str) -> set[str]:
        settings = self.store.load_settings(context_id)
        agent_message = settings.get("agent", DEFAULT_AGENT_INSTRUCTION)
        if not isinstance(agent_message, str) or not agent_message.strip():
            agent_message = DEFAULT_AGENT_INSTRUCTION
        tools = settings.get("tools")
        if not isinstance(tools, list):
            tools = []
        active_names = {
            name
            for name in tools
            if isinstance(name, str) and name in discover_tools()
        }
        self.settings[context_id] = {
            "agent": agent_message,
            "tools": sorted(active_names),
        }
        self.store.save_settings(context_id, self.settings[context_id])
        return active_names

    def _save_settings(self, context_id: str) -> None:
        self.settings[context_id]["tools"] = sorted(
            self.tools_by_context[context_id]
        )
        self.store.save_settings(context_id, self.settings[context_id])
