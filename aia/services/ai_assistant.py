# This file is part of AIA. Copyright (C) 2026 Christian Rauch.
# Distributed under terms of the GPL3 license.

from collections.abc import Callable
from pathlib import Path

from aia.infrastructure.context_manager import (
    DEFAULT_AGENT_INSTRUCTION,
    ContextManager,
)
from aia.infrastructure.model_manager import ensure_model
from aia.infrastructure.model_runtime import ModelRuntime
from aia.infrastructure.output import configure_logging
from aia.services.conversation_workflow import ConversationWorkflow
from aia.services.file_workflow import FileWorkflow
from aia.services.tool_runtime import ToolRuntime


MEMORY_TOOL_NAMES = {
    "save_memory",
    "append_memory",
    "load_memory",
    "list_memories",
    "delete_memory",
}


class AIAssistant:
    def __init__(
        self,
        models_dir: Path,
        model_id: str,
        context_dir: Path,
        safe_context: bool = True,
        agent_instruction: str = DEFAULT_AGENT_INSTRUCTION,
        allow_external_files: bool = False,
        show_progress: bool = False,
        log_dir: Path | None = None,
    ) -> None:
        if log_dir is not None:
            configure_logging(log_dir)
        self.context_manager = ContextManager(
            context_dir,
            safe_context,
        )
        model_path = ensure_model(
            model_id,
            models_dir,
            context_id=self.context_manager.context_id,
        )
        self.store = self.context_manager.store
        self.context_dir = context_dir
        self.safe_context = safe_context
        self.allow_external_files = allow_external_files
        self.show_progress = show_progress
        self.agent_instruction = agent_instruction
        self.contexts = self.context_manager.contexts
        self.settings = self.context_manager.settings
        self.tools_by_context = self.context_manager.tools_by_context
        self.context_id = self.context_manager.context_id
        self.agent_instruction = self.context_manager.agent_instruction
        self.runtime = ModelRuntime(
            model_path,
            log_dir,
            self.context_manager.context_id,
            show_progress=show_progress,
        )
        self.tools = ToolRuntime(self.runtime.stream)
        self.file_workflow = FileWorkflow(
            self.context_manager,
            self.tools.respond,
            allow_external_files,
        )
        self.conversation_workflow = ConversationWorkflow(
            self.context_manager,
            self.runtime.stream,
            self.tools.respond,
            allow_external_files,
        )
        self.device = self.runtime.device
        self.processor = self.runtime.processor
        self.model = self.runtime.model

    def create_context(self, system_content: str | None = None) -> str:
        return self.context_manager.create_context(system_content)

    def latest_context_id(self) -> str | None:
        return self.context_manager.latest_context_id()

    def context_ids(self) -> list[str]:
        return self.context_manager.context_ids()

    def latest_context_file(self, context_id: str) -> Path | None:
        return self.context_manager.latest_context_file(context_id)

    def context_timestamp(self, context_id: str) -> str:
        return self.context_manager.context_timestamp(context_id)

    def context_last_response(self, context_id: str) -> str:
        return self.context_manager.context_last_response(context_id)

    def set_agent_instruction(
        self,
        context_id: str,
        instruction: str,
    ) -> tuple[str, str]:
        result = self.context_manager.set_agent_instruction(context_id, instruction)
        self.agent_instruction = self.context_manager.agent_instruction
        return result

    def activate_context(self, context_id: str) -> None:
        self.context_manager.activate(context_id)
        self.context_id = self.context_manager.context_id
        self.agent_instruction = self.context_manager.agent_instruction

    def delete_context(self, context_id: str) -> None:
        self.context_manager.delete(context_id)

    def delete_unanswered_contexts(self) -> list[str]:
        return self.context_manager.delete_unanswered()

    def delete_all_contexts(self) -> None:
        self.context_manager.delete_all()

    def truncate_context(self, context_id: str) -> int:
        return self.context_manager.truncate(context_id)

    def compact_context(
        self,
        context_id: str,
        on_text: Callable[[str], None] | None = None,
        instruction: str | None = None,
    ) -> tuple[Path | None, str, float, int]:
        return self.conversation_workflow.compact(
            context_id,
            on_text=on_text,
            instruction=instruction,
        )

    def respond(
        self,
        context_id: str,
        prompt: str,
        on_text: Callable[[str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
        max_new_tokens: int = 512,
    ) -> tuple[str, float, int]:
        return self.conversation_workflow.respond(
            context_id,
            prompt,
            on_text=on_text,
            on_thinking=on_thinking,
            max_new_tokens=max_new_tokens,
        )

    def available_tools(self) -> list[str]:
        return self.context_manager.available_tools()

    def active_tool_names(self) -> list[str]:
        return self.context_manager.active_tool_names()

    def set_tool_active(self, name: str, active: bool) -> None:
        self.context_manager.set_tool_active(name, active)

    def activate_all_tools(self) -> None:
        self.context_manager.activate_all_tools()

    def deactivate_all_tools(self) -> None:
        self.context_manager.deactivate_all_tools()

    def activate_memory_tools(self) -> None:
        self.context_manager.activate_tools(MEMORY_TOOL_NAMES)

    def deactivate_memory_tools(self) -> None:
        self.context_manager.deactivate_tools(MEMORY_TOOL_NAMES)

    def _messages_for_model(self, context_id: str) -> list[dict[str, object]]:
        return self.context_manager.messages_for_model(context_id)

    def revise_file(
        self,
        file_path: str,
        additional_instruction: str | None = None,
        on_text: Callable[[str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
        on_diff: Callable[[str], None] | None = None,
        max_new_tokens: int = 2048,
    ) -> tuple[str, float, int]:
        return self.file_workflow.revise(
            file_path,
            additional_instruction,
            on_text=on_text,
            on_thinking=on_thinking,
            on_diff=on_diff,
            max_new_tokens=max_new_tokens,
        )

    def append_file(self, file_path: str) -> Path | None:
        return self.file_workflow.append(file_path)

    def unappend_file(self, file_path: str) -> tuple[int, Path | None]:
        return self.file_workflow.unappend(file_path)

    def appended_files(self) -> list[str]:
        return self.file_workflow.appended_files()

    def _save_context(
        self,
        context_id: str,
        compacted: bool = False,
    ) -> Path | None:
        return self.context_manager.save(context_id, compacted)
