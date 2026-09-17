# This file is part of AIA. Copyright (C) 2026 Christian Rauch.
# Distributed under terms of the GPL3 license.

from collections.abc import Callable
from pathlib import Path

from aia.domain.context import Context
from aia.infrastructure.context_manager import ContextManager
from aia.infrastructure.model_runtime import GenerationCancelled


DEFAULT_COMPACTION_INSTRUCTION = (
    "Preserve the user's goals, important facts, decisions, constraints, "
    "and unresolved questions. Before summarizing, review the available "
    "memories and save durable facts, decisions, constraints, preferences, "
    "or goals that would otherwise be lost. Do not save temporary details."
)


class ConversationWorkflow:
    def __init__(
        self,
        context_manager: ContextManager,
        generate: Callable[..., tuple[str, float, int]],
        respond_with_tools: Callable[..., tuple[str, float, int]],
        allow_external_files: bool,
    ) -> None:
        self.context_manager = context_manager
        self.generate = generate
        self.respond_with_tools = respond_with_tools
        self.allow_external_files = allow_external_files

    def compact(
        self,
        context_id: str,
        on_text: Callable[[str], None] | None = None,
        instruction: str | None = None,
    ) -> tuple[Path | None, str, float, int]:
        context = self.context_manager.contexts[context_id]
        self.context_manager.store.backup(context_id)
        instruction = instruction or DEFAULT_COMPACTION_INSTRUCTION
        summary_request = [
            *self.context_manager.messages_for_model(context_id),
            {
                "role": "user",
                "content": (
                    "Summarize the entire conversation so far. "
                    "Return only the summary. "
                    f"{instruction}"
                ),
            },
        ]
        summary, elapsed_seconds, chunk_count = self.generate(
            summary_request,
            on_text=on_text,
            add_tools=False,
            context_id=context_id,
        )

        compacted_context = Context()
        compacted_context.messages.append({"role": "system", "content": summary})
        self.context_manager.contexts[context_id] = compacted_context
        context_file = self.context_manager.save(context_id, compacted=True)
        return context_file, summary, elapsed_seconds, chunk_count

    def respond(
        self,
        context_id: str,
        prompt: str,
        on_text: Callable[[str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
        max_new_tokens: int = 512,
    ) -> tuple[str, float, int]:
        context = self.context_manager.contexts[context_id]
        original_messages = context.messages.copy()
        context.add_user_message(prompt)
        try:
            response, elapsed_seconds, chunk_count = self.respond_with_tools(
                self.context_manager.messages_for_model(context_id),
                add_tools=True,
                tools=self.context_manager.tools_by_context[context_id],
                context_dir=self.context_manager.context_dir,
                context_id=context_id,
                allow_external_files=self.allow_external_files,
                on_text=on_text,
                on_thinking=on_thinking,
                max_new_tokens=max_new_tokens,
            )
        except GenerationCancelled:
            context.messages = original_messages
            raise
        context.add_assistant_message(response)
        self.context_manager.save(context_id)
        return response, elapsed_seconds, chunk_count
