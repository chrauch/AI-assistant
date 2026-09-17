# This file is part of AIA. Copyright (C) 2026 Christian Rauch.
# Distributed under terms of the GPL3 license.

from collections.abc import Callable
from pathlib import Path
import shutil
import subprocess

from aia.infrastructure.context_manager import ContextManager
from aia.tools import execute_tool
from aia.tools.read_file import ReadFileTool


DEFAULT_REVISION_INSTRUCTION = (
    "Revise the requested file to remove spelling and grammar errors, and fix "
    "phrasing only when it is unidiomatic. If the file contains code, also "
    "fix syntax errors. Preserve the original meaning, structure, formatting, "
    "and style as much as possible. Return only the complete revised file "
    "content, without explanations, markdown fences, or a file path."
)


class FileWorkflow:
    def __init__(
        self,
        context_manager: ContextManager,
        respond: Callable[..., tuple[str, float, int]],
        allow_external_files: bool,
    ) -> None:
        self.context_manager = context_manager
        self.respond = respond
        self.allow_external_files = allow_external_files

    def revise(
        self,
        file_path: str,
        additional_instruction: str | None = None,
        on_text: Callable[[str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
        on_diff: Callable[[str], None] | None = None,
        max_new_tokens: int = 2048,
    ) -> tuple[str, float, int]:
        revision_instruction = DEFAULT_REVISION_INSTRUCTION
        if additional_instruction and additional_instruction.strip():
            revision_instruction = (
                f"{revision_instruction}\n\n"
                "IMPORTANT: "
                f"{additional_instruction.strip()}"
            )
        original_content = execute_tool(
            "read_file",
            {"file_path": file_path},
            allow_external_files=self.allow_external_files,
        )
        context_id = self.context_manager.context_id
        context_messages = self.context_manager.messages_for_model(context_id)
        context_messages[0]["content"] = (
            f"{context_messages[0].get('content', '')}\n\n"
            f"{revision_instruction}"
        )
        context_messages.append(
            {
                "role": "user",
                "content": (
                    f"Revise the file `{file_path}` according to the revision "
                    "instructions. The complete source file content is below.\n\n"
                    "--- BEGIN FILE CONTENT ---\n"
                    f"{original_content}\n"
                    "--- END FILE CONTENT ---"
                ),
            }
        )
        response, elapsed_seconds, token_count = self.respond(
            context_messages,
            add_tools=False,
            context_id=context_id,
            on_text=on_text,
            on_thinking=on_thinking,
            max_new_tokens=max_new_tokens,
        )
        generated_path = execute_tool(
            "write_file",
            {
                "file_path": file_path,
                "content": response,
            },
            allow_external_files=self.allow_external_files,
        )
        self._show_diff(file_path, generated_path, on_diff)
        return generated_path, elapsed_seconds, token_count

    def append(self, file_path: str) -> Path | None:
        content = execute_tool(
            "read_file",
            {"file_path": file_path},
            allow_external_files=self.allow_external_files,
        )
        return self.context_manager.append_file(file_path, content)

    def unappend(self, file_path: str) -> tuple[int, Path | None]:
        return self.context_manager.unappend_file(file_path)

    def appended_files(self) -> list[str]:
        return self.context_manager.appended_files()

    def _show_diff(
        self,
        file_path: str,
        generated_path: str,
        on_diff: Callable[[str], None] | None,
    ) -> None:
        diff_executable = shutil.which("diff")
        if diff_executable is None:
            return
        original_path = self._resolve_file_path(file_path)
        generated_file = self._resolve_file_path(generated_path)
        if not generated_file.is_file() and not Path(generated_path).is_absolute():
            generated_file = original_path.parent / Path(generated_path).name
        diff_process = subprocess.run(
            [
                diff_executable,
                "-u",
                "-U",
                "1",
                "-L",
                f"Original: {file_path}",
                "-L",
                f"Revised: {generated_path}",
                str(original_path),
                str(generated_file),
            ],
            capture_output=True,
            check=False,
            text=True,
        )
        if diff_process.returncode <= 1:
            diff_result = diff_process.stdout
        else:
            diff_result = (
                f"diff failed with exit code {diff_process.returncode}: "
                f"{diff_process.stderr.strip()}"
            )
        if on_diff is not None:
            on_diff(diff_result)

    def _resolve_file_path(self, file_path: str) -> Path:
        path = Path(file_path).expanduser()
        if path.is_absolute():
            return path.resolve()
        base_directory = (
            Path.cwd() if self.allow_external_files else ReadFileTool.data_dir
        )
        return (base_directory / path).resolve()
