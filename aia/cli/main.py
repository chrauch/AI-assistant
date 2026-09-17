# This file is part of AIA. Copyright (C) 2026 Christian Rauch.
# Distributed under terms of the GPL3 license.

import json
from collections.abc import Callable
from pathlib import Path
import platform
import signal
import sys
import termios
import time

try:
    import readline
except ImportError:
    readline = None

import torch
import transformers

from aia.infrastructure.config import load_config
from aia.infrastructure.model_runtime import GenerationCancelled, request_cancel
from aia.infrastructure.output import BANNER, LICENSE_NOTICE, configure_logging, out
from aia.services.ai_assistant import AIAssistant, MEMORY_TOOL_NAMES


CONFIG = load_config()
configure_logging(CONFIG.log_dir)
COMMANDS_DIR = CONFIG.commands_dir


CONTINUATION_PROMPT = ""
CHARACTER_DELAY_SECONDS = 0.03


COMMAND_DESCRIPTIONS = {
    "/help or /?": "List available commands.",
    "/clear": "Clear the active conversation and start a new one.",
    "/clean": "Delete inactive conversations with no response.",
    "/truncate [GUID]": "Delete saved messages from a conversation.",
    "/history": "Show the active conversation history.",
    "/delete GUID": "Delete an inactive conversation.",
    "/new [INSTRUCTION]": "Create a conversation with an optional instruction.",
    "/agent [INSTRUCTION]": "Show or change the agent instruction.",
    "/load [GUID|NUMBER]": "List or resume a saved conversation.",
    "/compact [INSTRUCTION]": "Summarize and compact the active conversation.",
    "/revise FILE [INSTRUCTION]": "Revise a file and show the generated diff.",
    "/file [+-] FILE": "List, append, or remove context files.",
    "/tool [+-] [NAME]": "List or activate/deactivate model tools.",
    "/memory [+-]": "List or activate/deactivate memory tools.",
    "/exit or //": "Exit the assistant.",
    "/system": "Show the system information.",
    "Ctrl+C": "Interrupt the current operation.",
}


def handle_interrupt(signum: int, frame: object) -> None:
    request_cancel()


def suppress_control_character_echo() -> list[int] | None:
    if not sys.stdin.isatty():
        return None
    terminal_attributes = termios.tcgetattr(sys.stdin)
    updated_attributes = terminal_attributes.copy()
    updated_attributes[3] &= ~termios.ECHOCTL
    termios.tcsetattr(sys.stdin, termios.TCSANOW, updated_attributes)
    return terminal_attributes


def restore_terminal_attributes(
    terminal_attributes: list[int] | None,
) -> None:
    if terminal_attributes is not None:
        termios.tcsetattr(sys.stdin, termios.TCSANOW, terminal_attributes)


def print_response(text: str, context_id: str | None = None) -> None:
    for character in text:
        out(
            "AIA",
            character,
            context_id=context_id,
            destination="console",
            end="",
            flush=True,
        )
        time.sleep(CHARACTER_DELAY_SECONDS)


def print_thinking(text: str, context_id: str | None = None) -> None:
    out(
        "AIA",
        text,
        context_id=context_id,
        destination="console",
        end="",
        flush=True,
    )


def print_diff(text: str, context_id: str | None = None) -> None:
    out(
        "SYS",
        "\n",
        context_id=context_id,
        destination="console",
        end="",
        flush=True,
    )
    out(
        "SYS",
        context_id=context_id,
        marker=True,
        end="",
        flush=True,
    )
    out(
        "SYS",
        f"{text.rstrip(chr(10))}\n",
        context_id=context_id,
        end="",
        flush=True,
    )


def read_prompt() -> str:
    lines = []
    out("YOU", destination="console", marker=True, end="")
    while True:
        line = input()
        if line.endswith("\\"):
            lines.append(line[:-1])
            out("YOU", CONTINUATION_PROMPT, destination="console", end="")
            continue
        lines.append(line)
        return "\n".join(lines)


def handle_help(assistant: AIAssistant, argument: str) -> None:
    lines = ["Builtin commands:"]
    lines.extend(
        f"  {command:<28} {description}"
        for command, description in COMMAND_DESCRIPTIONS.items()
    )
    command_files = sorted(path.stem for path in COMMANDS_DIR.glob("*.json"))
    if command_files:
        lines.extend(("", "User commands:"))
        for name in command_files:
            command_file = COMMANDS_DIR / f"{name}.json"
            try:
                with command_file.open(encoding="utf-8") as file:
                    script = json.load(file)
                description = " ".join(
                    " ".join(command_line for command_line in script).split()
                )
            except (OSError, json.JSONDecodeError, TypeError):
                description = "Unable to read command script."
            if len(description) > 60:
                description = f"{description[:57]}..."
            lines.append(f"  /{name:<28}{description}")

    lines.append(f"\n{LICENSE_NOTICE}")
    out("AIA", "\n".join(lines), context_id=assistant.context_id)


def handle_system(assistant: AIAssistant, argument: str) -> None:
    runtime = assistant.runtime
    model = runtime.model
    processor = runtime.processor
    tokenizer = getattr(processor, "tokenizer", None)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    cuda_lines = [
        f"  Available: {torch.cuda.is_available()}",
    ]
    if torch.cuda.is_available():
        cuda_lines.extend(
            [
                f"  Device count: {torch.cuda.device_count()}",
                f"  Device name: {torch.cuda.get_device_name()}",
                f"  Allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB",
                f"  Reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GiB",
            ]
        )
    tokenizer_lines = [
        f"  Processor class: {type(processor).__name__}",
    ]
    if tokenizer is not None:
        tokenizer_lines.extend(
            [
                f"  Tokenizer class: {type(tokenizer).__name__}",
                f"  Vocabulary size: {len(tokenizer.get_vocab())}",
            ]
        )
    lines = [
        "Runtime:",
        f"  Python: {sys.version.split()[0]}",
        f"  Platform: {platform.platform()}",
        f"  PyTorch: {torch.__version__}",
        f"  Transformers: {transformers.__version__}",
        f"  Device: {runtime.device}",
        f"  Dtype: {runtime.dtype}",
        "  CUDA:",
        *cuda_lines,
        "",
        "Model:",
        f"  Model path: {runtime.model_path}",
        f"  Model class: {type(runtime.model).__name__}",
        f"  Model type: {model.config.model_type}",
        f"  Architecture: {model.config.architectures}",
        f"  Parameters: {parameter_count:,}",
        "",
        "Tokenizer and processor:",
        *tokenizer_lines,
        "",
        "Assistant:",
        f"  Context ID: {assistant.context_id}",
        f"  Context directory: {assistant.context_dir}",
        f"  Log directory: {CONFIG.log_dir}",
        f"  Safe context: {assistant.safe_context}",
        f"  External files: {assistant.allow_external_files}",
        f"  Agent instruction: {assistant.agent_instruction}",
        f"  Active tools: {assistant.active_tool_names()}",
        "  Model config:",
        json.dumps(runtime.model.config.to_dict(), indent=4, default=str),
        "  Generation config:",
        json.dumps(runtime.model.generation_config.to_dict(), indent=4, default=str),
    ]
    out("SYS", "\n".join(lines), context_id=assistant.context_id)


def handle_exit(assistant: AIAssistant, argument: str) -> None:
    raise SystemExit


def handle_clear(assistant: AIAssistant, argument: str) -> None:
    assistant.delete_context(assistant.context_id)
    assistant.activate_context(assistant.create_context())
    out("AIA", "Conversation context cleared.", context_id=assistant.context_id)


def handle_clean(assistant: AIAssistant, argument: str) -> None:
    deleted_context_ids = assistant.delete_unanswered_contexts()
    if deleted_context_ids:
        out(
            "AIA",
            f"Deleted {len(deleted_context_ids)} unanswered conversation(s).",
            context_id=assistant.context_id,
        )
    else:
        out(
            "AIA",
            "No unanswered inactive conversations to delete.",
            context_id=assistant.context_id,
        )


def handle_truncate(assistant: AIAssistant, argument: str) -> None:
    context_id = argument.strip() or assistant.context_id
    try:
        deleted_count = assistant.truncate_context(context_id)
    except ValueError as error:
        out("SYS", f"Error: {error}", context_id=assistant.context_id, file=sys.stderr)
        return
    out(
        "AIA",
        f"Truncated conversation context: {context_id} "
        f"({deleted_count} JSON file(s) deleted).",
        context_id=assistant.context_id,
    )


def handle_history(assistant: AIAssistant, argument: str) -> None:
    context_file = assistant.latest_context_file(assistant.context_id)
    if context_file is None:
        out("AIA", "No saved conversation history.", context_id=assistant.context_id)
        return
    try:
        with context_file.open() as file:
            history = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        out(
            "SYS",
            f"Could not read conversation history: {error}",
            context_id=assistant.context_id,
            file=sys.stderr,
        )
        return
    out("AIA", json.dumps(history, indent=2), context_id=assistant.context_id)


def handle_delete(assistant: AIAssistant, argument: str) -> None:
    context_id = argument.strip()
    if not context_id:
        out("SYS", "Usage: /delete GUID", context_id=assistant.context_id, file=sys.stderr)
        return
    if context_id == assistant.context_id:
        out(
            "SYS",
            "Cannot delete the currently active conversation.",
            context_id=assistant.context_id,
            file=sys.stderr,
        )
        return
    if context_id not in assistant.contexts:
        out(
            "SYS",
            f"Unknown conversation context: {context_id}",
            context_id=assistant.context_id,
            file=sys.stderr,
        )
        return
    assistant.delete_context(context_id)
    out("AIA", f"Deleted conversation context: {context_id}", context_id=assistant.context_id)


def handle_new(assistant: AIAssistant, argument: str) -> None:
    assistant.activate_context(assistant.create_context())
    out(
        "AIA",
        f"New conversation context created: {assistant.context_id}",
        context_id=assistant.context_id,
    )
    requested_instruction = argument.strip()
    if not requested_instruction:
        return
    try:
        previous_instruction, new_instruction = assistant.set_agent_instruction(
            assistant.context_id,
            requested_instruction,
        )
    except ValueError as error:
        out("SYS", f"Error: {error}", context_id=assistant.context_id, file=sys.stderr)
        return
    out("AIA", f"Previous agent instruction: {previous_instruction}", context_id=assistant.context_id)
    out("AIA", f"New agent instruction: {new_instruction}", context_id=assistant.context_id)


def handle_agent(assistant: AIAssistant, argument: str) -> None:
    requested_instruction = argument.strip()
    if not requested_instruction:
        out("AIA", f"Agent instruction: {assistant.agent_instruction}", context_id=assistant.context_id)
        return
    try:
        previous_instruction, new_instruction = assistant.set_agent_instruction(
            assistant.context_id,
            requested_instruction,
        )
    except ValueError as error:
        out("SYS", f"Error: {error}", context_id=assistant.context_id, file=sys.stderr)
        return
    out("AIA", f"Previous agent instruction: {previous_instruction}", context_id=assistant.context_id)
    out("AIA", f"New agent instruction: {new_instruction}", context_id=assistant.context_id)


def handle_load(assistant: AIAssistant, argument: str) -> None:
    requested_context_id = argument.strip()
    if not requested_context_id:
        context_ids = assistant.context_ids()
        if context_ids:
            lines = [
                "Available conversation contexts:",
                f"{'NR':>2}  {'TIMESTAMP':<15}  GUID  LAST RESPONSE",
            ]
            for number, context_id in enumerate(context_ids):
                marker = "<- ACTIVE" if context_id == assistant.context_id else ""
                lines.append(
                    f"{number:>2}  "
                    f"{assistant.context_timestamp(context_id):<15}  "
                    f"{context_id}  "
                    f"{assistant.context_last_response(context_id):<30}  "
                    f"{marker}"
                )
            out("AIA", "\n".join(lines), context_id=assistant.context_id)
        else:
            out("AIA", "No saved conversation contexts.", context_id=assistant.context_id)
        return
    context_ids = assistant.context_ids()
    if requested_context_id.isdecimal():
        context_number = int(requested_context_id)
        if context_number >= len(context_ids):
            out(
                "SYS",
                f"Unknown conversation context number: {context_number}",
                context_id=assistant.context_id,
                file=sys.stderr,
            )
            return
        requested_context_id = context_ids[context_number]
    if requested_context_id not in assistant.contexts:
        out(
            "SYS",
            f"Unknown conversation context: {requested_context_id}",
            context_id=assistant.context_id,
            file=sys.stderr,
        )
        return
    assistant.activate_context(requested_context_id)
    out(
        "AIA",
        f"Resumed conversation context: {assistant.context_id}",
        context_id=assistant.context_id,
    )


def handle_compact(assistant: AIAssistant, argument: str) -> None:
    compact_instruction = argument.strip()
    out(
        "AIA",
        context_id=assistant.context_id,
        marker=True,
        end="",
        flush=True,
    )
    context_file, _, elapsed_seconds, token_count = assistant.compact_context(
        assistant.context_id,
        on_text=lambda text: print_response(text, assistant.context_id),
        instruction=compact_instruction or None,
    )
    out(
        "SYS",
        f"\nCompacted in {elapsed_seconds:.1f}s "
        f"({token_count} streamed chunks).",
        context_id=assistant.context_id,
        destination="log",
    )
    out(
        "SYS",
        "\n",
        context_id=assistant.context_id,
        destination="console",
        end="",
        flush=True,
    )
    out(
        "SYS",
        f"Snapshot: {context_file}",
        context_id=assistant.context_id,
        marker=True,
    )


def handle_revise(assistant: AIAssistant, argument: str) -> None:
    file_path, separator, additional_instruction = argument.strip().partition(" ")
    if not file_path:
        out(
            "SYS",
            "Usage: /revise FILE [ADDITIONALINSTRUCTION]",
            file=sys.stderr,
        )
        return
    out(
        "AIA",
        context_id=assistant.context_id,
        marker=True,
        end="",
        flush=True,
    )
    _, elapsed_seconds, token_count = assistant.revise_file(
        file_path,
        additional_instruction.strip() if separator else None,
        on_text=lambda text: print_response(text, assistant.context_id),
        on_thinking=lambda text: print_thinking(text, assistant.context_id),
        on_diff=lambda text: print_diff(text, assistant.context_id),
    )
    out(
        "SYS",
        f"\nCompleted in {elapsed_seconds:.1f}s "
        f"({token_count} streamed chunks).",
        context_id=assistant.context_id,
        destination="log",
    )


def handle_file(assistant: AIAssistant, argument: str) -> None:
    operation, _, file_path = argument.strip().partition(" ")
    operation = operation.strip()
    file_path = file_path.strip()
    if not operation:
        operation = "list"
    if operation == "list":
        _print_appended_files(assistant)
        return
    if operation not in {"+", "-"}:
        _print_appended_files(assistant)
        return
    if operation == "+":
        if not file_path:
            out(
                "SYS",
                "Usage: /file + FILE",
                context_id=assistant.context_id,
                file=sys.stderr,
            )
            _print_appended_files(assistant)
            return
        try:
            context_file = assistant.append_file(file_path)
        except (OSError, ValueError) as error:
            out(
                "SYS",
                f"Could not append file '{file_path}': {error}",
                context_id=assistant.context_id,
                file=sys.stderr,
            )
        else:
            out("AIA", f"Appended {file_path} to the context.", context_id=assistant.context_id)
            out(
                "SYS",
                f"Context saved: {context_file}",
                context_id=assistant.context_id,
                file=sys.stderr,
            )
        _print_appended_files(assistant)
        return
    if not file_path:
        for appended_file in assistant.appended_files():
            assistant.unappend_file(appended_file)
        out("AIA", "Removed all appended files from the context.", context_id=assistant.context_id)
        _print_appended_files(assistant)
        return
    removed_count, context_file = assistant.unappend_file(file_path)
    if removed_count == 0:
        out("AIA", f"No appended content found for {file_path}.", context_id=assistant.context_id)
    else:
        out("AIA", f"Removed {removed_count} appended message(s) for {file_path}.", context_id=assistant.context_id)
        out(
            "SYS",
            f"Context saved: {context_file}",
            context_id=assistant.context_id,
            file=sys.stderr,
        )
    _print_appended_files(assistant)


def _print_appended_files(assistant: AIAssistant) -> None:
    appended_files = assistant.appended_files()
    out("AIA", "Currently appended files:", context_id=assistant.context_id)
    if not appended_files:
        out("AIA", "(none)", context_id=assistant.context_id)
        return
    for file_path in appended_files:
        out("AIA", file_path, context_id=assistant.context_id)


def handle_tool(assistant: AIAssistant, argument: str) -> None:
    operation, _, requested_tool = argument.strip().partition(" ")
    operation = operation.strip()
    requested_tool = requested_tool.strip()
    if operation == "+" and not requested_tool:
        assistant.activate_all_tools()
    elif operation == "-" and not requested_tool:
        assistant.deactivate_all_tools()
    elif operation in {"+", "-"} and requested_tool:
        try:
            assistant.set_tool_active(
                requested_tool,
                operation == "+",
            )
        except ValueError as error:
            out("SYS", f"Error: {error}", context_id=assistant.context_id, file=sys.stderr)
            return
    out("AIA", "Available tools:", context_id=assistant.context_id)
    for tool_name in assistant.available_tools():
        marker = "*" if tool_name in assistant.active_tool_names() else " "
        out("AIA", f"{marker} {tool_name}", context_id=assistant.context_id)


def handle_memory(assistant: AIAssistant, argument: str) -> None:
    operation = argument.strip()
    if operation == "+":
        assistant.activate_memory_tools()
    elif operation == "-":
        assistant.deactivate_memory_tools()
    out("AIA", "Memory tools:", context_id=assistant.context_id)
    for tool_name in sorted(MEMORY_TOOL_NAMES):
        marker = "*" if tool_name in assistant.active_tool_names() else " "
        out("AIA", f"{marker} {tool_name}", context_id=assistant.context_id)


def handle_response(assistant: AIAssistant, prompt: str) -> None:
    if not prompt:
        return
    out(
        "AIA",
        context_id=assistant.context_id,
        marker=True,
        end="",
        flush=True,
    )
    _, elapsed_seconds, token_count = assistant.respond(
        assistant.context_id,
        prompt,
        on_text=lambda text: print_response(text, assistant.context_id),
        on_thinking=lambda text: print_thinking(text, assistant.context_id),
    )
    out(
        "AIA",
        "\n",
        context_id=assistant.context_id,
        destination="console",
        end="",
        flush=True,
    )
    out(
        "SYS",
        f"\nCompleted in {elapsed_seconds:.1f}s "
        f"({token_count} streamed chunks).",
        context_id=assistant.context_id,
        destination="log",
    )


CommandHandler = Callable[[AIAssistant, str], None]


def command_handlers(assistant: AIAssistant) -> dict[str, CommandHandler]:
    return {
        "/help": handle_help,
        "/?": handle_help,
        "/system": handle_system,
        "//": handle_exit,
        "/clear": handle_clear,
        "/clean": handle_clean,
        "/truncate": handle_truncate,
        "/history": handle_history,
        "/delete": handle_delete,
        "/new": handle_new,
        "/agent": handle_agent,
        "/resume": handle_load,
        "/continue": handle_load,
        "/load": handle_load,
        "/compact": handle_compact,
        "/revise": handle_revise,
        "/file": handle_file,
        "/tool": handle_tool,
        "/memory": handle_memory,
    }


def command_file_path(command: str) -> Path | None:
    if not command.startswith("/") or not command[1:]:
        return None
    requested_path = (COMMANDS_DIR / f"{command[1:]}.json").resolve()
    try:
        requested_path.relative_to(COMMANDS_DIR.resolve())
    except ValueError:
        return None
    if not requested_path.is_file():
        return None
    return requested_path


def execute_instruction(
    assistant: AIAssistant,
    instruction: str,
    handlers: dict[str, CommandHandler],
    command_stack: tuple[Path, ...] = (),
) -> None:
    command, _, argument = instruction.partition(" ")
    handler = handlers.get(command)
    if handler is not None:
        if command not in {"/compact", "/revise"}:
            out(
                "SYS",
                context_id=assistant.context_id,
                marker=True,
                end="",
                flush=True,
            )
        handler(assistant, argument)
        return
    command_path = command_file_path(command)
    if command_path is not None:
        if command_path in command_stack:
            out(
                "SYS",
                f"Recursive command file: {command_path}",
                context_id=assistant.context_id,
                file=sys.stderr,
            )
            return
        try:
            with command_path.open(encoding="utf-8") as command_file:
                command_lines = json.load(command_file)
        except (OSError, json.JSONDecodeError) as error:
            out(
                "SYS",
                f"Could not read command file '{command_path}': {error}",
                context_id=assistant.context_id,
                file=sys.stderr,
            )
            return
        if not isinstance(command_lines, list) or not all(
            isinstance(command_line, str) for command_line in command_lines
        ):
            out(
                "SYS",
                f"Command file must contain a JSON list of strings: {command_path}",
                context_id=assistant.context_id,
                file=sys.stderr,
            )
            return
        for command_line in command_lines:
            execute_instruction(
                assistant,
                command_line,
                handlers,
                (*command_stack, command_path),
            )
        return
    if command.startswith("/"):
        out(
            "SYS",
            f"Unknown command: {command}",
            context_id=assistant.context_id,
            file=sys.stderr,
        )
        return
    handle_response(assistant, instruction)


def main(argv: list[str] | None = None) -> None:
    out("SYS", BANNER, destination="console", end="\n", flush=True)
    signal.signal(signal.SIGINT, handle_interrupt)
    assistant = AIAssistant(
        CONFIG.models_dir,
        CONFIG.model_id,
        CONFIG.context_dir,
        CONFIG.safe_context,
        allow_external_files=CONFIG.allow_external_files,
        show_progress=CONFIG.show_progress,
        log_dir=CONFIG.log_dir,
    )
    handlers = command_handlers(assistant)
    instructions = sys.argv[1:] if argv is None else argv
    if instructions:
        for instruction in instructions:
            execute_instruction(assistant, instruction, handlers)
        return

    terminal_attributes = suppress_control_character_echo()
    try:
        while True:
            try:
                prompt = read_prompt().strip()
            except (EOFError, KeyboardInterrupt):
                out("SYS", "\n", context_id=assistant.context_id, end="", flush=True)
                continue
            if prompt == "/exit":
                break
            try:
                execute_instruction(assistant, prompt, handlers)
            except GenerationCancelled:
                out(
                    "SYS",
                    "\n",
                    context_id=assistant.context_id,
                    destination="console",
                    end="",
                    flush=True,
                )
                out(
                    "SYS",
                    "Thinking interrupted.",
                    context_id=assistant.context_id,
                    marker=True,
                )
    finally:
        restore_terminal_attributes(terminal_attributes)


if __name__ == "__main__":
    main()
