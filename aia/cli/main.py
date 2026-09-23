# This file is part of AIA. Copyright (C) 2026 Christian Rauch.
# Distributed under terms of the GPL3 license.

import json
from collections.abc import Callable
import os
from pathlib import Path
import platform
import shlex
import signal
import subprocess
import sys
import termios
import threading
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
from aia.tools.read_file import ReadFileTool


CONFIG = load_config()
configure_logging(CONFIG.log_dir)
COMMANDS_DIR = CONFIG.commands_dir


CHARACTER_DELAY_SECONDS = 0.03
PROMPT_HISTORY: list[str] = []
EXIT_REQUESTED = False


COMMAND_DESCRIPTIONS = {
    "/help or /?": "List available commands.",
    "/clear": "Clear the active conversation and start a new one.",
    "/clean": "Delete inactive conversations with no response.",
    "/truncate [GUID]": "Delete saved messages from a conversation.",
    "/history": "Show the active conversation history.",
    "/delete GUID": "Delete an inactive conversation.",
    "/new [INSTRUCTION]": "Create a conversation with an optional instruction.",
    "/fork": "Fork the current conversation context.",
    "/agent [INSTRUCTION]": "Show or change the agent instruction.",
    "/load [GUID|NUMBER]": "List or resume a saved conversation.",
    "/compact [INSTRUCTION]": "Summarize and compact the active conversation.",
    "/revise FILE [INSTRUCTION]": "Revise a file and show the generated diff.",
    "/edit FILE": "Open a file in the configured terminal editor.",
    "/browse [PATH]": "Open a directory in the configured file explorer.",
    "/file [+-] FILE": "List, append, or remove context files.",
    "/tool [+-] [NAME]": "List or activate/deactivate model tools.",
    "/memory [+-]": "List or activate/deactivate memory tools.",
    "/system": "Show the system information.",
    "Up/Down": "Navigate prompt history.",
    "Ctrl+O": "Switch to the editor with the current prompt.",
    "Ctrl+C": "Clear the current prompt or interrupt generation.",
    "Ctrl+D": "Exit the assistant.",
}


def handle_interrupt(signum: int, frame: object) -> None:
    request_cancel()


def handle_quit(signum: int, frame: object) -> None:
    global EXIT_REQUESTED
    EXIT_REQUESTED = True
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


def suppress_input_echo() -> list[int] | None:
    if not sys.stdin.isatty():
        return None
    terminal_attributes = termios.tcgetattr(sys.stdin)
    updated_attributes = terminal_attributes.copy()
    updated_attributes[3] &= ~(termios.ECHO | termios.ECHONL)
    updated_attributes[6][termios.VQUIT] = b"\x04"
    termios.tcsetattr(sys.stdin, termios.TCSANOW, updated_attributes)
    return terminal_attributes


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


class StatusSpinner:
    #frames = ("▄▀", " █", "▀▄", "▄▄", "▄▀", "█ ", "▀▄", "▀▀")
    #frames = ("█  ", "▄▀ ", " █ ", " ▄▀", "  █", " ▀▄", " █ ", "▀▄ ")
    #frames = ("·", "∘", "○", "◌", "◎", "◉", "●", "◉", "◎", "◌", "○", "∘", "·", " ", "·", "∘")
    #frames = ( "·", ":", "∙", ":", "✦", ":", "∙", ":", "·", " ", "·", ":", "∙", ":", "✦", ":", "∙", ":")
    frames = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

    def __init__(self) -> None:
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0

    def start(self) -> None:
        if not sys.stdout.isatty():
            return
        self._started_at = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join()
        sys.stdout.write("\r\033[2K")
        sys.stdout.flush()
        self._thread = None

    def _run(self) -> None:
        frame_index = 0
        while not self._stop_event.is_set():
            elapsed = time.monotonic() - self._started_at
            sys.stdout.write(
                f"\r{self.frames[frame_index]} Thinking... {elapsed:.0f}s"
            )
            sys.stdout.flush()
            frame_index = (frame_index + 1) % len(self.frames)
            self._stop_event.wait(0.1)


def stop_spinner(spinner: StatusSpinner, callback: Callable[[str], None]) -> Callable[[str], None]:
    def wrapped(text: str) -> None:
        spinner.stop()
        callback(text)

    return wrapped


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


def read_prompt(assistant: AIAssistant) -> tuple[str, Path | None]:
    if not sys.stdin.isatty():
        out("YOU", destination="console", marker=True, end="")
        return input(), None

    prompt_lines: list[str] = []
    prompt_path: Path | None = None
    current_line = ""
    cursor_index = 0
    history_index = len(PROMPT_HISTORY)
    history_draft = ""
    displayed_line_count = 1

    def clear_prompt_display() -> None:
        nonlocal displayed_line_count
        sys.stdout.write("\r\033[2K")
        for _ in range(displayed_line_count - 1):
            sys.stdout.write("\033[1A\r\033[2K")
        displayed_line_count = 1

    def read_escape_sequence() -> bytes:
        sequence = bytearray(b"\x1b")
        while len(sequence) < 16:
            character = os.read(sys.stdin.fileno(), 1)
            sequence.extend(character)
            if character in b"~ABCD":
                break
        return bytes(sequence)

    out("YOU", destination="console", marker=True, end="")
    previous_handler = signal.getsignal(signal.SIGINT)
    terminal_attributes = termios.tcgetattr(sys.stdin)
    raw_attributes = terminal_attributes.copy()
    raw_attributes[3] &= ~(termios.ICANON | termios.ECHO | termios.ISIG)
    raw_attributes[6][termios.VMIN] = 1
    raw_attributes[6][termios.VTIME] = 0
    signal.signal(signal.SIGINT, signal.default_int_handler)
    termios.tcsetattr(sys.stdin, termios.TCSANOW, raw_attributes)
    try:
        while True:
            character = os.read(sys.stdin.fileno(), 1)
            if character == b"\x04":
                raise EOFError
            if character == b"\x03":
                raise KeyboardInterrupt
            if character == b"\x0f":
                prompt = "\n".join([*prompt_lines, current_line])
                clear_prompt_display()
                sys.stdout.flush()
                termios.tcsetattr(sys.stdin, termios.TCSANOW, terminal_attributes)
                try:
                    prompt_path, edited_prompt = edit_prompt(assistant, prompt)
                finally:
                    termios.tcsetattr(sys.stdin, termios.TCSANOW, raw_attributes)
                prompt_lines = edited_prompt.split("\n")
                current_line = ""
                cursor_index = 0
                sys.stdout.write(f"{edited_prompt}\n")
                sys.stdout.flush()
                displayed_line_count = len(prompt_lines) + 1
                continue
            if character == b"\x1b":
                sequence = read_escape_sequence()
                if sequence in {b"\x1b[1;5C", b"\x1b[5C"}:
                    old_cursor_index = cursor_index
                    while cursor_index < len(current_line) and current_line[
                        cursor_index
                    ].isspace():
                        cursor_index += 1
                    while cursor_index < len(current_line) and not current_line[
                        cursor_index
                    ].isspace():
                        cursor_index += 1
                    distance = cursor_index - old_cursor_index
                    if distance:
                        sys.stdout.write(f"\033[{distance}C")
                    sys.stdout.flush()
                    continue
                if sequence in {b"\x1b[1;5D", b"\x1b[5D"}:
                    old_cursor_index = cursor_index
                    while cursor_index > 0 and current_line[cursor_index - 1].isspace():
                        cursor_index -= 1
                    while cursor_index > 0 and not current_line[cursor_index - 1].isspace():
                        cursor_index -= 1
                    distance = old_cursor_index - cursor_index
                    if distance:
                        sys.stdout.write(f"\033[{distance}D")
                    sys.stdout.flush()
                    continue
                if sequence in {b"\x1b[C", b"\x1b[D"}:
                    if sequence == b"\x1b[C" and cursor_index < len(current_line):
                        cursor_index += 1
                        sys.stdout.write("\033[C")
                    elif sequence == b"\x1b[D" and cursor_index > 0:
                        cursor_index -= 1
                        sys.stdout.write("\033[D")
                    sys.stdout.flush()
                    continue
                if sequence not in {b"\x1b[A", b"\x1b[B"}:
                    continue
                current_prompt = "\n".join([*prompt_lines, current_line])
                if sequence == b"\x1b[A":
                    if history_index == len(PROMPT_HISTORY):
                        history_draft = current_prompt
                    if history_index > 0:
                        history_index -= 1
                        current_prompt = PROMPT_HISTORY[history_index]
                elif history_index < len(PROMPT_HISTORY):
                    history_index += 1
                    current_prompt = (
                        history_draft
                        if history_index == len(PROMPT_HISTORY)
                        else PROMPT_HISTORY[history_index]
                    )
                else:
                    continue
                prompt_lines = current_prompt.split("\n")[:-1]
                current_line = current_prompt.split("\n")[-1]
                cursor_index = len(current_line)
                clear_prompt_display()
                sys.stdout.write(current_prompt)
                displayed_line_count = max(1, len(current_prompt.split("\n")))
                sys.stdout.flush()
                continue
            if character in {b"\r", b"\n"}:
                sys.stdout.write("\n")
                sys.stdout.flush()
                prompt = "\n".join([*prompt_lines, current_line])
                if prompt and (not PROMPT_HISTORY or PROMPT_HISTORY[-1] != prompt):
                    PROMPT_HISTORY.append(prompt)
                return prompt, prompt_path
            if character in {b"\x08", b"\x7f"}:  # backspace, delete
                if cursor_index > 0:
                    suffix = current_line[cursor_index:]
                    current_line = (
                        current_line[: cursor_index - 1]
                        + current_line[cursor_index:]
                    )
                    cursor_index -= 1
                    sys.stdout.write(
                        "\b" + suffix + " " + "\b" * (len(suffix) + 1)
                    )
                    sys.stdout.flush()
                continue
            try:
                text = character.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if text.isprintable():
                suffix = current_line[cursor_index:]
                current_line = (
                    current_line[:cursor_index] + text + current_line[cursor_index:]
                )
                cursor_index += len(text)
                sys.stdout.write(text + suffix + "\b" * len(suffix))
                sys.stdout.flush()
    except (EOFError, KeyboardInterrupt):
        remove_prompt_file(assistant, prompt_path)
        raise
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSANOW, terminal_attributes)
        signal.signal(signal.SIGINT, previous_handler)


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


def handle_fork(assistant: AIAssistant, argument: str) -> None:
    new_context_id = assistant.fork_context(assistant.context_id)
    out(
        "AIA",
        f"Forked conversation context: {new_context_id}",
        context_id=assistant.context_id,
    )


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
    spinner = StatusSpinner()
    spinner.start()
    try:
        context_file, _, elapsed_seconds, token_count = assistant.compact_context(
            assistant.context_id,
            on_text=stop_spinner(
                spinner,
                lambda text: print_response(text, assistant.context_id),
            ),
            instruction=compact_instruction or None,
        )
    finally:
        spinner.stop()
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
    spinner = StatusSpinner()
    spinner.start()
    try:
        _, elapsed_seconds, token_count = assistant.revise_file(
            file_path,
            additional_instruction.strip() if separator else None,
            on_text=stop_spinner(
                spinner,
                lambda text: print_response(text, assistant.context_id),
            ),
            on_thinking=stop_spinner(
                spinner,
                lambda text: print_thinking(text, assistant.context_id),
            ),
            on_diff=stop_spinner(
                spinner,
                lambda text: print_diff(text, assistant.context_id),
            ),
        )
    finally:
        spinner.stop()
    out(
        "SYS",
        f"\nCompleted in {elapsed_seconds:.1f}s "
        f"({token_count} streamed chunks).",
        context_id=assistant.context_id,
        destination="log",
    )


def handle_edit(assistant: AIAssistant, argument: str) -> None:
    file_path = argument.strip()
    if not file_path:
        out(
            "SYS",
            "Usage: /edit FILE",
            context_id=assistant.context_id,
            file=sys.stderr,
        )
        return

    requested_path = Path(file_path).expanduser()
    if not requested_path.is_absolute():
        base_directory = (
            Path.cwd() if assistant.allow_external_files else ReadFileTool.data_dir
        )
        requested_path = base_directory / requested_path
    requested_path = requested_path.resolve()
    if not assistant.allow_external_files:
        try:
            requested_path.relative_to(ReadFileTool.data_dir.resolve())
        except ValueError:
            out(
                "SYS",
                "File path must remain inside the data directory.",
                context_id=assistant.context_id,
                file=sys.stderr,
            )
            return

    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or CONFIG.editor
    try:
        editor_command = shlex.split(editor)
        if not editor_command:
            raise ValueError("Editor command is empty")
        result = subprocess.run(
            [*editor_command, str(requested_path)],
            check=False,
        )
    except (OSError, ValueError) as error:
        out(
            "SYS",
            f"Could not open editor: {error}",
            context_id=assistant.context_id,
            file=sys.stderr,
        )
        return
    if result.returncode != 0:
        out(
            "SYS",
            f"Editor exited with status {result.returncode}.",
            context_id=assistant.context_id,
            file=sys.stderr,
        )
        return
    out("AIA", f"Finished editing {requested_path}.", context_id=assistant.context_id)


def edit_prompt(
    assistant: AIAssistant,
    initial_prompt: str = "",
) -> tuple[Path | None, str]:
    prompt_path = assistant.context_dir / assistant.context_id / "prompt"
    try:
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(initial_prompt, encoding="utf-8")
        editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or CONFIG.editor
        editor_command = shlex.split(editor)
        if not editor_command:
            raise ValueError("Editor command is empty")
        result = subprocess.run(
            [*editor_command, str(prompt_path)],
            check=False,
        )
        if result.returncode != 0:
            out(
                "SYS",
                f"Editor exited with status {result.returncode}.",
                context_id=assistant.context_id,
                file=sys.stderr,
            )
            return prompt_path, ""
        return prompt_path, prompt_path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError, ValueError) as error:
        out(
            "SYS",
            f"Could not edit prompt: {error}",
            context_id=assistant.context_id,
            file=sys.stderr,
        )
        return prompt_path, ""


def remove_prompt_file(assistant: AIAssistant, prompt_path: Path | None) -> None:
    if prompt_path is None:
        return
    try:
        prompt_path.unlink()
    except FileNotFoundError:
        pass
    except OSError as error:
        out(
            "SYS",
            f"Could not remove temporary prompt file: {error}",
            context_id=assistant.context_id,
            file=sys.stderr,
        )


def confirm_edited_prompt(assistant: AIAssistant, prompt: str) -> bool:
    out(
        "YOU",
        prompt,
        context_id=assistant.context_id,
        destination="console",
        marker=True,
    )
    previous_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        input()
    except KeyboardInterrupt:
        return False
    except EOFError:
        return False
    finally:
        signal.signal(signal.SIGINT, previous_handler)
    return True


def handle_browse(assistant: AIAssistant, argument: str) -> None:
    requested_path = Path(argument.strip() or ".").expanduser()
    if not requested_path.is_absolute():
        base_directory = (
            Path.cwd() if assistant.allow_external_files else ReadFileTool.data_dir
        )
        requested_path = base_directory / requested_path
    requested_path = requested_path.resolve()
    if not assistant.allow_external_files:
        try:
            requested_path.relative_to(ReadFileTool.data_dir.resolve())
        except ValueError:
            out(
                "SYS",
                "Browser path must remain inside the data directory.",
                context_id=assistant.context_id,
                file=sys.stderr,
            )
            return
    if not requested_path.is_dir():
        out(
            "SYS",
            f"Directory not found: {requested_path}",
            context_id=assistant.context_id,
            file=sys.stderr,
        )
        return

    explorer = os.environ.get("FILE_EXPLORER") or CONFIG.file_explorer
    try:
        explorer_command = shlex.split(explorer)
        if not explorer_command:
            raise ValueError("File explorer command is empty")
        result = subprocess.run(
            [*explorer_command, str(requested_path)],
            check=False,
        )
    except (OSError, ValueError) as error:
        out(
            "SYS",
            f"Could not open file explorer: {error}",
            context_id=assistant.context_id,
            file=sys.stderr,
        )
        return
    if result.returncode != 0:
        out(
            "SYS",
            f"File explorer exited with status {result.returncode}.",
            context_id=assistant.context_id,
            file=sys.stderr,
        )
        return
    out(
        "AIA",
        f"Finished browsing {requested_path}.",
        context_id=assistant.context_id,
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
    spinner = StatusSpinner()
    spinner.start()
    try:
        _, elapsed_seconds, token_count = assistant.respond(
            assistant.context_id,
            prompt,
            on_text=stop_spinner(
                spinner,
                lambda text: print_response(text, assistant.context_id),
            ),
            on_thinking=stop_spinner(
                spinner,
                lambda text: print_thinking(text, assistant.context_id),
            ),
        )
    finally:
        spinner.stop()
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
        "/clear": handle_clear,
        "/clean": handle_clean,
        "/truncate": handle_truncate,
        "/history": handle_history,
        "/delete": handle_delete,
        "/new": handle_new,
        "/fork": handle_fork,
        "/agent": handle_agent,
        "/resume": handle_load,
        "/continue": handle_load,
        "/load": handle_load,
        "/compact": handle_compact,
        "/revise": handle_revise,
        "/edit": handle_edit,
        "/browse": handle_browse,
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
    global EXIT_REQUESTED
    EXIT_REQUESTED = False
    out("SYS", BANNER, destination="console", end="\n", flush=True)
    signal.signal(signal.SIGINT, handle_interrupt)
    terminal_attributes = suppress_control_character_echo()
    try:
        loading_quit_handler = signal.getsignal(signal.SIGQUIT)
        signal.signal(signal.SIGQUIT, handle_quit)
        loading_terminal_attributes = suppress_input_echo()
        try:
            assistant = AIAssistant(
                CONFIG.models_dir,
                CONFIG.model_id,
                CONFIG.context_dir,
                CONFIG.safe_context,
                allow_external_files=CONFIG.allow_external_files,
                show_progress=CONFIG.show_progress,
                log_dir=CONFIG.log_dir,
            )
        finally:
            restore_terminal_attributes(loading_terminal_attributes)
            signal.signal(signal.SIGQUIT, loading_quit_handler)
        if EXIT_REQUESTED:
            return

        handlers = command_handlers(assistant)
        instructions = sys.argv[1:] if argv is None else argv
        if instructions:
            for instruction in instructions:
                execute_instruction(assistant, instruction, handlers)
            return

        while True:
            if sys.stdin.isatty():
                termios.tcflush(sys.stdin, termios.TCIFLUSH)
            try:
                prompt, prompt_path = read_prompt(assistant)
                prompt = prompt.strip()
            except EOFError:
                break
            except KeyboardInterrupt:
                out("SYS", "\n", context_id=assistant.context_id, end="", flush=True)
                continue
            previous_quit_handler = signal.getsignal(signal.SIGQUIT)
            signal.signal(signal.SIGQUIT, handle_quit)
            response_terminal_attributes = suppress_input_echo()
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
                restore_terminal_attributes(response_terminal_attributes)
                signal.signal(signal.SIGQUIT, previous_quit_handler)
                remove_prompt_file(assistant, prompt_path)
            if EXIT_REQUESTED:
                break
    finally:
        restore_terminal_attributes(terminal_attributes)


if __name__ == "__main__":
    main()
