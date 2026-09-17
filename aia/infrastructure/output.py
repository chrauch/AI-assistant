# This file is part of AIA. Copyright (C) 2026 Christian Rauch.
# Distributed under terms of the GPL3 license.

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
from typing import Literal, TextIO

from aia.infrastructure.logger import FileLogger


_logger: FileLogger | None = None
OutputType = Literal["AIA", "YOU", "SYS"]
OUTPUT_COLORS: dict[OutputType, str] = {
    "AIA": "\033[38;5;158m",
    "YOU": "\033[38;5;187m",
    "SYS": "\033[38;5;152m",
}
RESET_COLOR = "\033[0m"
LICENSE_NOTICE = (
    "Copyright (C) 2026 Christian Rauch.\n"
    "Distributed under terms of the GPL3 license."
)


def format_marker(output_type: OutputType, colored: bool = True) -> str:
    marker = f"──── [ {output_type} ] ────"
    if not colored:
        return marker
    return f"{OUTPUT_COLORS[output_type]}{marker}{RESET_COLOR}"


BANNER = f"""{OUTPUT_COLORS['AIA']}
    ▄▀█ █ █▀▄
▀▀▀ █▀█ █ █▀█ ▀▀▀
your AI-assistant
free, open, local

{format_marker('SYS')}"""


def configure_logging(directory: Path) -> FileLogger:
    global _logger
    _logger = FileLogger(directory)
    return _logger


def get_logger() -> FileLogger | None:
    return _logger


def out(
    output_type: OutputType,
    *values: object,
    context_id: str | None = None,
    destination: str = "console+log",
    marker: bool = False,
    file: TextIO | None = None,
    sep: str = " ",
    end: str = "\n",
    flush: bool = False,
) -> None:
    if output_type not in OUTPUT_COLORS:
        raise ValueError("output_type must be 'AIA', 'YOU', or 'SYS'")
    if destination not in {"console", "console+log", "log"}:
        raise ValueError(
            "destination must be 'console', 'console+log', or 'log'"
        )

    if destination in {"console", "console+log"}:
        if marker:
            print(format_marker(output_type), file=file)
        print(*values, file=file, sep=sep, end=end, flush=flush)
    if destination == "console" or _logger is None:
        return

    rendered = StringIO()
    if marker:
        print(format_marker(output_type, colored=False), file=rendered)
    with redirect_stdout(rendered):
        print(*values, sep=sep, end=end)
    _logger.log(
        "output",
        context_id=context_id,
        stream="stderr" if file is sys.stderr else "stdout",
        text=rendered.getvalue(),
    )
