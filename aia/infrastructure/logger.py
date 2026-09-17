# This file is part of AIA. Copyright (C) 2026 Christian Rauch.
# Distributed under terms of the GPL3 license.

from datetime import datetime, timedelta
import json
from pathlib import Path
import threading
from typing import Any


class FileLogger:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._lock = threading.Lock()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._clean_old_files()

    def log(
        self,
        event: str,
        *,
        context_id: str | None = None,
        **fields: Any,
    ) -> None:
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        formatted_fields = json.dumps(
            fields,
            ensure_ascii=True,
            default=str,
            indent=2,
        )
        context_suffix = f" | {context_id}" if context_id is not None else ""
        block = (
            f"\n{'=' * 80}\n"
            f"{timestamp} | {event}{context_suffix}\n"
            f"{'-' * 80}\n"
            f"{formatted_fields}\n"
        )
        path = self.directory / f"{self._hour_key()}.txt"
        try:
            with self._lock:
                with path.open("a", encoding="utf-8") as log_file:
                    log_file.write(block)
        except OSError:
            pass

    def _clean_old_files(self) -> None:
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        retained_files = {
            f"{(current_hour - timedelta(hours=offset)).strftime('%Y-%m-%d-%H')}.txt"
            for offset in (0, 1)
        }
        for pattern in ("*.txt", "*.jsonl"):
            for path in self.directory.glob(pattern):
                if path.name in retained_files:
                    continue
                try:
                    path.unlink()
                except OSError:
                    pass

    @staticmethod
    def _hour_key() -> str:
        return datetime.now().strftime("%Y-%m-%d-%H")
