# This file is part of AIA. Copyright (C) 2026 Christian Rauch.
# Distributed under terms of the GPL3 license.

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .base import Tool


class CurrentTimeTool(Tool):
    name = "get_current_time"
    description = (
        "Get the current date and time. "
        "Optionally provide an IANA timezone name."
    )
    parameters = {
        "type": "object",
        "properties": {
            "timezone_name": {
                "type": "string",
                "description": (
                    "IANA timezone, such as Europe/London or "
                    "America/New_York. Omit this for the local timezone."
                ),
            }
        },
        "additionalProperties": False,
    }

    def run(self, timezone_name: str | None = None, **_: Any) -> str:
        if timezone_name is None:
            current_time = datetime.now().astimezone()
        else:
            try:
                timezone = ZoneInfo(timezone_name)
            except ZoneInfoNotFoundError as error:
                raise ValueError(f"Unknown timezone: {timezone_name}") from error
            current_time = datetime.now(timezone)
        return current_time.isoformat(timespec="seconds")
