# This file is part of AIA. Copyright (C) 2026 Christian Rauch.
# Distributed under terms of the GPL3 license.

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Context:
    system_content: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.system_content is not None:
            self.messages.append(
                {
                    "role": "system",
                    "content": self.system_content,
                }
            )

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def to_dict(self) -> dict[str, Any]:
        return {"messages": self.messages}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Context":
        return cls(messages=list(data.get("messages", [])))

    def __iter__(self):
        return iter(self.messages)
