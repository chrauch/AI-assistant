# This file is part of AIA. Copyright (C) 2026 Christian Rauch.
# Distributed under terms of the GPL3 license.

import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Any

from .base import Tool


def discover_tools() -> dict[str, Tool]:
    discovered: dict[str, Tool] = {}
    for module_info in pkgutil.iter_modules(__path__):
        if module_info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{__name__}.{module_info.name}")
        for _, tool_class in inspect.getmembers(module, inspect.isclass):
            if (
                tool_class is Tool
                or not issubclass(tool_class, Tool)
                or inspect.isabstract(tool_class)
                or tool_class.__module__ != module.__name__
            ):
                continue
            tool = tool_class()
            if tool.name in discovered:
                raise ValueError(f"Duplicate tool name: {tool.name}")
            discovered[tool.name] = tool
    return discovered


def tool_schemas(
    allowed_names: set[str] | None = None,
    *,
    allow_external_files: bool = False,
) -> list[dict[str, Any]]:
    tools = discover_tools()
    if allowed_names is not None:
        tools = {
            name: tool for name, tool in tools.items() if name in allowed_names
        }
    schemas = [tool.schema() for tool in tools.values()]
    if allow_external_files:
        for schema in schemas:
            if schema["name"] in {"read_file", "write_file", "list_directory"}:
                schema["description"] += (
                    " External file access is enabled for this conversation."
                )
    return schemas


def execute_tool(
    name: str,
    arguments: dict[str, Any],
    allowed_names: set[str] | None = None,
    *,
    context_dir: Path | None = None,
    context_id: str | None = None,
    allow_external_files: bool = False,
) -> str:
    tools = discover_tools()
    if allowed_names is not None and name not in allowed_names:
        raise ValueError(f"Tool is not active: {name}")
    try:
        tool = tools[name]
    except KeyError as error:
        raise ValueError(f"Unknown tool: {name}") from error
    return tool.run(
        **arguments,
        context_dir=context_dir,
        context_id=context_id,
        allow_external_files=allow_external_files,
    )
