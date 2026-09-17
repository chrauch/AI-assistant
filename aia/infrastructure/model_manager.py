# This file is part of AIA. Copyright (C) 2026 Christian Rauch.
# Distributed under terms of the GPL3 license.

import json
import math
import os
from pathlib import Path

# Xet transfer can fail on some Python/platform combinations. The regular
# HTTPS downloader supports resuming the partially downloaded model.
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, GenerationConfig
from aia.infrastructure.output import out


def list_available_models(models_dir: Path) -> list[Path]:
    if not models_dir.is_dir():
        return []
    return sorted(
        model_path
        for model_path in models_dir.iterdir()
        if model_path.is_dir() and model_is_complete(model_path)
    )


def model_is_complete(model_path: Path) -> bool:
    return (
        (model_path / "config.json").is_file()
        and (
            (model_path / "model.safetensors.index.json").is_file()
            or any(model_path.glob("*.safetensors"))
        )
    )


def ensure_model(
    model_id: str,
    models_dir: Path,
    context_id: str | None = None,
) -> Path:
    model_path = models_dir / model_id.replace("/", "_")
    if model_is_complete(model_path):
        log_model_metadata(model_path, context_id=context_id)
        return model_path

    model_path.mkdir(parents=True, exist_ok=True)
    out(
        "SYS",
        f"Downloading {model_id} to {model_path}...",
        context_id=context_id,
    )
    previous_offline = os.environ.pop("HF_HUB_OFFLINE", None)
    try:
        snapshot_download(repo_id=model_id, local_dir=model_path, max_workers=1)
    finally:
        if previous_offline is not None:
            os.environ["HF_HUB_OFFLINE"] = previous_offline
    out("SYS", "Download complete.", context_id=context_id)
    log_model_metadata(model_path, context_id=context_id)
    return model_path


def log_model_metadata(
    model_path: Path,
    model_id: str | None = None,
    context_id: str | None = None,
) -> None:
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    generation_config = GenerationConfig.from_pretrained(
        model_path,
        local_files_only=True,
    )

    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open() as file:
            weight_map = json.load(file)["weight_map"]
        shard_names = set(weight_map.values())
    else:
        weight_map = None
        shard_names = {path.name for path in model_path.glob("*.safetensors")}

    parameter_count = 0
    for shard_name in shard_names:
        with safe_open(model_path / shard_name, framework="pt", device="cpu") as shard:
            parameter_count += sum(
                math.prod(shard.get_slice(name).get_shape())
                for name in shard.keys()
                if weight_map is None or weight_map[name] == shard_name
            )

    total_size = sum(
        (model_path / shard_name).stat().st_size
        for shard_name in shard_names
    )

    metadata_lines = ["Model metadata:"]
    if model_id is not None:
        metadata_lines.append(f"  Model: {model_id}")
    metadata_lines.extend(
        [
            f"  Model path: {model_path}",
            f"  Architecture: {config.architectures}",
            f"  Model type: {config.model_type}",
            f"  Parameters: {parameter_count:,}",
            f"  Weight size: {total_size / 1024**3:.2f} GiB",
            f"  Torch dtype: {config.torch_dtype}",
        ]
    )
    text_config = getattr(config, "text_config", config)
    metadata_lines.extend(
        [
            f"  Text layers: {text_config.num_hidden_layers}",
            f"  Text hidden size: {text_config.hidden_size}",
        ]
    )
    vision_config = getattr(config, "vision_config", None)
    if vision_config is not None:
        metadata_lines.append(f"  Vision config: {vision_config}")
    metadata_lines.append(f"  Generation config: {generation_config}")
    out(
        "SYS",
        "\n".join(metadata_lines),
        context_id=context_id,
        destination="log",
    )


