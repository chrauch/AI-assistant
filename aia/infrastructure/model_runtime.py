# This file is part of AIA. Copyright (C) 2026 Christian Rauch.
# Distributed under terms of the GPL3 license.

import os
import threading
import time
from collections.abc import Callable
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)
from transformers.utils import logging as transformers_logging
from transformers.utils.logging import disable_progress_bar, enable_progress_bar

from aia.infrastructure.logger import FileLogger
from aia.infrastructure.output import get_logger, out


transformers_logging.set_verbosity_error()


_cancel_event = threading.Event()


class GenerationCancelled(Exception):
    """Raised when the operator stops the active model generation."""


class CancelOnRequest(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return _cancel_event.is_set()


def request_cancel() -> None:
    _cancel_event.set()


def cancel_requested() -> bool:
    return _cancel_event.is_set()


def clear_cancel() -> None:
    _cancel_event.clear()


def configure_offline_mode() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


class ModelRuntime:
    def __init__(
        self,
        model_path: Path,
        log_dir: Path | None = None,
        context_id: str | None = None,
        show_progress: bool = False,
    ) -> None:
        configure_offline_mode()
        if show_progress:
            enable_progress_bar()
        else:
            disable_progress_bar()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
        else:
            dtype = torch.bfloat16
        self.model_path = model_path
        self.dtype = dtype
        if log_dir is not None and get_logger() is None:
            self.logger = FileLogger(log_dir)
        else:
            self.logger = get_logger()
        self.context_id = context_id
        out(
            "SYS",
            f"Initializing {model_path.name} on {self.device.upper()}.",
            #f"from {model_path}...",
            context_id=self.context_id,
        )

        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
        )
        if config.model_type == "idefics3":
            from transformers.models.idefics3.modeling_idefics3 import (
                Idefics3ForConditionalGeneration,
            )

            model_class = Idefics3ForConditionalGeneration
        else:
            model_class = AutoModelForCausalLM
        self.model = model_class.from_pretrained(
            model_path,
            dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        out(
            "SYS",
            "System ready. Use /help to list commands.",
            context_id=self.context_id,
        )
        self.model.eval()
        if self.logger is not None:
            self.logger.log(
                "model_metadata",
                context_id=self.context_id,
                model_path=str(model_path),
                device=self.device,
                model_config=self.model.config.to_dict(),
                generation_config=self.model.generation_config.to_dict(),
                processor_type=type(self.processor).__name__,
                model_type=type(self.model).__name__,
            )

    def stream(
        self,
        messages: list[dict[str, object]],
        *,
        add_tools: bool,
        context_id: str | None = None,
        on_text: Callable[[str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
        max_new_tokens: int = 512,
    ) -> tuple[str, float, int]:
        if add_tools:
            raise ValueError(
                "ModelRuntime does not add tools; use ToolRuntime for add_tools=True."
            )
        clear_cancel()
        messages = self._format_messages(messages)
        if self.logger is not None:
            self.logger.log(
                "messages_sent",
                context_id=context_id,
                messages=messages,
                add_tools=add_tools,
                max_new_tokens=max_new_tokens,
            )
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generation_args = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "streamer": streamer,
            "stopping_criteria": StoppingCriteriaList([CancelOnRequest()]),
        }
        generation_thread = threading.Thread(
            target=self._generate,
            kwargs=generation_args,
            daemon=True,
        )
        start_time = time.perf_counter()
        generation_thread.start()

        response_parts: list[str] = []
        pending_text = ""
        thinking = False
        for text in streamer:
            pending_text += text
            while pending_text:
                if thinking:
                    end_index = pending_text.find("</think>")
                    if end_index < 0:
                        thinking_text = pending_text[:-len("</think>") + 1]
                        pending_text = pending_text[-len("</think>") + 1:]
                        if thinking_text and on_thinking is not None:
                            on_thinking(thinking_text)
                        break
                    thinking_text = pending_text[:end_index]
                    pending_text = pending_text[end_index + len("</think>"):]
                    if thinking_text and on_thinking is not None:
                        on_thinking(thinking_text)
                    if on_thinking is not None:
                        on_thinking("</think>")
                    thinking = False
                    continue

                start_index = pending_text.find("<think>")
                if start_index < 0:
                    safe_length = len(pending_text) - len("<think>") + 1
                    visible_text = pending_text[:safe_length]
                    pending_text = pending_text[safe_length:]
                    if visible_text:
                        response_parts.append(visible_text)
                        if on_text is not None:
                            on_text(visible_text)
                    break

                visible_text = pending_text[:start_index]
                pending_text = pending_text[start_index + len("<think>"):]
                thinking = True
                if on_thinking is not None:
                    on_thinking("<think>")
                if visible_text:
                    response_parts.append(visible_text)
                    if on_text is not None:
                        on_text(visible_text)

        if pending_text and not thinking:
            response_parts.append(pending_text)
            if on_text is not None:
                on_text(pending_text)

        generation_thread.join()
        elapsed_seconds = time.perf_counter() - start_time
        if cancel_requested():
            if self.logger is not None:
                self.logger.log(
                    "generation_cancelled",
                    context_id=context_id,
                    elapsed_seconds=elapsed_seconds,
                )
            clear_cancel()
            raise GenerationCancelled()
        response = "".join(response_parts).strip()
        if self.logger is not None:
            self.logger.log(
                "response_received",
                context_id=context_id,
                response=response,
                elapsed_seconds=elapsed_seconds,
                streamed_chunks=len(response_parts),
            )
        return response, elapsed_seconds, len(response_parts)

    def _format_messages(
        self,
        messages: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        if self.model.config.model_type != "idefics3":
            return list(messages)
        formatted_messages = []
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            formatted_messages.append({**message, "content": content})
        return formatted_messages

    def _generate(self, **generation_args: object) -> None:
        with torch.inference_mode():
            self.model.generate(**generation_args)
