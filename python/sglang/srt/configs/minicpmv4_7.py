# Copyright 2026 The SGLang team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Sglang-side ``PretrainedConfig`` class for MiniCPM-V 4.7 (OCR).

4.7 reuses the 4.6 stack unchanged — SigLip NaViT tower with a mid-ViT 2x2
window-attention merger, an MLP connector and a Qwen3.5 hybrid backbone — but
its released checkpoints ship a remote-code ``MiniCPMVConfig`` that keeps the
backbone fields flat at the top level and tags the release through a numeric
``version`` instead of nesting ``text_config`` / ``vision_config`` the way the
4.6 config does.

``MiniCPMV4_7Config.from_flat_config`` performs that conversion once, at config
load time, so everything downstream (hybrid GDN cache sizing, attention backend
selection, model dispatch) sees the nesting it already understands.
"""

from typing import Any, Dict, Optional, Tuple

from transformers import PretrainedConfig

from sglang.srt.configs.minicpmv4_6 import MiniCPMV4_6Config
from sglang.srt.configs.qwen3_5 import Qwen3_5TextConfig

# The ``version`` an OCR checkpoint tags itself with.
OCR_VERSION: Tuple[int, int] = (4, 7)

# Flat-config keys owned by the MiniCPM-V wrapper; everything else describes
# the Qwen3.5 backbone and is forwarded to ``text_config``.
_WRAPPER_KEYS = frozenset(
    {
        "_name_or_path",
        "architectures",
        "auto_map",
        "batch_3d_resampler",
        "batch_vision_input",
        "downsample_mode",
        "drop_vision_last_layer",
        "image_size",
        "insert_layer_id",
        "merge_kernel_size",
        "merger_times",
        "model_type",
        "patch_size",
        "query_num",
        "slice_config",
        "slice_mode",
        "transformers_version",
        "use_image_id",
        "version",
        "vision_batch_size",
        "vision_config",
    }
)

# Wrapper-only fields the image processor / vision stack reads back off the
# config; copied verbatim when present.
_PASSTHROUGH_KEYS = (
    "batch_3d_resampler",
    "batch_vision_input",
    "query_num",
    "slice_config",
    "slice_mode",
    "use_image_id",
    "vision_batch_size",
)


def parse_version(config: PretrainedConfig) -> Optional[Tuple[int, ...]]:
    """``config.version`` (e.g. ``4.7``) as a tuple, or ``None`` if absent."""
    return _parse_version(getattr(config, "version", None))


def _parse_version(version: Any) -> Optional[Tuple[int, ...]]:
    if version is None:
        return None
    try:
        return tuple(int(part) for part in str(version).split("."))
    except ValueError:
        return None


def is_flat_ocr_config(config: PretrainedConfig) -> bool:
    """Whether *config* is a released, not-yet-converted OCR checkpoint config."""
    return (
        getattr(config, "model_type", None) == "minicpmv"
        and parse_version(config) == OCR_VERSION
        and not hasattr(config, "text_config")
    )


def is_flat_ocr_config_dict(config: Dict[str, Any]) -> bool:
    """``is_flat_ocr_config`` for a raw ``config.json`` dict."""
    return (
        config.get("model_type") == "minicpmv"
        and _parse_version(config.get("version")) == OCR_VERSION
        and "text_config" not in config
    )


class MiniCPMV4_7Config(MiniCPMV4_6Config):
    model_type = "minicpmv4_7"

    # Explicit ``__init__``: transformers 5 wraps ``PretrainedConfig``
    # subclasses with ``@dataclass(kw_only=True)``, whose generated ``__init__``
    # would bypass ``MiniCPMV4_6Config.__init__`` and leave the sub-configs as
    # raw dicts.
    def __init__(
        self,
        version: float = 4.7,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.version = version

    @classmethod
    def from_flat_config(cls, config: PretrainedConfig) -> "MiniCPMV4_7Config":
        data: Dict[str, Any] = config.to_dict()

        vision_config = dict(data.get("vision_config") or {})
        vision_config.pop("model_type", None)
        insert_layer_id = data.get(
            "insert_layer_id", vision_config.pop("insert_layer_id", 6)
        )

        text_config = {
            key: value for key, value in data.items() if key not in _WRAPPER_KEYS
        }
        text_config["model_type"] = Qwen3_5TextConfig.model_type

        converted = cls(
            version=data.get("version", 4.7),
            text_config=text_config,
            vision_config=vision_config,
            insert_layer_id=insert_layer_id,
            image_size=data.get("image_size", 448),
            drop_vision_last_layer=data.get("drop_vision_last_layer", False),
            tie_word_embeddings=data.get("tie_word_embeddings", False),
            downsample_mode=data.get("downsample_mode", "16x"),
            dtype=data.get("dtype"),
            bos_token_id=data.get("bos_token_id"),
            eos_token_id=data.get("eos_token_id"),
            pad_token_id=data.get("pad_token_id"),
        )
        for key in _PASSTHROUGH_KEYS:
            if key in data:
                setattr(converted, key, data[key])

        # The released checkpoints load their image processor / tokenizer
        # through remote code, so the map has to survive the conversion.
        if getattr(config, "auto_map", None):
            converted.auto_map = config.auto_map
        converted._name_or_path = getattr(config, "_name_or_path", "")
        return converted


__all__ = [
    "MiniCPMV4_7Config",
    "OCR_VERSION",
    "is_flat_ocr_config",
    "is_flat_ocr_config_dict",
    "parse_version",
]
