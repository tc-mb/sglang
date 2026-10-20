# Copyright 2026 The SGLang team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Config classes for MiniCPM-OCR 1.0.

The checkpoint reuses the MiniCPM-V 4.6 stack (SigLIP vision + Qwen3.5 text)
with ``model_type=minicpm_ocr`` and document-density defaults: ``downsample_mode``
``"4x"`` and every vision layer kept.
"""

from typing import Any

from sglang.srt.configs.minicpmv4_6 import MiniCPMV4_6Config, MiniCPMV4_6VisionConfig


class MiniCPMOcrVisionConfig(MiniCPMV4_6VisionConfig):
    model_type = "minicpm_ocr_vision"


class MiniCPMOcrConfig(MiniCPMV4_6Config):
    model_type = "minicpm_ocr"
    sub_configs = {
        "vision_config": MiniCPMOcrVisionConfig,
        "text_config": MiniCPMV4_6Config.sub_configs["text_config"],
    }

    def __init__(self, downsample_mode: str = "4x", **kwargs: Any) -> None:
        super().__init__(downsample_mode=downsample_mode, **kwargs)
        # OCR reads dense text and does not drop the last vision layer.
        self.drop_vision_last_layer = False


__all__ = ["MiniCPMOcrConfig", "MiniCPMOcrVisionConfig"]
