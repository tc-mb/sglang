# Copyright 2026 The SGLang team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""SGLang multimodal processor for MiniCPM-V 4.7 OCR.

The 4.7 checkpoint uses the same image slicing and visual-token expansion as
4.6, so the backend-native 4.6 processor can be reused unchanged. In
particular, this avoids depending on modifications to the checkpoint's remote
``processing_minicpmv.py`` file: users only need to set ``version`` to ``4.7``.
"""

from sglang.srt.models.minicpmv import MiniCPMV4_7ForConditionalGeneration
from sglang.srt.multimodal.processors.minicpmv4_6 import (
    MiniCPMV4_6MultimodalProcessor,
)


class MiniCPMV4_7MultimodalProcessor(MiniCPMV4_6MultimodalProcessor):
    models = [MiniCPMV4_7ForConditionalGeneration]
