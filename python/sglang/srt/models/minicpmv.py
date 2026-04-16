# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The SGLang team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only MiniCPM-V model compatible with HuggingFace weights."""

import types
from functools import partial
from itertools import chain
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    TypedDict,
    Union,
)

import numpy as np
import torch
import torch.types
from PIL import Image
from torch import nn
from torch.nn.init import trunc_normal_
from transformers import PretrainedConfig

from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternTokenPairs,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    MultimodalDataItem,
    MultimodalInputFormat,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.idefics2 import Idefics2VisionTransformer
from sglang.srt.models.llama import LlamaConfig, LlamaForCausalLM
from sglang.srt.models.qwen2 import Qwen2Config, Qwen2ForCausalLM
from sglang.srt.models.qwen3 import Qwen3Config, Qwen3ForCausalLM
from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM
from sglang.srt.utils import add_prefix, flatten_nested_list

RawImageType = Union[Image.Image, torch.Tensor]


# sin/cos positional embedding helpers are adapted from:
# https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py#L20
def get_1d_sincos_pos_embed_from_grid(
    embed_dim: int, pos: np.ndarray, version: Tuple[int, int] = (2, 0)
) -> torch.Tensor:
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,) / (H, W)
    out: (M, D) / (H, W, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    if version == (2, 0):
        pos = pos.reshape(-1)  # (M,)
        out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product
        emb_sin = np.sin(out)  # (M, D/2)
        emb_cos = np.cos(out)  # (M, D/2)
        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    else:
        out = np.einsum("hw,d->hwd", pos, omega)  # (H, W, D/2), outer product
        emb_sin = np.sin(out)  # (H, W, D/2)
        emb_cos = np.cos(out)  # (H, W, D/2)
        emb = np.concatenate([emb_sin, emb_cos], axis=-1)  # (H, W, D)
    return emb


def get_2d_sincos_pos_embed_from_grid(
    embed_dim: int, grid: np.ndarray, version: Tuple[int, int] = (2, 0)
) -> torch.Tensor:
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0], version
    )  # (H*W, D/2) or (H, W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1], version
    )  # (H*W, D/2) or (H, W, D/2)

    if version == (2, 0):
        emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    else:
        emb = np.concatenate([emb_h, emb_w], axis=-1)  # (H, W, D)
    return emb


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: Union[int, Tuple[int, int]],
    cls_token: bool = False,
    version: Tuple[int, int] = (2, 0),
) -> torch.Tensor:
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or
                [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_h_size, grid_w_size = grid_size, grid_size
    else:
        grid_h_size, grid_w_size = grid_size[0], grid_size[1]

    grid_h = np.arange(grid_h_size, dtype=np.float32)
    grid_w = np.arange(grid_w_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    assert isinstance(grid, np.ndarray) and grid.shape == (2, grid_h_size, grid_w_size)

    if version == (2, 0):
        grid = grid.reshape([2, 1, grid_h_size, grid_w_size])
        pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid, version)
        if cls_token:
            pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    else:
        pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid, version)
    return pos_embed


class MiniCPMVImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: List[torch.Tensor]
    """
    Shape: `(batch_size * num_images, num_channels, height, width)`

    Note that the image size may vary, so we pass it as a list
    instead of a batched tensor.
    """

    image_bounds: torch.Tensor
    """
    Shape: `(batch_size * num_images, 2)`

    This should be in `(start, stop)` format.
    """

    tgt_sizes: torch.Tensor
    """
    Shape: `(batch_size * num_images, 2)`

    This should be in `(height, width)` format.
    """


class MiniCPMVImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    data: torch.Tensor
    """
    Shape: `(batch_size * num_images, image_feature_size, hidden_size)`

    `hidden_size` must match the hidden size of language model backbone.
    instead of a batched tensor.
    """

    image_bounds: torch.Tensor
    """
    Shape: `(batch_size * num_images, 2)`

    This should be in `(start, stop)` format.
    """


MiniCPMVImageInputs = Union[MiniCPMVImagePixelInputs, MiniCPMVImageEmbeddingInputs]

DEFAULT_LN = partial(nn.LayerNorm, eps=1e-6)


class BaseResampler(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
        (grid_size**2) learnable queries and 2d sincos pos_emb.
    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)
    """

    def __init__(
        self,
        num_queries: int,
        embed_dim: int,
        num_heads: int,
        kv_dim: Optional[int] = None,
        norm_layer: Callable[[int], nn.LayerNorm] = DEFAULT_LN,
        do_post_projection: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.query = nn.Parameter(torch.zeros(self.num_queries, embed_dim))
        trunc_normal_(self.query, std=0.02)
        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = ReplicatedLinear(
                kv_dim,
                embed_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("kv_proj", prefix),
            )
        else:
            # Maintain the same return value with ReplicatedLinear.forward
            self.kv_proj = lambda *args, **kwargs: (  # type: ignore # noqa
                nn.Identity()(*args, **kwargs),
                None,
            )
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)
        self.do_post_projection = do_post_projection
        self.ln_post = norm_layer(embed_dim) if do_post_projection else None
        self.proj = (
            nn.Parameter((embed_dim**-0.5) * torch.randn(embed_dim, embed_dim))
            if do_post_projection
            else None
        )

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)


class Resampler2_5(BaseResampler):

    def __init__(
        self,
        num_queries: int,
        embed_dim: int,
        num_heads: int,
        kv_dim: Optional[int] = None,
        norm_layer: Callable[[int], nn.LayerNorm] = DEFAULT_LN,
        max_size: Tuple[int, int] = (70, 70),
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            num_queries,
            embed_dim,
            num_heads,
            kv_dim,
            norm_layer,
            quant_config=quant_config,
            prefix=prefix,
        )

        self.max_size = max_size
        self._set_2d_pos_cache(self.max_size)

        self.apply(self._init_weights)

    def _set_2d_pos_cache(
        self, max_size: Tuple[int, int], device: torch.types.Device = "cpu"
    ) -> None:
        pos_embed_arr = get_2d_sincos_pos_embed(
            self.embed_dim, max_size, version=(2, 5)
        )
        pos_embed = torch.from_numpy(pos_embed_arr).float().to(device)
        self.register_buffer("pos_embed", pos_embed, persistent=False)

    def _adjust_pos_cache(
        self, tgt_sizes: torch.Tensor, device: torch.types.Device
    ) -> None:
        max_h = tgt_sizes[:, 0].max().item()
        max_w = tgt_sizes[:, 1].max().item()
        assert isinstance(max_h, int) and isinstance(max_w, int)

        if max_h > self.max_size[0] or max_w > self.max_size[1]:
            self.max_size = (
                max(max_h, self.max_size[0]),
                max(max_w, self.max_size[1]),
            )
            self._set_2d_pos_cache(self.max_size, device)

    def forward(self, x: torch.Tensor, tgt_sizes: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == tgt_sizes.shape[0]
        bs = x.shape[0]

        device = x.device
        dtype = x.dtype

        patch_len = tgt_sizes[:, 0] * tgt_sizes[:, 1]

        self._adjust_pos_cache(tgt_sizes, device=device)

        max_patch_len = patch_len.max().item()
        assert isinstance(max_patch_len, int)

        key_padding_mask = torch.zeros(
            (bs, max_patch_len), dtype=torch.bool, device=device
        )

        pos_embed = []
        for i in range(bs):
            tgt_h, tgt_w = tgt_sizes[i].tolist()
            pos_embed.append(
                self.pos_embed[:tgt_h, :tgt_w, :].reshape((tgt_h * tgt_w, -1)).to(dtype)
            )  # patches * D
            key_padding_mask[i, patch_len[i] :] = True
        pos_embed = torch.nn.utils.rnn.pad_sequence(
            pos_embed, batch_first=True, padding_value=0.0
        ).permute(
            1, 0, 2
        )  # BLD => L * B * D
        x, _ = self.kv_proj(x)  # B * L * D
        x = self.ln_kv(x).permute(1, 0, 2)  # L * B * D

        q = self.ln_q(self.query)  # Q * D

        out = self.attn(
            self._repeat(q, bs),  # Q * B * D
            x + pos_embed,  # L * B * D +  L * B * D
            x,
            key_padding_mask=key_padding_mask,
        )[0]
        #  out: Q * B * D
        x = out.permute(1, 0, 2)  # B * Q * D

        x = self.ln_post(x)
        x = x @ self.proj
        return x


class Resampler4_5(BaseResampler):

    def __init__(
        self,
        num_queries: int,
        embed_dim: int,
        num_heads: int,
        kv_dim: Optional[int] = None,
        norm_layer: Callable[[int], nn.LayerNorm] = DEFAULT_LN,
        max_size: tuple[int, int] = (70, 70),
        max_temporal_size=36000,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            num_queries,
            embed_dim,
            num_heads,
            kv_dim,
            norm_layer,
            quant_config=quant_config,
            prefix=prefix,
        )

        self.max_size = max_size
        self.max_temporal_size = max_temporal_size

        self._set_2d_pos_cache(self.max_size)
        self._set_temporal_pos_cache(self.max_temporal_size)
        self.apply(self._init_weights)

    def get_1d_sincos_pos_embed_from_temporal_size(
        self, embed_dim: int, pos: np.ndarray
    ):
        """
        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)
        out: (M, D)
        """
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float32)
        omega /= embed_dim / 2.0
        omega = 1.0 / 10000**omega  # (D/2,)

        pos = pos.reshape(-1)  # (M,)
        out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

        emb_sin = np.sin(out)  # (M, D/2)
        emb_cos = np.cos(out)  # (M, D/2)

        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
        return emb

    def _set_2d_pos_cache(
        self, max_size: tuple[int, int], device: torch.types.Device = "cpu"
    ) -> None:
        pos_embed_arr = get_2d_sincos_pos_embed(
            self.embed_dim, max_size, version=(2, 5)
        )
        pos_embed = torch.from_numpy(pos_embed_arr).float().to(device)
        self.register_buffer("pos_embed", pos_embed, persistent=False)

    def _adjust_pos_cache(
        self, tgt_sizes: torch.Tensor, device: torch.types.Device
    ) -> None:
        max_h = tgt_sizes[:, 0].max().item()
        max_w = tgt_sizes[:, 1].max().item()
        assert isinstance(max_h, int) and isinstance(max_w, int)

        if max_h > self.max_size[0] or max_w > self.max_size[1]:
            self.max_size = (
                max(max_h, self.max_size[0]),
                max(max_w, self.max_size[1]),
            )
            self._set_2d_pos_cache(self.max_size, device)

    def _set_temporal_pos_cache(
        self, max_temporal_size: int, device: torch.types.Device = "cpu"
    ) -> None:
        temporal_size = np.arange(max_temporal_size, dtype=np.float32)
        pos_embed = (
            torch.from_numpy(
                self.get_1d_sincos_pos_embed_from_temporal_size(
                    self.embed_dim, temporal_size
                )
            )
            .float()
            .to(device)
        )
        self.register_buffer("temporal_pos_embed", pos_embed, persistent=False)

    def _adjust_temporal_pos_cache(
        self, max_temporal_size: int, device: torch.types.Device = "cpu"
    ):
        if max_temporal_size > self.max_temporal_size:
            self.max_temporal_size = max_temporal_size
            self._set_temporal_pos_cache(self.max_temporal_size, device)

    def forward(
        self, x: torch.Tensor, tgt_sizes: torch.Tensor, temporal_ids=None
    ) -> torch.Tensor:
        assert x.shape[0] == tgt_sizes.shape[0]
        bs = x.shape[0]

        device = x.device
        dtype = x.dtype

        patch_len = tgt_sizes[:, 0] * tgt_sizes[:, 1]

        self._adjust_pos_cache(tgt_sizes, device=device)

        temporal_pos_emb = False
        temporal_ids_flatten = None
        if temporal_ids is not None:
            # example: [[-1], [-1], [2, 6, 9]]
            temporal_ids_flatten = list(chain.from_iterable(temporal_ids))
            max_temporal_size = max(temporal_ids_flatten)
            if max_temporal_size > -1:
                temporal_pos_emb = True
            if max_temporal_size > self.max_temporal_size:
                self._adjust_temporal_pos_cache(max_temporal_size, device)

        max_patch_len = patch_len.max().item()
        assert isinstance(max_patch_len, int)

        key_padding_mask = torch.zeros(
            (bs, max_patch_len), dtype=torch.bool, device=device
        )

        x, _ = self.kv_proj(x)  # B * L * D
        x = self.ln_kv(x).permute(1, 0, 2)  # L * B * D
        q = self.ln_q(self.query)  # Q * D

        pos_embed_2d = []
        pos_embed_temporal = []
        for i in range(bs):
            tgt_h, tgt_w = tgt_sizes[i]
            if temporal_pos_emb:
                if temporal_ids_flatten[i] == -1:
                    pos_embed_temporal.append(
                        torch.zeros(self.embed_dim, dtype=dtype, device=device)
                    )
                else:
                    pos_embed_temporal.append(
                        self.temporal_pos_embed[temporal_ids_flatten[i]].to(dtype)
                    )  # D

            pos_embed_2d.append(
                self.pos_embed[:tgt_h, :tgt_w, :].reshape((tgt_h * tgt_w, -1)).to(dtype)
            )  # patches * D
            key_padding_mask[i, patch_len[i] :] = True

        pos_embed_2d = torch.nn.utils.rnn.pad_sequence(
            pos_embed_2d, batch_first=True, padding_value=0.0
        ).permute(
            1, 0, 2
        )  # BLD => L * B * D

        k = x
        v = x + pos_embed_2d

        if pos_embed_temporal:
            k += torch.stack(pos_embed_temporal, dim=0)
            bs = len(temporal_ids)
            merge_k = []
            merge_v = []
            merge_key_padding_mask = []

            start = 0
            for tp in temporal_ids:
                end = start + len(tp)
                # # L * (end-start) * D -> (end-start) * L * D -> 1 * L*(end-start) * D
                merge_k.append(
                    k[:, start:end, :].permute(1, 0, 2).reshape(-1, self.embed_dim)
                )
                merge_v.append(
                    v[:, start:end, :].permute(1, 0, 2).reshape(-1, self.embed_dim)
                )
                merge_key_padding_mask.append(
                    key_padding_mask[start:end, :].reshape(-1, 1)
                )

                start = end

            k = torch.nn.utils.rnn.pad_sequence(
                merge_k, batch_first=True, padding_value=0.0
            ).permute(
                1, 0, 2
            )  # L*(end-start)
            v = torch.nn.utils.rnn.pad_sequence(
                merge_v, batch_first=True, padding_value=0.0
            ).permute(
                1, 0, 2
            )  # L*(end-start)
            key_padding_mask = torch.nn.utils.rnn.pad_sequence(
                merge_key_padding_mask, batch_first=True, padding_value=True
            ).squeeze(-1)

        out = self.attn(
            self._repeat(q, bs),  # Q * B * D
            k,  # L * B * D +  L * B * D
            v,
            key_padding_mask=key_padding_mask,
        )[0]
        #  out: Q * B * D
        x = out.permute(1, 0, 2)  # B * Q * D

        x = self.ln_post(x)
        x = x @ self.proj
        return x


def get_version_by_config(config: PretrainedConfig) -> Tuple[int, ...]:
    version_float = getattr(config, "version", None)

    # The old configs do not include version number
    # TODO: Remove this after the HF repos are updated
    if version_float is None:
        if config.hidden_size == 2304 and config.query_num == 64:
            return 2, 0
        return 2, 5

    version_str = str(version_float)
    return tuple(int(x) for x in version_str.split("."))


class MiniCPMBaseModel(nn.Module):
    """
    The abstract class of MiniCPMV can only be inherited, but cannot be
    instantiated.
    """

    def __init__(
        self,
        *,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        # All MiniCPM-V models disable `tie_word_embeddings` but
        # `PretrainedConfig.tie_word_embeddings` defaults to True; we cannot
        # check `tie_word_embeddings` until SGLang integrate MiniCPM-V model
        # and config class
        self.config = config

        self.version = get_version_by_config(self.config)
        self.llm = self.init_llm(
            config=config, quant_config=quant_config, prefix=add_prefix("llm", prefix)
        )
        self.vpm = self.init_vision_module(
            config, quant_config, add_prefix("vpm", prefix)
        )
        self.vision_dim = (
            self.vpm.embed_dim
            if self.version == (2, 0)
            else self.vpm.embeddings.embed_dim
        )
        self.embed_dim = self.config.hidden_size

        self.resampler = self.init_resampler(
            self.embed_dim,
            self.vision_dim,
            quant_config=quant_config,
            prefix=add_prefix("resampler", prefix),
        )

        self.logits_processor = LogitsProcessor(config)

    def _get_image_bounds(
        self,
        input_ids: torch.Tensor,
        pad_values: List[int],
        im_start_id: int,
        im_end_id: int,
        slice_start_id: Optional[int] = None,
        slice_end_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Returns a tensor indicating the bounds (start and end token ids) of the images
        """
        # All the images in the batch should share the same special image
        # bound token ids.
        start_cond = input_ids == im_start_id
        end_cond = input_ids == im_end_id
        if slice_start_id is not None:
            start_cond |= input_ids == slice_start_id
            end_cond |= input_ids == slice_end_id

        (image_start_tokens,) = torch.where(start_cond)
        image_start_tokens += 1
        (image_end_tokens,) = torch.where(end_cond)

        # the im_start_id sometimes can be cached as prefix, but it is needed for the embedding of the images
        if len(image_start_tokens) != len(image_end_tokens):
            if (
                len(image_start_tokens) + 1 == len(image_end_tokens)
                and input_ids[0] in pad_values
                and len(image_start_tokens) != 0
                and len(image_end_tokens) != 0
                and image_end_tokens[0] < image_start_tokens[0]
            ):
                image_start_tokens = torch.cat(
                    [
                        torch.tensor([0], device=image_start_tokens.device),
                        image_start_tokens,
                    ]
                )
        valid_image_nums = min(len(image_start_tokens), len(image_end_tokens))

        if valid_image_nums == 0:
            return torch.zeros((0, 2), device=input_ids.device)

        # Filter out pairs where start_token >= end_token
        valid_pairs = []
        for i in range(valid_image_nums):
            start_token = image_start_tokens[i]
            end_token = image_end_tokens[i]
            if start_token < end_token:
                valid_pairs.append((start_token, end_token))

        if not valid_pairs:
            return torch.zeros((0, 2), device=input_ids.device)

        # Convert valid pairs to tensor
        valid_pairs_tensor = torch.tensor(valid_pairs, device=input_ids.device)
        return valid_pairs_tensor

    def _parse_and_validate_inputs(
        self,
        input_ids: torch.Tensor,
        **kwargs: object,
    ) -> Optional[MiniCPMVImageInputs]:
        pixel_values = kwargs.pop("pixel_values", [])
        tgt_sizes = kwargs.pop("tgt_sizes", [])
        im_start_id = kwargs.pop("im_start_id", None)
        im_end_id = kwargs.pop("im_end_id", None)
        slice_start_id = kwargs.pop("slice_start_id", None)
        slice_end_id = kwargs.pop("slice_end_id", None)
        image_embeds = kwargs.pop("image_embeds", None)
        pad_values = kwargs.pop("pad_values", None)

        if image_embeds is not None:
            image_bounds = self._get_image_bounds(
                input_ids=input_ids,
                pad_values=pad_values,
                im_start_id=im_start_id,
                im_end_id=im_end_id,
                slice_start_id=slice_start_id,
                slice_end_id=slice_end_id,
            )
            if not isinstance(image_embeds, (torch.Tensor, list)):
                raise ValueError(
                    f"Incorrect type of image embeds. "
                    f"Got type: {type(image_embeds)}"
                )

            if isinstance(image_embeds, list):
                image_embeds = torch.cat(image_embeds)

            return MiniCPMVImageEmbeddingInputs(
                image_bounds=image_bounds,
                data=image_embeds,
                type="image_embeds",
            )

        image_bounds = self._get_image_bounds(
            input_ids=input_ids,
            pad_values=pad_values,
            im_start_id=im_start_id,
            im_end_id=im_end_id,
            slice_start_id=slice_start_id,
            slice_end_id=slice_end_id,
        )
        return MiniCPMVImagePixelInputs(
            image_bounds=image_bounds.to(device=input_ids.device),
            data=pixel_values,
            tgt_sizes=tgt_sizes,
            type="pixel_values",
        )

    def get_embedding(
        self,
        input_ids: torch.Tensor,
        image_inputs: Optional[MiniCPMVImageInputs],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        vlm_embedding: torch.Tensor = self.llm.get_input_embeddings(input_ids)

        if image_inputs is None:  # No image
            vision_hidden_states = torch.tensor([], device=input_ids.device)
        else:
            if image_inputs["type"] == "image_embeds":
                vision_hidden_states = (
                    image_inputs["data"]
                    .type(vlm_embedding.dtype)
                    .to(vlm_embedding.device)
                )
            else:
                vision_hidden_states = self.get_vision_hidden_states(image_inputs)
            # See NOTE in _parse_and_validate_inputs
            image_bounds = image_inputs["image_bounds"]
            if len(image_bounds) > 0:
                image_indices = torch.stack(
                    [
                        torch.arange(start, end, dtype=torch.long)
                        for start, end in image_bounds.tolist()
                    ]
                ).to(vlm_embedding.device)

                vlm_embedding.scatter_(
                    0,
                    image_indices.view(-1, 1).repeat(1, vlm_embedding.shape[-1]),
                    vision_hidden_states.view(-1, vision_hidden_states.shape[-1]),
                )

        return vlm_embedding, vision_hidden_states

    def get_input_embeddings(self) -> nn.Embedding:
        return self.llm.get_input_embeddings()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ) -> torch.Tensor:
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            multimodal_model=self,
            language_model=self.llm,
            positions=positions,
        )
        return hidden_states

    def init_llm(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> nn.Module:
        raise NotImplementedError

    def init_vision_module(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str = "",
    ) -> nn.Module:
        raise NotImplementedError

    def init_resampler(
        self,
        embed_dim: int,
        vision_dim: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> nn.Module:
        raise NotImplementedError

    def get_vision_embedding(
        self,
        pixel_values: List[torch.Tensor],
        patch_attn_mask: Optional[torch.Tensor] = None,
        tgt_sizes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        raise NotImplementedError


class MiniCPMV2_6(MiniCPMBaseModel):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }
    # LoRA specific attributes
    supported_lora_modules = [
        # vision encoder
        "fc1",
        "fc2",
        "out_proj",
        # language model
        "qkv_proj",  # same name with vision encoder
        "o_proj",
        "gate_up_proj",
        "down_proj",
        # resampler
        "kv_proj",
    ]

    # BitandBytes specific attributes
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    embedding_modules = {}
    embedding_padding_modules = []

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        assert self.version == (2, 6)

    def init_llm(
        self,
        config: Qwen2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> nn.Module:
        return Qwen2ForCausalLM(config=config, quant_config=quant_config, prefix=prefix)

    def init_vision_module(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str = "",
    ) -> nn.Module:
        model = Idefics2VisionTransformer(
            config=config.vision_config, quant_config=quant_config, prefix=prefix
        )
        if self.config.drop_vision_last_layer:
            model.encoder.layers = model.encoder.layers[:-1]

        setattr(model, "embed_dim", model.embeddings.embed_dim)
        setattr(model, "patch_size", model.embeddings.patch_size)
        return model

    def init_resampler(
        self,
        embed_dim: int,
        vision_dim: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> nn.Module:
        with set_default_torch_dtype(torch.float16):
            # The resampler in 2.6 remains consistent with the one in 2.5.
            resampler = Resampler2_5(
                num_queries=self.config.query_num,
                embed_dim=embed_dim,
                num_heads=embed_dim // 128,
                kv_dim=vision_dim,
                quant_config=quant_config,
                prefix=prefix,
            )

        return resampler.to(device="cuda", dtype=torch.get_default_dtype())

    def get_vision_embedding(
        self,
        pixel_values: List[torch.Tensor],
        patch_attn_mask: Optional[torch.Tensor] = None,
        tgt_sizes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        vision_embedding = self.vpm(
            pixel_values,
            patch_attention_mask=patch_attn_mask,
            tgt_sizes=tgt_sizes,
        )
        return vision_embedding

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        if items and items[0].format == MultimodalInputFormat.PRECOMPUTED_EMBEDDING:
            result = torch.cat([item.feature for item in items])
            return result.reshape(-1, result.shape[-1])

        # list of tensors
        pixel_values = flatten_nested_list([item.feature for item in items])
        tgt_sizes = torch.stack(
            flatten_nested_list([item.tgt_size for item in items]), dim=0
        )
        assert len(pixel_values) == tgt_sizes.shape[0]

        device = self.vpm.embeddings.position_embedding.weight.device
        dtype = self.vpm.embeddings.position_embedding.weight.dtype
        all_pixel_values_lst = [
            i.flatten(end_dim=1).permute(1, 0) for i in pixel_values
        ]

        max_patches = (tgt_sizes[:, 0] * tgt_sizes[:, 1]).max().item()
        assert isinstance(max_patches, int)
        all_pixel_values = torch.nn.utils.rnn.pad_sequence(
            all_pixel_values_lst, batch_first=True, padding_value=0.0
        )

        B, L, _ = all_pixel_values.shape
        all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)
        patch_attn_mask = torch.zeros(
            (B, 1, max_patches), dtype=torch.bool, device=device
        )

        tgt_sizes_tensor = tgt_sizes.clone().to(device=patch_attn_mask.device)
        mask_shapes = tgt_sizes_tensor[:, 0] * tgt_sizes_tensor[:, 1]
        patch_attn_mask[:, 0, :] = torch.arange(
            patch_attn_mask.size(2), device=patch_attn_mask.device
        ).unsqueeze(0) < mask_shapes.unsqueeze(1)

        vision_embedding = self.vpm(
            all_pixel_values.type(dtype),
            patch_attention_mask=patch_attn_mask,
            tgt_sizes=tgt_sizes,
        )
        return self.resampler(vision_embedding, tgt_sizes)

    def pad_input_ids(self, input_ids: List[int], image_inputs: MultimodalInputs):
        # Get all special token IDs
        im_start_id: int = image_inputs.im_start_id
        im_end_id: int = image_inputs.im_end_id
        slice_start_id: int = image_inputs.slice_start_id
        slice_end_id: int = image_inputs.slice_end_id

        media_token_pairs = [(im_start_id, im_end_id), (slice_start_id, slice_end_id)]
        pattern = MultiModalityDataPaddingPatternTokenPairs(media_token_pairs)

        return pattern.pad_input_tokens(input_ids, image_inputs)


class MiniCPMV4_0(MiniCPMBaseModel):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }
    # LoRA specific attributes
    supported_lora_modules = [
        # vision encoder
        "fc1",
        "fc2",
        "out_proj",
        # language model
        "qkv_proj",  # same name with vision encoder
        "o_proj",
        "gate_up_proj",
        "down_proj",
        # resampler
        "kv_proj",
    ]

    # BitandBytes specific attributes
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    embedding_modules = {}
    embedding_padding_modules = []

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        assert self.version == (4, 0)

    def init_llm(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> nn.Module:
        return LlamaForCausalLM(config=config, quant_config=quant_config, prefix=prefix)

    def init_vision_module(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str = "",
    ) -> nn.Module:
        model = Idefics2VisionTransformer(
            config=config.vision_config, quant_config=quant_config, prefix=prefix
        )
        if self.config.drop_vision_last_layer:
            model.encoder.layers = model.encoder.layers[:-1]

        setattr(model, "embed_dim", model.embeddings.embed_dim)
        setattr(model, "patch_size", model.embeddings.patch_size)
        return model

    def init_resampler(
        self,
        embed_dim: int,
        vision_dim: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> nn.Module:
        with set_default_torch_dtype(torch.float16):
            # The resampler in 2.6 remains consistent with the one in 2.5.
            resampler = Resampler2_5(
                num_queries=self.config.query_num,
                embed_dim=embed_dim,
                num_heads=embed_dim // 128,
                kv_dim=vision_dim,
                quant_config=quant_config,
                prefix=prefix,
            )

        return resampler.to(device="cuda", dtype=torch.get_default_dtype())

    def get_vision_embedding(
        self,
        pixel_values: List[torch.Tensor],
        patch_attn_mask: Optional[torch.Tensor] = None,
        tgt_sizes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        vision_embedding = self.vpm(
            pixel_values,
            patch_attention_mask=patch_attn_mask,
            tgt_sizes=tgt_sizes,
        )
        return vision_embedding

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        if items and items[0].format == MultimodalInputFormat.PRECOMPUTED_EMBEDDING:
            result = torch.cat([item.feature for item in items])
            return result.reshape(-1, result.shape[-1])

        # list of tensors
        pixel_values = flatten_nested_list([item.feature for item in items])
        tgt_sizes = torch.stack(
            flatten_nested_list([item.tgt_size for item in items]), dim=0
        )
        assert len(pixel_values) == tgt_sizes.shape[0]

        device = self.vpm.embeddings.position_embedding.weight.device
        dtype = self.vpm.embeddings.position_embedding.weight.dtype
        all_pixel_values_lst = [
            i.flatten(end_dim=1).permute(1, 0) for i in pixel_values
        ]

        max_patches = (tgt_sizes[:, 0] * tgt_sizes[:, 1]).max().item()
        assert isinstance(max_patches, int)
        all_pixel_values = torch.nn.utils.rnn.pad_sequence(
            all_pixel_values_lst, batch_first=True, padding_value=0.0
        )

        B, L, _ = all_pixel_values.shape
        all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)
        patch_attn_mask = torch.zeros(
            (B, 1, max_patches), dtype=torch.bool, device=device
        )

        tgt_sizes_tensor = tgt_sizes.clone().to(device=patch_attn_mask.device)
        mask_shapes = tgt_sizes_tensor[:, 0] * tgt_sizes_tensor[:, 1]
        patch_attn_mask[:, 0, :] = torch.arange(
            patch_attn_mask.size(2), device=patch_attn_mask.device
        ).unsqueeze(0) < mask_shapes.unsqueeze(1)

        vision_embedding = self.vpm(
            all_pixel_values.type(dtype),
            patch_attention_mask=patch_attn_mask,
            tgt_sizes=tgt_sizes,
        )
        return self.resampler(vision_embedding, tgt_sizes)

    def pad_input_ids(self, input_ids: List[int], image_inputs: MultimodalInputs):
        # Get all special token IDs
        im_start_id: int = image_inputs.im_start_id
        im_end_id: int = image_inputs.im_end_id
        slice_start_id: int = image_inputs.slice_start_id
        slice_end_id: int = image_inputs.slice_end_id

        media_token_pairs = [(im_start_id, im_end_id), (slice_start_id, slice_end_id)]
        pattern = MultiModalityDataPaddingPatternTokenPairs(media_token_pairs)

        return pattern.pad_input_tokens(input_ids, image_inputs)


class MiniCPMV4_5(MiniCPMBaseModel):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }
    # LoRA specific attributes
    supported_lora_modules = [
        # vision encoder
        "fc1",
        "fc2",
        "out_proj",
        # language model
        "qkv_proj",  # same name with vision encoder
        "o_proj",
        "gate_up_proj",
        "down_proj",
        # resampler
        "kv_proj",
    ]

    # BitandBytes specific attributes
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    embedding_modules = {}
    embedding_padding_modules = []

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        assert self.version == (4, 5)

    def init_llm(
        self,
        config: Qwen3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> nn.Module:
        llm = Qwen3ForCausalLM(config=config, quant_config=quant_config, prefix=prefix)
        llm.get_input_embeddings = types.MethodType(
            lambda self: self.model.get_input_embeddings(), llm
        )
        return llm

    def init_vision_module(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str = "",
    ) -> nn.Module:
        model = Idefics2VisionTransformer(
            config=config.vision_config, quant_config=quant_config, prefix=prefix
        )
        if self.config.drop_vision_last_layer:
            model.encoder.layers = model.encoder.layers[:-1]

        setattr(model, "embed_dim", model.embeddings.embed_dim)
        setattr(model, "patch_size", model.embeddings.patch_size)
        return model

    def init_resampler(
        self,
        embed_dim: int,
        vision_dim: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> nn.Module:
        with set_default_torch_dtype(torch.float16):
            # The resampler in 2.6 remains consistent with the one in 2.5.
            resampler = Resampler4_5(
                num_queries=self.config.query_num,
                embed_dim=embed_dim,
                num_heads=embed_dim // 128,
                kv_dim=vision_dim,
                quant_config=quant_config,
                prefix=prefix,
            )

        return resampler.to(device="cuda", dtype=torch.get_default_dtype())

    def get_vision_embedding(
        self,
        pixel_values: List[torch.Tensor],
        patch_attn_mask: Optional[torch.Tensor] = None,
        tgt_sizes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        vision_embedding = self.vpm(
            pixel_values,
            patch_attention_mask=patch_attn_mask,
            tgt_sizes=tgt_sizes,
        )
        return vision_embedding

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        if items and items[0].format == MultimodalInputFormat.PRECOMPUTED_EMBEDDING:
            result = torch.cat([item.feature for item in items])
            return result.reshape(-1, result.shape[-1])

        # list of tensors
        pixel_values = flatten_nested_list([item.feature for item in items])
        tgt_sizes = torch.stack(
            flatten_nested_list([item.tgt_size for item in items]), dim=0
        )
        assert len(pixel_values) == tgt_sizes.shape[0]

        device = self.vpm.embeddings.position_embedding.weight.device
        dtype = self.vpm.embeddings.position_embedding.weight.dtype
        all_pixel_values_lst = [
            i.flatten(end_dim=1).permute(1, 0) for i in pixel_values
        ]

        max_patches = (tgt_sizes[:, 0] * tgt_sizes[:, 1]).max().item()
        assert isinstance(max_patches, int)
        all_pixel_values = torch.nn.utils.rnn.pad_sequence(
            all_pixel_values_lst, batch_first=True, padding_value=0.0
        )

        B, L, _ = all_pixel_values.shape
        all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)
        patch_attn_mask = torch.zeros(
            (B, 1, max_patches), dtype=torch.bool, device=device
        )

        tgt_sizes_tensor = tgt_sizes.clone().to(device=patch_attn_mask.device)
        mask_shapes = tgt_sizes_tensor[:, 0] * tgt_sizes_tensor[:, 1]
        patch_attn_mask[:, 0, :] = torch.arange(
            patch_attn_mask.size(2), device=patch_attn_mask.device
        ).unsqueeze(0) < mask_shapes.unsqueeze(1)

        vision_embedding = self.vpm(
            all_pixel_values.type(dtype),
            patch_attention_mask=patch_attn_mask,
            tgt_sizes=tgt_sizes,
        )
        return self.resampler(vision_embedding, tgt_sizes)

    def pad_input_ids(self, input_ids: List[int], image_inputs: MultimodalInputs):
        # Get all special token IDs
        im_start_id: int = image_inputs.im_start_id
        im_end_id: int = image_inputs.im_end_id
        slice_start_id: int = image_inputs.slice_start_id
        slice_end_id: int = image_inputs.slice_end_id

        media_token_pairs = [(im_start_id, im_end_id), (slice_start_id, slice_end_id)]
        pattern = MultiModalityDataPaddingPatternTokenPairs(media_token_pairs)

        return pattern.pad_input_tokens(input_ids, image_inputs)

    def eval(self):
        super().eval()
        return self


_SUPPORT_VERSION = {(2, 6): MiniCPMV2_6, (4, 0): MiniCPMV4_0, (4, 5): MiniCPMV4_5}


class MiniCPMV:
    """
    Different versions of MiniCPMV use different visual encoders and LLMs,
    which is not conducive to the current integration logic of LoRA and
    bitsandbytes in SGLang. Therefore, it is necessary to separate them.
    """

    # Ensure that the LoRA support check passes when the class is not
    # initialized, but set all these attributes to empty.
    packed_modules_mapping = {}
    supported_lora_modules = []
    embedding_modules = {}
    embedding_padding_modules = []

    minicpmv: nn.Module

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        if not hasattr(config, "version"):
            version = (2, 6)
        else:
            version = str(config.version).split(".")
            version = tuple([int(x) for x in version])
        # Dispatch class based on version
        instance_class = _SUPPORT_VERSION.get(version)
        if instance_class is None:
            supported_versions = ", ".join(
                [f"{v[0]}.{v[1]}" for v in sorted(_SUPPORT_VERSION.keys())]
            )
            raise ValueError(
                f"Currently, MiniCPMV only supports versions "
                f"{supported_versions}. Got version: {version}"
            )

        try:
            minicpmv = instance_class(
                config=config, quant_config=quant_config, prefix=prefix
            )
            self.minicpmv = minicpmv
        except Exception as e:
            print(f"Failed to instantiate MiniCPMV: {e}")
            raise e
        self.config = config

    def __getattr__(self, name):
        if name == "minicpmv":
            return None
        return getattr(self.minicpmv, name)

    def __call__(self, *args, **kwargs):
        return self.minicpmv(*args, **kwargs)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.minicpmv.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq~" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if name.startswith("model.vision_tower") and name not in params_dict:
                continue

            # adapt to VisionAttention
            name = name.replace(r"self_attn.out_proj", r"self_attn.proj")

            if "sampler" in name:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                # replace the name and load with customized loader
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


# =================== MiniCPM-V 4.6 (independent architecture) ===================
#
# MiniCPM-V 4.6 introduces a new vision stack on top of the V4.5 family:
#   * Qwen3.5 LLM (dense + gated delta-net hybrid) instead of Qwen3.
#   * SigLIP (Idefics2) ViT encoder kept, but a window-attention + MLP merger
#     is injected after ``config.insert_layer_id`` to halve the spatial grid.
#   * The perceiver resampler is replaced by a lightweight MLP merger
#     (``merger``) that maps vision features to the LLM embedding space.
#   * Vision token count is dynamic (depends on image size and
#     ``downsample_mode``).
#
# Because the HF architecture string is ``MiniCPMV4_6ForConditionalGeneration``,
# this class is registered as its own SGLang entry-class (distinct from the
# versioned ``MiniCPMV`` dispatcher above).


class MiniCPMV4_6ViTWindowAttentionSelfAttn(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, L, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(
            B, L, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(hidden_states).view(
            B, L, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(hidden_states).view(
            B, L, self.num_heads, self.head_dim
        ).transpose(1, 2)

        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=self.scale
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, L, self.embed_dim)
        return self.out_proj(attn_out)


class MiniCPMV4_6ViTWindowAttentionMerger(nn.Module):
    """Window-attention + MLP downsampling merger injected mid-ViT.

    Each 2x2 window does a single self-attention pass (residual), then the
    window is flattened and projected through a 2-layer MLP down to the ViT
    hidden size, collapsing 4 tokens into 1.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.window_kernel_size = (2, 2)
        self.embed_dim = config.hidden_size

        self.self_attn = MiniCPMV4_6ViTWindowAttentionSelfAttn(config)
        self.layer_norm1 = nn.LayerNorm(
            self.embed_dim, eps=config.layer_norm_eps,
        )

        hidden_4x = self.embed_dim * 4
        inter_4x = config.intermediate_size * 4

        self.pre_norm = nn.LayerNorm(hidden_4x, eps=config.layer_norm_eps)
        self.linear_1 = nn.Linear(hidden_4x, inter_4x, bias=True)
        self.act = nn.GELU(approximate="tanh")
        self.linear_2 = nn.Linear(inter_4x, self.embed_dim, bias=True)

    def _apply_window_attention(
        self, valid_states: torch.Tensor, H: int, W: int,
    ) -> torch.Tensor:
        D = valid_states.shape[-1]
        wh, ww = self.window_kernel_size
        nh, nw = H // wh, W // ww
        num_windows = nh * nw

        x = valid_states.view(H, W, D)
        x = x.view(nh, wh, nw, ww, D).permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(num_windows, wh * ww, D)

        x = self.self_attn(x)

        x = x.view(nh, nw, wh, ww, D).permute(0, 2, 1, 3, 4).contiguous()
        return x.view(H * W, D)

    def _apply_mlp_downsample(
        self, valid_states: torch.Tensor, H: int, W: int,
    ) -> torch.Tensor:
        D = valid_states.shape[-1]
        wh, ww = self.window_kernel_size
        nh, nw = H // wh, W // ww

        x = valid_states.view(H, W, D)
        x = x.view(nh, wh, nw, ww, D).permute(0, 2, 1, 3, 4).contiguous()

        residual = x.reshape(nh * nw, wh * ww, D).mean(dim=1)
        x = x.reshape(nh * nw, wh * ww * D)

        x = self.pre_norm(x)
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        return x + residual

    def forward(
        self,
        hidden_states: torch.Tensor,
        tgt_sizes: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        B, _L, D = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        all_merged: List[torch.Tensor] = []
        new_tgt_sizes = torch.zeros_like(tgt_sizes)

        for b in range(B):
            H, W = tgt_sizes[b].tolist()
            H, W = int(H), int(W)
            hs = hidden_states[b, : H * W, :]

            residual = hs
            hs = self.layer_norm1(hs)
            hs = residual + self._apply_window_attention(hs, H, W)

            wh, ww = self.window_kernel_size
            new_H, new_W = H // wh, W // ww
            all_merged.append(self._apply_mlp_downsample(hs, H, W))
            new_tgt_sizes[b] = torch.tensor(
                [new_H, new_W], device=device, dtype=tgt_sizes.dtype,
            )

        new_num_patches = new_tgt_sizes[:, 0] * new_tgt_sizes[:, 1]
        new_max_patches = int(new_num_patches.max().item())
        new_hidden = torch.zeros(
            B, new_max_patches, D, device=device, dtype=dtype,
        )
        for b, merged in enumerate(all_merged):
            new_hidden[b, : merged.shape[0], :] = merged

        new_attention_mask: Optional[torch.Tensor] = None
        if attention_mask is not None:
            mask = torch.zeros(
                B, new_max_patches, dtype=torch.bool, device=device,
            )
            for b in range(B):
                mask[b, : int(new_num_patches[b].item())] = True
            min_val = torch.finfo(dtype).min
            new_attention_mask = (~mask).to(dtype=dtype) * min_val
            new_attention_mask = new_attention_mask[:, None, None, :]

        return new_hidden, new_tgt_sizes, new_attention_mask


class MiniCPMV4_6DownsampleMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        llm_embed_dim: int,
        merge_kernel_size: Tuple[int, int] = (2, 2),
    ):
        super().__init__()
        self.merge_kernel_size = merge_kernel_size
        self.hidden_size = (
            hidden_size * merge_kernel_size[0] * merge_kernel_size[1]
        )
        self.pre_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(self.hidden_size, llm_embed_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.pre_norm(x))


class MiniCPMV4_6Merger(nn.Module):
    """Final 2x2 merger projecting ViT features into the LLM embedding space.

    Returns a list of per-slice tensors (dynamic vision token counts).
    """

    def __init__(
        self,
        hidden_size: int,
        llm_embed_dim: int,
        merge_kernel_size: Tuple[int, int] = (2, 2),
        times: int = 1,
    ):
        super().__init__()
        self.merge_kernel_size = merge_kernel_size
        self.times = times
        self.mlp = nn.ModuleList([
            MiniCPMV4_6DownsampleMLP(
                hidden_size,
                llm_embed_dim if i == times - 1 else hidden_size,
                merge_kernel_size,
            )
            for i in range(times)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        tgt_sizes: torch.Tensor,
    ) -> List[torch.Tensor]:
        m1, m2 = self.merge_kernel_size
        results: List[torch.Tensor] = []

        for b in range(len(tgt_sizes)):
            h, w = tgt_sizes[b].tolist()
            h, w = int(h), int(w)
            n_patches = h * w
            hs = hidden_states[b, :n_patches, :]

            hs = hs.reshape(h // m1, m1, w // m2, m2, -1)
            hs = hs.permute(0, 2, 1, 3, 4).reshape(
                (h // m1) * (w // m2), m1 * m2 * hs.shape[-1],
            )
            hs = self.mlp[0](hs)

            if self.times > 1:
                cur_h, cur_w = h // m1, w // m2
                for t in range(1, self.times):
                    cur_h, cur_w = cur_h // m1, cur_w // m2
                    hs = hs.reshape(cur_h, m1, cur_w, m2, -1)
                    hs = hs.permute(0, 2, 1, 3, 4).reshape(
                        cur_h * cur_w, m1 * m2 * hs.shape[-1],
                    )
                    hs = self.mlp[t](hs)

            results.append(hs)

        return results


class MiniCPMV4_6ForConditionalGeneration(nn.Module):
    """SGLang entry class for MiniCPM-V 4.6.

    Module layout (matches Qwen3-VL-style multimodal models):
        self.vpm              - SigLIP ViT encoder (Idefics2VisionTransformer)
        self.vit_merger       - mid-ViT window-attention + MLP merger
        self.merger           - final 2x2 MLP merger -> LLM hidden size
        self.model            - Qwen3_5ForCausalLM backbone (hidden states only)
        self.lm_head          - ParallelLMHead, tied to model.embed_tokens
                                when ``config.tie_word_embeddings`` is True
        self.logits_processor - SGLang logits processor
    """

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }
    supported_lora_modules = [
        "fc1",
        "fc2",
        "out_proj",
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]
    embedding_modules = {}
    embedding_padding_modules = []

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
        from sglang.srt.distributed import get_pp_group

        self.config = config
        self.text_config = config.text_config
        self.quant_config = quant_config
        self.pp_group = get_pp_group()

        # ---- Vision tower ----
        self.vpm = Idefics2VisionTransformer(
            config=config.vision_config,
            quant_config=quant_config,
            prefix=add_prefix("vpm", prefix),
        )
        if getattr(config, "drop_vision_last_layer", False):
            self.vpm.encoder.layers = self.vpm.encoder.layers[:-1]
        setattr(self.vpm, "embed_dim", self.vpm.embeddings.embed_dim)
        setattr(self.vpm, "patch_size", self.vpm.embeddings.patch_size)

        # ---- Mid-encoder + final merger ----
        self.vit_merger = MiniCPMV4_6ViTWindowAttentionMerger(
            config.vision_config,
        )
        self.merger = MiniCPMV4_6Merger(
            hidden_size=config.vision_config.hidden_size,
            llm_embed_dim=config.text_config.hidden_size,
        )

        # ---- Language model (Qwen3.5) ----
        # ``Qwen3_5ForCausalLM`` here returns hidden states only; the LM head
        # and logits processor live on this outer module (mirroring
        # ``Qwen3VLForConditionalGeneration``).
        self.model = Qwen3_5ForCausalLM(
            config=config.text_config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

        if self.pp_group.is_last_rank:
            if (
                self.pp_group.world_size == 1
                and getattr(config, "tie_word_embeddings", False)
            ):
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.text_config.vocab_size,
                    config.text_config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                )
        else:
            from sglang.srt.layers.vocab_parallel_embedding import (
                PPMissingLayer,
            )
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.text_config)

        self.insert_layer_id = getattr(config, "insert_layer_id", -1)
        self.default_downsample_mode = getattr(
            config, "downsample_mode", "16x",
        )

    # --------- Embedding helpers ---------

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    # --------- Vision forward ---------

    def get_vision_hidden_states(
        self,
        pixel_values: List[torch.Tensor],
        tgt_sizes: torch.Tensor,
        downsample_mode: Optional[str] = None,
    ) -> List[torch.Tensor]:
        B = len(pixel_values)
        P = pixel_values[0].shape[-2]
        L = max(item.shape[-1] for item in pixel_values)
        device = pixel_values[0].device
        dtype = pixel_values[0].dtype

        all_pixel_values = torch.zeros(
            (B, 3, P, L), dtype=dtype, device=device,
        )
        for i, pv in enumerate(pixel_values):
            all_pixel_values[i, ..., : pv.shape[-1]] = pv

        num_patches = tgt_sizes[:, 0] * tgt_sizes[:, 1]
        max_patches = int(num_patches.max().item())
        patch_attn_mask = torch.zeros(
            (B, max_patches), dtype=torch.bool, device=device,
        )
        for i in range(B):
            patch_attn_mask[i, : int(num_patches[i].item())] = True

        hidden_states = self.vpm.embeddings(
            pixel_values=all_pixel_values,
            patch_attention_mask=patch_attn_mask.unsqueeze(1),
            tgt_sizes=tgt_sizes,
        )

        # Build attention mask for the mid-ViT merger (bool -> additive).
        if torch.any(~patch_attn_mask):
            min_val = torch.finfo(dtype).min
            attention_mask = (~patch_attn_mask).to(dtype=dtype) * min_val
            attention_mask = attention_mask[:, None, None, :]
        else:
            attention_mask = None

        cu_seqlens = self.vpm.compute_cu_seqlens(tgt_sizes, hidden_states)

        # SGLang's ``Idefics2EncoderLayer`` consumes ``cu_seqlens`` instead of
        # a 4D attention mask; the mask is only used to drive the merger's
        # padding-aware reshape, not the transformer layers themselves.
        ds_mode = downsample_mode or self.default_downsample_mode
        use_vit_merger = ds_mode != "4x" and self.insert_layer_id >= 0

        for layer in self.vpm.encoder.layers[: self.insert_layer_id + 1]:
            hidden_states = layer(hidden_states, cu_seqlens=cu_seqlens)

        if use_vit_merger:
            hidden_states, tgt_sizes, attention_mask = self.vit_merger(
                hidden_states, tgt_sizes, attention_mask,
            )
            cu_seqlens = self.vpm.compute_cu_seqlens(tgt_sizes, hidden_states)

        for layer in self.vpm.encoder.layers[self.insert_layer_id + 1:]:
            hidden_states = layer(hidden_states, cu_seqlens=cu_seqlens)

        hidden_states = self.vpm.post_layernorm(hidden_states)

        return self.merger(hidden_states, tgt_sizes)

    def get_image_feature(
        self, items: List[MultimodalDataItem],
    ) -> torch.Tensor:
        if items and items[0].format == MultimodalInputFormat.PRECOMPUTED_EMBEDDING:
            result = torch.cat([item.feature for item in items])
            return result.reshape(-1, result.shape[-1])

        pixel_values = flatten_nested_list([item.feature for item in items])
        tgt_sizes = torch.stack(
            flatten_nested_list([item.tgt_size for item in items]), dim=0,
        ).to(device=self.vpm.embeddings.position_embedding.weight.device)
        assert len(pixel_values) == tgt_sizes.shape[0]

        device = self.vpm.embeddings.position_embedding.weight.device
        dtype = self.vpm.embeddings.position_embedding.weight.dtype

        # Per-slice pixel_values come in as (3, patch_size, patch_size * n)
        # tensors; reorder to (3, P, L) and stack with padding in
        # ``get_vision_hidden_states``.
        pixel_values = [pv.to(device=device, dtype=dtype) for pv in pixel_values]

        # No downsample_mode plumbing through SGLang multimodal items yet, so
        # fall back to the model default ("16x", i.e. use the ViT merger).
        slice_features = self.get_vision_hidden_states(
            pixel_values, tgt_sizes,
        )
        return torch.cat(slice_features, dim=0)

    # --------- Multimodal forward ---------

    def pad_input_ids(
        self, input_ids: List[int], image_inputs: MultimodalInputs,
    ):
        im_start_id: int = image_inputs.im_start_id
        im_end_id: int = image_inputs.im_end_id
        slice_start_id: int = image_inputs.slice_start_id
        slice_end_id: int = image_inputs.slice_end_id

        media_token_pairs = [
            (im_start_id, im_end_id),
            (slice_start_id, slice_end_id),
        ]
        pattern = MultiModalityDataPaddingPatternTokenPairs(media_token_pairs)
        return pattern.pad_input_tokens(input_ids, image_inputs)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            multimodal_model=self,
            language_model=self.model,
            positions=positions,
        )

        if self.pp_group.is_last_rank and not get_embedding:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch,
            )
        return hidden_states

    # --------- Weight loading ---------

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        """Load HF weights into the SGLang parameter tree.

        HF layout                              SGLang layout
        -----------------------------------------------------------
          model.vpm.*                    ->    vpm.*
          model.vit_merger.*             ->    vit_merger.*
          model.merger.*                 ->    merger.*
          model.language_model.*         ->    model.*
          lm_head.*                      ->    lm_head.*   (only when not tied)
        When ``tie_word_embeddings`` is true the HF checkpoint omits
        ``lm_head.weight`` entirely; we tie ``self.lm_head`` to
        ``self.model.embed_tokens`` at init time, so embed_tokens loading
        is sufficient.
        """
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            ("in_proj_qkvz.", "in_proj_qkv.", (0, 1, 2)),
            ("in_proj_qkvz.", "in_proj_z.", 3),
            ("in_proj_ba.", "in_proj_b.", 0),
            ("in_proj_ba.", "in_proj_a.", 1),
        ]

        tied = bool(getattr(self.config, "tie_word_embeddings", False))

        params_dict = dict(self.named_parameters())
        loaded_names: set = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if (
                "rotary_emb.cos_cached" in name
                or "rotary_emb.sin_cached" in name
            ):
                continue

            # Map HF prefixes into our tree.
            if name.startswith("model.language_model."):
                # Qwen3_5ForCausalLM is a flat module (no extra ``model.``
                # wrapper inside our ``self.model``), so drop the HF
                # ``model.language_model.`` level and remap to ``model.``.
                name = "model." + name[len("model.language_model."):]
                # Qwen3.5 full-attention layers store Q/K/V/O/norm directly on
                # the layer (see sglang Qwen3_5ForCausalLM.load_weights); drop
                # the ``.self_attn.`` level from HF names.
                if ".self_attn." in name:
                    name = name.replace(".self_attn.", ".")
            elif name.startswith("model.vpm."):
                name = "vpm." + name[len("model.vpm."):]
            elif name.startswith("model.vit_merger."):
                name = "vit_merger." + name[len("model.vit_merger."):]
            elif name.startswith("model.merger."):
                name = "merger." + name[len("model.merger."):]
            elif name == "lm_head.weight":
                if tied:
                    # lm_head is tied to model.embed_tokens, skip to avoid
                    # redundant copy (and to avoid a shape mismatch when the
                    # head is a VocabParallelEmbedding reference).
                    continue
                # else keep as "lm_head.weight"

            # ``VisionAttention`` renames ``out_proj`` -> ``proj`` for the
            # SigLIP tower only. Do not touch our standalone ``vit_merger``
            # self-attention (plain nn.Linear with bias ``out_proj``).
            if name.startswith("vpm."):
                name = name.replace(
                    "self_attn.out_proj", "self_attn.proj",
                )

            matched_stacked = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if mapped.endswith(".bias") and mapped not in params_dict:
                    continue
                if mapped not in params_dict:
                    continue
                param = params_dict[mapped]
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader,
                )
                weight_loader(param, loaded_weight, shard_id)
                loaded_names.add(mapped)
                matched_stacked = True
                break
            if matched_stacked:
                continue

            if name.endswith(".bias") and name not in params_dict:
                continue
            if name not in params_dict:
                continue

            param = params_dict[name]
            weight_loader = getattr(
                param, "weight_loader", default_weight_loader,
            )
            weight_loader(param, loaded_weight)
            loaded_names.add(name)

        missing = sorted(set(params_dict) - loaded_names)
        if missing:
            import logging as _logging
            _logging.getLogger("sglang.minicpmv4_6").warning(
                "MiniCPMV4_6: %d parameter(s) not loaded from checkpoint "
                "(showing first 10): %s",
                len(missing),
                missing[:10],
            )


EntryClass = [MiniCPMV, MiniCPMV4_6ForConditionalGeneration]
