# Adapted from https://github.com/NERSC/swin_v2_weather/blob/main/networks/swinv2_global.py
# under Apache-2.0 license
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#  Which is the PyTorch code for the Preprint of `Analyzing and Exploring Training Recipes for Large-Scale Transformer-Based Weather Prediction`:
#     - https://arxiv.org/abs/2404.19630
# Adapted from timm v0.9.2:
#  https://github.com/huggingface/pytorch-image-models/blob/v0.9.2/timm/models/swin_transformer_v2_cr.py
# under MIT License:
# --------------------------------------------------------
# Swin Transformer V2 reimplementation
# Copyright (c) 2021 Christoph Reich
# Licensed under The MIT License [see LICENSE for details]
# Written by Christoph Reich
# --------------------------------------------------------
#  which is a PyTorch impl of : `Swin Transformer V2: Scaling Up Capacity and Resolution`
#     - https://arxiv.org/pdf/2111.09883


import math
from typing import Any, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, Mlp, _assert, to_2tuple
from torch.utils.checkpoint import checkpoint

import modulus
import nvtx
from dataclasses import dataclass
import numpy as np
import xarray as xr

@dataclass
class SwinTransformerV2CrModulusMetaData(modulus.ModelMetaData):
    name: str = "SwinTransformerV2CrModulus"
    # Optimization
    jit: bool = True
    cuda_graphs: bool = True
    amp_cpu: bool = False
    amp_gpu: bool = False


def bchw_to_bhwc(x: torch.Tensor) -> torch.Tensor:
    """Permutes a tensor from the shape (B, C, H, W) to (B, H, W, C)."""
    return x.permute(0, 2, 3, 1)


def bhwc_to_bchw(x: torch.Tensor) -> torch.Tensor:
    """Permutes a tensor from the shape (B, H, W, C) to (B, C, H, W)."""
    return x.permute(0, 3, 1, 2)


# def swinv2net(params, checkpoint_stages=False):
#     act_ckpt = checkpoint_stages or params.activation_ckpt
#     return SwinTransformerV2Cr(
#         img_size=params.img_size,
#         patch_size=params.patch_size,
#         depths=(params.depth,),
#         num_heads=(params.num_heads,),
#         in_chans=params.n_in_channels,
#         out_chans=params.n_out_channels,
#         embed_dim=params.embed_dim,
#         img_window_ratio=params.window_ratio,
#         drop_path_rate=params.drop_path_rate,
#         full_pos_embed=params.full_pos_embed,
#         rel_pos=params.rel_pos,
#         mlp_ratio=params.mlp_ratio,
#         checkpoint_stages=act_ckpt,
#         residual=params.residual,
#     )


def bchw_to_bhwc(x: torch.Tensor) -> torch.Tensor:
    """Permutes a tensor from the shape (B, C, H, W) to (B, H, W, C)."""
    return x.permute(0, 2, 3, 1)


def bhwc_to_bchw(x: torch.Tensor) -> torch.Tensor:
    """Permutes a tensor from the shape (B, H, W, C) to (B, C, H, W)."""
    return x.permute(0, 3, 1, 2)


def window_partition(x, window_size: Tuple[int, int]):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(
        B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C
    )
    windows = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size[0], window_size[1], C)
    )
    return windows


def window_reverse(windows, window_size: Tuple[int, int], img_size: Tuple[int, int]):
    """
    Args:
        windows: (num_windows * B, window_size[0], window_size[1], C)
        window_size (Tuple[int, int]): Window size
        img_size (Tuple[int, int]): Image size

    Returns:
        x: (B, H, W, C)
    """
    H, W = img_size
    C = windows.shape[-1]
    x = windows.view(
        -1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    return x


class WindowMultiHeadAttentionNoPos(nn.Module):
    r"""This class implements window-based Multi-Head-Attention with log-spaced continuous position bias.

    Args:
        dim (int): Number of input features
        window_size (int): Window size
        num_heads (int): Number of attention heads
        drop_attn (float): Dropout rate of attention map
        drop_proj (float): Dropout rate after projection
        meta_hidden_dim (int): Number of hidden features in the two layer MLP meta network
        sequential_attn (bool): If true sequential self-attention is performed
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int],
        drop_attn: float = 0.0,
        drop_proj: float = 0.0,
        sequential_attn: bool = False,
    ) -> None:
        super(WindowMultiHeadAttentionNoPos, self).__init__()
        assert (
            dim % num_heads == 0
        ), "The number of input features (in_features) are not divisible by the number of heads (num_heads)."
        self.in_features: int = dim
        self.window_size: Tuple[int, int] = window_size
        self.num_heads: int = num_heads
        self.sequential_attn: bool = sequential_attn

        self.qkv = nn.Linear(in_features=dim, out_features=dim * 3, bias=True)
        self.attn_drop = nn.Dropout(drop_attn)
        self.proj = nn.Linear(in_features=dim, out_features=dim, bias=True)
        self.proj_drop = nn.Dropout(drop_proj)
        # NOTE old checkpoints used inverse of logit_scale ('tau') following the paper, see conversion fn
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones(num_heads)))

    def update_input_size(self, new_window_size: int, **kwargs: Any) -> None:
        """Method updates the window size and so the pair-wise relative positions

        Args:
            new_window_size (int): New window size
            kwargs (Any): Unused
        """
        # Set new window size and new pair-wise relative positions
        self.window_size: int = new_window_size

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        Args:
            x (torch.Tensor): Input tensor of the shape (B * windows, N, C)
            mask (Optional[torch.Tensor]): Attention mask for the shift case

        Returns:
            Output tensor of the shape [B * windows, N, C]
        """
        Bw, L, C = x.shape

        qkv = (
            self.qkv(x)
            .view(Bw, L, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv.unbind(0)

        # compute attention map with scaled cosine attention
        attn = F.normalize(query, dim=-1) @ F.normalize(key, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(
            self.logit_scale.reshape(1, self.num_heads, 1, 1), max=math.log(1.0 / 0.01)
        ).exp()
        attn = attn * logit_scale

        if mask is not None:
            # Apply mask if utilized
            num_win: int = mask.shape[0]
            attn = attn.view(Bw // num_win, num_win, self.num_heads, L, L)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, L, L)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ value).transpose(1, 2).reshape(Bw, L, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class WindowMultiHeadAttentionRPB(nn.Module):
    r"""This class implements window-based Multi-Head-Attention with log-spaced continuous position bias.

    Args:
        dim (int): Number of input features
        window_size (int): Window size
        num_heads (int): Number of attention heads
        drop_attn (float): Dropout rate of attention map
        drop_proj (float): Dropout rate after projection
        meta_hidden_dim (int): Number of hidden features in the two layer MLP meta network
        sequential_attn (bool): If true sequential self-attention is performed
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int],
        drop_attn: float = 0.0,
        drop_proj: float = 0.0,
        meta_hidden_dim: int = 384,  # FIXME what's the optimal value?
        sequential_attn: bool = False,
    ) -> None:
        super(WindowMultiHeadAttentionRPB, self).__init__()
        assert (
            dim % num_heads == 0
        ), "The number of input features (in_features) are not divisible by the number of heads (num_heads)."
        self.in_features: int = dim
        self.window_size: Tuple[int, int] = window_size
        self.num_heads: int = num_heads
        self.sequential_attn: bool = sequential_attn

        self.qkv = nn.Linear(in_features=dim, out_features=dim * 3, bias=True)
        self.attn_drop = nn.Dropout(drop_attn)
        self.proj = nn.Linear(in_features=dim, out_features=dim, bias=True)
        self.proj_drop = nn.Dropout(drop_proj)
        # meta network for positional encodings
        self.meta_mlp = Mlp(
            2,  # x, y
            hidden_features=meta_hidden_dim,
            out_features=num_heads,
            act_layer=nn.ReLU,
            drop=(
                0.125,
                0.0,
            ),  # FIXME should there be stochasticity, appears to 'overfit' without?
        )
        # NOTE old checkpoints used inverse of logit_scale ('tau') following the paper, see conversion fn
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones(num_heads)))
        self._make_pair_wise_relative_positions()

    def _make_pair_wise_relative_positions(self) -> None:
        """Method initializes the pair-wise relative positions to compute the positional biases."""
        device = self.logit_scale.device
        coordinates = torch.stack(
            torch.meshgrid(
                [
                    torch.arange(self.window_size[0], device=device),
                    torch.arange(self.window_size[1], device=device),
                ]
            ),
            dim=0,
        ).flatten(1)
        relative_coordinates = coordinates[:, :, None] - coordinates[:, None, :]
        relative_coordinates = (
            relative_coordinates.permute(1, 2, 0).reshape(-1, 2).float()
        )
        relative_coordinates_log = torch.sign(relative_coordinates) * torch.log(
            1.0 + relative_coordinates.abs()
        )
        self.register_buffer(
            "relative_coordinates_log", relative_coordinates_log, persistent=False
        )

    def update_input_size(self, new_window_size: int, **kwargs: Any) -> None:
        """Method updates the window size and so the pair-wise relative positions

        Args:
            new_window_size (int): New window size
            kwargs (Any): Unused
        """
        # Set new window size and new pair-wise relative positions
        self.window_size: int = new_window_size
        self._make_pair_wise_relative_positions()

    def _relative_positional_encodings(self) -> torch.Tensor:
        """Method computes the relative positional encodings

        Returns:
            relative_position_bias (torch.Tensor): Relative positional encodings
            (1, number of heads, window size ** 2, window size ** 2)
        """
        window_area = self.window_size[0] * self.window_size[1]
        relative_position_bias = self.meta_mlp(self.relative_coordinates_log)
        relative_position_bias = relative_position_bias.transpose(1, 0).reshape(
            self.num_heads, window_area, window_area
        )
        relative_position_bias = relative_position_bias.unsqueeze(0)
        return relative_position_bias

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        Args:
            x (torch.Tensor): Input tensor of the shape (B * windows, N, C)
            mask (Optional[torch.Tensor]): Attention mask for the shift case

        Returns:
            Output tensor of the shape [B * windows, N, C]
        """
        Bw, L, C = x.shape

        qkv = (
            self.qkv(x)
            .view(Bw, L, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv.unbind(0)

        # compute attention map with scaled cosine attention
        attn = F.normalize(query, dim=-1) @ F.normalize(key, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(
            self.logit_scale.reshape(1, self.num_heads, 1, 1), max=math.log(1.0 / 0.01)
        ).exp()
        attn = attn * logit_scale
        attn = attn + self._relative_positional_encodings()

        if mask is not None:
            # Apply mask if utilized
            num_win: int = mask.shape[0]
            attn = attn.view(Bw // num_win, num_win, self.num_heads, L, L)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, L, L)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ value).transpose(1, 2).reshape(Bw, L, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def _rope_rotate_half(x: torch.Tensor) -> torch.Tensor:
    # (.., d) -> (.., d) with (x0,x1,x2,x3,...) -> (-x1,x0,-x3,x2,...)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    x_rot = torch.stack((-x_odd, x_even), dim=-1)
    return x_rot.flatten(-2)


def _rope_apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x, cos, sin broadcastable to (..., d)
    return (x * cos) + (_rope_rotate_half(x) * sin)


def _rope_init_random_2d_freqs(
    head_dim: int, num_heads: int, theta: float = 100.0, rotate: bool = True
) -> torch.Tensor:
    """Init 2D RoPE frequencies with random per-head axis rotation.

    Returns a tensor of shape (2, num_heads, complex_dim=head_dim//2):
      freqs[0] -> x-axis frequencies per head
      freqs[1] -> y-axis frequencies per head

    This matches a common Swin+RoPE "mixed" init:
    - build a base magnitude schedule with step 4 over head_dim
    - allocate half the complex dims to one axis and half to an orthogonal axis
    - optionally rotate axes by a random angle per head
    """
    if head_dim % 4 != 0:
        raise ValueError(
            f"RoPE mixed random-rotation init needs head_dim % 4 == 0, got head_dim={head_dim}"
        )
    mag = 1.0 / (
        float(theta)
        ** (
            torch.arange(0, head_dim, 4, dtype=torch.float32)[: (head_dim // 4)]
            / float(head_dim)
        )
    )  # (head_dim//4,)

    freqs_x = []
    freqs_y = []
    for _ in range(num_heads):
        angle = (
            torch.rand(1, dtype=torch.float32) * 2.0 * math.pi
            if rotate
            else torch.zeros(1, dtype=torch.float32)
        )
        fx = torch.cat(
            [mag * torch.cos(angle), mag * torch.cos(angle + math.pi / 2.0)], dim=-1
        )
        fy = torch.cat(
            [mag * torch.sin(angle), mag * torch.sin(angle + math.pi / 2.0)], dim=-1
        )
        freqs_x.append(fx)
        freqs_y.append(fy)

    freqs_x = torch.stack(freqs_x, dim=0)  # (H, complex_dim)
    freqs_y = torch.stack(freqs_y, dim=0)  # (H, complex_dim)
    return torch.stack([freqs_x, freqs_y], dim=0)  # (2, H, complex_dim)


class WindowMultiHeadAttentionRoPE(nn.Module):
    r"""Window-based Multi-Head Attention with 2D RoPE (Axial or Mixed).

    This replaces the additive relative position bias in Swin-V2 with RoPE,
    applying rotary embeddings to *query* and *key* before scaled cosine attention.

    Args:
        dim: embedding dimension
        num_heads: number of heads
        window_size: (Wh, Ww)
        rope_mode: 'mixed' or 'axial'
        rope_theta: base for inverse frequencies (vision papers often use ~100)
        drop_attn, drop_proj: dropout rates
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int],
        rope_mode: str = "mixed",
        rope_theta: float = 100.0,
        drop_attn: float = 0.0,
        drop_proj: float = 0.0,
        sequential_attn: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, (
            "The number of input features (in_features) are not divisible by num_heads."
        )
        self.in_features: int = dim
        self.window_size: Tuple[int, int] = window_size
        self.num_heads: int = num_heads
        self.sequential_attn: bool = sequential_attn

        self.rope_mode = rope_mode
        self.rope_theta = float(rope_theta)

        self.qkv = nn.Linear(in_features=dim, out_features=dim * 3, bias=True)
        self.attn_drop = nn.Dropout(drop_attn)
        self.proj = nn.Linear(in_features=dim, out_features=dim, bias=True)
        self.proj_drop = nn.Dropout(drop_proj)
        # Swin-V2 scaled cosine attention
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones(num_heads)))

        head_dim = dim // num_heads
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE needs even head_dim, got head_dim={head_dim}")
        if self.rope_mode == "axial" and (head_dim % 4 != 0):
            raise ValueError(
                f"Axial 2D RoPE needs head_dim divisible by 4, got head_dim={head_dim}"
            )

        # Fixed inverse frequencies (used for axial); keep as a fallback init
        complex_dim = head_dim // 2  # number of complex pairs
        freq_seq = torch.arange(complex_dim, dtype=torch.float32)
        inv_freq = self.rope_theta ** (-freq_seq / float(complex_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        if self.rope_mode == "mixed":
            # Learnable mixed-axis frequencies per head.
            # Recommended init: random-rotated orthogonal axes per head (common Swin+RoPE practice),
            # computed in fp32 for stability.
            if head_dim % 4 == 0:
                freqs = _rope_init_random_2d_freqs(
                    head_dim=head_dim, num_heads=num_heads, theta=self.rope_theta, rotate=True
                )  # (2, H, complex_dim)
                self.theta_x = nn.Parameter(freqs[0].clone(), requires_grad=True)
                self.theta_y = nn.Parameter(freqs[1].clone(), requires_grad=True)
            else:
                # Fallback: deterministic init if head_dim is not compatible with the orthogonal packing
                theta0 = inv_freq[None, :].repeat(num_heads, 1)  # (H, complex_dim)
                self.theta_x = nn.Parameter(theta0.clone(), requires_grad=True)
                self.theta_y = nn.Parameter(theta0.clone(), requires_grad=True)
        else:
            self.theta_x = None
            self.theta_y = None

        self._make_window_positions()

    def _make_window_positions(self) -> None:
        device = self.logit_scale.device
        Wh, Ww = self.window_size
        yy, xx = torch.meshgrid(
            torch.arange(Wh, device=device),
            torch.arange(Ww, device=device),
            indexing="ij",
        )
        self.register_buffer("pos_x", xx.reshape(-1).float(), persistent=False)  # (L,)
        self.register_buffer("pos_y", yy.reshape(-1).float(), persistent=False)  # (L,)

    def update_input_size(self, new_window_size: Tuple[int, int], **kwargs: Any) -> None:
        self.window_size = new_window_size
        self._make_window_positions()


    def _rope_sincos(
        self, dtype: torch.dtype, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute RoPE sin/cos in fp32 for stability, then cast to `dtype`.

        Returns cos, sin broadcastable to (Bw, H, L, head_dim).
        """
        head_dim = self.in_features // self.num_heads
        compute_dtype = torch.float32
        out_dtype = dtype

        if self.rope_mode == "axial":
            # Split channels: first half for x, second half for y (each half uses 1D RoPE)
            half = head_dim // 2
            inv = self.inv_freq[: half // 2].to(device=device, dtype=compute_dtype)  # (half/2,)

            pos_x = self.pos_x.to(device=device, dtype=compute_dtype)  # (L,)
            pos_y = self.pos_y.to(device=device, dtype=compute_dtype)  # (L,)

            ang_x = pos_x[:, None] * inv[None, :]  # (L, half/2)
            ang_y = pos_y[:, None] * inv[None, :]  # (L, half/2)

            cos_x = torch.cos(ang_x).repeat_interleave(2, dim=-1)  # (L, half)
            sin_x = torch.sin(ang_x).repeat_interleave(2, dim=-1)
            cos_y = torch.cos(ang_y).repeat_interleave(2, dim=-1)  # (L, half)
            sin_y = torch.sin(ang_y).repeat_interleave(2, dim=-1)

            cos = torch.cat([cos_x, cos_y], dim=-1).unsqueeze(0).unsqueeze(0)  # (1,1,L,head_dim)
            sin = torch.cat([sin_x, sin_y], dim=-1).unsqueeze(0).unsqueeze(0)
            return cos.to(dtype=out_dtype), sin.to(dtype=out_dtype)

        # mixed: per-head angles, learnable (theta_x, theta_y)
        pos_x = self.pos_x.to(device=device, dtype=compute_dtype)[None, :, None]  # (1,L,1)
        pos_y = self.pos_y.to(device=device, dtype=compute_dtype)[None, :, None]
        theta_x = self.theta_x.to(device=device, dtype=compute_dtype)[:, None, :]  # (H,1,complex_dim)
        theta_y = self.theta_y.to(device=device, dtype=compute_dtype)[:, None, :]
        ang = pos_x * theta_x + pos_y * theta_y  # (H,L,complex_dim)

        cos = torch.cos(ang).repeat_interleave(2, dim=-1).unsqueeze(0)  # (1,H,L,head_dim)
        sin = torch.sin(ang).repeat_interleave(2, dim=-1).unsqueeze(0)
        return cos.to(dtype=out_dtype), sin.to(dtype=out_dtype)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        Bw, L, C = x.shape

        qkv = (
            self.qkv(x)
            .view(Bw, L, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv.unbind(0)  # (Bw, H, L, head_dim)

        # RoPE on Q,K (rotation preserves norms; normalize after is fine)
        cos, sin = self._rope_sincos(dtype=query.dtype, device=query.device)
        query = _rope_apply(query, cos, sin)
        key = _rope_apply(key, cos, sin)

        # scaled cosine attention (Swin-V2)
        attn = F.normalize(query, dim=-1) @ F.normalize(key, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(
            self.logit_scale.reshape(1, self.num_heads, 1, 1), max=math.log(1.0 / 0.01)
        ).exp()
        attn = attn * logit_scale

        if mask is not None:
            num_win: int = mask.shape[0]
            attn = attn.view(Bw // num_win, num_win, self.num_heads, L, L)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, L, L)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ value).transpose(1, 2).reshape(Bw, L, -1)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class SwinTransformerV2CrBlock(nn.Module):
    r"""This class implements the Swin transformer block.

    Args:
        dim (int): Number of input channels
        num_heads (int): Number of attention heads to be utilized
        feat_size (Tuple[int, int]): Input resolution
        window_size (Tuple[int, int]): Window size to be utilized
        shift_size (int): Shifting size to be used
        mlp_ratio (int): Ratio of the hidden dimension in the FFN to the input channels
        proj_drop (float): Dropout in input mapping
        drop_attn (float): Dropout rate of attention map
        drop_path (float): Dropout in main path
        extra_norm (bool): Insert extra norm on 'main' branch if True
        sequential_attn (bool): If true sequential self-attention is performed
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        feat_size: Tuple[int, int],
        window_size: Tuple[int, int],
        shift_size: Tuple[int, int] = (0, 0),
        mlp_ratio: float = 4.0,
        init_values: Optional[float] = 0,
        proj_drop: float = 0.0,
        drop_attn: float = 0.0,
        drop_path: float = 0.0,
        extra_norm: bool = False,
        sequential_attn: bool = False,
        rel_pos: bool = True,
        pos_encoding: str = "rope_mixed",
        rope_theta: float = 100.0,
    ) -> None:
        super(SwinTransformerV2CrBlock, self).__init__()
        self.dim: int = dim
        self.feat_size: Tuple[int, int] = feat_size
        self.target_shift_size: Tuple[int, int] = to_2tuple(shift_size)
        self.window_size, self.shift_size = self._calc_window_shift(
            to_2tuple(window_size)
        )
        self.window_area = self.window_size[0] * self.window_size[1]
        self.init_values: Optional[float] = init_values
        # attn branch
        if (not rel_pos) or (pos_encoding == "none"):
            self.attn = WindowMultiHeadAttentionNoPos(
                dim=dim,
                num_heads=num_heads,
                window_size=self.window_size,
                drop_attn=drop_attn,
                drop_proj=proj_drop,
                sequential_attn=sequential_attn,
            )
        elif pos_encoding == "rpb":
            self.attn = WindowMultiHeadAttentionRPB(
                dim=dim,
                num_heads=num_heads,
                window_size=self.window_size,
                drop_attn=drop_attn,
                drop_proj=proj_drop,
                sequential_attn=sequential_attn,
            )
        else:
            rope_mode = "mixed" if ("mixed" in pos_encoding) else "axial"
            self.attn = WindowMultiHeadAttentionRoPE(
                dim=dim,
                num_heads=num_heads,
                window_size=self.window_size,
                rope_mode=rope_mode,
                rope_theta=rope_theta,
                drop_attn=drop_attn,
                drop_proj=proj_drop,
                sequential_attn=sequential_attn,
            )
        self.norm1 = nn.LayerNorm(dim)
        self.drop_path1 = (
            DropPath(drop_prob=drop_path) if drop_path > 0.0 else nn.Identity()
        )

        # mlp branch
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=proj_drop,
            out_features=dim,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.drop_path2 = (
            DropPath(drop_prob=drop_path) if drop_path > 0.0 else nn.Identity()
        )

        # Extra main branch norm layer mentioned for Huge/Giant models in V2 paper.
        # Also being used as final network norm and optional stage ending norm while still in a C-last format.
        # Dropped for ERA5
        self.norm3 = nn.Identity()

        self._make_attention_mask()
        self.init_weights()

    def _calc_window_shift(self, target_window_size):
        window_size = [
            f if f <= w else w for f, w in zip(self.feat_size, target_window_size)
        ]
        shift_size = [
            0 if f <= w else s
            for f, w, s in zip(self.feat_size, window_size, self.target_shift_size)
        ]
        return tuple(window_size), tuple(shift_size)

    def _make_attention_mask(self) -> None:
        """Method generates the attention mask used in shift case."""
        # Make masks for shift case

        if any(self.shift_size):
            # calculate attention mask for SW-MSA
            H, W = self.feat_size
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            cnt = 0
            for h in (slice(0, -self.window_size[0]), slice(-self.shift_size[0], None)):
                img_mask[:, h, :, :] = cnt
                cnt += 1
            mask_windows = window_partition(
                img_mask, self.window_size
            )  # num_windows, window_size, window_size, 1
            mask_windows = mask_windows.view(
                -1, self.window_area
            )  # num_windows, window_size*window_size
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask, persistent=False)

    def init_weights(self):
        # extra, module specific weight init
        if self.init_values is not None:
            nn.init.constant_(self.norm1.weight, self.init_values)
            nn.init.constant_(self.norm2.weight, self.init_values)

    def update_input_size(
        self, new_window_size: Tuple[int, int], new_feat_size: Tuple[int, int]
    ) -> None:
        """Method updates the image resolution to be processed and window size and so the pair-wise relative positions.

        Args:
            new_window_size (int): New window size
            new_feat_size (Tuple[int, int]): New input resolution
        """
        # Update input resolution
        self.feat_size: Tuple[int, int] = new_feat_size
        self.window_size, self.shift_size = self._calc_window_shift(
            to_2tuple(new_window_size)
        )
        self.window_area = self.window_size[0] * self.window_size[1]
        self.attn.update_input_size(new_window_size=self.window_size)
        self._make_attention_mask()

    def _shifted_window_attn(self, x):
        B, H, W, C = x.shape

        # cyclic shift
        sh, sw = self.shift_size
        do_shift: bool = any(self.shift_size)
        if do_shift:
            # FIXME PyTorch XLA needs cat impl, roll not lowered
            # x = torch.cat([x[:, sh:], x[:, :sh]], dim=1)
            # x = torch.cat([x[:, :, sw:], x[:, :, :sw]], dim=2)

            x = torch.roll(x, shifts=(-sh, -sw), dims=(1, 2))

        # partition windows
        x_windows = window_partition(
            x, self.window_size
        )  # num_windows * B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(
            x_windows, mask=self.attn_mask
        )  # num_windows * B, window_size * window_size, C

        # merge windows
        attn_windows = attn_windows.view(
            -1, self.window_size[0], self.window_size[1], C
        )
        x = window_reverse(attn_windows, self.window_size, self.feat_size)  # B H' W' C

        # reverse cyclic shift
        if do_shift:
            # FIXME PyTorch XLA needs cat impl, roll not lowered
            # x = torch.cat([x[:, -sh:], x[:, :-sh]], dim=1)
            # x = torch.cat([x[:, :, -sw:], x[:, :, :-sw]], dim=2)
            x = torch.roll(x, shifts=(sh, sw), dims=(1, 2))

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of the shape [B, C, H, W]

        Returns:
            output (torch.Tensor): Output tensor of the shape [B, C, H, W]
        """
        # post-norm branches (op -> norm -> drop)
        x = x + self.drop_path1(self.norm1(self._shifted_window_attn(x)))

        B, H, W, C = x.shape
        x = x.reshape(B, -1, C)
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        x = self.norm3(
            x
        )  # main-branch norm enabled for some blocks / stages (every 6 for Huge/Giant)
        x = x.reshape(B, H, W, C)
        return x


class PatchMerging(nn.Module):
    """This class implements the patch merging as a strided convolution with a normalization before.
    Args:
        dim (int): Number of input channels
    """

    def __init__(self, dim: int) -> None:
        super(PatchMerging, self).__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(
            in_features=4 * dim, out_features=2 * dim, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        Args:
            x (torch.Tensor): Input tensor of the shape [B, C, H, W]
        Returns:
            output (torch.Tensor): Output tensor of the shape [B, 2 * C, H // 2, W // 2]
        """
        B, H, W, C = x.shape
        x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self, img_size=224, patch_size=16, in_chans=3, embed_dim=768
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        _assert(
            H == self.img_size[0],
            f"Input image height ({H}) doesn't match model ({self.img_size[0]}).",
        )
        _assert(
            W == self.img_size[1],
            f"Input image width ({W}) doesn't match model ({self.img_size[1]}).",
        )
        x = self.proj(x)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x

class VerticalPatchEmbed(nn.Module):
    """
    Vertical-aware patch embedding

    Assumes channel layout (v2):
      first 182 channels = 7 multilevel vars x 26 levels
      remaining channels = single-level / broadcast vars
    """
    def __init__(
        self,
        img_size=224,
        patch_size=1,
        in_chans=200,
        embed_dim=768,
        nlev=26,
        n3d_vars=7,
        vert_dim=8,
        surf_dim=16,
        use_level_emb=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        if patch_size != (1, 1):
            raise ValueError(
                f"VerticalPatchEmbed currently expects patch_size=1, got {patch_size}"
            )

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0], img_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.nlev = nlev
        self.n3d_vars = n3d_vars
        self.n3d_ch = n3d_vars * nlev
        self.n2d_ch = in_chans - self.n3d_ch
        self.vert_dim = vert_dim
        self.surf_dim = surf_dim

        if self.n2d_ch < 0:
            raise ValueError(
                f"in_chans={in_chans} is too small for n3d_vars={n3d_vars}, nlev={nlev}"
            )

        # Learned level embedding, separate for each 3D variable
        if use_level_emb:
            self.level_emb = nn.Parameter(torch.zeros(1, n3d_vars, nlev))
        else:
            self.level_emb = None

        # Cheap vertical mixer:
        # input shape will be [B*H*W, n3d_vars, nlev]
        self.vert_in = nn.Conv1d(n3d_vars, vert_dim, kernel_size=3, padding=1)
        self.vert_mid = nn.Conv1d(vert_dim, vert_dim, kernel_size=3, padding=1)
        self.vert_out = nn.Conv1d(vert_dim, vert_dim, kernel_size=1)

        # Surface / single-level branch
        if self.n2d_ch > 0:
            self.sfc_proj = nn.Conv2d(self.n2d_ch, surf_dim, kernel_size=1)
        else:
            self.sfc_proj = None
            self.surf_dim = 0

        # Fuse back to backbone width
        self.proj = nn.Conv2d(
            vert_dim * nlev + self.surf_dim,
            embed_dim,
            kernel_size=1,
            stride=1,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        _assert(
            H == self.img_size[0],
            f"Input image height ({H}) doesn't match model ({self.img_size[0]}).",
        )
        _assert(
            W == self.img_size[1],
            f"Input image width ({W}) doesn't match model ({self.img_size[1]}).",
        )

        # Split 3D and 2D channels
        x3d = x[:, :self.n3d_ch, :, :]   # [B, 182, H, W]
        x2d = x[:, self.n3d_ch:, :, :]   # [B, 18, H, W] for v2

        # Reshape to [B*H*W, n3d_vars, nlev]
        x3d = x3d.view(B, self.n3d_vars, self.nlev, H, W)
        x3d = x3d.permute(0, 3, 4, 1, 2).contiguous()
        x3d = x3d.view(B * H * W, self.n3d_vars, self.nlev)

        # Add learned level embedding
        if self.level_emb is not None:
            x3d = x3d + self.level_emb

        # Vertical mixing
        x3d = F.gelu(self.vert_in(x3d))
        x3d = F.gelu(self.vert_mid(x3d))
        x3d = F.gelu(self.vert_out(x3d))

        # Back to [B, vert_dim*nlev, H, W]
        x3d = x3d.view(B, H, W, self.vert_dim, self.nlev)
        x3d = x3d.permute(0, 3, 4, 1, 2).contiguous()
        x3d = x3d.view(B, self.vert_dim * self.nlev, H, W)

        # Single-level branch
        if self.sfc_proj is not None:
            x2d = F.gelu(self.sfc_proj(x2d))
            x = torch.cat([x3d, x2d], dim=1)
        else:
            x = x3d

        # Final projection to embed_dim
        x = self.proj(x)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x


class SwinTransformerV2CrStage(nn.Module):
    r"""This class implements a stage of the Swin transformer including multiple layers.

    Args:
        embed_dim (int): Number of input channels
        depth (int): Depth of the stage (number of layers)
        downscale (bool): If true input is downsampled (see Fig. 3 or V1 paper)
        feat_size (Tuple[int, int]): input feature map size (H, W)
        num_heads (int): Number of attention heads to be utilized
        window_size (int): Window size to be utilized
        mlp_ratio (int): Ratio of the hidden dimension in the FFN to the input channels
        proj_drop (float): Dropout in input mapping
        drop_attn (float): Dropout rate of attention map
        drop_path (float): Dropout in main path
        extra_norm_period (int): Insert extra norm layer on main branch every N (period) blocks
        extra_norm_stage (bool): End each stage with an extra norm layer in main branch
        sequential_attn (bool): If true sequential self-attention is performed
    """

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        downscale: bool,
        num_heads: int,
        feat_size: Tuple[int, int],
        window_size: Tuple[int, int],
        mlp_ratio: float = 4.0,
        init_values: Optional[float] = 0.0,
        proj_drop: float = 0.0,
        drop_attn: float = 0.0,
        drop_path: Union[List[float], float] = 0.0,
        extra_norm_period: int = 0,
        extra_norm_stage: bool = False,
        sequential_attn: bool = False,
        rel_pos: bool = True,
        pos_encoding: str = "rope_mixed",
        rope_theta: float = 100.0,
        grad_checkpointing: bool = False,
        random_shift: bool = False,
    ) -> None:
        super(SwinTransformerV2CrStage, self).__init__()
        self.downscale: bool = downscale
        self.feat_size: Tuple[int, int] = (
            (feat_size[0] // 2, feat_size[1] // 2) if downscale else feat_size
        )
        self.grad_checkpointing = grad_checkpointing

        if downscale:
            self.downsample = PatchMerging(embed_dim)
            embed_dim = embed_dim * 2
        else:
            self.downsample = nn.Identity()

        def _extra_norm(index):
            i = index + 1
            if extra_norm_period and i % extra_norm_period == 0:
                return True
            return i == depth if extra_norm_stage else False

        def _calc_shift_size(index):
            if random_shift and index > 0:
                return [torch.randint(0, w, size = ()).item() for w in window_size]
            elif (index % 2) != 0:
                return [w // 2 for w in window_size]
            else:
                return 0

        self.blocks = nn.Sequential(
            *[
                SwinTransformerV2CrBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    feat_size=self.feat_size,
                    window_size=window_size,
                    shift_size=_calc_shift_size(index),
                    mlp_ratio=mlp_ratio,
                    init_values=init_values,
                    proj_drop=proj_drop,
                    drop_attn=drop_attn,
                    drop_path=(
                        drop_path[index] if isinstance(drop_path, list) else drop_path
                    ),
                    extra_norm=_extra_norm(index),
                    sequential_attn=sequential_attn,
                    rel_pos=rel_pos,
                    pos_encoding=pos_encoding,
                    rope_theta=rope_theta,
                )
                for index in range(depth)
            ]
        )

    def update_input_size(
        self, new_window_size: int, new_feat_size: Tuple[int, int]
    ) -> None:
        """Method updates the resolution to utilize and the window size and so the pair-wise relative positions.

        Args:
            new_window_size (int): New window size
            new_feat_size (Tuple[int, int]): New input resolution
        """
        self.feat_size: Tuple[int, int] = (
            (new_feat_size[0] // 2, new_feat_size[1] // 2)
            if self.downscale
            else new_feat_size
        )
        for block in self.blocks:
            block.update_input_size(
                new_window_size=new_window_size, new_feat_size=self.feat_size
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        Args:
            x (torch.Tensor): Input tensor of the shape [B, C, H, W] or [B, L, C]
        Returns:
            output (torch.Tensor): Output tensor of the shape [B, 2 * C, H // 2, W // 2]
        """
        x = bchw_to_bhwc(x)
        x = self.downsample(x)
        for block in self.blocks:
            # Perform checkpointing if utilized
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = bhwc_to_bchw(x)
        return x


class SwinTransformerV2CrModulus_polepadding_conserve(modulus.Module):
    r"""Swin Transformer V2
        A PyTorch impl of : `Swin Transformer V2: Scaling Up Capacity and Resolution`  -
          https://arxiv.org/pdf/2111.09883

    Args:
        img_size: Input resolution.
        window_size: Window size. If None, img_size // window_div
        img_window_ratio: Window size to image size ratio.
        patch_size: Patch size.
        in_chans: Number of input channels.
        depths: Depth of the stage (number of layers).
        num_heads: Number of attention heads to be utilized.
        embed_dim: Patch embedding dimension.
        num_classes: Number of output classes.
        mlp_ratio:  Ratio of the hidden dimension in the FFN to the input channels.
        drop_rate: Dropout rate.
        proj_drop_rate: Projection dropout rate.
        attn_drop_rate: Dropout rate of attention map.
        drop_path_rate: Stochastic depth rate.
        extra_norm_period: Insert extra norm layer on main branch every N (period) blocks in stage
        extra_norm_stage: End each stage with an extra norm layer in main branch
        sequential_attn: If true sequential self-attention is performed.
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (224, 224),
        patch_size: int = 4,
        window_size: Optional[int] = None,
        img_window_ratio: int = 32,
        in_chans: int = 3,
        out_chans: int = 3,
        embed_dim: int = 96,
        depths: Tuple[int, ...] = (2, 2, 6, 2),
        num_heads: Tuple[int, ...] = (3, 6, 12, 24),
        mlp_ratio: float = 4.0,
        init_values: Optional[float] = 0.0,
        drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        extra_norm_period: int = 0,
        extra_norm_stage: bool = False,
        sequential_attn: bool = False,
        global_pool: str = "avg",
        full_pos_embed: bool = False,
        rel_pos: bool = True,
        pos_encoding: str = "rope_mixed",
        rope_theta: float = 10.0,
        checkpoint_stages: bool = False,
        residual: bool = False,
        random_shift: bool = False,
        pole_padding: bool = True,
        pole_padding_value: int = 6,
        pole_tqmean: bool = True,
        conserve_water: bool = False,
        conserve_heat: bool = False,
        input_mean = None,
        input_std = None,
        target_mean = None,
        target_std = None,
        grid_info = None,
        pressure_index: int = 130,
        sdiff_std_file = None,
        qdiff_std_file = None,
        vertical_embed: bool = True,
        nlev: int = 26,
        n3d_vars: int = 7,
        vert_dim: int = 8,
        surf_dim: int = 16,
        **kwargs: Any,
    ) -> None:
        # super(SwinTransformerV2Cr, self).__init__()
        super().__init__(meta=SwinTransformerV2CrModulusMetaData())
        img_size = to_2tuple(img_size)
        self.pole_padding = pole_padding
        self.pole_padding_value = pole_padding_value
        self.pole_tqmean = pole_tqmean
        
        if self.pole_padding:
            img_size = (img_size[0]+2*self.pole_padding_value, img_size[1])

        window_size = (
            tuple([s // img_window_ratio for s in img_size])
            if window_size is None
            else to_2tuple(window_size)
        )

        self.patch_size: int = patch_size
        self.img_size: Tuple[int, int] = img_size
        self.window_size: int = window_size
        self.num_features: int = int(embed_dim)
        self.out_chans: int = out_chans
        self.feature_info = []
        self.full_pos_embed = full_pos_embed
        self.checkpoint_stages = checkpoint_stages
        self.residual = residual
        self.depth = len(depths)
        self.conserve_water = conserve_water
        self.conserve_heat = conserve_heat

        def _load_maybe(fname):
            if fname is None:
                return None
            arr = np.load(fname)            # or xr.open_dataarray, h5py, …
            return torch.as_tensor(arr, dtype=torch.float32)
        
        def _load_grid_info(fname):
            if fname is None:
                return None, None, None
            ds_grid = xr.open_dataset(fname)
            hyai = ds_grid.hyai.values
            hybi = ds_grid.hybi.values
            gw = ds_grid.gw.values
            hyai = torch.tensor(hyai, dtype=torch.float32)
            hybi = torch.tensor(hybi, dtype=torch.float32)
            gw = torch.tensor(gw, dtype=torch.float32)
            return hyai, hybi, gw
        
        self.register_buffer("input_mean", _load_maybe(input_mean))
        self.register_buffer("input_std", _load_maybe(input_std))
        self.register_buffer("target_mean", _load_maybe(target_mean))
        self.register_buffer("target_std", _load_maybe(target_std))
        hyai, hybi, gw = _load_grid_info(grid_info)
        self.register_buffer("hyai", hyai)
        self.register_buffer("hybi", hybi)
        self.register_buffer("gw", gw)
        self.pressure_index = pressure_index
        self.register_buffer("sdiff_std", _load_maybe(sdiff_std_file)) # of shape (26,96)
        self.register_buffer("qdiff_std", _load_maybe(qdiff_std_file)) # of shape (26,96)


        if vertical_embed:
            self.patch_embed = VerticalPatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                nlev=nlev,
                n3d_vars=n3d_vars,
                vert_dim=vert_dim,
                surf_dim=surf_dim,
                use_level_emb=True,
            )
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        patch_grid_size: Tuple[int, int] = self.patch_embed.grid_size

        dpr = [
            x.tolist()
            for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)
        ]
        stages = []
        in_dim = embed_dim
        in_scale = 1
        for stage_idx, (depth, num_heads) in enumerate(zip(depths, num_heads)):
            stages += [
                SwinTransformerV2CrStage(
                    embed_dim=in_dim,
                    depth=depth,
                    downscale=False,
                    feat_size=(
                        patch_grid_size[0] // in_scale,
                        patch_grid_size[1] // in_scale,
                    ),
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    init_values=init_values,
                    proj_drop=proj_drop_rate,
                    drop_attn=attn_drop_rate,
                    drop_path=dpr[stage_idx],
                    extra_norm_period=extra_norm_period,
                    extra_norm_stage=extra_norm_stage
                    or (stage_idx + 1) == len(depths),  # last stage ends w/ norm
                    sequential_attn=sequential_attn,
                    rel_pos=rel_pos,
                    pos_encoding=pos_encoding,
                    rope_theta=rope_theta,
                    grad_checkpointing=self.checkpoint_stages,
                    random_shift=random_shift,
                )
            ]
            self.feature_info += [
                dict(
                    num_chs=in_dim, reduction=4 * in_scale, module=f"stages.{stage_idx}"
                )
            ]

        self.stages = nn.Sequential(*stages)
        self.head = nn.Linear(
            embed_dim, self.out_chans * self.patch_size * self.patch_size, bias=False
        )

        if self.full_pos_embed:
            raise NotImplementedError("Full positional embedding not implemented")
            # self.pos_embed = nn.Parameter(
            #     torch.randn(1, embed_dim, patch_grid_size[0], patch_grid_size[1]) * 0.02
            # )
        # else:
        #     self.pos_embed = None
        

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.full_pos_embed:
            # x = x + self.pos_embed
            raise NotImplementedError("Full positional embedding not implemented")
        x = self.stages(x)
        return x

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        B, _, h, w = x.shape
        x = bchw_to_bhwc(x)
        x = self.head(x)

        x = x.reshape(shape=(B, h, w, self.patch_size, self.patch_size, self.out_chans))
        x = torch.einsum("nhwpqc->nchpwq", x)
        x = x.reshape(shape=(B, self.out_chans, self.img_size[0], self.img_size[1]))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x is in shape (B, C, H, W) where H is latitude and W is longitude
        # retrieve PS first (B, 1, H, W)
        ps = x[:,self.pressure_index,:,:].unsqueeze(1)*self.input_std[self.pressure_index] + self.input_mean[self.pressure_index]

        if self.residual:
            skip = x
        else:
            skip = torch.zeros_like(x)

        if self.pole_padding:
            # x is in shape (B, C, H, W) where H is latitude and W is longitude
            # For North pole padding (reflecting top rows)
            north_data = x[:, :, 0:self.pole_padding_value, :]
            # Flip latitudes and rotate longitudes by 180 degrees
            north_padded = torch.flip(north_data, dims=[2])  # Flip latitude only (dim 2)
            north_padded = torch.roll(north_padded, shifts=north_padded.shape[3]//2, dims=3)  # Rotate longitude by 180°
            
            # For South pole padding (reflecting bottom rows)
            south_data = x[:, :, -self.pole_padding_value:, :]
            # Flip latitudes and rotate longitudes by 180 degrees
            south_padded = torch.flip(south_data, dims=[2])  # Flip latitude only (dim 2)
            south_padded = torch.roll(south_padded, shifts=south_padded.shape[3]//2, dims=3)  # Rotate longitude by 180°
            
            # Concatenate padded data with original
            x = torch.cat((north_padded, x, south_padded), dim=2)

        x = self.forward_features(x)
        x = self.forward_head(x)
        if self.pole_padding:
            # x is in shape (B, C, H, W)
            x = x[:,:,self.pole_padding_value:-self.pole_padding_value,:]

        if self.pole_tqmean:
            # take the zonal average for T and Q at two poles 
            half_channels = self.out_chans // 2
            # Top pole 
            top_mean = x[:, :half_channels, 0, :].mean(dim=-1, keepdim=True)
            x[:, :half_channels, 0, :] = top_mean.expand(-1, -1, x.shape[-1])
            
            # Bottom pole
            bottom_mean = x[:, :half_channels, -1, :].mean(dim=-1, keepdim=True)
            x[:, :half_channels, -1, :] = bottom_mean.expand(-1, -1, x.shape[-1])
        x = x + skip[:, : self.out_chans, :, :]

        if self.conserve_water or self.conserve_heat:
            pi = self.hyai.view(1,-1,1,1) * 1e5 + ps * self.hybi.view(1,-1,1,1)
            dp = pi[:, 1:, :, :] - pi[:, :-1, :, :] # dimension is (B, 26, H, W)
            weight = dp * self.gw.view(1,1,-1,1) # dimension is (B, 26, H, W)
            weight = weight / weight.sum(dim=(1,2,3), keepdim=True) # dimension is (B, 26, H, W)
            # weight_square_sum = (weight * weight).sum(dim=(1,2,3), keepdim=True) # dimension is (B, 1, 1, 1)

            if self.conserve_heat:
                x_t = x[:, 0:26, :, :] * self.target_std[0:26].view(1,-1,1,1) + self.target_mean[0:26].view(1,-1,1,1)
                # get the weighted mean t
                x_t_mean = (x_t*weight).sum(dim=(1,2,3), keepdim=True)
                # x_t_new = x_t - x_t_mean * weight / weight_square_sum
                # x[:, 0:26, :, :] = (x_t_new - self.target_mean[0:26].view(1,-1,1,1)) / self.target_std[0:26].view(1,-1,1,1)
                sigma_t = self.sdiff_std.unsqueeze(0).unsqueeze(3)
                sigmat_w_sum = (sigma_t * weight).sum(dim=(1,2,3), keepdim=True)
                x_t_new = x_t - x_t_mean * sigma_t / sigmat_w_sum
                x[:, 0:26, :, :] = (x_t_new - self.target_mean[0:26].view(1,-1,1,1)) / self.target_std[0:26].view(1,-1,1,1)
            
            if self.conserve_water:
                x_q = x[:, 26:52, :, :] * self.target_std[26:52].view(1,-1,1,1) + self.target_mean[26:52].view(1,-1,1,1)
                # get the weighted mean q
                x_q_mean = (x_q*weight).sum(dim=(1,2,3), keepdim=True)
                # x_q_new = x_q - x_q_mean * weight / weight_square_sum
                # x[:, 26:52, :, :] = (x_q_new - self.target_mean[26:52].view(1,-1,1,1)) / self.target_std[26:52].view(1,-1,1,1)
                sigma_q = self.qdiff_std.unsqueeze(0).unsqueeze(3)
                sigmaq_w_sum = (sigma_q * weight).sum(dim=(1,2,3), keepdim=True)
                x_q_new = x_q - x_q_mean * sigma_q / sigmaq_w_sum
                x[:, 26:52, :, :] = (x_q_new - self.target_mean[26:52].view(1,-1,1,1)) / self.target_std[26:52].view(1,-1,1,1)
                
        return x

    def update_input_size(
        self,
        new_img_size: Optional[Tuple[int, int]] = None,
        new_window_size: Optional[int] = None,
        img_window_ratio: int = 32,
    ) -> None:
        """Method updates the image resolution to be processed and window size and so the pair-wise relative positions.

        Args:
            new_window_size (Optional[int]): New window size, if None based on new_img_size // window_div
            new_img_size (Optional[Tuple[int, int]]): New input resolution, if None current resolution is used
            img_window_ratio (int): divisor for calculating window size from image size
        """
        # Check parameters
        if new_img_size is None:
            new_img_size = self.img_size
        else:
            new_img_size = to_2tuple(new_img_size)
        if new_window_size is None:
            new_window_size = tuple([s // img_window_ratio for s in new_img_size])
        # Compute new patch resolution & update resolution of each stage
        new_patch_grid_size = (
            new_img_size[0] // self.patch_size,
            new_img_size[1] // self.patch_size,
        )
        for index, stage in enumerate(self.stages):
            stage_scale = 2 ** max(index - 1, 0)
            stage.update_input_size(
                new_window_size=new_window_size,
                new_img_size=(
                    new_patch_grid_size[0] // stage_scale,
                    new_patch_grid_size[1] // stage_scale,
                ),
            )

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r"^patch_embed",  # stem and embed
            blocks=(
                r"^stages\.(\d+)"
                if coarse
                else [
                    (r"^stages\.(\d+).downsample", (0,)),
                    (r"^stages\.(\d+)\.\w+\.(\d+)", None),
                ]
            ),
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore()
    def get_classifier(self) -> nn.Module:
        """Method returns the classification head of the model.
        Returns:
            head (nn.Module): Current classification head
        """
        return self.head.fc

    def reset_classifier(
        self, num_classes: int, global_pool: Optional[str] = None
    ) -> None:
        """Method results the classification head

        Args:
            num_classes (int): Number of classes to be predicted
            global_pool (str): Unused
        """
        self.num_classes = num_classes
        self.head.reset(num_classes, global_pool)


def init_weights(module: nn.Module, name: str = ""):
    # FIXME WIP determining if there's a better weight init
    if isinstance(module, nn.Linear):
        if "qkv" in name:
            # treat the weights of Q, K, V separately
            val = math.sqrt(
                6.0 / float(module.weight.shape[0] // 3 + module.weight.shape[1])
            )
            nn.init.uniform_(module.weight, -val, val)
        elif "head" in name:
            nn.init.zeros_(module.weight)
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()
