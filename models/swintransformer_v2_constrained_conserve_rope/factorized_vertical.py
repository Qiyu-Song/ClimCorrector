"""Factorized vertical-attention corrector (v1, NO conservation ops).

Explicit vertical axis L=26 instead of flattening levels into channels. Each layer alternates:
  - HORIZONTAL: existing SwinTransformerV2CrBlock (windowed+shifted, 2D RoPE) applied per level
    (batched over L) -> reuses the tested horizontal attention/RoPE.
  - VERTICAL:   MHA over the L level-tokens per column (+ learned level embedding).
Per-level head -> 104 = 4 vars x 26 lev (channel = var*26+lev). Pole-pad + pole_tqmean as baseline;
conservation intentionally omitted. Gradient checkpointing on both block types (batching over L makes
the horizontal effective batch B*L, so checkpointing is needed to fit memory during training).
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import modulus
from swintransformer_modulus_polepadding_conserve_rope import SwinTransformerV2CrBlock, SwinTransformerV2CrModulusMetaData


class VerticalTokenEmbed(nn.Module):
    def __init__(self, in_chans=200, nlev=26, n3d_vars=7, d=256):
        super().__init__()
        self.nlev = nlev; self.n3d = n3d_vars; self.n3d_ch = n3d_vars * nlev
        self.n2d = in_chans - self.n3d_ch
        assert self.n2d >= 0
        self.lin3d = nn.Linear(n3d_vars, d)
        self.lin2d = nn.Linear(self.n2d, d) if self.n2d > 0 else None
        self.level_emb = nn.Parameter(torch.zeros(1, 1, 1, nlev, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        B, C, H, W = x.shape
        x3d = x[:, :self.n3d_ch].view(B, self.n3d, self.nlev, H, W).permute(0, 3, 4, 2, 1)
        tok = self.lin3d(x3d) + self.level_emb
        if self.lin2d is not None:
            x2d = x[:, self.n3d_ch:].permute(0, 2, 3, 1)
            tok = tok + self.lin2d(x2d).unsqueeze(3)
        return self.norm(tok)


class VerticalAttnBlock(nn.Module):
    def __init__(self, d, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        h = int(d * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, d))

    def forward(self, z):                     # (B,H,W,L,d)
        B, H, W, L, d = z.shape
        y = z.reshape(B * H * W, L, d)
        yn = self.norm1(y)
        a, _ = self.attn(yn, yn, yn, need_weights=False)
        y = y + a
        y = y + self.mlp(self.norm2(y))
        return y.reshape(B, H, W, L, d)


class FactorizedVerticalCorrector(modulus.Module):
    def __init__(self, img_size=(96, 144), in_chans=200, out_chans=104, d=256, depth=8,
                 num_heads=8, window_size=(4, 6), nlev=26, n3d_vars=7, mlp_ratio=4.0,
                 pos_encoding="rope_mixed", rope_theta=10.0, pole_padding=True,
                 pole_padding_value=6, pole_tqmean=True, grad_checkpoint=True):
        super().__init__(meta=SwinTransformerV2CrModulusMetaData())
        self.pp = pole_padding_value if pole_padding else 0
        self.pole_padding = pole_padding; self.pole_tqmean = pole_tqmean
        self.out_chans = out_chans; self.nlev = nlev; self.nvar_out = out_chans // nlev
        self.grad_checkpoint = grad_checkpoint
        self.wshift = img_size[1] // 2   # constant: torch.jit.trace rejects a Tensor `shifts`
        Hp = img_size[0] + 2 * self.pp
        feat = (Hp, img_size[1])
        self.embed = VerticalTokenEmbed(in_chans, nlev, n3d_vars, d)
        self.horiz = nn.ModuleList([
            SwinTransformerV2CrBlock(
                dim=d, num_heads=num_heads, feat_size=feat, window_size=window_size,
                shift_size=tuple(w // 2 for w in window_size) if (i % 2) else (0, 0),
                mlp_ratio=mlp_ratio, pos_encoding=pos_encoding, rope_theta=rope_theta,
            ) for i in range(depth)])
        self.vert = nn.ModuleList([VerticalAttnBlock(d, num_heads, mlp_ratio) for _ in range(depth)])
        self.head = nn.Linear(d, self.nvar_out)

    def _run(self, blk, t):
        if self.grad_checkpoint and torch.is_grad_enabled():
            return checkpoint(blk, t, use_reentrant=False)
        return blk(t)

    def forward(self, x):                      # (B, in_chans, 96, 144)
        if self.pole_padding:
            north = torch.roll(torch.flip(x[:, :, :self.pp, :], dims=[2]), shifts=self.wshift, dims=3)
            south = torch.roll(torch.flip(x[:, :, -self.pp:, :], dims=[2]), shifts=self.wshift, dims=3)
            x = torch.cat((north, x, south), dim=2)
        z = self.embed(x)                       # (B,Hp,W,L,d)
        B, H, W, L, d = z.shape
        for hb, vb in zip(self.horiz, self.vert):
            zh = z.permute(0, 3, 1, 2, 4).reshape(B * L, H, W, d)   # (B*L,H,W,d) BHWC
            zh = self._run(hb, zh)
            z = zh.reshape(B, L, H, W, d).permute(0, 2, 3, 1, 4).contiguous()
            z = self._run(vb, z)
        o = self.head(z)                        # (B,H,W,L,nvar_out)
        o = o.permute(0, 4, 3, 1, 2).reshape(B, self.nvar_out * L, H, W)
        if self.pole_padding:
            o = o[:, :, self.pp:-self.pp, :]
        if self.pole_tqmean:
            o = o.clone()
            half = self.out_chans // 2
            o[:, :half, 0, :] = o[:, :half, 0, :].mean(-1, keepdim=True)
            o[:, :half, -1, :] = o[:, :half, -1, :].mean(-1, keepdim=True)
        return o
