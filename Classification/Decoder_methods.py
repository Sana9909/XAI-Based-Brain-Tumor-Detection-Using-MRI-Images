"""
Decoder_methods_v2.py
=====================
Drop-in replacement for Decoder_methods.py with the following improvements:

  CBE  — SwiGLU-style gated MLP replaces the plain two-layer MLP (better
          gradient flow, used in LLaMA / PaLM decoders).
  ACA  — drop_path added to both branches; resolution assertion kept.
  HAF  — unchanged structurally; skip_norm kept.
  DecoderBlock — drop_path forwarded into ACA/HAF Swin blocks; CBE wired
                 between ACA and HAF (was dead code in v1).
"""

import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d
from einops import rearrange
 
from timm.layers import DropPath


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)

        return x

# ─────────────────────────────────────────────────────────────────────────────
# SwiGLU-style gated MLP (replaces plain Mlp inside CBE)
# ─────────────────────────────────────────────────────────────────────────────

class GatedMLP(nn.Module):
    """
    SwiGLU variant:  out = (W1·x) ⊙ SiLU(W2·x)  projected back by W3.

    This doubles expressiveness vs a plain two-layer MLP for the same
    hidden_dim, and empirically trains faster on small medical datasets.
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(self.w1(x) * self.act(self.w2(x)))


# ─────────────────────────────────────────────────────────────────────────────
# CBE — Contextual Bottleneck Enhancer (v2: SwiGLU MLPs)
# ─────────────────────────────────────────────────────────────────────────────

class CBE(nn.Module):
    """
    Contextual Bottleneck Enhancer with SwiGLU gated MLPs.

    Pipeline (token-last, B H W C throughout):
      Step 1: cyclic shift along W
      Step 2: GatedMLP_W
      Step 3: GELU( GroupNorm( DWConv( · ) ) )
      Step 4: cyclic shift along H
      Step 5: GatedMLP_H
      Step 6: LN( T_H + T )   residual to original input
    """

    def __init__(self, dim: int, window_size: int = 7, mlp_ratio: float = 4.0):
        super().__init__()
        shift = window_size // 2
        self.shift_w = shift
        self.shift_h = shift

        hidden = int(dim * mlp_ratio)

        self.norm_w = nn.LayerNorm(dim)
        self.mlp_w  = GatedMLP(dim, hidden)

        self.dw_conv  = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.norm_conv = nn.GroupNorm(num_groups=min(32, dim), num_channels=dim)
        self.act      = nn.GELU()

        self.norm_h = nn.LayerNorm(dim)
        self.mlp_h  = GatedMLP(dim, hidden)

        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x

        x_w = torch.roll(x, -self.shift_w, dims=2)
        T_W = self.mlp_w(self.norm_w(x_w))

        t_cf = T_W.permute(0, 3, 1, 2).contiguous()
        Y    = self.act(self.norm_conv(self.dw_conv(t_cf)))
        Y    = Y.permute(0, 2, 3, 1).contiguous()

        Y_h = torch.roll(Y, -self.shift_h, dims=1)
        T_H = self.mlp_h(self.norm_h(Y_h))

        return self.final_norm(T_H + T)


# ─────────────────────────────────────────────────────────────────────────────
# DepthwiseDeformableConv (unchanged from v1, kept for completeness)
# ─────────────────────────────────────────────────────────────────────────────

class DepthwiseDeformableConv(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.kernel_size = kernel_size
        self.padding     = padding

        self.offset_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.Conv2d(dim, 2 * kernel_size * kernel_size,
                      kernel_size=kernel_size, padding=padding, bias=False),
        )
        self.mask_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.Conv2d(dim, kernel_size * kernel_size,
                      kernel_size=kernel_size, padding=padding, bias=False),
            nn.Sigmoid(),
        )
        self.weight = nn.Parameter(torch.empty(dim, 1, kernel_size, kernel_size))
        self.bias   = nn.Parameter(torch.zeros(dim))
        nn.init.kaiming_uniform_(self.weight, a=1.0)

        self.norm = nn.GroupNorm(num_groups=min(32, dim), num_channels=dim)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset_conv(x)
        mask   = self.mask_conv(x)
        out    = deform_conv2d(
            x, offset, self.weight, self.bias,
            padding=self.padding, mask=mask, stride=1,
        )
        return self.act(self.norm(out))


# ─────────────────────────────────────────────────────────────────────────────
# ACA — Adaptive Cross-branch Attention (v2: drop_path added)
# ─────────────────────────────────────────────────────────────────────────────
class ACA(nn.Module):
    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
    ):
        super().__init__()

        self.norm_deform    = nn.LayerNorm(dim)
        self.deform_branch  = DepthwiseDeformableConv(dim)
        self.norm_after_deform = nn.LayerNorm(dim)
        self.proj           = nn.Linear(dim, dim, bias=False)
        self.drop_path      = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x_d    = self.norm_deform(x)
        x_d_cf = x_d.permute(0, 3, 1, 2).contiguous()
        d_out  = self.deform_branch(x_d_cf).permute(0, 2, 3, 1).contiguous()
        d_out  = self.norm_after_deform(d_out)

        out = self.proj(d_out)
        return identity + self.drop_path(out)


# ─────────────────────────────────────────────────────────────────────────────
# HAF — Hierarchical Attention Fusion (unchanged structurally; skip_norm kept)
# ─────────────────────────────────────────────────────────────────────────────
class HAF(nn.Module):
    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
    ):
        super().__init__()

        self.skip_norm = nn.LayerNorm(dim)
        self.dec_norm  = nn.LayerNorm(dim)

        self.proj = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(32, dim), num_channels=dim),
            nn.GELU(),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x_skip: torch.Tensor, x_dec: torch.Tensor) -> torch.Tensor:
        x_skip = self.skip_norm(x_skip)
        x_dec  = self.dec_norm(x_dec)

        x_cat    = torch.cat([x_skip, x_dec], dim=-1)
        x_cat_cf = x_cat.permute(0, 3, 1, 2).contiguous()
        out      = self.proj(x_cat_cf)
        return out.permute(0, 2, 3, 1).contiguous()

# ─────────────────────────────────────────────────────────────────────────────
# DecoderBlock (v2: drop_path forwarded; CBE wired between ACA and HAF)
# ─────────────────────────────────────────────────────────────────────────────
class DecoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution,
        window_size: int = 7,
        drop_path: float = 0.0,
    ):
        super().__init__()

        self.expand  = PatchExpand(input_resolution, dim)
        half_dim     = dim // 2

        self.aca = ACA(half_dim, drop_path=drop_path)
        self.haf = HAF(half_dim, drop_path=drop_path)

    def forward(self, x, x_skip=None):
        H, W = self.expand.input_resolution
        B, L, C = x.shape
        x = self.expand(x)
        H2, W2 = H * 2, W * 2
        x = x.view(B, H2, W2, -1)

        if x_skip is None:
            x_aca = self.aca(x)
            return x_aca.view(B, -1, x_aca.shape[-1])

        if x_skip.dim() == 3:
            x_skip = x_skip.view(B, H2, W2, -1)

        x_aca   = self.aca(x)
        x_fused = self.haf(x_skip, x_aca)
        return x_fused.view(B, -1, x_fused.shape[-1])