 

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath                   # FIX 1: use timm, not local
from torchvision.ops import deform_conv2d          # FIX 2: proper deformable conv


# ─────────────────────────────────────────────────────────────────────────────
# 1.  ADAPTIVE KERNEL CONVOLUTION  (AKConv)
# ─────────────────────────────────────────────────────────────────────────────
class AdaptiveKernelConv(nn.Module):
    """
    Adaptive Kernel Convolution.

    Combines:
      - Deformable depthwise conv (via torchvision deform_conv2d)
      - Soft kernel-size gating: learns a weighted mixture over
        odd kernel sizes {1, 3, 5, ..., max_kernel}
      - Pointwise branch for global channel mixing

    Args
    ----
    in_channels  : input channels
    out_channels : output channels
    max_kernel   : largest kernel size (must be odd, e.g. 7)
    stride       : spatial stride (default 1)
    reduction    : channel reduction ratio for gate MLP
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        max_kernel: int = 7,
        stride: int = 1,
        reduction: int = 4,
    ):
        super().__init__()
        assert max_kernel % 2 == 1, "max_kernel must be odd"
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.max_kernel   = max_kernel
        self.stride       = stride
        self.padding      = max_kernel // 2

        # ── depthwise conv weight (used by deform_conv2d directly)
        self.dw_weight = nn.Parameter(
            torch.empty(in_channels, 1, max_kernel, max_kernel)
        )
        nn.init.kaiming_uniform_(self.dw_weight, a=1.0)

        # ── pointwise projection
        self.pw_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)

        # FIX 2: offset_conv now properly consumed by deform_conv2d
        # deform_conv2d needs 2*K*K offset channels (no mask)
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * max_kernel * max_kernel,   # 2*K² for (dy, dx) per position
            kernel_size=3, padding=1, bias=True,
        )
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)

        # ── kernel-size gate: soft mixture over odd sizes {1,3,...,max_kernel}
        self.gate_pool = nn.AdaptiveAvgPool2d(1)
        hidden    = max(in_channels // reduction, 16)
        num_sizes = (max_kernel + 1) // 2
        self.gate_fc = nn.Sequential(
            nn.Linear(in_channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_sizes, bias=False),
        )

        # ── per-size 1×1 projections applied to deformable dw output
        self.size_projs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            for _ in range(num_sizes)
        ])

        self.bn  = nn.GroupNorm(min(32, out_channels), out_channels)  # FIX: GroupNorm works correctly at batch_size=1
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # FIX 2: compute offset and feed to deform_conv2d (not discarded)
        offset = self.offset_conv(x)                          # (B, 2K², H, W)

        # deformable depthwise conv — groups=in_channels makes it depthwise
        dw_out = deform_conv2d(
            input=x,
            offset=offset,
            weight=self.dw_weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            mask=None,                  # no modulation mask (standard DCN v1)
        )                                                      # (B, C, H', W')

        # kernel-size gating
        g     = self.gate_pool(x).view(B, C)
        gates = torch.softmax(self.gate_fc(g), dim=-1)        # (B, num_sizes)

        # soft mixture of per-size projections
        out = sum(
            gates[:, i].view(B, 1, 1, 1) * self.size_projs[i](dw_out)
            for i in range(gates.shape[1])
        )

        # add pointwise branch
        out = out + self.pw_conv(dw_out)

        return self.act(self.bn(out))


# ─────────────────────────────────────────────────────────────────────────────
# 2.  ENHANCED CONVOLUTION  (EnhConv)
# ─────────────────────────────────────────────────────────────────────────────
class EnhancedConv(nn.Module):
    """
    Enhanced convolution block:
      (a) Depth-wise separable convolution
      (b) Multi-scale dilated convolutions (parallel branches)
      (c) Local residual connection

    Args
    ----
    channels       : in == out (residual block)
    dilation_rates : dilation factors for parallel branches.
                     Default (1,2,3) — safe for 14×14 bottleneck maps.
                     Do NOT use dilation=4+ on spatial sizes <= 14.
    """

    def __init__(
        self,
        channels: int,
        dilation_rates: tuple = (1, 2, 3),    # FIX 4: was (1,2,4), unsafe at 14x14
    ):
        super().__init__()

        # FIX 3: guard against non-divisible channel splits
        assert channels % len(dilation_rates) == 0, (
            f"channels ({channels}) must be divisible by "
            f"number of dilation rates ({len(dilation_rates)})"
        )
        branch_ch = channels // len(dilation_rates)

        # ── depth-wise separable block
        self.dw = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1,
                      groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(min(32, channels), channels),  # FIX: GroupNorm works correctly at batch_size=1
            nn.GELU(),
        )

        # ── dilated parallel branches
        self.dilated_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, branch_ch,
                          kernel_size=3,
                          padding=d, dilation=d,
                          groups=branch_ch,
                          bias=False),
                nn.GroupNorm(min(32, branch_ch), branch_ch),  # FIX: GroupNorm works correctly at batch_size=1
                nn.GELU(),
            )
            for d in dilation_rates
        ])

        # fuse concatenated branches back to full channels
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(min(32, channels), channels),  # FIX: GroupNorm works correctly at batch_size=1
        )

        self.norm = nn.GroupNorm(1, channels)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        y = self.dw(x)

        dilated_outs = torch.cat([b(y) for b in self.dilated_branches], dim=1)
        y = y + self.fuse(dilated_outs)

        return self.act(self.norm(y + identity))


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CHANNEL ATTENTION  (CA)
# ─────────────────────────────────────────────────────────────────────────────
class ChannelAttention(nn.Module):
    """
    Dual-pool Squeeze-and-Excitation channel attention.
    Combines avg-pool and max-pool descriptors before excitation MLP.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.excitation = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        avg   = self.avg_pool(x).view(B, C)
        mx    = self.max_pool(x).view(B, C)
        scale = self.sigmoid(
            self.excitation(avg) + self.excitation(mx)
        ).view(B, C, 1, 1)
        return x * scale


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SPATIAL ATTENTION  (SA)
# ─────────────────────────────────────────────────────────────────────────────
class SpatialAttention(nn.Module):
    """CBAM-style spatial attention."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size % 2 == 1
        self.conv    = nn.Conv2d(2, 1, kernel_size,
                                 padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg        = x.mean(dim=1, keepdim=True)
        mx, _      = x.max(dim=1, keepdim=True)
        descriptor = torch.cat([avg, mx], dim=1)
        mask       = self.sigmoid(self.conv(descriptor))
        return x * mask


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MAFR BLOCK
# ─────────────────────────────────────────────────────────────────────────────
class MAFRBlock(nn.Module):
    """
    Multi-Attention Feature Refinement Block.

    Pipeline:
        input
          │
          ├─ AKConv  (deformable + gated kernel-size selection)
          │
          ├─ EnhConv (DW-sep + multi-dilated, safe dilation rates)
          │
          ├─ ChannelAttention (dual-pool SE)
          │
          ├─ SpatialAttention (CBAM spatial)
          │
          └─ shortcut residual + drop-path → output

    Args
    ----
    in_channels    : input channels
    out_channels   : output channels
    max_kernel     : max kernel for AKConv (default 7)
    dilation_rates : for EnhConv (default (1,2,3) — safe for 14×14)
    ca_reduction   : SE reduction ratio
    drop_path_rate : stochastic depth rate (set 0.0 for batch_size=1)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        max_kernel: int = 7,
        dilation_rates: tuple = (1, 2, 3),
        ca_reduction: int = 16,
        drop_path_rate: float = 0.0,           # FIX 5: default 0.0 for small batches
    ):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels

        self.ak_conv      = AdaptiveKernelConv(in_channels, out_channels,
                                               max_kernel=max_kernel)
        self.enh_conv     = EnhancedConv(out_channels, dilation_rates=dilation_rates)
        self.channel_attn = ChannelAttention(out_channels, reduction=ca_reduction)
        self.spatial_attn = SpatialAttention(kernel_size=7)

        # shortcut projection if channels change
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.GroupNorm(min(32, out_channels), out_channels),  # FIX: GroupNorm works correctly at batch_size=1
            )
            if in_channels != out_channels
            else nn.Identity()
        )

        # FIX 1: DropPath from timm (local class removed)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        self.out_norm = nn.GroupNorm(min(32, out_channels), out_channels)  # FIX: GroupNorm works correctly at batch_size=1
        self.out_act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        y = self.ak_conv(x)
        y = self.enh_conv(y)
        y = self.channel_attn(y)
        y = self.spatial_attn(y)

        return self.out_act(self.out_norm(identity + self.drop_path(y)))


# ─────────────────────────────────────────────────────────────────────────────
# 6.  MAFR BOTTLENECK  — Swin-UNet adapter
# ─────────────────────────────────────────────────────────────────────────────
class MAFRBottleneck(nn.Module):
    """
    Wraps MAFRBlock to accept Swin-UNet token sequences (B, L, C).

    Reshapes tokens → spatial map → runs MAFR blocks → reshapes back.

    Args
    ----
    channels         : embedding dim at bottleneck (e.g. 768 for 4-layer Swin)
    input_resolution : (H, W) of bottleneck feature map (e.g. (14, 14))
    depth            : number of stacked MAFRBlocks (default 2)
    **mafr_kwargs    : forwarded to MAFRBlock
    """

    def __init__(
        self,
        channels: int,
        input_resolution: tuple,
        depth: int = 2,
        **mafr_kwargs,
    ):
        super().__init__()
        self.H, self.W = input_resolution
        self.blocks = nn.Sequential(*[
            MAFRBlock(channels, channels, **mafr_kwargs)
            for _ in range(depth)
        ])

    def tokens_to_map(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, H*W, C) → (B, C, H, W)"""
        B, L, C = tokens.shape
        assert L == self.H * self.W, (
            f"Token count {L} != H*W = {self.H}*{self.W} = {self.H * self.W}. "
            f"Check input_resolution matches your encoder output."
        )
        return tokens.transpose(1, 2).view(B, C, self.H, self.W)

    def map_to_tokens(self, feat_map: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, H*W, C)"""
        B, C, H, W = feat_map.shape
        return feat_map.view(B, C, H * W).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x       : (B, H*W, C)  — Swin-UNet token format
        returns : (B, H*W, C)
        """
        feat = self.tokens_to_map(x)
        feat = self.blocks(feat)
        return self.map_to_tokens(feat)