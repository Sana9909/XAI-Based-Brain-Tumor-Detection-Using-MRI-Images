# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import numpy as np
from typing import Sequence, Tuple, Union,Optional
from monai.networks.blocks import UpSample
from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer
from dynunet_defconv_block import SABlock
from monai.networks.layers.factories import Conv
from monai.utils import ensure_tuple_rep
from torch.nn import functional as F
from monai.networks.blocks.convolutions import Convolution
from monai.networks.utils import pixelshuffle
from timm.layers import DropPath
 

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class SubPixelConv(nn.Module):
    def __init__(
            self,
            spatial_dims: int,
            in_channels: Optional[int] = None,
            out_channels: Optional[int] = None,
            scale_factor: Union[Sequence[float], float] = 2,
            bias: bool = True,
        ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
            in_channels: number of channels of the input image.
            out_channels: number of channels of the output image. Defaults to `in_channels`.
            scale_factor: multiplier for spatial size. Has to match input size if it is a tuple. Defaults to 2.
            bias: whether to have a bias term in the conv layers. Defaults to True.
        """
        super().__init__()
        if scale_factor <= 0:
            raise ValueError(f"The `scale_factor` multiplier must be an integer greater than 0, got {scale_factor}.")
        self.scale_factor = ensure_tuple_rep(scale_factor, spatial_dims)
        self.dimensions = spatial_dims
        conv_out_channels = out_channels * np.prod(self.scale_factor)
        self.conv =  Conv[Conv.CONV, self.dimensions](
                in_channels=in_channels, out_channels=conv_out_channels, kernel_size=3, stride=1, padding=1, bias=bias
            )
        self.tanh = nn.Tanh()
        if self.dimensions == 2:
            self.pixelshuffle = nn.PixelShuffle(upscale_factor=self.scale_factor)
        elif self.dimensions == 3:
            print("Invalid dimension")

    def forward(self, x):
        x = self.conv(x)
        x = self.tanh(x)
        x = self.pixelshuffle(x)
        return x

class HAF(nn.Module):
    def __init__(self, dim, drop_path=0.0):
        super().__init__()

        groups = min(32, dim)
        while dim % groups != 0:
            groups -= 1

        self.proj = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            nn.GroupNorm(groups, dim),
            nn.GELU(),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x_skip, x_dec):

        x_cat = torch.cat([x_skip, x_dec], dim=1)

        out = self.proj(x_cat)

        out = out + self.drop_path(x_dec)   # FIX: was x_skip — decoder needs its own residual path

        return out

class UnetrUpBlockWithAttention(nn.Module):
    """
    An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
            self,
            upsample: str,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            upsample_kernel_size: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            sa_block: bool = False,
            res_block: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """

        super().__init__()

        if upsample == 'transconv':
            upsample_stride = upsample_kernel_size
            self.upsample = get_conv_layer(
                spatial_dims,
                in_channels,
                out_channels,
                kernel_size=upsample_kernel_size,
                stride=upsample_stride,
                conv_only=True,
                is_transposed=True,
            )
        elif upsample == 'subpixelconv':
            self.upsample = SubPixelConv(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
            )
        elif upsample == 'nontrainable':
            self.upsample = UpSample(
                spatial_dims,
                in_channels,
                out_channels,
                mode='nontrainable'
            )

        if sa_block:
            self.conv_block = SABlock(
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
        elif res_block:
            self.conv_block = UnetResBlock(
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
        else:
            self.conv_block = UnetBasicBlock(  # type: ignore
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
         
        self.skipatten = HAF(
            dim=out_channels
        )

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.upsample(inp)
        skip = self.skipatten(skip,out)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out
                                # import torch
                                # import torch.nn as nn
                                # import numpy as np
                                # from typing import Sequence, Tuple, Union,Optional
                                # from monai.networks.blocks import UpSample
                                # from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer
                                # from dynunet_defconv_block import SABlock
                                # from monai.networks.layers.factories import Conv
                                # from monai.utils import ensure_tuple_rep
                                # from torch.nn import functional as F
                                # from monai.networks.blocks.convolutions import Convolution
                                # from monai.networks.utils import pixelshuffle
                                # from timm.layers import DropPath
                                

                                # def normal_init(module, mean=0, std=1, bias=0):
                                #     if hasattr(module, 'weight') and module.weight is not None:
                                #         nn.init.normal_(module.weight, mean, std)
                                #     if hasattr(module, 'bias') and module.bias is not None:
                                #         nn.init.constant_(module.bias, bias)


                                # def constant_init(module, val, bias=0):
                                #     if hasattr(module, 'weight') and module.weight is not None:
                                #         nn.init.constant_(module.weight, val)
                                #     if hasattr(module, 'bias') and module.bias is not None:
                                #         nn.init.constant_(module.bias, bias)


                                # class SubPixelConv(nn.Module):
                                #     def __init__(
                                #             self,
                                #             spatial_dims: int,
                                #             in_channels: Optional[int] = None,
                                #             out_channels: Optional[int] = None,
                                #             scale_factor: Union[Sequence[float], float] = 2,
                                #             bias: bool = True,
                                #         ) -> None:
                                #         """
                                #         Args:
                                #             spatial_dims: number of spatial dimensions of the input image.
                                #             in_channels: number of channels of the input image.
                                #             out_channels: number of channels of the output image. Defaults to `in_channels`.
                                #             scale_factor: multiplier for spatial size. Has to match input size if it is a tuple. Defaults to 2.
                                #             bias: whether to have a bias term in the conv layers. Defaults to True.
                                #         """
                                #         super().__init__()
                                #         if scale_factor <= 0:
                                #             raise ValueError(f"The `scale_factor` multiplier must be an integer greater than 0, got {scale_factor}.")
                                #         self.scale_factor = ensure_tuple_rep(scale_factor, spatial_dims)
                                #         self.dimensions = spatial_dims
                                #         conv_out_channels = out_channels * np.prod(self.scale_factor)
                                #         self.conv =  Conv[Conv.CONV, self.dimensions](
                                #                 in_channels=in_channels, out_channels=conv_out_channels, kernel_size=3, stride=1, padding=1, bias=bias
                                #             )
                                #         self.tanh = nn.Tanh()
                                #         if self.dimensions == 2:
                                #             self.pixelshuffle = nn.PixelShuffle(upscale_factor=self.scale_factor)
                                #         elif self.dimensions == 3:
                                #             print("Invalid dimension")

                                #     def forward(self, x):
                                #         x = self.conv(x)
                                #         x = self.tanh(x)
                                #         x = self.pixelshuffle(x)
                                #         return x

                                # class HAF(nn.Module):
                                #     def __init__(self, dim, drop_path=0.0):
                                #         super().__init__()

                                #         groups = min(32, dim)
                                #         while dim % groups != 0:
                                #             groups -= 1

                                #         self.proj = nn.Sequential(
                                #             nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
                                #             nn.GroupNorm(groups, dim),
                                #             nn.GELU(),
                                #         )

                                #         self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

                                #     def forward(self, x_skip, x_dec):

                                #         x_cat = torch.cat([x_skip, x_dec], dim=1)

                                #         out = self.proj(x_cat)

                                #         out = out + self.drop_path(x_skip)

                                #         return out

                                # class UnetrUpBlockWithAttention(nn.Module):
                                #     """
                                #     An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
                                #     UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
                                #     """

                                #     def __init__(
                                #             self,
                                #             upsample: str,
                                #             spatial_dims: int,
                                #             in_channels: int,
                                #             out_channels: int,
                                #             kernel_size: Union[Sequence[int], int],
                                #             upsample_kernel_size: Union[Sequence[int], int],
                                #             norm_name: Union[Tuple, str],
                                #             sa_block: bool = False,
                                #             res_block: bool = False,
                                #     ) -> None:
                                #         """
                                #         Args:
                                #             spatial_dims: number of spatial dimensions.
                                #             in_channels: number of input channels.
                                #             out_channels: number of output channels.
                                #             kernel_size: convolution kernel size.
                                #             upsample_kernel_size: convolution kernel size for transposed convolution layers.
                                #             norm_name: feature normalization type and arguments.
                                #             res_block: bool argument to determine if residual block is used.

                                #         """

                                #         super().__init__()

                                #         if upsample == 'transconv':
                                #             upsample_stride = upsample_kernel_size
                                #             self.upsample = get_conv_layer(
                                #                 spatial_dims,
                                #                 in_channels,
                                #                 out_channels,
                                #                 kernel_size=upsample_kernel_size,
                                #                 stride=upsample_stride,
                                #                 conv_only=True,
                                #                 is_transposed=True,
                                #             )
                                #         elif upsample == 'subpixelconv':
                                #             self.upsample = SubPixelConv(
                                #                 spatial_dims=spatial_dims,
                                #                 in_channels=in_channels,
                                #                 out_channels=out_channels,
                                #             )
                                #         elif upsample == 'nontrainable':
                                #             self.upsample = UpSample(
                                #                 spatial_dims,
                                #                 in_channels,
                                #                 out_channels,
                                #                 mode='nontrainable'
                                #             )

                                #         if sa_block:
                                #             self.conv_block = SABlock(
                                #                 spatial_dims,
                                #                 out_channels + out_channels,
                                #                 out_channels,
                                #                 kernel_size=kernel_size,
                                #                 stride=1,
                                #                 norm_name=norm_name,
                                #             )
                                #         elif res_block:
                                #             self.conv_block = UnetResBlock(
                                #                 spatial_dims,
                                #                 out_channels + out_channels,
                                #                 out_channels,
                                #                 kernel_size=kernel_size,
                                #                 stride=1,
                                #                 norm_name=norm_name,
                                #             )
                                #         else:
                                #             self.conv_block = UnetBasicBlock(  # type: ignore
                                #                 spatial_dims,
                                #                 out_channels + out_channels,
                                #                 out_channels,
                                #                 kernel_size=kernel_size,
                                #                 stride=1,
                                #                 norm_name=norm_name,
                                #             )
                                        
                                #         self.skipatten = HAF(
                                #             dim=out_channels
                                #         )

                                #     def forward(self, inp, skip):
                                #         # number of channels for skip should equals to out_channels
                                #         out = self.upsample(inp)
                                #         skip = self.skipatten(skip,out)
                                #         out = torch.cat((out, skip), dim=1)
                                #         out = self.conv_block(out)
                                #         return out

# from SwinDER.upsample.onsampling import Onsampling
# from SwinDER.upsample.subpixelconv import SubPixelConv
# # import itertools
# from typing import Optional, Sequence, Tuple, Type, Union

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.utils.checkpoint as checkpoint
# from torch.nn import LayerNorm

# from monai.networks.blocks import MLPBlock as Mlp, UpSample
# from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock
# from monai.networks.layers import DropPath, trunc_normal_
# from monai.utils import ensure_tuple_rep, look_up_option, optional_import


# rearrange, _ = optional_import("einops", name="rearrange")

# from SwinDER.network.dynunet_defconv_block import SABlock
# from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer

# from SwinDER.upsample.onsampling import Onsampling
# from SwinDER.upsample.subpixelconv import SubPixelConv


# __all__ = [
#     "SwinDER",
#     "window_partition",
#     "window_reverse",
#     "WindowAttention",
#     "SwinTransformerBlock",
#     "PatchMerging",
#     "PatchMergingV2",
#     "MERGING_MODE",
#     "BasicLayer",
#     "SwinTransformer",
# ]


# class SwinDER(nn.Module):
#     """
#     Swin UNETR based on: "Hatamizadeh et al.,
#     Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
#     <https://arxiv.org/abs/2201.01266>"
#     """

#     def __init__(
#             self,
#             img_size: Union[Sequence[int], int],
#             in_channels: int,
#             out_channels: int,
#             depths: Sequence[int] = (2, 2, 2, 2),
#             num_heads: Sequence[int] = (3, 6, 12, 24),
#             feature_size: int = 24,
#             norm_name: Union[Tuple, str] = "instance",
#             drop_rate: float = 0.0,
#             attn_drop_rate: float = 0.0,
#             dropout_path_rate: float = 0.0,
#             normalize: bool = True,
#             use_checkpoint: bool = False,
#             spatial_dims: int = 3,
#             downsample="merging",
#             upsample: str = "transconv",
#             deep_supervision: bool = False,
#     ) -> None:
#         """
#         Args:
#             img_size: dimension of input image.
#             in_channels: dimension of input channels.
#             out_channels: dimension of output channels.
#             feature_size: dimension of network feature size.
#             depths: number of layers in each stage.
#             num_heads: number of attention heads.
#             norm_name: feature normalization type and arguments.
#             drop_rate: dropout rate.
#             attn_drop_rate: attention dropout rate.
#             dropout_path_rate: drop path rate.
#             normalize: normalize output intermediate features in each stage.
#             use_checkpoint: use gradient checkpointing for reduced memory usage.
#             spatial_dims: number of spatial dims.
#             downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
#                 user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
#                 The default is currently `"merging"` (the original version defined in v0.9.0).

#         Examples::

#             # for 3D single channel input with size (96,96,96), 4-channel output and feature size of 48.
#             >>> net = SwinDER(img_size=(96,96,96), in_channels=1, out_channels=4, feature_size=48)

#             # for 3D 4-channel input with size (128,128,128), 3-channel output and (2,4,2,2) layers in each stage.
#             >>> net = SwinDER(img_size=(128,128,128), in_channels=4, out_channels=3, depths=(2,4,2,2))

#             # for 2D single channel input with size (96,96), 2-channel output and gradient checkpointing.
#             >>> net = SwinDER(img_size=(96,96), in_channels=3, out_channels=2, use_checkpoint=True, spatial_dims=2)

#         """

#         super().__init__()

#         img_size = ensure_tuple_rep(img_size, spatial_dims)  # img_size = img_size
#         patch_size = ensure_tuple_rep(2, spatial_dims)  # patch_size = (2,2,2) for 3D input
#         window_size = ensure_tuple_rep(7, spatial_dims)  # window_size = (7,7,7) for 3D input

#         if spatial_dims not in (2, 3):
#             raise ValueError("spatial dimension should be 2 or 3.")

#         for m, p in zip(img_size, patch_size):
#             for i in range(5):
#                 if m % np.power(p, i + 1) != 0:  # 输入图像尺寸必须是patch_size的整数倍
#                     raise ValueError("input image size (img_size) should be divisible by stage-wise image resolution.")

#         if not (0 <= drop_rate <= 1):
#             raise ValueError("dropout rate should be between 0 and 1.")

#         if not (0 <= attn_drop_rate <= 1):
#             raise ValueError("attention dropout rate should be between 0 and 1.")

#         if not (0 <= dropout_path_rate <= 1):
#             raise ValueError("drop path rate should be between 0 and 1.")

#         if feature_size % 12 != 0:
#             raise ValueError("feature_size should be divisible by 12.")

#         self.normalize = normalize
#         self.deep_supervision = deep_supervision

#         self.swinViT = SwinTransformer(
#             in_chans=in_channels,
#             embed_dim=feature_size,
#             window_size=window_size,
#             patch_size=patch_size,
#             depths=depths,
#             num_heads=num_heads,
#             mlp_ratio=4.0,
#             qkv_bias=True,
#             drop_rate=drop_rate,
#             attn_drop_rate=attn_drop_rate,
#             drop_path_rate=dropout_path_rate,
#             norm_layer=nn.LayerNorm,
#             use_checkpoint=use_checkpoint,
#             spatial_dims=spatial_dims,
#             downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample,
#         )

#         self.encoder1 = UnetrBasicBlock(
#             spatial_dims=spatial_dims,
#             in_channels=in_channels,
#             out_channels=feature_size,
#             kernel_size=3,
#             stride=1,
#             norm_name=norm_name,
#             res_block=True,
#         )

#         self.encoder2 = UnetrBasicBlock(
#             spatial_dims=spatial_dims,
#             in_channels=feature_size,
#             out_channels=feature_size,
#             kernel_size=3,
#             stride=1,
#             norm_name=norm_name,
#             res_block=True,
#         )

#         self.encoder3 = UnetrBasicBlock(
#             spatial_dims=spatial_dims,
#             in_channels=2 * feature_size,
#             out_channels=2 * feature_size,
#             kernel_size=3,
#             stride=1,
#             norm_name=norm_name,
#             res_block=True,
#         )

#         self.encoder4 = UnetrBasicBlock(
#             spatial_dims=spatial_dims,
#             in_channels=4 * feature_size,
#             out_channels=4 * feature_size,
#             kernel_size=3,
#             stride=1,
#             norm_name=norm_name,
#             res_block=True,
#         )

#         self.encoder5 = UnetrBasicBlock(
#             spatial_dims=spatial_dims,
#             in_channels=8 * feature_size,
#             out_channels=8 * feature_size,
#             kernel_size=3,
#             stride=1,
#             norm_name=norm_name,
#             res_block=True,
#         )

#         self.encoder10 = UnetrBasicBlock(
#             spatial_dims=spatial_dims,
#             in_channels=16 * feature_size,
#             out_channels=16 * feature_size,
#             kernel_size=3,
#             stride=1,
#             norm_name=norm_name,
#             res_block=True,
#         )

#         self.decoder5 = UnetrUpBlockWithAttention(
#             upsample=upsample,
#             spatial_dims=spatial_dims,
#             in_channels=16 * feature_size,
#             out_channels=8 * feature_size,
#             kernel_size=3,
#             upsample_kernel_size=2,
#             norm_name=norm_name,
#             sa_block=True,
#             res_block=False,
#         )

#         self.decoder4 = UnetrUpBlockWithAttention(
#             upsample=upsample,
#             spatial_dims=spatial_dims,
#             in_channels=feature_size * 8,
#             out_channels=feature_size * 4,
#             kernel_size=3,
#             upsample_kernel_size=2,
#             norm_name=norm_name,
#             sa_block=True,
#             res_block=False,
#         )

#         self.decoder3 = UnetrUpBlockWithAttention(
#             upsample=upsample,
#             spatial_dims=spatial_dims,
#             in_channels=feature_size * 4,
#             out_channels=feature_size * 2,
#             kernel_size=3,
#             upsample_kernel_size=2,
#             norm_name=norm_name,
#             sa_block=True,
#             res_block=False,
#         )
#         self.decoder2 = UnetrUpBlockWithAttention(
#             upsample=upsample,
#             spatial_dims=spatial_dims,
#             in_channels=feature_size * 2,
#             out_channels=feature_size,
#             kernel_size=3,
#             upsample_kernel_size=2,
#             norm_name=norm_name,
#             sa_block=True,
#             res_block=False,
#         )

#         self.decoder1 = UnetrUpBlockWithAttention(
#             upsample=upsample,
#             spatial_dims=spatial_dims,
#             in_channels=feature_size,
#             out_channels=feature_size,
#             kernel_size=3,
#             upsample_kernel_size=2,
#             norm_name=norm_name,
#             sa_block=True,
#             res_block=False,
#         )
#         self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
#         self.out1 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
#         self.out2 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size * 2, out_channels=out_channels)
#         self.out3 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size * 4, out_channels=out_channels)
#         self.out4 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size * 8, out_channels=out_channels)

#     def load_from(self, weights):

#         with torch.no_grad():
#             self.swinViT.patch_embed.proj.weight.copy_(weights["state_dict"]["module.patch_embed.proj.weight"])
#             self.swinViT.patch_embed.proj.bias.copy_(weights["state_dict"]["module.patch_embed.proj.bias"])
#             for bname, block in self.swinViT.layers1[0].blocks.named_children():
#                 block.load_from(weights, n_block=bname, layer="layers1")
#             self.swinViT.layers1[0].downsample.reduction.weight.copy_(
#                 weights["state_dict"]["module.layers1.0.downsample.reduction.weight"]
#             )
#             self.swinViT.layers1[0].downsample.norm.weight.copy_(
#                 weights["state_dict"]["module.layers1.0.downsample.norm.weight"]
#             )
#             self.swinViT.layers1[0].downsample.norm.bias.copy_(
#                 weights["state_dict"]["module.layers1.0.downsample.norm.bias"]
#             )
#             for bname, block in self.swinViT.layers2[0].blocks.named_children():
#                 block.load_from(weights, n_block=bname, layer="layers2")
#             self.swinViT.layers2[0].downsample.reduction.weight.copy_(
#                 weights["state_dict"]["module.layers2.0.downsample.reduction.weight"]
#             )
#             self.swinViT.layers2[0].downsample.norm.weight.copy_(
#                 weights["state_dict"]["module.layers2.0.downsample.norm.weight"]
#             )
#             self.swinViT.layers2[0].downsample.norm.bias.copy_(
#                 weights["state_dict"]["module.layers2.0.downsample.norm.bias"]
#             )
#             for bname, block in self.swinViT.layers3[0].blocks.named_children():
#                 block.load_from(weights, n_block=bname, layer="layers3")
#             self.swinViT.layers3[0].downsample.reduction.weight.copy_(
#                 weights["state_dict"]["module.layers3.0.downsample.reduction.weight"]
#             )
#             self.swinViT.layers3[0].downsample.norm.weight.copy_(
#                 weights["state_dict"]["module.layers3.0.downsample.norm.weight"]
#             )
#             self.swinViT.layers3[0].downsample.norm.bias.copy_(
#                 weights["state_dict"]["module.layers3.0.downsample.norm.bias"]
#             )
#             for bname, block in self.swinViT.layers4[0].blocks.named_children():
#                 block.load_from(weights, n_block=bname, layer="layers4")
#             self.swinViT.layers4[0].downsample.reduction.weight.copy_(
#                 weights["state_dict"]["module.layers4.0.downsample.reduction.weight"]
#             )
#             self.swinViT.layers4[0].downsample.norm.weight.copy_(
#                 weights["state_dict"]["module.layers4.0.downsample.norm.weight"]
#             )
#             self.swinViT.layers4[0].downsample.norm.bias.copy_(
#                 weights["state_dict"]["module.layers4.0.downsample.norm.bias"]
#             )

#     def forward(self, x_in):
#         hidden_states_out = self.swinViT(x_in, self.normalize)
#         enc0 = self.encoder1(x_in)
#         enc1 = self.encoder2(hidden_states_out[0])
#         enc2 = self.encoder3(hidden_states_out[1])
#         enc3 = self.encoder4(hidden_states_out[2])
#         enc4 = self.encoder5(hidden_states_out[3])
#         dec4 = self.encoder10(hidden_states_out[4])
#         dec3 = self.decoder5(dec4, enc4)
#         dec2 = self.decoder4(dec3, enc3)
#         dec1 = self.decoder3(dec2, enc2)
#         dec0 = self.decoder2(dec1, enc1)
#         out = self.decoder1(dec0, enc0)
#         output0 = self.out(out)
#         output1 = self.out1(dec0)
#         output2 = self.out2(dec1)
#         output3 = self.out3(dec2)
#         output4 = self.out4(dec3)
#         logits = [output0, output1, output2, output3, output4]

#         if self.deep_supervision:
#             r = logits
#         else:
#             r = logits[0]
#         return r
 