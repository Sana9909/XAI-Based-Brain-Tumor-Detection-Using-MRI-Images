# # These four are your foundation
# import torch
# import torch.nn as nn
# from swin_methods import (
#     PatchEmbed,
#     BasicLayer,
#     PatchMerging,
#     PatchExpand
# )

# from Decoder_methods import (
#     CBE,
#     DecoderBlock,
# )


# device = "cuda" if torch.cuda.is_available() else "cpu"

# class SwinHAFUNet(nn.Module):
#     def __init__(self,img_size=224, in_ch=1, num_classes=2, embed_dim=96, window_size=7, patch_size=4):
#         super().__init__()

#         C  = embed_dim
#         H0 = img_size // patch_size  # 56 after patch embed
#         # patch_dim = in_ch * ps * ps

#         self.patch_embed = PatchEmbed(img_size=img_size ,patch_size=patch_size,in_chans=in_ch,embed_dim=embed_dim)

#         self.enc1 = BasicLayer(
#             dim=C,
#             input_resolution=(H0, H0),
#             depth=2,
#             num_heads=3,
#             window_size=window_size,
#             downsample=PatchMerging   # merging happens inside BasicLayer
#         )

#         self.enc2 = BasicLayer(
#             dim=2*C,
#             input_resolution=(H0//2, H0//2),
#             depth=2,
#             num_heads=6,
#             window_size=window_size,
#             downsample=PatchMerging
#         )

#         self.enc3 = BasicLayer(
#             dim=4*C,
#             input_resolution=(H0//4, H0//4),
#             depth=2,
#             num_heads=12,
#             window_size=window_size,
#             downsample=PatchMerging
#         )
#         self.norm = nn.LayerNorm(8*C)

#         # ───────── Bottleneck ─────────
#         self.cbe = CBE(8*C, window_size=window_size)

#         # ───────── Decoder ─────────
#         h3 = H0 // 8
#         h2 = H0 // 4
#         h1 = H0 // 2

#         self.dec_block1 = DecoderBlock(8*C, (h3, h3), num_heads=12, window_size=window_size)
#         self.dec_block2 = DecoderBlock(4*C, (h2, h2), num_heads=6,  window_size=window_size)
#         self.dec_block3 = DecoderBlock(2*C, (h1, h1), num_heads=3,  window_size=window_size)

#         # ───────── Final Upsampling ─────────
#         self.final_expand1 = PatchExpand((H0, H0), C)
#         self.final_expand2 = PatchExpand((H0*2, H0*2), C // 2)

#         # ───────── Segmentation Head ─────────
#         self.head = nn.Conv2d(C // 4, num_classes, kernel_size=1)
#     def tokens_to_image(self, x):
#         B, L, C = x.shape
#         H = W = int(L ** 0.5)
#         return x.view(B, H, W, C)


#     def forward(self, img):
#         B = img.shape[0]

#         # ── Encoder ──────────────────────────────────────────
#         x = self.patch_embed(img)   # (B, 3136, C)

#         s1 = self.tokens_to_image(x)    # (B,56,56,C)
#         x = self.enc1(x)

#         s2 = self.tokens_to_image(x)    # (B,28,28,2C)
#         x = self.enc2(x)

#         s3 = self.tokens_to_image(x)    # (B,14,14,4C)
#         x = self.enc3(x)

#         # ── Bottleneck ───────────────────────────────────────
#         x = self.norm(x)
#         x = self.tokens_to_image(x)     # (B,7,7,8C)
#         x = self.cbe(x)

#         # ── Decoder ──────────────────────────────────────────
#         x = self.dec_block1(x, s3)      # (B,14,14,4C)
#         x = self.dec_block2(x, s2)      # (B,28,28,2C)
#         x = self.dec_block3(x, s1)      # (B,56,56,C)

#         # ── Final Upsampling ─────────────────────────────────
#         x = x.view(B, -1, x.shape[-1])   # tokens

#         x = self.final_expand1(x)        # (B,12544,C/2)
#         x = self.final_expand2(x)        # (B,50176,C/4)

#         # ── Back to image ────────────────────────────────────
#         H = W = int(x.shape[1] ** 0.5)
#         x = x.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

#         # ── Head ─────────────────────────────────────────────
#         logits = self.head(x)

#         return logits

    
# def main(epoch=5):
#     print("CUDA Available:", torch.cuda.is_available())
#     if torch.cuda.is_available():
#         print("CUDA Device Name:", torch.cuda.get_device_name(0))

    
#     from torchinfo import summary
#     print("Before summary")
#     model = SwinHAFUNet(img_size=448,num_classes=1).to(device)
#     summary(model, input_size=(1, 1, 448, 448), device=device)
#     print("After summary")


# if __name__ == "__main__":
#     epoch = 30
#     main(epoch)

  
import torch
import torch.nn as nn
from swin_methods import SwinTransformerSys

device = "cuda" if torch.cuda.is_available() else "cpu"

class SwinUnet(nn.Module):
    def __init__(self, img_size=448, num_classes=1):
        super(SwinUnet, self).__init__()

        self.swin_unet = SwinTransformerSys(
            img_size=img_size,
            patch_size=4,
            in_chans=1,
            num_classes=num_classes,

            embed_dim=96,
            depths=[2, 2, 2, 2],
            depths_decoder=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],

            window_size=7,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            drop_path_rate=0.1,

            ape=False,
            patch_norm=True,
            use_checkpoint=False
        )

    def forward(self, x):
        return self.swin_unet(x)

def main():
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA Device Name:", torch.cuda.get_device_name(0))

    
    from torchinfo import summary
    print("Before summary")
    model = SwinUnet(img_size=448,num_classes=1).to(device)
    summary(model, input_size=(1, 1, 448, 448), device=device)
    print("After summary")


if __name__ == "__main__":
    main()