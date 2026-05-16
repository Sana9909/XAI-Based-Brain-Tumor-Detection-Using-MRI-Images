 
  
import torch
import torch.nn as nn
from swin_methods import SwinTransformerSys

device = "cuda" if torch.cuda.is_available() else "cpu"

class SwinDSANet(nn.Module):
    def __init__(self, img_size=448, num_classes=1):
        super(SwinDSANet, self).__init__()

        self.swin_unet = SwinTransformerSys(
            img_size=img_size,
            patch_size=4,
            in_chans=1,
            num_classes=num_classes,

            embed_dim=96,
            depths=[2, 2, 6,2],
            depths_decoder=[2, 2, 6,2],
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
    model = SwinDSANet(img_size=448,num_classes=1).to(device)
    summary(model, input_size=(1, 1, 448, 448), device=device)
    print("After summary")


if __name__ == "__main__":
    main()

 