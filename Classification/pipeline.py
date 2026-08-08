# =========================================================
# COMPLETE MRI -> SEGMENTATION -> GRADCAM ->
# MRI+MASK STACK -> CLASSIFICATION PIPELINE
# =========================================================

import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)

from swin_custom import SwinUnet

# =========================================================
# DEVICE
# =========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

# =========================================================
# CLASS NAMES
# =========================================================

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary"
]

# =========================================================
# Z-SCORE NORMALIZATION
# =========================================================

def zscore(img):

    img = img.astype(np.float32)

    mean = img.mean()

    std = img.std()

    if std < 1e-6:
        std = 1e-6

    img = (img - mean) / std

    return img

# =========================================================
# LOAD SEGMENTATION MODEL
# =========================================================

print("\nLoading segmentation model...")

seg_model = SwinUnet().to(device)

ckpt = torch.load(
    "SwinHafNet_model_8535.pth",
    map_location=device,
    weights_only=False
)

seg_model.load_state_dict(
    ckpt["model_state_dict"]
)

seg_model.eval()

print("Segmentation model loaded.")

# =========================================================
# GRADCAM
# =========================================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model

        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(
            self.forward_hook
        )

        target_layer.register_full_backward_hook(
            self.backward_hook
        )

    def forward_hook(self, module, inp, out):

        self.activations = out.detach()

    def backward_hook(self, module, grad_in, grad_out):

        self.gradients = grad_out[0].detach()

    def generate(self, x):

        self.gradients = None
        self.activations = None

        self.model.zero_grad()

        final = self.model(x)

        prob = torch.sigmoid(final)

        score = (prob * (prob > 0.5)).sum()

        score.backward()

        grads = self.gradients
        acts = self.activations

        # =================================================
        # SWIN TOKEN CASE
        # =================================================

        if grads.dim() == 3:

            weights = grads.mean(dim=1, keepdim=True)

            cam = (weights * acts).sum(dim=2)

            B, N = cam.shape

            H = W = int(np.sqrt(N))

            cam = cam.reshape(B, H, W)

        # =================================================
        # CNN FEATURE MAP CASE
        # =================================================

        else:

            weights = grads.mean(
                dim=(2,3),
                keepdim=True
            )

            cam = (weights * acts).sum(dim=1)

        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()

        cam = cam - cam.min()

        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

target_layer = seg_model.swin_unet.layers_up[-1]

gradcam = GradCAM(
    seg_model,
    target_layer
)

print("GradCAM initialized.")

# =========================================================
# CLASSIFICATION MODEL
# =========================================================

def make_resnet18_2ch(num_classes):

    weights = ResNet18_Weights.DEFAULT

    model = resnet18(weights=weights)

    old_conv = model.conv1

    # =====================================================
    # CHANGE INPUT CHANNELS 3 -> 2
    # =====================================================

    new_conv = nn.Conv2d(
        in_channels=2,
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )

    with torch.no_grad():

        old_weights = old_conv.weight.data

        mean_weight = old_weights.mean(
            dim=1,
            keepdim=True
        )

        new_conv.weight[:, 0:1] = mean_weight
        new_conv.weight[:, 1:2] = mean_weight

    model.conv1 = new_conv

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model

print("\nLoading classification model...")

clf_model = make_resnet18_2ch(4)

clf_model.load_state_dict(
    torch.load(
        "./checkpoints_2ch/best_resnet18_2ch.pth",
        map_location=device
    )
)

clf_model = clf_model.to(device)

clf_model.eval()

print("Classification model loaded.")

# =========================================================
# PIPELINE
# =========================================================

def predict_pipeline(
    image_path,
    threshold=0.3,
    img_size=448
):

    # =====================================================
    # LOAD MRI
    # =====================================================

    img = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    original = cv2.resize(
        img,
        (img_size, img_size)
    )

    img_float = original.astype(np.float32) / 255.0

    # =====================================================
    # SEGMENTATION INPUT
    # =====================================================

    norm_img = zscore(img_float)

    seg_tensor = torch.from_numpy(
        norm_img
    ).unsqueeze(0).unsqueeze(0).float().to(device)

    # =====================================================
    # GRADCAM
    # =====================================================

    cam_map = gradcam.generate(seg_tensor)

    cam_map = cv2.resize(
        cam_map,
        (img_size, img_size)
    )

    cam_uint8 = (cam_map * 255).astype(np.uint8)

    cam_color = cv2.applyColorMap(
        cam_uint8,
        cv2.COLORMAP_JET
    )

    original_bgr = cv2.cvtColor(
        original,
        cv2.COLOR_GRAY2BGR
    )

    grad_overlay = cv2.addWeighted(
        original_bgr,
        0.6,
        cam_color,
        0.4,
        0
    )

    # =====================================================
    # SEGMENTATION PREDICTION
    # =====================================================

    with torch.no_grad():

        seg_out = seg_model(seg_tensor)

        seg_out = F.interpolate(
            seg_out,
            size=(img_size, img_size),
            mode='bilinear',
            align_corners=False
        )

        prob_map = torch.sigmoid(
            seg_out
        ).squeeze().cpu().numpy()

    # =====================================================
    # THRESHOLD MASK
    # =====================================================

    mask = (prob_map >= threshold).astype(np.float32)

    mask_vis = (mask * 255).astype(np.uint8)

    # =====================================================
    # MRI + MASK STACK
    # =====================================================

    stacked = np.stack(
        [
            img_float,
            mask
        ],
        axis=0
    )

    clf_tensor = torch.tensor(
        stacked,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    # =====================================================
    # CLASSIFICATION
    # =====================================================

    with torch.no_grad():

        logits = clf_model(clf_tensor)

        probs = torch.softmax(
            logits,
            dim=1
        )

        pred_idx = probs.argmax(dim=1).item()

        confidence = probs[0][pred_idx].item()

    prediction = CLASS_NAMES[pred_idx]

    # =====================================================
    # PRINT RESULTS
    # =====================================================

    print("\n===================================")
    print("FINAL PREDICTION")
    print("===================================")

    print(f"Class      : {prediction}")
    print(f"Confidence : {confidence:.4f}")

    # =====================================================
    # VISUALIZATION
    # =====================================================

    plt.figure(figsize=(15,5))

    # MRI
    plt.subplot(1,3,1)

    plt.imshow(original, cmap='gray')

    plt.title("Input MRI")

    plt.axis('off')

    # MASK
    plt.subplot(1,3,2)

    plt.imshow(mask_vis, cmap='gray')

    plt.title("Predicted Mask")

    plt.axis('off')

    # GRADCAM
    plt.subplot(1,3,3)

    plt.imshow(
        cv2.cvtColor(
            grad_overlay,
            cv2.COLOR_BGR2RGB
        )
    )

    plt.title("GradCAM Overlay")

    plt.axis('off')

    plt.tight_layout()

    plt.show()

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--img",
        type=str,
        required=True,
        help="Path to MRI image"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Segmentation threshold"
    )

    args = parser.parse_args()

    predict_pipeline(
        image_path=args.img,
        threshold=args.threshold
    )