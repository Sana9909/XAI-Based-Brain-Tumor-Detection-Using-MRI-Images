 
import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torchvision.models import resnet18, ResNet18_Weights
from swin_custom import SwinUnet

# ─── SHAP module (same directory) ─────────────────────────────────────────────
from shap_explainer import build_background_tensor, glob_mri_paths, SHAPExplainer

# =============================================================================
# DEVICE
# =============================================================================
torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# =============================================================================
# CLASS NAMES
# =============================================================================

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary",
]

# =============================================================================
# Z-SCORE NORMALIZATION
# =============================================================================

def zscore(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    mean, std = img.mean(), img.std()
    if std < 1e-6:
        std = 1e-6
    return (img - mean) / std


# =============================================================================
# LOAD SEGMENTATION MODEL
# =============================================================================

print("\nLoading segmentation model...")
seg_model = SwinUnet().to(device)
ckpt = torch.load(
    "SwinHafNet_model_8535.pth",
    map_location=device,
    weights_only=False,
)
seg_model.load_state_dict(ckpt["model_state_dict"])
seg_model.eval()
print("Segmentation model loaded.")

# =============================================================================
# GRADCAM  (applied to the segmentation model)
# =============================================================================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, x: torch.Tensor) -> np.ndarray:
        self.gradients = self.activations = None
        self.model.zero_grad()

        final = self.model(x)
        prob  = torch.sigmoid(final)
        score = (prob * (prob > 0.5)).sum()
        score.backward()

        grads = self.gradients
        acts  = self.activations

        # Swin token path
        if grads.dim() == 3:
            weights = grads.mean(dim=1, keepdim=True)
            cam     = (weights * acts).sum(dim=2)
            B, N    = cam.shape
            H = W   = int(np.sqrt(N))
            cam     = cam.reshape(B, H, W)
        else:
            # CNN feature-map path
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam     = (weights * acts).sum(dim=1)

        cam = F.relu(cam).squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


target_layer = seg_model.swin_unet.layers_up[-1]
gradcam      = GradCAM(seg_model, target_layer)
print("GradCAM initialised.")


# =============================================================================
# HELPER CALLABLES  (used by build_background_tensor in shap_explainer.py)
# These thin wrappers let the SHAP module produce background samples without
# knowing the internals of the seg model or GradCAM object.
# =============================================================================

def _run_segmentation(img_float: np.ndarray, img_size: int = 448) -> np.ndarray:
    """
    img_float : (H, W) float32 [0,1]
    returns   : (H, W) float32 binary mask [0,1]
    """
    norm  = zscore(img_float)
    t     = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        out  = seg_model(t)
        out  = F.interpolate(out, size=(img_size, img_size),
                             mode="bilinear", align_corners=False)
        mask = (torch.sigmoid(out) >= 0.3).squeeze().cpu().numpy().astype(np.float32)
    return mask


def _run_gradcam(img_float: np.ndarray, img_size: int = 448) -> np.ndarray:
    """
    img_float : (H, W) float32 [0,1]
    returns   : (H, W) float32 heatmap [0,1]
    """
    norm   = zscore(img_float)
    t      = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).float().to(device)
    cam    = gradcam.generate(t)
    cam    = cv2.resize(cam, (img_size, img_size))
    cam   -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()
    return cam.astype(np.float32)


# =============================================================================
# CLASSIFICATION MODEL (ResNet-18, 2-channel input)
# =============================================================================

def make_resnet18_2ch(num_classes: int) -> nn.Module:
    model    = resnet18(weights=ResNet18_Weights.DEFAULT)
    old_conv = model.conv1
    new_conv = nn.Conv2d(
        in_channels=2, out_channels=64,
        kernel_size=7, stride=2, padding=3, bias=False,
    )
    with torch.no_grad():
        mean_w = old_conv.weight.data.mean(dim=1, keepdim=True)
        new_conv.weight[:, 0:1] = mean_w
        new_conv.weight[:, 1:2] = mean_w
    model.conv1 = new_conv
    model.fc    = nn.Linear(model.fc.in_features, num_classes)
    return model


print("\nLoading classification model...")
clf_model = make_resnet18_2ch(4)
clf_model.load_state_dict(
    torch.load("./checkpoints_2ch/best_resnet18_2ch.pth", map_location=device)
)
clf_model = clf_model.to(device)
clf_model.eval()
print("Classification model loaded.")


# NOTE ─────────────────────────────────────────────────────────────────────────
# The pipeline uses a 2-channel classifier (MRI + mask).
# The SHAP module is written for the 3-channel variant
# (resnet18_colab_train_3ch.py).  If you switch to the 3-ch model, replace the
# stack below with:
#     stacked = np.stack([img_float, mask, cam_map], axis=0)  # (3, H, W)
# and load the 3-ch checkpoint instead.
# ──────────────────────────────────────────────────────────────────────────────


# =============================================================================
# SHAP EXPLAINER — lazy-initialised the first time it is needed
# =============================================================================

_shap_explainer: SHAPExplainer | None = None


def init_shap(
    bg_dir: str,
    n:      int  = 32,
    img_size: int = 448,
) -> None:
    """
    Call once before predict_pipeline() when --shap is enabled.
    Builds the background tensor and creates the global SHAPExplainer.
    """
    global _shap_explainer

    print(f"\n[SHAP] Building background tensor from '{bg_dir}' (n={n})…")
    paths = glob_mri_paths(bg_dir)
    if not paths:
        raise RuntimeError(
            f"[SHAP] No images found in {bg_dir}. "
            "Provide a directory with at least a few MRI samples."
        )

    # Wrap the pipeline helpers into fixed-size callables
    seg_fn  = lambda img: _run_segmentation(img, img_size=img_size)
    gcam_fn = lambda img: _run_gradcam(img, img_size=img_size)

    background = build_background_tensor(
        image_paths = paths,
        seg_fn      = seg_fn,
        gcam_fn     = gcam_fn,
        n           = n,
        img_size    = img_size,
        device      = device,
    )

    # ── Adapt background to match the actual classifier input ──────────
    # The 2-channel clf uses (MRI, mask).  Drop the gradcam channel if
    # the background was built with 3 channels.
    if background.shape[1] == 3 and clf_model.conv1.in_channels == 2:
        background = background[:, :2, :, :]

    _shap_explainer = SHAPExplainer(
        model       = clf_model,
        background  = background,
        class_names = CLASS_NAMES,
        device      = device,
    )
    print("[SHAP] Explainer ready.")


# =============================================================================
# MAIN PREDICTION PIPELINE
# =============================================================================

def predict_pipeline(
    image_path: str,
    threshold:  float = 0.5,
    img_size:   int   = 448,
    run_shap:   bool  = False,
    shap_out:   str   = "./shap_outputs",
    sample_tag: str   = "sample",
) -> dict:

    # ─── 1. Load & resize MRI ────────────────────────────────────────────────
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    original  = cv2.resize(img, (img_size, img_size))
    img_float = original.astype(np.float32) / 255.0

    # ─── 2. Segmentation input tensor ────────────────────────────────────────
    norm_img   = zscore(img_float)
    seg_tensor = (
        torch.from_numpy(norm_img)
        .unsqueeze(0).unsqueeze(0)
        .float().to(device)
    )

    # ─── 3. GradCAM (from seg model) ─────────────────────────────────────────
    cam_map = gradcam.generate(seg_tensor)
    cam_map = cv2.resize(cam_map, (img_size, img_size))

    cam_uint8   = (cam_map * 255).astype(np.uint8)
    cam_color   = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    original_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    grad_overlay = cv2.addWeighted(original_bgr, 0.6, cam_color, 0.4, 0)

    # ─── 4. Segmentation prediction ──────────────────────────────────────────
    with torch.no_grad():
        seg_out  = seg_model(seg_tensor)
        seg_out  = F.interpolate(seg_out, size=(img_size, img_size),
                                 mode="bilinear", align_corners=False)
        prob_map = torch.sigmoid(seg_out).squeeze().cpu().numpy()

    mask     = (prob_map >= threshold).astype(np.float32)
    mask_vis = (mask * 255).astype(np.uint8)

    # ─── 5. Build classifier input (2-ch: MRI + mask) ────────────────────────
    stacked    = np.stack([img_float, mask], axis=0)           # (2, H, W)
    clf_tensor = (
        torch.tensor(stacked, dtype=torch.float32)
        .unsqueeze(0).to(device)
    )

    # ─── 6. Classification ───────────────────────────────────────────────────
    with torch.no_grad():
        logits     = clf_model(clf_tensor)
        probs      = torch.softmax(logits, dim=1)
        pred_idx   = probs.argmax(dim=1).item()
        confidence = probs[0][pred_idx].item()

    prediction = CLASS_NAMES[pred_idx]

    # ─── 7. Print results ────────────────────────────────────────────────────
    print("\n===================================")
    print("FINAL PREDICTION")
    print("===================================")
    print(f"Class      : {prediction}")
    print(f"Confidence : {confidence:.4f}")
    all_probs = {
        CLASS_NAMES[i]: round(float(probs[0][i].item()), 4)
        for i in range(len(CLASS_NAMES))
    }
    for k, v in all_probs.items():
        print(f"  {k:<14}: {v:.4f}")

    # ─── 8. Standard visualisation ───────────────────────────────────────────
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap="gray")
    plt.title("Input MRI"); plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask_vis, cmap="gray")
    plt.title("Predicted Mask"); plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(grad_overlay, cv2.COLOR_BGR2RGB))
    plt.title("GradCAM Overlay"); plt.axis("off")

    plt.suptitle(
        f"Prediction: {prediction}  |  Confidence: {confidence:.2%}",
        fontsize=14, fontweight="bold",
    )

    os.makedirs("outputs", exist_ok=True)

    save_path = os.path.join(
        "outputs",
        f"{prediction}_{sample_tag}.png"
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    print(f"\nVisualization saved to: {save_path}")

    plt.close()

    # ─── 9. SHAP explanation ──────────────────────────────────────────────────
    shap_result = None
    if run_shap:
        if _shap_explainer is None:
            raise RuntimeError(
                "SHAPExplainer is not initialised. "
                "Call init_shap() before predict_pipeline()."
            )
        print("\n[SHAP] Running explanation…")
        shap_result = _shap_explainer.visualize(
            input_tensor = clf_tensor,
            save_dir     = shap_out,
            sample_tag   = sample_tag,
        )
        print(
            f"[SHAP] Done. Plots saved to: {shap_result['save_dir']}"
        )

    return {
        "prediction"      : prediction,
        "confidence"      : confidence,
        "all_probs"       : all_probs,
        "shap_result"     : shap_result,
    }


# =============================================================================
# CLI ENTRY POINT
# =============================================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="MRI Brain Tumour Classification Pipeline with SHAP"
    )

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

    parser.add_argument(
        "--img-size",
        type=int,
        default=448,
        help="Image size"
    )

    # SHAP arguments
    parser.add_argument(
        "--shap",
        action="store_true",
        help="Enable SHAP explanation"
    )

    parser.add_argument(
        "--shap-bg-dir",
        type=str,
        default=None,
        help="Background MRI directory for SHAP"
    )

    parser.add_argument(
        "--shap-bg-n",
        type=int,
        default=32,
        help="Number of SHAP background samples"
    )

    parser.add_argument(
        "--shap-out",
        type=str,
        default="./shap_outputs",
        help="Directory to save SHAP outputs"
    )

    args = parser.parse_args()

    # Initialize SHAP
    if args.shap:

        if args.shap_bg_dir is None:
            parser.error(
                "--shap-bg-dir is required when using --shap"
            )

        init_shap(
            bg_dir=args.shap_bg_dir,
            n=args.shap_bg_n,
            img_size=args.img_size,
        )

    from pathlib import Path

    sample_tag = Path(args.img).stem

    result = predict_pipeline(
        image_path=args.img,
        threshold=args.threshold,
        img_size=args.img_size,
        run_shap=args.shap,
        shap_out=args.shap_out,
        sample_tag=sample_tag,
    )