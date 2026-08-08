# =============# =========================================================
# RESEARCH-GRADE EXPLAINABLE MRI PIPELINE
# MRI -> SEGMENTATION -> GRADCAM -> LIME -> SHAP
# -> MRI+MASK STACK -> CLASSIFICATION -> PER-CLASS XAI
#
# Key improvements over baseline:
#   - LIME explains the CLASSIFIER (not segmentor)
#     on the correct 2-channel [MRI, Mask] input space
#   - SHAP uses real-distribution background (not random)
#     with per-channel attribution decomposition
#   - Per-class SHAP attribution maps for all 4 classes
#   - Faithfulness / sufficiency / comprehensiveness metrics
#   - LIME superpixel stability check (Jaccard across runs)
#   - Structured JSON export of all quantitative results
#   - Publication-ready figure layout with colorbars
# =========================================================

import argparse
import json
import os
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shap

from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

from torchvision.models import resnet18, ResNet18_Weights

from swin_custom import SwinUnet

warnings.filterwarnings("ignore")
matplotlib.use("Agg")   # headless / non-interactive backend

# =========================================================
# DEVICE
# =========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Device] {device}")

# =========================================================
# CONSTANTS
# =========================================================

CLASS_NAMES   = ["glioma", "meningioma", "no_tumor", "pituitary"]
NUM_CLASSES   = len(CLASS_NAMES)
CHANNEL_NAMES = ["MRI Intensity", "Tumor Mask"]   # 2-ch classifier channels

# =========================================================
# UTILITIES
# =========================================================

def zscore(img: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalisation."""
    img = img.astype(np.float32)
    std = img.std()
    return (img - img.mean()) / (std if std > 1e-6 else 1e-6)


def minmax(arr: np.ndarray) -> np.ndarray:
    """Normalise array to [0, 1]."""
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def iou_binary(a: np.ndarray, b: np.ndarray, thr: float = 0.5) -> float:
    """Binary IoU between two soft maps thresholded at `thr`."""
    A = (a >= thr).astype(bool)
    B = (b >= thr).astype(bool)
    inter = (A & B).sum()
    union = (A | B).sum()
    return float(inter / union) if union > 0 else 1.0


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Pixel-level Spearman rank correlation between two maps."""
    corr, _ = spearmanr(a.ravel(), b.ravel())
    return float(corr)


# =========================================================
# LOAD SEGMENTATION MODEL
# =========================================================

print("\n[Segmentation] Loading SwinUNet...")

seg_model = SwinUnet().to(device)
ckpt = torch.load(
    "SwinHafNet_model_8535.pth",
    map_location=device,
    weights_only=False,
)
seg_model.load_state_dict(ckpt["model_state_dict"])
seg_model.eval()

print("[Segmentation] Model ready.")

# =========================================================
# CLASSIFICATION MODEL  (2-channel ResNet-18)
# =========================================================

def make_resnet18_2ch(num_classes: int) -> nn.Module:
    model   = resnet18(weights=ResNet18_Weights.DEFAULT)
    old_w   = model.conv1.weight.data                        # (64, 3, 7, 7)
    mean_w  = old_w.mean(dim=1, keepdim=True)                # (64, 1, 7, 7)
    new_conv = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        new_conv.weight[:, 0:1] = mean_w
        new_conv.weight[:, 1:2] = mean_w
    model.conv1 = new_conv
    model.fc    = nn.Linear(model.fc.in_features, num_classes)
    return model


print("\n[Classifier] Loading ResNet-18 2-channel...")

clf_model = make_resnet18_2ch(NUM_CLASSES)
clf_model.load_state_dict(
    torch.load("./checkpoints_2ch/best_resnet18_2ch.pth", map_location=device)
)
clf_model = clf_model.to(device).eval()

print("[Classifier] Model ready.")

# =========================================================
# GRAD-CAM  (vanilla, Selvaraju et al. 2017)
# =========================================================

class GradCAM:
    """
    Vanilla Gradient-weighted Class Activation Mapping.

    Works for both:
      - CNN classifiers   : output (1, C)     -> uses target class logit
      - Segmentation nets : output (1,1,H,W)  -> uses mean sigmoid score

    The target layer must be in the forward graph and output either:
      - 4-D (B, C, H, W)  for standard CNNs
      - 3-D (B, N, C)     for transformer token outputs (SwinUNet)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model       = model
        self.activations = None
        self.gradients   = None
        target_layer.register_forward_hook(self._fwd_hook)
        target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, _module, _inp, out):
        # Some SwinUNet blocks return a tuple; take the first element
        self.activations = (out[0] if isinstance(out, tuple) else out).detach()

    def _bwd_hook(self, _module, _grad_in, grad_out):
        g = grad_out[0] if isinstance(grad_out, tuple) else grad_out
        self.gradients = g.detach()

    def generate(
        self,
        x: torch.Tensor,
        class_idx: int | None = None,
    ) -> np.ndarray:
        """
        Returns a (H, W) GradCAM saliency map normalised to [0, 1].

        Parameters
        ----------
        x         : input tensor (gradients must be enabled)
        class_idx : target class for classification models;
                    ignored for segmentation models
        """
        self.gradients   = None
        self.activations = None
        self.model.zero_grad()

        out = self.model(x)

        # ── Scalar score for backward ──────────────────────
        if out.dim() == 4:
            # Segmentation output (B,1,H,W) -> mean sigmoid prob
            score = torch.sigmoid(out).mean()
        else:
            # Classification output (B, num_classes)
            if class_idx is None:
                class_idx = int(out.argmax(dim=1).item())
            score = out[0, class_idx]

        score.backward()

        grads = self.gradients
        acts  = self.activations

        if grads is None or acts is None:
            raise RuntimeError(
                "GradCAM hooks returned None — make sure the target "
                "layer is actually in the forward path."
            )

        # ── Vanilla GradCAM weights (global average pooling) ──
        if grads.dim() == 3:
            # Transformer token layout (B, N, C)
            weights = grads.mean(dim=1, keepdim=True)       # (B,1,C)
            cam     = (weights * acts).sum(dim=2)           # (B,N)
            N       = cam.shape[-1]
            H = W   = int(np.sqrt(N))
            cam     = cam.reshape(-1, H, W)                 # (B,H,W)
        else:
            # Standard CNN (B, C, H, W)
            weights = grads.mean(dim=(2, 3), keepdim=True)  # (B,C,1,1)
            cam     = (weights * acts).sum(dim=1)           # (B,H,W)

        cam = F.relu(cam).squeeze().cpu().numpy()           # (H,W)
        return minmax(cam)


# ── One GradCAM instance per model ────────────────────────
target_seg_layer = seg_model.swin_unet.layers_up[-1]
gradcam_seg      = GradCAM(seg_model, target_seg_layer)

target_clf_layer = clf_model.layer4[-1]
gradcam_clf      = GradCAM(clf_model, target_clf_layer)

print("[GradCAM] Initialized for both models.")

# =========================================================
# LIME  — explains the CLASSIFIER on the 2-channel input
# =========================================================

# Module-level slot for the fixed mask injected into the callback
_lime_fixed_mask: np.ndarray | None = None


def _clf_predict_lime(images_rgb: np.ndarray) -> np.ndarray:
    """
    LIME predict callback — maps LIME's RGB perturbations to the
    2-channel classifier.

    LIME internally zeroes superpixels on an (H,W,3) RGB image.
    We take channel-0 as the perturbed MRI intensity and keep
    the segmentation mask fixed (_lime_fixed_mask), so LIME
    attributes importance to MRI texture regions w.r.t. the
    4-class diagnosis, not the segmentor.

    Returns
    -------
    np.ndarray  shape (N, NUM_CLASSES) — softmax probabilities
    """
    global _lime_fixed_mask
    tensors = []
    for img in images_rgb:
        mri_ch  = img[:, :, 0].astype(np.float32)
        stacked = np.stack([mri_ch, _lime_fixed_mask], axis=0)  # (2,H,W)
        tensors.append(stacked)

    batch = torch.tensor(
        np.stack(tensors), dtype=torch.float32
    ).to(device)

    with torch.no_grad():
        probs = torch.softmax(clf_model(batch), dim=1)
    return probs.cpu().numpy()


def explain_lime(
    img_float: np.ndarray,
    mask: np.ndarray,
    pred_class: int,
    num_samples: int = 512,
    num_features: int = 8,
    n_stability_runs: int = 3,
) -> dict:
    """
    LIME explanation of the 4-class classifier decision.

    Runs LIME `n_stability_runs` times and reports pairwise
    Jaccard IoU to quantify explanation stability — a metric
    required by XAI evaluation frameworks.

    Returns
    -------
    dict
        positive_vis    (H,W,3)  yellow-boundary overlay
        negative_vis    (H,W,3)  red-boundary overlay
        heatmap         (H,W)    soft positive-weight map in [0,1]
        stability_iou   float    mean pairwise Jaccard across runs
        top_label       int      explained class index
        seg_weights     dict     {superpixel_id: weight}
    """
    global _lime_fixed_mask
    _lime_fixed_mask = mask

    lime_input = np.stack([img_float, img_float, img_float], axis=-1)  # (H,W,3)

    segmenter = SegmentationAlgorithm(
        "slic",
        n_segments=80,
        compactness=10,
        sigma=1,
        start_label=0,
    )

    explainer = lime_image.LimeImageExplainer(
        feature_selection="highest_weights",
        verbose=False,
    )

    def _run() -> object:
        return explainer.explain_instance(
            lime_input,
            _clf_predict_lime,
            top_labels=NUM_CLASSES,
            hide_color=0,
            num_samples=num_samples,
            segmentation_fn=segmenter,
        )

    # ── Stability runs ────────────────────────────────────
    explanations   = [_run() for _ in range(n_stability_runs)]
    positive_masks = []
    for exp in explanations:
        _, pm = exp.get_image_and_mask(
            pred_class,
            positive_only=True,
            num_features=num_features,
            hide_rest=False,
        )
        positive_masks.append(pm.astype(float))

    jaccards = [
        iou_binary(positive_masks[i], positive_masks[j])
        for i in range(len(positive_masks))
        for j in range(i + 1, len(positive_masks))
    ]
    stability_iou = float(np.mean(jaccards)) if jaccards else 1.0

    # ── Visualisation from first run ──────────────────────
    exp0 = explanations[0]

    pos_img, pos_mask = exp0.get_image_and_mask(
        pred_class, positive_only=True,
        num_features=num_features, hide_rest=False,
    )
    neg_img, neg_mask = exp0.get_image_and_mask(
        pred_class, positive_only=False, negative_only=True,
        num_features=num_features, hide_rest=False,
    )

    positive_vis = mark_boundaries(pos_img, pos_mask, color=(1, 0.8, 0))
    negative_vis = mark_boundaries(neg_img, neg_mask, color=(1, 0,   0))

    # ── Soft heatmap from superpixel weights ──────────────
    segments    = exp0.segments
    seg_weights = dict(exp0.local_exp[pred_class])
    heatmap     = np.zeros(segments.shape, dtype=np.float32)
    for seg_id, w in seg_weights.items():
        heatmap[segments == seg_id] = max(w, 0.0)   # positive only
    heatmap = minmax(heatmap)

    return {
        "positive_vis" : positive_vis,
        "negative_vis" : negative_vis,
        "heatmap"      : heatmap,
        "stability_iou": stability_iou,
        "top_label"    : pred_class,
        "seg_weights"  : seg_weights,
    }


# =========================================================
# SHAP  — GradientExplainer on the 2-channel classifier
# =========================================================

def build_shap_background(
    img_float: np.ndarray,
    mask: np.ndarray,
    n_bg: int = 16,
) -> torch.Tensor:
    """
    Construct a realistic SHAP background distribution.

    Using a random tensor (as in the original code) is off the
    data manifold and causes attribution leakage — pixels get
    credit for pushing the model away from nonsense inputs.
    Instead we augment the actual image with Gaussian blurs
    at multiple scales, plus zero and mean references.

    Returns
    -------
    torch.Tensor  shape (n_bg, 2, H, W) on `device`
    """
    H, W    = img_float.shape
    bg_list = []

    for sigma in np.linspace(0, 8, n_bg // 2):
        bm = gaussian_filter(img_float, sigma=sigma)
        bk = gaussian_filter(mask,      sigma=max(sigma / 2, 0))
        bg_list.append(np.stack([bm, bk], axis=0))

    # Zero reference
    bg_list.append(np.zeros((2, H, W), dtype=np.float32))

    # Mean-intensity reference
    mean_mri = np.full((H, W), img_float.mean(), dtype=np.float32)
    bg_list.append(np.stack([mean_mri, mask], axis=0))

    while len(bg_list) < n_bg:
        bg_list.append(bg_list[-1])
    bg_list = bg_list[:n_bg]

    bg = np.stack(bg_list, axis=0).astype(np.float32)   # (n_bg,2,H,W)
    return torch.tensor(bg, dtype=torch.float32).to(device)


def _normalise_shap_output(raw: object, num_classes: int) -> list:
    """
    Normalise shap.GradientExplainer output to a consistent
    list of `num_classes` numpy arrays each shaped (1, 2, H, W).

    SHAP changed its return convention between versions:
      <  0.42  list[num_classes] of (1,2,H,W)    <- old
      >= 0.42  single ndarray (1,2,H,W,num_classes)
               or             (num_classes,1,2,H,W)
    """
    if isinstance(raw, list):
        return raw

    if isinstance(raw, np.ndarray):
        if raw.ndim == 5:
            if raw.shape[-1] == num_classes:
                # (1,2,H,W,C)
                return [raw[..., c] for c in range(num_classes)]
            if raw.shape[0] == num_classes:
                # (C,1,2,H,W)
                return [raw[c] for c in range(num_classes)]
        # Fallback: split along last axis
        if raw.shape[-1] == num_classes:
            return [raw[..., c] for c in range(num_classes)]

    raise ValueError(
        f"Cannot parse SHAP output of type {type(raw)}, "
        f"shape {getattr(raw, 'shape', 'N/A')}. "
        "Check your shap version."
    )


def explain_shap(
    clf_tensor: torch.Tensor,
    img_float: np.ndarray,
    mask: np.ndarray,
    pred_class: int,
    n_bg: int = 16,
) -> dict:
    """
    SHAP GradientExplainer explanation of the 2-channel classifier.

    Produces per-class and per-channel attribution maps, answering:
      - Which spatial regions drove the decision?  (per_class_maps)
      - Did the model rely on MRI texture or the tumor mask?
        (channel_importance fractions)

    Returns
    -------
    dict
        per_class_maps      (NUM_CLASSES, H, W)  normalised |SHAP|
        channel_maps        list[NUM_CLASSES] of (2,H,W) signed SHAP
        combined_map        (H, W)  all-class sum, normalised
        pred_class_map      (H, W)  |SHAP| for predicted class
        channel_importance  list of dicts {class, ch0_frac, ch1_frac, ...}
    """
    background = build_shap_background(img_float, mask, n_bg=n_bg)

    shap_explainer = shap.GradientExplainer(clf_model, background)
    raw_shap       = shap_explainer.shap_values(clf_tensor)

    sv_list = _normalise_shap_output(raw_shap, NUM_CLASSES)

    # Safety: pad with zeros if fewer classes returned than expected
    while len(sv_list) < NUM_CLASSES:
        sv_list.append(np.zeros_like(sv_list[0]))

    per_class_maps     = []
    channel_maps       = []
    channel_importance = []

    for c in range(NUM_CLASSES):
        sv = np.array(sv_list[c])      # ensure ndarray
        if sv.ndim == 4:
            sv = sv[0]                 # (1,2,H,W) -> (2,H,W)

        sv_abs       = np.abs(sv)
        combined_abs = sv_abs.sum(axis=0)

        per_class_maps.append(minmax(combined_abs))
        channel_maps.append(sv)        # signed (2,H,W)

        total = sv_abs.sum() + 1e-8
        channel_importance.append({
            "class"        : CLASS_NAMES[c],
            "ch0_frac"     : float(sv_abs[0].sum() / total),
            "ch1_frac"     : float(sv_abs[1].sum() / total),
            "ch0_mean_abs" : float(sv_abs[0].mean()),
            "ch1_mean_abs" : float(sv_abs[1].mean()),
        })

    per_class_maps = np.stack(per_class_maps, axis=0)   # (C,H,W)
    combined_map   = minmax(per_class_maps.sum(axis=0))

    return {
        "per_class_maps"    : per_class_maps,
        "channel_maps"      : channel_maps,
        "combined_map"      : combined_map,
        "pred_class_map"    : per_class_maps[pred_class],
        "channel_importance": channel_importance,
    }


# =========================================================
# FAITHFULNESS METRICS
# =========================================================

def compute_faithfulness_metrics(
    clf_tensor: torch.Tensor,
    shap_map: np.ndarray,
    lime_heatmap: np.ndarray,
    gradcam_map: np.ndarray,
    pred_class: int,
    top_k_fracs: list | None = None,
) -> dict:
    """
    Standard XAI faithfulness metrics:

    Sufficiency      — Keep only top-k% most important pixels;
                       measure retained confidence.
                       Higher = those pixels are truly predictive.

    Comprehensiveness — Remove top-k% pixels; measure confidence drop.
                        Higher drop = explainer found real signal.

    Inter-method agreement — SSIM (SHAP vs LIME) and Spearman ρ
                              between all three method pairs.

    NOTE: All masking is done in numpy (CPU) to avoid the
          "can't convert cuda tensor to numpy" error.

    Parameters
    ----------
    clf_tensor   : (1,2,H,W) on device  — original classifier input
    shap_map     : (H,W) normalised |SHAP| for pred_class
    lime_heatmap : (H,W) LIME soft positive-weight map
    gradcam_map  : (H,W) GradCAM saliency map
    pred_class   : int
    top_k_fracs  : area fractions to evaluate  [default 5%, 10%, 20%]
    """
    if top_k_fracs is None:
        top_k_fracs = [0.05, 0.10, 0.20]

    # Pull input to CPU numpy once — avoids all CUDA conversion issues
    x_np = clf_tensor.detach().cpu().numpy()   # (1,2,H,W)

    def _run_clf(arr_np: np.ndarray) -> float:
        """Inference on a (1,2,H,W) numpy array; returns pred_class prob."""
        t = torch.tensor(arr_np, dtype=torch.float32).to(device)
        with torch.no_grad():
            return float(
                torch.softmax(clf_model(t), dim=1)[0, pred_class].item()
            )

    orig_prob = _run_clf(x_np)

    def _masked_prob(imp_map: np.ndarray, frac: float, keep: bool) -> float:
        """
        keep=True  : zero everything OUTSIDE the top-frac region (sufficiency)
        keep=False : zero the top-frac region itself            (comprehensiveness)
        """
        flat   = imp_map.ravel()
        k      = max(1, int(frac * len(flat)))
        thresh = np.sort(flat)[-k]
        region = (imp_map >= thresh).astype(np.float32)   # (H,W)

        masked = x_np.copy()                              # (1,2,H,W) numpy
        if keep:
            masked[0, 0] *= region
            masked[0, 1] *= region
        else:
            masked[0, 0] *= (1.0 - region)
            masked[0, 1] *= (1.0 - region)

        return _run_clf(masked)

    metrics = {
        "orig_confidence"           : round(orig_prob, 5),
        "top_k_fracs"               : top_k_fracs,
        "sufficiency_shap"          : [],
        "comprehensiveness_shap"    : [],
        "sufficiency_lime"          : [],
        "comprehensiveness_lime"    : [],
        "sufficiency_gradcam"       : [],
        "comprehensiveness_gradcam" : [],
        "shap_lime_ssim"     : float(ssim(shap_map, lime_heatmap, data_range=1.0)),
        "shap_gradcam_spearman"  : spearman(shap_map,    gradcam_map),
        "lime_gradcam_spearman"  : spearman(lime_heatmap, gradcam_map),
        "shap_lime_spearman"     : spearman(shap_map,    lime_heatmap),
    }

    for frac in top_k_fracs:
        for key, imp in [
            ("shap",    shap_map),
            ("lime",    lime_heatmap),
            ("gradcam", gradcam_map),
        ]:
            suf  = _masked_prob(imp, frac, keep=True)
            comp = orig_prob - _masked_prob(imp, frac, keep=False)
            metrics[f"sufficiency_{key}"].append(round(suf,  4))
            metrics[f"comprehensiveness_{key}"].append(round(comp, 4))

    return metrics


# =========================================================
# MAIN PIPELINE
# =========================================================

def predict_pipeline(
    image_path: str,
    threshold: float = 0.3,
    img_size: int = 448,
    lime_samples: int = 512,
    lime_features: int = 8,
    lime_stability_runs: int = 3,
    shap_bg: int = 16,
    output_dir: str = "xai_outputs",
) -> dict:
    """
    Full explainable MRI classification pipeline.

    Saves:
      <output_dir>/<stem>_xai.png      — publication figure
      <output_dir>/<stem>_metrics.json — all quantitative results

    Returns the metrics dict.
    """
    os.makedirs(output_dir, exist_ok=True)
    stem = Path(image_path).stem

    # ── 0. Load & preprocess ─────────────────────────────
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")

    original  = cv2.resize(img, (img_size, img_size))
    img_float = original.astype(np.float32) / 255.0
    norm_img  = zscore(img_float)

    seg_tensor = (
        torch.from_numpy(norm_img)
        .unsqueeze(0).unsqueeze(0)
        .float().to(device)
    )

    # ── 1. Segmentation ──────────────────────────────────
    print("\n[1/5] Segmentation...")
    with torch.no_grad():
        seg_out  = seg_model(seg_tensor)
        seg_out  = F.interpolate(
            seg_out, (img_size, img_size),
            mode="bilinear", align_corners=False,
        )
        prob_map = torch.sigmoid(seg_out).squeeze().cpu().numpy()

    mask     = (prob_map >= threshold).astype(np.float32)
    mask_vis = (mask * 255).astype(np.uint8)

    # ── 2. GradCAM on segmentor ──────────────────────────
    print("[2/5] GradCAM (segmentation model)...")
    seg_tensor_g = seg_tensor.clone().requires_grad_(True)
    cam_seg      = gradcam_seg.generate(seg_tensor_g)
    cam_seg      = cv2.resize(cam_seg.astype(np.float32), (img_size, img_size))

    # ── 3. Classification ────────────────────────────────
    print("[3/5] Classification...")
    stacked    = np.stack([img_float, mask], axis=0)      # (2,H,W)
    clf_tensor = (
        torch.tensor(stacked, dtype=torch.float32)
        .unsqueeze(0).to(device)
    )

    with torch.no_grad():
        logits   = clf_model(clf_tensor)
        probs    = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        pred_idx = int(probs.argmax())

    # GradCAM on classifier
    clf_tensor_g = clf_tensor.clone().requires_grad_(True)
    cam_clf      = gradcam_clf.generate(clf_tensor_g, class_idx=pred_idx)
    cam_clf      = cv2.resize(cam_clf.astype(np.float32), (img_size, img_size))

    # ── 4. LIME ──────────────────────────────────────────
    print("[4/5] LIME explanation (classifier, 2-ch)...")
    lime_result = explain_lime(
        img_float,
        mask,
        pred_idx,
        num_samples=lime_samples,
        num_features=lime_features,
        n_stability_runs=lime_stability_runs,
    )

    # ── 5. SHAP ──────────────────────────────────────────
    print("[5/5] SHAP explanation (classifier, per-class + per-channel)...")
    shap_result = explain_shap(
        clf_tensor,
        img_float,
        mask,
        pred_idx,
        n_bg=shap_bg,
    )

    # ── 6. Faithfulness metrics ──────────────────────────
    print("[Metrics] Computing faithfulness metrics...")
    faith_metrics = compute_faithfulness_metrics(
        clf_tensor,
        shap_result["pred_class_map"],
        lime_result["heatmap"],
        cam_clf,
        pred_idx,
    )

    # ── Print summary ─────────────────────────────────────
    confidence = float(probs[pred_idx])
    print("\n" + "=" * 52)
    print("  PREDICTION RESULTS")
    print("=" * 52)
    print(f"  Class      : {CLASS_NAMES[pred_idx]}")
    print(f"  Confidence : {confidence:.4f}")
    print(f"\n  All-class probabilities:")
    for i, (cn, p) in enumerate(zip(CLASS_NAMES, probs)):
        tag = "  <-- predicted" if i == pred_idx else ""
        print(f"    {cn:<14}: {p:.4f}{tag}")
    print(f"\n  LIME stability (mean Jaccard IoU): {lime_result['stability_iou']:.4f}")
    print(f"  SHAP-LIME SSIM                   : {faith_metrics['shap_lime_ssim']:.4f}")
    print(f"  SHAP-GradCAM Spearman rho        : {faith_metrics['shap_gradcam_spearman']:.4f}")
    print(f"  LIME-GradCAM Spearman rho        : {faith_metrics['lime_gradcam_spearman']:.4f}")
    ci = shap_result["channel_importance"][pred_idx]
    print(f"\n  Channel importance (predicted class):")
    print(f"    MRI channel  : {ci['ch0_frac']*100:.1f}%"
          f"  (mean|SHAP|={ci['ch0_mean_abs']:.5f})")
    print(f"    Mask channel : {ci['ch1_frac']*100:.1f}%"
          f"  (mean|SHAP|={ci['ch1_mean_abs']:.5f})")

    # ── Render figure ─────────────────────────────────────
    print("\n[Figure] Rendering...")
    fig_path = _render_figure(
        stem          = stem,
        original      = original,
        prob_map      = prob_map,
        mask_vis      = mask_vis,
        cam_seg       = cam_seg,
        cam_clf       = cam_clf,
        probs         = probs,
        pred_idx      = pred_idx,
        lime_result   = lime_result,
        shap_result   = shap_result,
        faith_metrics = faith_metrics,
        output_dir    = output_dir,
    )

    # ── Export JSON ───────────────────────────────────────
    export = {
        "image"      : image_path,
        "prediction" : CLASS_NAMES[pred_idx],
        "confidence" : round(confidence, 5),
        "all_probs"  : {
            cn: round(float(p), 5)
            for cn, p in zip(CLASS_NAMES, probs)
        },
        "lime" : {
            "stability_iou"  : lime_result["stability_iou"],
            "num_samples"    : lime_samples,
            "num_features"   : lime_features,
            "stability_runs" : lime_stability_runs,
        },
        "shap" : {
            "channel_importance" : shap_result["channel_importance"],
            "background_samples" : shap_bg,
        },
        "faithfulness" : faith_metrics,
    }
    json_path = os.path.join(output_dir, f"{stem}_metrics.json")
    with open(json_path, "w") as f:
        json.dump(export, f, indent=2)
    print(f"[JSON] Metrics saved  -> {json_path}")
    print(f"[Done] Figure saved   -> {fig_path}")

    return export


# =========================================================
# PUBLICATION FIGURE
# =========================================================

def _cam_overlay(gray: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """Blend GradCAM saliency map over a grayscale MRI -> (H,W,3) RGB uint8."""
    cam_u8  = (cam * 255).astype(np.uint8)
    cam_col = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    base    = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    blend   = cv2.addWeighted(base, 0.55, cam_col, 0.45, 0)
    return cv2.cvtColor(blend, cv2.COLOR_BGR2RGB)


def _render_figure(
    stem, original, prob_map, mask_vis,
    cam_seg, cam_clf, probs, pred_idx,
    lime_result, shap_result, faith_metrics, output_dir,
) -> str:
    """
    3-row, 5-column publication figure.

    Row 1 — Segmentation stage:
        Input MRI | Tumor prob map | Binary mask |
        GradCAM (seg model) | GradCAM (clf model)

    Row 2 — LIME stage:
        LIME positive | LIME weight heatmap | LIME negative |
        Class prob bar chart | Quantitative metrics panel

    Row 3 — SHAP stage:
        SHAP |pred class| combined | SHAP MRI ch (signed) |
        SHAP Mask ch (signed) | 4-class SHAP grid

    Returns saved figure path (str).
    """
    BG   = "#0D1117"
    TC   = "white"
    HEAT = "inferno"
    DIV  = "RdBu_r"

    fig = plt.figure(figsize=(24, 14), facecolor=BG)
    fig.suptitle(
        f"Explainable MRI Analysis  |  Prediction: "
        f"{CLASS_NAMES[pred_idx].upper()}  "
        f"(conf: {probs[pred_idx]:.3f})",
        fontsize=15, color=TC, fontweight="bold", y=0.985,
    )

    gs = gridspec.GridSpec(
        3, 5, figure=fig,
        hspace=0.40, wspace=0.12,
        top=0.94, bottom=0.05, left=0.03, right=0.97,
    )

    def _show(ax, arr, title, cmap="gray",
              vmin=None, vmax=None, cbar=False):
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, color=TC, fontsize=8.5, pad=4)
        ax.axis("off")
        if cbar:
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.ax.yaxis.set_tick_params(color=TC)
            plt.setp(cb.ax.yaxis.get_ticklabels(), color=TC, fontsize=7)
        return im

    # ── Row 1 ─────────────────────────────────────────────
    _show(fig.add_subplot(gs[0, 0]), original, "Input MRI")
    _show(fig.add_subplot(gs[0, 1]), prob_map,
          "Tumor Probability Map", cmap=HEAT, vmin=0, vmax=1, cbar=True)
    _show(fig.add_subplot(gs[0, 2]), mask_vis, "Segmentation Mask")
    _show(fig.add_subplot(gs[0, 3]),
          _cam_overlay(original, cam_seg), "GradCAM (Seg model)")
    _show(fig.add_subplot(gs[0, 4]),
          _cam_overlay(original, cam_clf), "GradCAM (Clf model)")

    # ── Row 2 : LIME ──────────────────────────────────────
    _show(fig.add_subplot(gs[1, 0]),
          lime_result["positive_vis"],
          "LIME Positive\n(supports prediction)")
    _show(fig.add_subplot(gs[1, 1]),
          lime_result["heatmap"],
          "LIME Weight Heatmap", cmap=HEAT, vmin=0, vmax=1, cbar=True)
    _show(fig.add_subplot(gs[1, 2]),
          lime_result["negative_vis"],
          "LIME Negative\n(contradicts prediction)")

    # Class probability bar chart
    ax_bar = fig.add_subplot(gs[1, 3])
    colors = [
        "#E74C3C" if i == pred_idx else "#5DADE2"
        for i in range(NUM_CLASSES)
    ]
    bars = ax_bar.barh(
        CLASS_NAMES, probs,
        color=colors, edgecolor="white", linewidth=0.4,
    )
    ax_bar.set_xlim(0, 1)
    ax_bar.set_facecolor(BG)
    ax_bar.tick_params(colors=TC, labelsize=7.5)
    ax_bar.spines[:].set_color("#444")
    for bar, p in zip(bars, probs):
        ax_bar.text(
            min(p + 0.02, 0.93),
            bar.get_y() + bar.get_height() / 2,
            f"{p:.3f}", va="center", ha="left", color=TC, fontsize=7.5,
        )
    ax_bar.set_title("Class Probabilities", color=TC, fontsize=8.5, pad=4)

    # Quantitative metrics text panel
    ax_txt = fig.add_subplot(gs[1, 4])
    ax_txt.set_facecolor("#161B22")
    ax_txt.axis("off")
    k10 = 1   # index for 10% in [0.05, 0.10, 0.20]
    lines = [
        ("-- LIME --", True),
        (f"Stability IoU : {lime_result['stability_iou']:.4f}", False),
        (f"Suf  @10%    : "
         f"{faith_metrics['sufficiency_lime'][k10]:.4f}", False),
        (f"Comp @10%    : "
         f"{faith_metrics['comprehensiveness_lime'][k10]:.4f}", False),
        ("", False),
        ("-- GradCAM --", True),
        (f"Suf  @10%    : "
         f"{faith_metrics['sufficiency_gradcam'][k10]:.4f}", False),
        (f"Comp @10%    : "
         f"{faith_metrics['comprehensiveness_gradcam'][k10]:.4f}", False),
        ("", False),
        ("-- Agreement --", True),
        (f"SHAP-LIME SSIM : "
         f"{faith_metrics['shap_lime_ssim']:.4f}", False),
        (f"SHAP-GradCAM r : "
         f"{faith_metrics['shap_gradcam_spearman']:.4f}", False),
        (f"LIME-GradCAM r : "
         f"{faith_metrics['lime_gradcam_spearman']:.4f}", False),
    ]
    for i, (txt, bold) in enumerate(lines):
        ax_txt.text(
            0.04, 0.97 - i * 0.073, txt,
            transform=ax_txt.transAxes,
            color=TC, fontsize=7.8,
            fontweight="bold" if bold else "normal",
            va="top", family="monospace",
        )

    # ── Row 3 : SHAP ──────────────────────────────────────
    _show(fig.add_subplot(gs[2, 0]),
          shap_result["pred_class_map"],
          f"SHAP |{CLASS_NAMES[pred_idx]}|\n(combined channels)",
          cmap=HEAT, vmin=0, vmax=1, cbar=True)

    sv_ch = shap_result["channel_maps"][pred_idx]   # (2,H,W) signed
    vext  = float(np.abs(sv_ch).max())

    _show(fig.add_subplot(gs[2, 1]),
          sv_ch[0],
          f"SHAP: {CHANNEL_NAMES[0]}\n(signed, pred class)",
          cmap=DIV, vmin=-vext, vmax=vext, cbar=True)
    _show(fig.add_subplot(gs[2, 2]),
          sv_ch[1],
          f"SHAP: {CHANNEL_NAMES[1]}\n(signed, pred class)",
          cmap=DIV, vmin=-vext, vmax=vext, cbar=True)

    # 4-class SHAP grid in the last 2 columns
    inner = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs[2, 3:5],
        hspace=0.08, wspace=0.06,
    )
    for c in range(NUM_CLASSES):
        r_, c_ = divmod(c, 2)
        ax_    = fig.add_subplot(inner[r_, c_])
        ax_.imshow(
            shap_result["per_class_maps"][c],
            cmap=HEAT, vmin=0, vmax=1,
        )
        ax_.set_title(CLASS_NAMES[c], color=TC, fontsize=7.5, pad=2)
        ax_.axis("off")
        if c == pred_idx:
            for spine in ax_.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("#E74C3C")
                spine.set_linewidth(2.0)

    fig.text(
        0.97, 0.17, "Per-class SHAP",
        color="#AAA", fontsize=7.5,
        ha="right", va="bottom", style="italic",
    )

    fig_path = os.path.join(output_dir, f"{stem}_xai.png")
    plt.savefig(
        fig_path, dpi=160,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    return fig_path


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Research-grade explainable MRI pipeline"
    )
    parser.add_argument("--img",           type=str,   required=True,
                        help="Path to MRI image (jpg/png)")
    parser.add_argument("--threshold",     type=float, default=0.3,
                        help="Segmentation probability threshold")
    parser.add_argument("--img_size",      type=int,   default=448,
                        help="Resize to this square size for both models")
    parser.add_argument("--lime_samples",  type=int,   default=512,
                        help="LIME perturbation samples (>=512 recommended)")
    parser.add_argument("--lime_features", type=int,   default=8,
                        help="Number of LIME superpixels to highlight")
    parser.add_argument("--lime_runs",     type=int,   default=3,
                        help="Stability repeat runs for LIME Jaccard score")
    parser.add_argument("--shap_bg",       type=int,   default=16,
                        help="SHAP background sample count")
    parser.add_argument("--out",           type=str,   default="xai_outputs",
                        help="Output directory for figure and JSON")
    args = parser.parse_args()

    predict_pipeline(
        image_path          = args.img,
        threshold           = args.threshold,
        img_size            = args.img_size,
        lime_samples        = args.lime_samples,
        lime_features       = args.lime_features,
        lime_stability_runs = args.lime_runs,
        shap_bg             = args.shap_bg,
        output_dir          = args.out,
    )