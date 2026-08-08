# shap_explainer.py
# =============================================================================
# SHAP EXPLAINABILITY MODULE (FIXED FOR 2-CHANNEL CLASSIFIER)
# =============================================================================

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import cv2
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

try:
    import shap
except ImportError as exc:
    raise ImportError(
        "Install SHAP first: pip install shap"
    ) from exc


# =============================================================================
# CONSTANTS
# =============================================================================

# 2-channel classifier
CHANNEL_NAMES = [
    "MRI (grayscale)",
    "Segmentation mask",
]

CHANNEL_CMAPS = [
    "gray",
    "hot",
]


# =============================================================================
# BACKGROUND DATASET BUILDER
# =============================================================================

def build_background_tensor(
    image_paths: List[str],
    seg_fn: Callable[[np.ndarray], np.ndarray],
    gcam_fn: Callable[[np.ndarray], np.ndarray],
    n: int = 32,
    img_size: int = 448,
    device: str = "cpu",
    seed: int = 42,
) -> torch.Tensor:

    rng = np.random.default_rng(seed)

    paths = list(image_paths)

    if len(paths) > n:
        chosen = rng.choice(
            paths,
            size=n,
            replace=False,
        ).tolist()
    else:
        chosen = paths

    samples = []

    for p in chosen:

        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)

        if img is None:
            warnings.warn(f"Could not read {p}")
            continue

        img = cv2.resize(
            img,
            (img_size, img_size),
        ).astype(np.float32) / 255.0

        mask = seg_fn(img)

        mask = cv2.resize(
            mask.astype(np.float32),
            (img_size, img_size),
        )

        # 2-channel input: MRI + segmentation mask
        stacked = np.stack(
            [img, mask],
            axis=0,
        )

        samples.append(stacked)

    if not samples:
        raise RuntimeError(
            "No valid SHAP background samples found."
        )

    bg = np.stack(samples, axis=0).astype(np.float32)

    return torch.tensor(
        bg,
        dtype=torch.float32,
    ).to(device)


# =============================================================================
# SHAP EXPLAINER
# =============================================================================

class SHAPExplainer:

    def __init__(
        self,
        model: nn.Module,
        background: torch.Tensor,
        class_names: Sequence[str],
        device: str = "cpu",
    ) -> None:

        self.model = model.eval().to(device)

        self.background = background.to(device)

        self.class_names = list(class_names)

        self.device = device

        self.num_classes = len(class_names)

        self._explainer = shap.GradientExplainer(
            model=self.model,
            data=self.background,
        )

        print(
            f"[SHAPExplainer] Ready — "
            f"{self.num_classes} classes, "
            f"background size = {self.background.shape[0]}"
        )

    # =========================================================================
    # COMPUTE SHAP VALUES
    # =========================================================================

    def explain(
        self,
        input_tensor: torch.Tensor,
        ranked_outputs: Optional[int] = None,
    ) -> np.ndarray:
        """
        Returns shap_values with shape (n_classes, C, H, W).

        shap.GradientExplainer can return values in several layouts depending
        on the SHAP version and whether ranked_outputs is set.  All known
        layouts are normalised here:

          Layout A — list of per-class arrays, each (batch, C, H, W)
                     → np.stack → (n_cls, batch, C, H, W)
                     → strip batch → (n_cls, C, H, W)          [5-D path]

          Layout B — single array (n_cls, batch, C, H, W)
                     → strip batch → (n_cls, C, H, W)          [5-D path]

          Layout C — single array (n_cls, C, H, W)             [4-D path, correct]

          Layout D — single array (batch, H, W, n_cls)         [channels-last,
                     returned by some SHAP versions for PyTorch] no batch dim to
                     strip; transpose → (n_cls, 1, H, W) then squeeze channel
                     → handled via axis-detection logic below

        The output is always (n_classes, C, H, W) where C == input channels.
        """

        x = input_tensor.to(self.device).float()
        n_input_ch  = x.shape[1]   # e.g. 2
        n_cls       = self.num_classes

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self._explainer.shap_values(
                X=x,
                ranked_outputs=ranked_outputs,
                output_rank_order="max",
            )

        # Unwrap tuple returned when ranked_outputs is set
        if isinstance(result, tuple):
            shap_values = result[0]
        else:
            shap_values = result

        # ── Layout A: list of per-class arrays ────────────────────────────────
        if isinstance(shap_values, list):
            # Each element: (batch, C, H, W)
            shap_values = np.stack(shap_values, axis=0)   # (n_cls, batch, C, H, W)

        shap_values = np.array(shap_values, dtype=np.float32)

        print(f"[SHAPExplainer] Raw shap_values shape from SHAP: {shap_values.shape}")

        # ── Normalise to (n_classes, C, H, W) ─────────────────────────────────
        # Known layouts from shap.GradientExplainer across versions:
        #
        #  ndim=5, last axis == n_cls : (batch, C, H, W, n_cls)  ← your version
        #  ndim=5, first axis == n_cls: (n_cls, batch, C, H, W)
        #  ndim=4, last axis == n_cls : (batch, H, W, n_cls)      [no C dim]
        #  ndim=4, first axis == n_cls: (n_cls, C, H, W)
        #
        sv = shap_values

        if sv.ndim == 5:
            if sv.shape[-1] == n_cls:
                # Layout: (batch, C, H, W, n_cls)
                sv = sv[0]                      # (C, H, W, n_cls)
                sv = sv.transpose(3, 0, 1, 2)   # (n_cls, C, H, W)
            elif sv.shape[0] == n_cls:
                # Layout: (n_cls, batch, C, H, W)
                sv = sv[:, 0]                   # (n_cls, C, H, W)
            elif sv.shape[1] == n_cls:
                # Layout: (batch, n_cls, C, H, W)
                sv = sv[0]                      # (n_cls, C, H, W)
            else:
                raise ValueError(
                    f"[SHAPExplainer] Cannot interpret 5-D shap_values shape "
                    f"{shap_values.shape} for n_cls={n_cls}, n_input_ch={n_input_ch}."
                )

        elif sv.ndim == 4:
            if sv.shape[-1] == n_cls:
                # Layout: (batch, H, W, n_cls) — no explicit C dim
                sv = sv[0]                          # (H, W, n_cls)
                sv = sv.transpose(2, 0, 1)          # (n_cls, H, W)
                sv = sv[:, np.newaxis, :, :]        # (n_cls, 1, H, W)
                sv = np.repeat(sv, n_input_ch, axis=1)  # (n_cls, C, H, W)
            elif sv.shape[0] == n_cls:
                # Layout: (n_cls, C, H, W) — already correct
                pass
            else:
                raise ValueError(
                    f"[SHAPExplainer] Cannot interpret 4-D shap_values shape "
                    f"{shap_values.shape} for n_cls={n_cls}, n_input_ch={n_input_ch}."
                )

        else:
            raise ValueError(
                f"[SHAPExplainer] Unexpected shap_values ndim={sv.ndim}, "
                f"shape={shap_values.shape}. Expected 4-D or 5-D."
            )

        # ── Final shape check ─────────────────────────────────────────────────
        assert sv.ndim == 4, f"Expected 4-D after normalisation, got {sv.shape}"

        if sv.shape[0] != n_cls:
            raise ValueError(
                f"[SHAPExplainer] Class axis mismatch: got {sv.shape[0]}, "
                f"expected {n_cls}. Shape: {sv.shape}"
            )
        if sv.shape[1] != n_input_ch:
            raise ValueError(
                f"[SHAPExplainer] Channel axis mismatch: got {sv.shape[1]}, "
                f"expected {n_input_ch}. Shape: {sv.shape}"
            )

        print(f"[SHAPExplainer] Normalised shap_values shape: {sv.shape}")
        return sv   # (n_classes, C, H, W)

    # =========================================================================
    # VISUALIZE
    # =========================================================================

    def visualize(
        self,
        input_tensor: torch.Tensor,
        save_dir: str,
        sample_tag: str = "sample",
    ) -> dict:

        save_dir = Path(save_dir)

        save_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        x = input_tensor.to(self.device).float()

        with torch.no_grad():

            logits = self.model(x)

            probs = torch.softmax(
                logits,
                dim=1,
            )[0].cpu().numpy()

        pred_idx = int(probs.argmax())

        pred_class = self.class_names[pred_idx]

        confidence = float(probs[pred_idx])

        print(
            f"\n[SHAPExplainer] Explaining: "
            f"{pred_class} "
            f"(conf={confidence:.4f})"
        )

        # sv shape: (n_classes, C, H, W)
        sv = self.explain(
            x,
            ranked_outputs=None,
        )

        inp_np = x[0].cpu().numpy()   # (C, H, W)

        self._plot_channel_overlay(
            inp_np=inp_np,
            sv_pred=sv[pred_idx],     # (C, H, W)
            pred_class=pred_class,
            confidence=confidence,
            save_path=save_dir / f"{sample_tag}_shap_channel_overlay.png",
        )

        self._plot_multiclass_grid(
            inp_np=inp_np,
            sv=sv,                    # (n_classes, C, H, W)
            probs=probs,
            save_path=save_dir / f"{sample_tag}_shap_multiclass_grid.png",
        )

        self._plot_channel_importance(
            sv_pred=sv[pred_idx],     # (C, H, W)
            pred_class=pred_class,
            save_path=save_dir / f"{sample_tag}_shap_channel_importance.png",
        )

        self._plot_summary(
            inp_np=inp_np,
            sv=sv,                    # (n_classes, C, H, W)
            save_path=save_dir / f"{sample_tag}_shap_summary.png",
        )

        print(
            f"[SHAPExplainer] Saved explanation plots to: {save_dir}"
        )

        return {
            "predicted_class": pred_class,
            "confidence": confidence,
            "save_dir": str(save_dir),
        }

    # =========================================================================
    # PLOT CHANNEL OVERLAY
    # =========================================================================

    def _plot_channel_overlay(
        self,
        inp_np,      # (C, H, W)
        sv_pred,     # (C, H, W)
        pred_class,
        confidence,
        save_path,
    ):
        # Guard: sv_pred must be (C, H, W)
        if sv_pred.ndim != 3:
            raise ValueError(
                f"_plot_channel_overlay: expected sv_pred (C, H, W), "
                f"got {sv_pred.shape}"
            )

        n_ch = min(len(CHANNEL_NAMES), inp_np.shape[0])

        fig, axes = plt.subplots(
            3,
            n_ch,
            figsize=(5 * n_ch, 12),
            facecolor="#0e0e0e",
        )

        # Ensure axes is always 2-D
        if n_ch == 1:
            axes = np.expand_dims(axes, axis=1)

        fig.suptitle(
            f"SHAP Attribution ▸ {pred_class.upper()} "
            f"({confidence:.2%})",
            color="white",
            fontsize=15,
            fontweight="bold",
        )

        for ch_idx in range(n_ch):

            channel  = inp_np[ch_idx]       # (H, W)
            shap_ch  = sv_pred[ch_idx]      # (H, W)
            abs_shap = np.abs(shap_ch)

            # Row 0 — raw input channel
            ax0 = axes[0][ch_idx]
            ax0.imshow(channel, cmap=CHANNEL_CMAPS[ch_idx], vmin=0, vmax=1)
            ax0.set_title(CHANNEL_NAMES[ch_idx], color="white")
            _style_ax(ax0)

            # Row 1 — absolute SHAP magnitude
            ax1 = axes[1][ch_idx]
            im1 = ax1.imshow(abs_shap, cmap="hot")
            plt.colorbar(im1, ax=ax1)
            _style_ax(ax1)

            # Row 2 — signed SHAP overlaid on input
            ax2 = axes[2][ch_idx]
            ax2.imshow(channel, cmap="gray", alpha=0.6)
            vmax_s = max(float(np.abs(shap_ch).max()), 1e-8)
            im2 = ax2.imshow(
                shap_ch,
                cmap="RdBu_r",
                alpha=0.65,
                vmin=-vmax_s,
                vmax=vmax_s,
            )
            plt.colorbar(im2, ax=ax2)
            _style_ax(ax2)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0e0e0e")
        plt.close(fig)
        print(f"[✓] {save_path.name}")

    # =========================================================================
    # MULTICLASS GRID
    # =========================================================================

    def _plot_multiclass_grid(
        self,
        inp_np,    # (C, H, W)
        sv,        # (n_classes, C, H, W)
        probs,
        save_path,
    ):
        n_cls_available = sv.shape[0]
        n_ch = min(len(CHANNEL_NAMES), inp_np.shape[0])
        class_names = self.class_names[:n_cls_available]

        fig, axes = plt.subplots(
            n_cls_available,
            n_ch,
            figsize=(4.5 * n_ch, 4 * n_cls_available),
            facecolor="#0e0e0e",
        )

        # Ensure axes is always 2-D
        if n_cls_available == 1 and n_ch == 1:
            axes = np.array([[axes]])
        elif n_cls_available == 1:
            axes = np.expand_dims(axes, axis=0)
        elif n_ch == 1:
            axes = np.expand_dims(axes, axis=1)

        for cls_i, cname in enumerate(class_names):

            for ch_i in range(n_ch):

                ax      = axes[cls_i][ch_i]
                sv_map  = sv[cls_i][ch_i]      # (H, W)
                channel = inp_np[ch_i]          # (H, W)

                ax.imshow(channel, cmap="gray", alpha=0.55, vmin=0, vmax=1)

                vmax_s = max(float(np.abs(sv_map).max()), 1e-8)
                ax.imshow(
                    sv_map,
                    cmap="RdBu_r",
                    alpha=0.6,
                    vmin=-vmax_s,
                    vmax=vmax_s,
                )

                ax.set_title(CHANNEL_NAMES[ch_i], color="white")
                ax.set_ylabel(
                    f"{cname}\n(p={probs[cls_i]:.3f})",
                    color="white",
                )
                _style_ax(ax)

        plt.tight_layout()
        plt.savefig(save_path, dpi=130, bbox_inches="tight", facecolor="#0e0e0e")
        plt.close(fig)
        print(f"[✓] {save_path.name}")

    # =========================================================================
    # CHANNEL IMPORTANCE
    # =========================================================================

    def _plot_channel_importance(
        self,
        sv_pred,     # (C, H, W)
        pred_class,
        save_path,
    ):
        # Guard: must be (C, H, W)
        if sv_pred.ndim != 3:
            raise ValueError(
                f"_plot_channel_importance: expected sv_pred (C, H, W), "
                f"got {sv_pred.shape}"
            )

        n_ch = sv_pred.shape[0]   # number of input channels, e.g. 2

        mean_abs = [
            float(np.abs(sv_pred[ch]).mean())
            for ch in range(n_ch)
        ]

        total = sum(mean_abs) or 1.0

        pct = [v / total * 100 for v in mean_abs]

        colors = ["#4fc3f7", "#ef5350"][:n_ch]

        labels = CHANNEL_NAMES[:n_ch]

        fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0e0e0e")

        bars = ax.barh(
            labels[::-1],   # y-axis labels  (length n_ch)
            pct[::-1],      # bar widths     (length n_ch)
            color=colors[::-1],
        )

        for bar, val in zip(bars, pct[::-1]):
            ax.text(
                bar.get_width() + 0.4,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%",
                color="white",
                va="center",
            )

        ax.set_title(
            f"Channel Importance ▸ {pred_class.upper()}",
            color="white",
        )
        ax.set_facecolor("#0e0e0e")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#555")
        ax.spines["left"].set_color("#555")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0e0e0e")
        plt.close(fig)
        print(f"[✓] {save_path.name}")

    # =========================================================================
    # SUMMARY PLOT
    # =========================================================================

    def _plot_summary(
        self,
        inp_np,    # (C, H, W)
        sv,        # (n_classes, C, H, W)
        save_path,
    ):
        n_cls = len(self.class_names)
        n_ch  = sv.shape[1]   # number of input channels

        matrix = np.zeros((n_cls, n_ch), dtype=np.float32)

        for c in range(n_cls):
            for ch in range(n_ch):
                matrix[c, ch] = float(np.abs(sv[c, ch]).mean())

        row_max = matrix.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1.0
        matrix_norm = matrix / row_max

        fig, axes = plt.subplots(
            1, 2,
            figsize=(13, 5),
            facecolor="#0e0e0e",
        )

        # — Heatmap —
        ax = axes[0]
        ax.set_facecolor("#0e0e0e")
        im = ax.imshow(matrix_norm, cmap="plasma", aspect="auto")
        ax.set_xticks(range(n_ch))
        ax.set_xticklabels(CHANNEL_NAMES[:n_ch], color="white")
        ax.set_yticks(range(n_cls))
        ax.set_yticklabels(self.class_names, color="white")
        ax.set_title("Norm. Mean |SHAP| per class & channel", color="white")
        plt.colorbar(im, ax=ax)

        # — Stacked horizontal bar —
        ax2 = axes[1]
        ax2.set_facecolor("#0e0e0e")
        colors   = ["#4fc3f7", "#ef5350"][:n_ch]
        bottoms  = np.zeros(n_cls)

        for ch_i, (ch_label, color) in enumerate(
            zip(CHANNEL_NAMES[:n_ch], colors)
        ):
            vals = matrix_norm[:, ch_i]
            ax2.barh(
                self.class_names[::-1],
                vals[::-1],
                left=bottoms[::-1],
                color=color,
                label=ch_label,
            )
            bottoms += vals

        ax2.set_title("Channel contribution per class", color="white")
        ax2.tick_params(colors="white")
        ax2.spines["bottom"].set_color("#555")
        ax2.spines["left"].set_color("#555")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        legend = ax2.legend(facecolor="#1e1e1e", labelcolor="white")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0e0e0e")
        plt.close(fig)
        print(f"[✓] {save_path.name}")


# =============================================================================
# UTILITIES
# =============================================================================

def _style_ax(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#555")


def glob_mri_paths(
    data_dir: str,
    exts=(".png", ".jpg", ".jpeg"),
) -> List[str]:

    root  = Path(data_dir)
    paths = []

    for ext in exts:
        paths.extend(root.rglob(f"*{ext}"))

    return [str(p) for p in sorted(paths)]
