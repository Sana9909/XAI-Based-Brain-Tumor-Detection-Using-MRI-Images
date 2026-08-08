"""
dataset_cleaning.py
===================
Identifies and removes hard-negative / mislabelled / unlearnable samples
from the BRISC-2025 brain-tumour segmentation dataset.

Five conservative criteria — only truly harmful images are discarded:

  C1  Tumour-class image with a completely EMPTY mask.
      (No ground truth at all → pure false negative supervision.)

  C2  Tumour-class image where the foreground mask is MICROSCOPICALLY SMALL
      after resize (< MIN_MASK_PIXELS at 448×448).
      A region this tiny is invisible to a patch-based transformer;
      keeping it adds noise, not signal.

  C3  "no_tumor" image that carries a NON-TRIVIAL mask.
      Any segmentation annotation on a no-tumour scan is a mislabel.

  C4  NEAR-BLANK / corrupted image (pixel std-dev < STD_THRESHOLD).
      No useful brain tissue is visible — the image cannot teach anything.

  C5  SPATIAL MISMATCH: the majority of mask pixels land in dark
      background regions of the MRI (likely an annotation offset error).

Everything else is KEPT.  The surviving pairs are resized to 448×448 and
saved to a clean output directory ready for training.

Usage
-----
    python dataset_cleaning.py

Outputs
-------
    ./archive/brisc2025/segmentation_cleaned/
        train/images/   ← cleaned + resized JPGs
        train/masks/    ← cleaned + resized PNGs
        test/images/
        test/masks/
    discard_report.txt  ← full log of every discarded image and why
"""

import os
from glob import glob
from collections import defaultdict
from typing import Optional

import numpy as np
from PIL import Image
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# PATHS  — edit only these if your directory layout differs
# ──────────────────────────────────────────────────────────────────────────────

TRAIN_IMG_DIR  = "./archive/brisc2025/segmentation_task/train/images"
TRAIN_MASK_DIR = "./archive/brisc2025/segmentation_task/train/masks"
TEST_IMG_DIR   = "./archive/brisc2025/segmentation_task/test/images"
TEST_MASK_DIR  = "./archive/brisc2025/segmentation_task/test/masks"

CLEAN_TRAIN_IMG  = "./archive/brisc2025/segmentation_cleaned/train/images"
CLEAN_TRAIN_MASK = "./archive/brisc2025/segmentation_cleaned/train/masks"
CLEAN_TEST_IMG   = "./archive/brisc2025/segmentation_cleaned/test/images"
CLEAN_TEST_MASK  = "./archive/brisc2025/segmentation_cleaned/test/masks"

REPORT_PATH = "./discard_report.txt"
TARGET_SIZE = (448, 448)   # (width, height) used for all thresholds


# ──────────────────────────────────────────────────────────────────────────────
# THRESHOLDS  — deliberately conservative
# ──────────────────────────────────────────────────────────────────────────────

# C2: minimum foreground pixels in the 448×448 mask for a tumour image.
#     64 px ≈ an 8×8 region — anything smaller is sub-patch and unlearnable.
MIN_MASK_PIXELS_BY_CLASS = {
    "glioma":      64,    # glioma can be large or small; keep anything ≥ 8×8
    "meningioma":  64,
    "pituitary":   48,    # pituitary tumours are naturally small — extra lenient
    "unknown":     64,
}

# C3: foreground pixels allowed on a "no_tumor" mask before flagging it.
#     Tiny white-pixel artefacts from compression are ignored; real lesions
#     are not.
NO_TUMOR_MAX_PIXELS = 80

# C4: minimum image std-dev (0-255).  Below this the scan is essentially blank.
STD_THRESHOLD = 7.0

# C5: intensity below which a pixel is treated as "dark background" in the MRI.
#     If >85 % of mask foreground pixels are dark, it is an annotation mismatch.
DARK_PIXEL_CUTOFF   = 12          # intensity value (0-255)
DARK_OVERLAP_RATIO  = 0.85        # fraction to trigger flag


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def get_class(filename: str) -> str:
    name = filename.lower()
    if "_gl_" in name:   return "glioma"
    if "_me_" in name:   return "meningioma"
    if "_pi_" in name:   return "pituitary"
    if "_no_" in name:   return "no_tumor"
    return "unknown"


def load_gray_448(path: str, nearest: bool = False) -> np.ndarray:
    """Open image, convert to grayscale, resize to TARGET_SIZE."""
    img = Image.open(path).convert("L")
    method = Image.NEAREST if nearest else Image.BILINEAR
    img = img.resize(TARGET_SIZE, method)
    return np.array(img, dtype=np.uint8)


def check_sample(img_path: str, mask_path: str) -> list[str]:
    """
    Returns a list of reason strings for discarding this sample.
    An empty list means the sample is clean.
    """
    reasons: list[str] = []
    filename = os.path.basename(img_path)
    cls = get_class(filename)

    # Load and resize to 448×448 (consistent threshold space)
    img_arr  = load_gray_448(img_path,  nearest=False)
    mask_arr = load_gray_448(mask_path, nearest=True)
    mask_bin = (mask_arr > 0).astype(np.uint8)

    fg_pixels = int(mask_bin.sum())
    min_px    = MIN_MASK_PIXELS_BY_CLASS.get(cls, 64)

    # ── C1: Tumour class with completely empty mask ───────────────────────────
    if cls != "no_tumor" and fg_pixels == 0:
        reasons.append(
            f"[C1] Tumour class '{cls}' but mask is COMPLETELY EMPTY "
            f"— no positive supervision at all."
        )

    # ── C2: Mask too small to learn from ─────────────────────────────────────
    elif cls != "no_tumor" and 0 < fg_pixels < min_px:
        pct = fg_pixels / (TARGET_SIZE[0] * TARGET_SIZE[1]) * 100
        reasons.append(
            f"[C2] Mask has only {fg_pixels} foreground pixels "
            f"({pct:.4f}% of 448×448) — below the {min_px}-px threshold "
            f"for class '{cls}'.  Sub-patch lesion; model cannot learn it."
        )

    # ── C3: no_tumor image with a real mask ───────────────────────────────────
    if cls == "no_tumor" and fg_pixels > NO_TUMOR_MAX_PIXELS:
        reasons.append(
            f"[C3] Class 'no_tumor' but mask contains {fg_pixels} foreground "
            f"pixels (threshold = {NO_TUMOR_MAX_PIXELS} px) "
            f"— almost certainly mislabelled."
        )

    # ── C4: Near-blank / corrupted image ─────────────────────────────────────
    std_val = float(img_arr.std())
    if std_val < STD_THRESHOLD:
        reasons.append(
            f"[C4] Image near-blank (pixel std = {std_val:.2f}, "
            f"threshold = {STD_THRESHOLD}) — no useful brain tissue visible."
        )

    # ── C5: Mask annotation lands in dark background pixels ──────────────────
    if fg_pixels > 0:
        mask_intensities = img_arr[mask_bin == 1]
        dark_ratio = float((mask_intensities < DARK_PIXEL_CUTOFF).mean())
        if dark_ratio > DARK_OVERLAP_RATIO:
            reasons.append(
                f"[C5] {dark_ratio*100:.1f}% of mask pixels fall in dark "
                f"background (intensity < {DARK_PIXEL_CUTOFF}) "
                f"— spatial annotation mismatch (annotation offset error)."
            )

    return reasons


def resize_and_save(src: str, dst: str, nearest: bool = False) -> None:
    """Resize image to TARGET_SIZE and save to dst."""
    img = Image.open(src)
    method = Image.NEAREST if nearest else Image.BILINEAR
    img = img.resize(TARGET_SIZE, method)
    img.save(dst)


# ──────────────────────────────────────────────────────────────────────────────
# PER-SPLIT PROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def process_split(
    img_dir:   str,
    mask_dir:  str,
    out_img:   str,
    out_mask:  str,
    split:     str,
) -> tuple[dict, list[tuple[str, list[str]]]]:
    """
    Scans all image–mask pairs, applies cleaning criteria, saves survivors.
    Returns (stats_dict, discarded_list).
    """
    os.makedirs(out_img,  exist_ok=True)
    os.makedirs(out_mask, exist_ok=True)

    img_paths = sorted(glob(os.path.join(img_dir,  "*.jpg")))

    # Map stem → mask path
    mask_map = {
        os.path.splitext(os.path.basename(p))[0]: p
        for p in sorted(glob(os.path.join(mask_dir, "*.png")))
    }

    stats     = defaultdict(int)
    discarded = []

    # Class distribution counters
    class_total   = defaultdict(int)
    class_kept    = defaultdict(int)
    class_dropped = defaultdict(int)

    sep = "─" * 70
    print(f"\n{sep}")
    print(f"  {split} split  ({len(img_paths)} images found)")
    print(sep)

    for img_path in tqdm(img_paths, desc=f"  Scanning {split}"):
        base      = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = mask_map.get(base)
        cls       = get_class(os.path.basename(img_path))

        class_total[cls] += 1

        if mask_path is None:
            tqdm.write(f"  ⚠  No mask for {base} — skipped.")
            stats["missing_mask"] += 1
            continue

        reasons = check_sample(img_path, mask_path)

        if reasons:
            discarded.append((os.path.basename(img_path), reasons))
            stats["discarded"] += 1
            class_dropped[cls] += 1
            for r in reasons:
                tqdm.write(f"  ✗  {os.path.basename(img_path)}  →  {r}")
        else:
            # Save resized pair to clean directory
            resize_and_save(
                img_path,  os.path.join(out_img,  os.path.basename(img_path)),
                nearest=False,
            )
            resize_and_save(
                mask_path, os.path.join(out_mask, os.path.basename(mask_path)),
                nearest=True,
            )
            stats["kept"] += 1
            class_kept[cls] += 1

    # Print per-class breakdown
    print(f"\n  {'Class':<14} {'Total':>7} {'Kept':>7} {'Dropped':>9}")
    print(f"  {'─'*40}")
    for c in ["glioma", "meningioma", "pituitary", "no_tumor", "unknown"]:
        if class_total[c] > 0:
            print(
                f"  {c:<14} {class_total[c]:>7} "
                f"{class_kept[c]:>7} {class_dropped[c]:>9}"
            )
    print(f"  {'─'*40}")
    print(
        f"  {'TOTAL':<14} {sum(class_total.values()):>7} "
        f"{stats['kept']:>7} {stats['discarded']:>9}"
    )

    return dict(stats), discarded


# ──────────────────────────────────────────────────────────────────────────────
# REPORT WRITER
# ──────────────────────────────────────────────────────────────────────────────

def write_report(
    all_discarded: dict[str, list[tuple[str, list[str]]]],
    all_stats:     dict[str, dict],
) -> None:
    lines = []
    lines.append("=" * 70)
    lines.append("  BRISC-2025  DATASET CLEANING REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Criteria applied:")
    lines.append(
        "  C1  Tumour class with a completely empty mask."
    )
    lines.append(
        f"  C2  Tumour mask < MIN_MASK_PIXELS "
        f"(glioma/meningioma: 64 px, pituitary: 48 px) after 448×448 resize."
    )
    lines.append(
        f"  C3  'no_tumor' image with > {NO_TUMOR_MAX_PIXELS} foreground pixels "
        f"in mask (mislabel)."
    )
    lines.append(
        f"  C4  Image std-dev < {STD_THRESHOLD}  (near-blank scan)."
    )
    lines.append(
        f"  C5  > {DARK_OVERLAP_RATIO*100:.0f}% of mask pixels in dark background "
        f"(annotation offset)."
    )
    lines.append("")

    for split in ["TRAIN", "TEST"]:
        items = all_discarded.get(split, [])
        stats = all_stats.get(split, {})
        lines.append(f"{'─'*70}")
        lines.append(f"  {split}  —  {len(items)} image(s) discarded "
                     f"(kept {stats.get('kept', 0)})")
        lines.append(f"{'─'*70}")
        if items:
            for fname, reasons in items:
                lines.append(f"\n  FILE : {fname}")
                for r in reasons:
                    lines.append(f"    → {r}")
        else:
            lines.append("  (none)")
        lines.append("")

    lines.append("=" * 70)
    lines.append("  Cleaned dataset saved to:")
    lines.append(f"  {CLEAN_TRAIN_IMG}")
    lines.append(f"  {CLEAN_TRAIN_MASK}")
    lines.append(f"  {CLEAN_TEST_IMG}")
    lines.append(f"  {CLEAN_TEST_MASK}")
    lines.append("=" * 70)

    report_text = "\n".join(lines)
    print("\n\n" + report_text)

    with open(REPORT_PATH, "w") as f:
        f.write(report_text)
    print(f"\n  📄  Report saved → {REPORT_PATH}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║           BRISC-2025  Dataset Cleaning & Resize                 ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  Output size  : {TARGET_SIZE[0]}×{TARGET_SIZE[1]}")
    print(f"  C2 threshold : glioma/meningioma ≥ 64 px, pituitary ≥ 48 px")
    print(f"  C3 threshold : no_tumor mask ≤ {NO_TUMOR_MAX_PIXELS} px")
    print(f"  C4 threshold : image std-dev ≥ {STD_THRESHOLD}")
    print(f"  C5 threshold : dark-mask overlap ≤ {DARK_OVERLAP_RATIO*100:.0f}%")

    all_discarded: dict = {}
    all_stats:     dict = {}

    for split, img_dir, mask_dir, out_img, out_mask in [
        ("TRAIN", TRAIN_IMG_DIR,  TRAIN_MASK_DIR,  CLEAN_TRAIN_IMG,  CLEAN_TRAIN_MASK),
        ("TEST",  TEST_IMG_DIR,   TEST_MASK_DIR,   CLEAN_TEST_IMG,   CLEAN_TEST_MASK),
    ]:
        stats, discarded = process_split(
            img_dir, mask_dir, out_img, out_mask, split
        )
        all_discarded[split] = discarded
        all_stats[split]     = stats

    write_report(all_discarded, all_stats)

    print("\n✅  Done.  Update your training script to point at:")
    print(f"     images : {CLEAN_TRAIN_IMG}")
    print(f"     masks  : {CLEAN_TRAIN_MASK}")


if __name__ == "__main__":
    main()
