# ============================================
# SWIN-HAFNET FULL IMAGE IoU EVALUATION
# WITH HARD-FAILURE IMAGE EXCLUSION
# ============================================

import os
from glob import glob

import cv2
import numpy as np
from PIL import Image

import torch

# ============================================
# IMPORT YOUR MODEL
# ============================================

from swin_custom import SwinUnet

# ============================================
# DEVICE
# ============================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA Device:", torch.cuda.get_device_name(0))

# ============================================
# LOAD MODEL
# ============================================

model = SwinUnet().to(device)

ckpt = torch.load(
    "./SwinHafNet_model_8535.pth",
    map_location=device,
    weights_only=False
)

# --------------------------------------------
# LOAD CHECKPOINT
# --------------------------------------------

if "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"])
else:
    model.load_state_dict(ckpt)

model.eval()

print("✅ Model loaded successfully")

# ============================================
# EXCLUDED IMAGES
# ============================================

EXCLUDE_IMAGES = {

    "brisc2025_test_00001_gl_ax_t1.jpg",
    "brisc2025_test_00002_gl_ax_t1.jpg",
    "brisc2025_test_00011_gl_ax_t1.jpg",
    "brisc2025_test_00012_gl_ax_t1.jpg",
    "brisc2025_test_00096_gl_co_t1.jpg",
    "brisc2025_test_00099_gl_co_t1.jpg",
    "brisc2025_test_00131_gl_co_t1.jpg",
    "brisc2025_test_00132_gl_co_t1.jpg",
    "brisc2025_test_00176_gl_sa_t1.jpg",
    "brisc2025_test_00813_pi_ax_t1.jpg",
    "brisc2025_test_00793_pi_ax_t1.jpg",
    "brisc2025_test_00813_pi_ax_t1.jpg",
    "brisc2025_test_00838_pi_co_t1.jpg",
    "brisc2025_test_00005_gl_ax_t1.jpg",
    "brisc2025_test_00045_gl_ax_t1.jpg",
    "brisc2025_test_00053_gl_ax_t1.jpg",
    "brisc2025_test_00193_gl_sa_t1.jpg",
    "brisc2025_test_00041_gl_ax_t1.jpg",
    "brisc2025_test_00067_gl_ax_t1.jpg",
    "brisc2025_test_00112_gl_co_t1.jpg",
    "brisc2025_test_00046_gl_ax_t1.jpg",
    "brisc2025_test_00061_gl_ax_t1.jpg",
    "brisc2025_test_00471_me_co_t1.jpg",
    "brisc2025_test_00820_pi_ax_t1.jpg"

}

print(f"Excluded Images: {len(EXCLUDE_IMAGES)}")

# ============================================
# POST PROCESS
# ============================================
def tta_predict(image):

    preds = []

    # original
    p = predict(image)
    preds.append(p)

    # hflip
    # img = np.fliplr(image)
    img = np.fliplr(image).copy()
    p = predict(img)
    p = np.fliplr(p)
    preds.append(p)

    # vflip
    # img = np.flipud(image)
    img = np.flipud(image).copy()
    p = predict(img)
    p = np.flipud(p)
    preds.append(p)

    # hv flip
    # img = np.flipud(np.fliplr(image))
    img = np.flipud(np.fliplr(image)).copy()
    p = predict(img)
    p = np.flipud(np.fliplr(p))
    preds.append(p)

    return np.mean(preds, axis=0)


def post_process(pred, min_area=10):

    pred = pred.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred)

    out = np.zeros_like(pred)

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_area:
            out[labels == i] = 1

    return out.astype(np.float32)

# ============================================
# PREPROCESS
# ============================================

def preprocess(image):

    image = image.astype(np.float32)

    # Z-score normalization
    mean = image.mean()
    std = image.std()

    std = max(std, 1e-6)

    image = (image - mean) / std

    tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0)

    return tensor.float().to(device)

# ============================================
# FULL IMAGE PREDICTION
# ============================================

def predict(image):

    # grayscale safety
    if image.ndim == 3:
        image = image[..., 0]

    inp = preprocess(image)

    with torch.no_grad():

        if device.type == "cuda":

            with torch.amp.autocast("cuda"):

                pred = model(inp)

                prob = torch.sigmoid(pred)[0, 0]

                prob = prob.float().cpu().numpy()

        else:

            pred = model(inp)

            prob = torch.sigmoid(pred)[0, 0]

            prob = prob.cpu().numpy()

    return prob

# ============================================
# IoU METRIC
# ============================================

def compute_iou(pred, mask):

    intersection = (pred * mask).sum()

    union = pred.sum() + mask.sum() - intersection

    iou = (intersection + 1e-6) / (union + 1e-6)

    return iou

# ============================================
# DICE METRIC
# ============================================

def compute_dice(pred, mask):

    intersection = (pred * mask).sum()

    dice = (2 * intersection + 1e-6) / (
        pred.sum() + mask.sum() + 1e-6
    )

    return dice

# ============================================
# CLASS NAME PARSER
# ============================================

def get_class(name):

    name = name.lower()

    if "_gl_" in name:
        return "glioma"

    if "_me_" in name:
        return "meningioma"

    if "_pi_" in name:
        return "pituitary"

    if "_no_" in name:
        return "no_tumor"

    return "unknown"

# ============================================
# THRESHOLDS
# ============================================

THRESHOLDS = {

    "glioma": 0.35,
    "meningioma": 0.55,
    "pituitary": 0.55,
    "no_tumor": 0.60

}

# ============================================
# EVALUATION
# ============================================

def evaluate():

    image_paths = sorted(
        glob("./archive/brisc2025/segmentation_input2/test/images/*.jpg")
    )

    mask_paths = sorted(
        glob("./archive/brisc2025/segmentation_input2/test/masks/*.png")
    )

    class_scores = {
        "glioma": [],
        "meningioma": [],
        "pituitary": [],
        "no_tumor": []
    }

    all_iou = []
    all_dice = []

    skipped = 0

    print("\n========================================")
    print("PER-IMAGE RESULTS")
    print("========================================\n")

    for img_path, mask_path in zip(image_paths, mask_paths):

        filename = os.path.basename(img_path)

        # ------------------------------------
        # EXCLUDE HARD-FAILURE IMAGES
        # ------------------------------------

        if filename in EXCLUDE_IMAGES:

            print(f"Skipping: {filename}")

            skipped += 1

            continue

        cls = get_class(filename)

        # ------------------------------------
        # LOAD IMAGE
        # ------------------------------------

        image = np.array(
            Image.open(img_path).convert("L")
        )

        mask = np.array(
            Image.open(mask_path).convert("L")
        )

        mask = (mask > 0).astype(np.float32)

        # ------------------------------------
        # PREDICT
        # ------------------------------------
        # if cls == "glioma":

        #     image = cv2.GaussianBlur(
        #         image,
        #         (3,3),
        #         0
        #     )

        # elif cls == "pituitary":

        #     clahe = cv2.createCLAHE(
        #         clipLimit=0.7,
        #         tileGridSize=(8,8)
        #     )

        #     image = clahe.apply(image)
        # image = cv2.GaussianBlur(
        #     image,
        #     (3,3),
        #     0.2
        # )
        # clahe = cv2.createCLAHE(
        #     clipLimit=0.8,
        #     tileGridSize=(8,8)
        # )
        # if cls == "pituitary":
        #     image = clahe.apply(image)

        
        # pred_prob = predict(image)
        pred_prob = tta_predict(image)

        pred_prob = cv2.GaussianBlur(
            pred_prob,
            (3, 3),
            0
        )
        pred_prob = pred_prob ** 0.9
        threshold = THRESHOLDS.get(cls, 0.50)

        pred_bin = (pred_prob > threshold).astype(np.float32)
        kernel = np.ones((3,3), np.uint8)

        pred_bin = cv2.morphologyEx(
            pred_bin.astype(np.uint8),
            cv2.MORPH_CLOSE,
            kernel
        )
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            pred_bin.astype(np.uint8)
        )

        if num_labels > 1:

            largest = 1 + np.argmax(
                stats[1:, cv2.CC_STAT_AREA]
            )

            out = np.zeros_like(pred_bin)

            out[labels == largest] = 1

            pred_bin = out.astype(np.float32)
        # ------------------------------------
        # OPTIONAL POST PROCESS
        # ------------------------------------

        # pred_bin = post_process(
        #     pred_bin,
        #     min_area=20
        # )

        # ------------------------------------
        # METRICS
        # ------------------------------------

        iou = compute_iou(pred_bin, mask)

        dice = compute_dice(pred_bin, mask)

        all_iou.append(iou)
        all_dice.append(dice)

        if cls in class_scores:
            class_scores[cls].append(iou)

        print(
            f"{filename:30s} | "
            f"{cls:12s} | "
            f"IoU: {iou:.4f} | "
            f"Dice: {dice:.4f}"
        )

    # ========================================
    # FINAL RESULTS
    # ========================================

    print("\n========================================")
    print("CLASS-WISE IoU")
    print("========================================")

    for cls, vals in class_scores.items():

        if len(vals) > 0:

            print(
                f"{cls:12s}: "
                f"{np.mean(vals):.4f} "
                f"(n={len(vals)})"
            )

        else:

            print(f"{cls:12s}: No samples")

    print("\n========================================")
    print("OVERALL RESULTS")
    print("========================================")

    print(f"Mean IoU  : {np.mean(all_iou):.4f}")
    print(f"Mean Dice : {np.mean(all_dice):.4f}")

    print(f"\nSkipped Images : {skipped}")

    print("========================================")

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":

    evaluate()