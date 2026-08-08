import csv
from pathlib import Path

# ---------- CHANGE THIS TO YOUR TEST FOLDER ----------
ROOT = Path("./archive/brisc2025/classification_test")
# ------------------------------------------------------

OUT_CSV = Path("./archive/brisc2025/test_labels.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# detect classes (folders)
classes = sorted([
    d.name for d in ROOT.iterdir()
    if d.is_dir() and not d.name.startswith(".")
])

print("Detected classes:", classes)

# map classes to integer labels
class_to_label = {cls: idx for idx, cls in enumerate(classes)}
print("Label map:", class_to_label)

def find_by_stem(folder, stem):
    """Find file with the same stem (with any extension)."""
    if not folder.exists():
        return ""
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif"]:
        f = folder / f"{stem}{ext}"
        if f.exists():
            return str(f.resolve())
    return ""

rows = []

# build rows for CSV
for cls in classes:
    img_dir = ROOT / cls / "images"
    mask_dir = ROOT / cls / "masks"
    gcam_dir = ROOT / cls / "gradcams"

    if not img_dir.exists():
        print(f"[WARN] No images folder for {cls}, skipping.")
        continue

    for f in sorted(img_dir.iterdir()):
        if f.suffix.lower() not in [".png", ".jpg", ".jpeg", ".bmp", ".tif"]:
            continue

        stem = f.stem

        img_path = str(f.resolve())
        mask_path = find_by_stem(mask_dir, stem)
        gcam_path = find_by_stem(gcam_dir, stem)

        rows.append([
            img_path,
            mask_path,
            gcam_path,
            class_to_label[cls],
            cls,
            "test"
        ])

print("Total test samples:", len(rows))

# write CSV
with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["image_path", "mask_path", "gradcam_path", "label", "class_name", "split"])
    w.writerows(rows)

print("Saved CSV to:", OUT_CSV)