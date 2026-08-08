

import argparse
from pathlib import Path
import csv
import random
from PIL import Image
import numpy as np

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')


def is_image(f: Path):
    return f.is_file() and f.suffix.lower() in IMG_EXTS


def collect_by_stem(folder: Path):
    if folder is None or not folder.exists():
        return {}

    m = {}
    for f in folder.rglob("*"):
        if is_image(f):
            m[f.stem] = f
    return m


def ensure_zero_png(path: Path, size=(224, 224)):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        arr = np.zeros((size[1], size[0]), dtype=np.uint8)
        Image.fromarray(arr).save(path)


# 🔥 NEW: identify base classes only
def get_base_classes(root: Path):
    classes = []

    for p in root.iterdir():
        if not p.is_dir():
            continue

        name = p.name

        # skip masks/gradcam folders
        if name.endswith("_masks") or name.endswith("_gradcam"):
            continue

        classes.append(name)

    return sorted(classes)


def build_rows(root: Path):
    rows = []
    classes = get_base_classes(root)

    print("\n===== DATASET SCAN (FIXED LOGIC) =====")

    for cname in classes:
        class_dir = root / cname
        mask_dir = root / f"{cname}_masks"
        gcam_dir = root / f"{cname}_gradcam"

        print(f"\n🔍 Class: {cname}")
        print(f"   Image dir: {class_dir}")
        print(f"   Mask dir: {mask_dir}")
        print(f"   GradCAM dir: {gcam_dir}")

        img_map = collect_by_stem(class_dir)
        mask_map = collect_by_stem(mask_dir)
        gcam_map = collect_by_stem(gcam_dir)

        print(f"   Images found: {len(img_map)}")
        print(f"   Masks found: {len(mask_map)}")
        print(f"   GradCAM found: {len(gcam_map)}")

        if not img_map:
            print(f"⚠️ No images for class {cname}, skipping")
            continue

        for stem in sorted(img_map.keys()):
            imgp = img_map.get(stem)
            maskp = mask_map.get(stem)
            gcamp = gcam_map.get(stem)

            rows.append({
                'image_path': str(imgp.resolve()),
                'mask_path': str(maskp.resolve()) if maskp else '',
                'gradcam_path': str(gcamp.resolve()) if gcamp else '',
                'class_name': cname
            })

    return rows


def write_csv_with_placeholders(rows, out_csv: str, root: Path,
                                val_fraction=0.12, seed=42, img_size=224):

    placeholders = root / 'placeholders_for_missing'
    placeholders.mkdir(exist_ok=True)

    for r in rows:
        cname = r['class_name']

        if r['mask_path'] == '':
            pm = placeholders / f"{cname}_missing_mask.png"
            ensure_zero_png(pm, size=(img_size, img_size))
            r['mask_path'] = str(pm.resolve())

        if r['gradcam_path'] == '':
            pg = placeholders / f"{cname}_missing_gradcam.png"
            ensure_zero_png(pg, size=(img_size, img_size))
            r['gradcam_path'] = str(pg.resolve())

    # stratified split
    by_class = {}
    for r in rows:
        by_class.setdefault(r['class_name'], []).append(r)

    train_rows, val_rows = [], []
    random.seed(seed)

    for cname, items in by_class.items():
        random.shuffle(items)
        n_val = int(len(items) * val_fraction)

        val_rows.extend(items[:n_val])
        train_rows.extend(items[n_val:])

    classes = sorted(by_class.keys())
    class_to_label = {c: i for i, c in enumerate(classes)}

    final_rows = []

    for r in train_rows:
        final_rows.append({
            **r,
            'label': class_to_label[r['class_name']],
            'split': 'train'
        })

    for r in val_rows:
        final_rows.append({
            **r,
            'label': class_to_label[r['class_name']],
            'split': 'val'
        })

    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'image_path',
                'mask_path',
                'gradcam_path',
                'label',
                'class_name',
                'split'
            ]
        )
        writer.writeheader()
        for r in final_rows:
            writer.writerow(r)

    print("\n===== DONE (FIXED) =====")
    print(f"✅ Wrote {len(final_rows)} rows to {out_csv}")
    print("📌 Label map:", class_to_label)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--val-fraction', type=float, default=0.12)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--img-size', type=int, default=224)

    args = parser.parse_args()

    root = Path(args.root)

    rows = build_rows(root)

    if not rows:
        raise RuntimeError("❌ No valid data found")

    write_csv_with_placeholders(
        rows,
        args.out,
        root,
        args.val_fraction,
        args.seed,
        args.img_size
    )


if __name__ == "__main__":
    main()