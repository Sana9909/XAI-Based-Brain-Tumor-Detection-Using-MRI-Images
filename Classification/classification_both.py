import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import cv2
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import seaborn as sns
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)

# =========================================================
# DATASET (MRI + MASK)
# =========================================================

class ImageMaskDataset(Dataset):

    def __init__(self, df, transforms=None, img_size=448):

        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def _read_image(self, path):

        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")

        img = cv2.resize(
            img,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_LINEAR
        )

        img = img.astype(np.float32) / 255.0

        return img

    def _read_mask(self, path):

        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise FileNotFoundError(f"Mask not found: {path}")

        mask = cv2.resize(
            mask,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_NEAREST
        )

        mask = mask.astype(np.float32)

        if mask.max() > 1:
            mask = mask / 255.0

        return mask

    def __getitem__(self, idx):

        row = self.df.loc[idx]

        img = self._read_image(row['image_path'])

        mask = self._read_mask(row['mask_path'])

        # =====================================================
        # 2 CHANNEL STACK
        # =====================================================

        stacked = np.stack([img, mask], axis=2)

        if self.transforms:

            transformed = self.transforms(image=stacked)

            stacked = transformed['image']

        else:

            stacked = torch.tensor(
                stacked.transpose(2,0,1),
                dtype=torch.float32
            )

        label = int(row['label'])

        return stacked, label


# =========================================================
# TRANSFORMS
# =========================================================

def get_test_transforms(img_size=448):

    return A.Compose([

        A.Resize(img_size, img_size),

        ToTensorV2()

    ])


# =========================================================
# MODEL
# =========================================================

def make_resnet18_2ch(num_classes):

    weights = ResNet18_Weights.DEFAULT

    model = resnet18(weights=weights)

    old_conv = model.conv1

    # =====================================================
    # CHANGE INPUT CHANNELS: 3 -> 2
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

        new_conv.weight[:, 0:1, :, :] = mean_weight
        new_conv.weight[:, 1:2, :, :] = mean_weight

    model.conv1 = new_conv

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model


# =========================================================
# INFERENCE
# =========================================================

@torch.no_grad()
def evaluate(model, loader, device):

    model.eval()

    preds_all = []
    labels_all = []

    for imgs, labels in tqdm(loader):

        imgs = imgs.to(device).float()

        outputs = model(imgs)

        preds = outputs.argmax(dim=1)

        preds_all.extend(preds.cpu().numpy())
        labels_all.extend(labels.numpy())

    return preds_all, labels_all


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data-csv',
        required=True
    )

    parser.add_argument(
        '--model-path',
        required=True
    )

    parser.add_argument(
        '--img-size',
        type=int,
        default=448
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32
    )

    args = parser.parse_args()

    # =====================================================
    # LOAD CSV
    # =====================================================

    df = pd.read_csv(args.data_csv)

    # =====================================================
    # TEST SPLIT
    # =====================================================

    test_df = df[df['split'] == 'test'].reset_index(drop=True)

    print(f"Test samples: {len(test_df)}")

    classes = sorted(df['class_name'].unique())

    print("Classes:", classes)

    # =====================================================
    # DATASET
    # =====================================================

    test_ds = ImageMaskDataset(
        test_df,
        transforms=get_test_transforms(args.img_size),
        img_size=args.img_size
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # =====================================================
    # DEVICE
    # =====================================================

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Device:", device)

    # =====================================================
    # MODEL
    # =====================================================

    num_classes = int(df['label'].max()) + 1

    model = make_resnet18_2ch(num_classes)

    model.load_state_dict(
        torch.load(args.model_path, map_location=device)
    )

    model = model.to(device)

    print("Loaded model:", args.model_path)

    # =====================================================
    # EVALUATE
    # =====================================================

    preds, labels = evaluate(
        model,
        test_loader,
        device
    )

    # =====================================================
    # METRICS
    # =====================================================

    acc = accuracy_score(labels, preds)

    print(f"\nTest Accuracy: {acc:.4f}")

    print("\n===== CLASSIFICATION REPORT =====\n")

    print(
        classification_report(
            labels,
            preds,
            target_names=classes,
            digits=4
        )
    )

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(8,6))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.title("Test Confusion Matrix")

    plt.tight_layout()

    plt.savefig("test_confusion_matrix_2ch.png")

    print("\nSaved: test_confusion_matrix_2ch.png")


if __name__ == "__main__":
    main()