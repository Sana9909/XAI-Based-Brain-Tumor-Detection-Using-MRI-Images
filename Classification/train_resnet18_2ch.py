import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.optim as optim
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# =========================================================
# SPEEDUP
# =========================================================

torch.backends.cudnn.benchmark = True

# =========================================================
# DATASET
# =========================================================

class ImageMaskDataset(Dataset):

    def __init__(self, df, transforms=None, img_size=256):
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

            return transformed['image'], int(row['label'])

        stacked = stacked.transpose(2, 0, 1)

        return torch.tensor(stacked, dtype=torch.float32), int(row['label'])


# =========================================================
# TRANSFORMS
# =========================================================

def get_train_transforms(img_size=256):

    return A.Compose([

        A.HorizontalFlip(p=0.5),

        A.VerticalFlip(p=0.2),

        A.RandomRotate90(p=0.3),

        A.Affine(
            translate_percent=0.05,
            scale=(0.90, 1.10),
            rotate=(-15, 15),
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5
        ),

        A.RandomBrightnessContrast(p=0.4),

        A.GaussNoise(p=0.12),

        A.Resize(img_size, img_size),

        ToTensorV2()

    ])


def get_valid_transforms(img_size=256):

    return A.Compose([

        A.Resize(img_size, img_size),

        ToTensorV2()

    ])


# =========================================================
# RESNET18 2 CHANNEL
# =========================================================

def make_resnet18_2ch(num_classes, pretrained=True):

    if pretrained:
        weights = ResNet18_Weights.DEFAULT
    else:
        weights = None

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

        mean_weight = old_weights.mean(dim=1, keepdim=True)

        new_conv.weight[:, 0:1, :, :] = mean_weight
        new_conv.weight[:, 1:2, :, :] = mean_weight

    model.conv1 = new_conv

    # =====================================================
    # FINAL FC
    # =====================================================

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model


# =========================================================
# UTILS
# =========================================================

def accuracy_from_logits(logits, labels):

    preds = logits.argmax(dim=1)

    correct = (preds == labels).sum().item()

    total = labels.size(0)

    return correct, total


def make_weighted_sampler(df):

    counts = df['label'].value_counts().to_dict()

    class_weights = {
        k: 1.0 / v for k, v in counts.items()
    }

    sample_weights = df['label'].map(class_weights).values

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler


# =========================================================
# TRAIN
# =========================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    loop = tqdm(loader, desc="Train", leave=False)

    for imgs, labels in loop:

        imgs = imgs.to(device).float()
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(imgs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

        c, t = accuracy_from_logits(outputs, labels)

        running_correct += c
        running_total += t

    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total

    return epoch_loss, epoch_acc


# =========================================================
# VALIDATE
# =========================================================

@torch.no_grad()
def validate(model, loader, criterion, device):

    model.eval()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    preds_all = []
    labels_all = []

    for imgs, labels in tqdm(loader, desc="Val", leave=False):

        imgs = imgs.to(device).float()
        labels = labels.to(device)

        outputs = model(imgs)

        loss = criterion(outputs, labels)

        preds = outputs.argmax(dim=1)

        running_loss += loss.item() * imgs.size(0)

        running_correct += (preds == labels).sum().item()

        running_total += imgs.size(0)

        preds_all.extend(preds.cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total

    return epoch_loss, epoch_acc, preds_all, labels_all


# =========================================================
# FINAL EVALUATION
# =========================================================

def final_evaluation(
    preds,
    labels,
    classes,
    checkpoint_dir
):

    print("\n===== CLASSIFICATION REPORT =====\n")

    print(
        classification_report(
            labels,
            preds,
            target_names=classes,
            digits=4
        )
    )

    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(8, 6))

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
    plt.title("Confusion Matrix")

    plt.tight_layout()

    save_path = Path(checkpoint_dir) / "confusion_matrix.png"

    plt.savefig(str(save_path))

    plt.close()

    print("Saved confusion matrix:", save_path)


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--data-csv', required=True)

    parser.add_argument('--batch-size', type=int, default=16)

    parser.add_argument('--epochs', type=int, default=20)

    parser.add_argument('--img-size', type=int, default=256)

    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--num-workers', type=int, default=2)

    parser.add_argument('--checkpoint-dir', default='checkpoints')

    parser.add_argument('--weighted-sampler', action='store_true')

    args = parser.parse_args()

    # =====================================================
    # LOAD CSV
    # =====================================================

    df = pd.read_csv(args.data_csv)

    train_df = df[df['split'] == 'train'].reset_index(drop=True)

    val_df = df[df['split'] == 'val'].reset_index(drop=True)

    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")

    classes = sorted(train_df['class_name'].unique())

    print("Classes:", classes)

    # =====================================================
    # DATASETS
    # =====================================================

    train_ds = ImageMaskDataset(
        train_df,
        transforms=get_train_transforms(args.img_size),
        img_size=args.img_size
    )

    val_ds = ImageMaskDataset(
        val_df,
        transforms=get_valid_transforms(args.img_size),
        img_size=args.img_size
    )

    # =====================================================
    # DATALOADERS
    # =====================================================

    if args.weighted_sampler:

        sampler = make_weighted_sampler(train_df)

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )

    else:

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
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

    model = make_resnet18_2ch(
        num_classes=num_classes,
        pretrained=True
    ).to(device)

    # =====================================================
    # LOSS
    # =====================================================

    criterion = nn.CrossEntropyLoss()

    # =====================================================
    # OPTIMIZER
    # =====================================================

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    # =====================================================
    # CHECKPOINT DIR
    # =====================================================

    ckpt_dir = Path(args.checkpoint_dir)

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0

    # =====================================================
    # TRAIN LOOP
    # =====================================================

    for epoch in range(1, args.epochs + 1):

        t0 = time.time()

        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_acc, preds, labels = validate(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step()

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f}"
        )

        print(
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        print(f"Time: {(time.time()-t0):.1f}s")

        # =================================================
        # SAVE BEST
        # =================================================

        if val_acc > best_acc:

            best_acc = val_acc

            torch.save(
                model.state_dict(),
                str(ckpt_dir / "best_resnet18_2ch.pth")
            )

            print("Saved best model.")

    # =====================================================
    # FINAL EVALUATION
    # =====================================================

    final_evaluation(
        preds,
        labels,
        classes,
        ckpt_dir
    )

    print("\nBest Validation Accuracy:", best_acc)


if __name__ == "__main__":
    main()