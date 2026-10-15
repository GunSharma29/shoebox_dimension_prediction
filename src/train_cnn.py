"""
Model 1: CNN category classifier (ResNet18, transfer learning).

Run:
    python src/train_cnn.py

Trains on ZAPPOS_ROOT only. Produces:
    outputs/models/cnn_category_classifier.pt
    outputs/cnn_training_curves.png
    outputs/cnn_confusion_matrix.png
    outputs/cnn_classes.json
"""
import os
import sys
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_utils import find_images_with_category, safe_imread

config.seed_everything()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ShoeCategoryDataset(Dataset):
    def __init__(self, dframe, tf):
        self.df = dframe.reset_index(drop=True)
        self.tf = tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = safe_imread(row["path"])
        if img is None:
            # skip corrupt/unreadable file -- try the next index instead of crashing
            return self.__getitem__((idx + 1) % len(self.df))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.tf(img), row["label"]


def _worker_init_fn(worker_id):
    # Without this, each DataLoader worker process gets its own unseeded
    # random state -- augmentation (RandomResizedCrop, ColorJitter) would
    # differ between runs even with everything else seeded. This was the
    # actual cause of "different accuracy every rerun." Must be a
    # module-level function (not defined inside main()) so Windows'
    # spawn-based multiprocessing can pickle it.
    seed = config.SEED + worker_id
    np.random.seed(seed)
    random.seed(seed)


def build_cnn(num_classes):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)


def run_epoch(model, loader, criterion, optimizer, train):
    model.train() if train else model.eval()
    total_loss, correct, n = 0.0, 0, 0
    torch.set_grad_enabled(train)
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        if train:
            optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        n += imgs.size(0)
    torch.set_grad_enabled(True)
    return total_loss / n, correct / n


def main():
    print(f"Device: {DEVICE}")
    print("Scanning ZAPPOS_ROOT ...")
    df = find_images_with_category(config.ZAPPOS_ROOT)
    if df.empty:
        print("!! No images found. Check config.ZAPPOS_ROOT and run "
              "`python src/data_utils.py` to inspect the real folder layout.")
        return
    print(df["category"].value_counts())

    classes = sorted(df["category"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    df["label"] = df["category"].map(class_to_idx)

    if config.MAX_IMAGES_PER_CLASS:
        df = (df.groupby("category", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), config.MAX_IMAGES_PER_CLASS), random_state=config.SEED)))
        print(f"Capped to {config.MAX_IMAGES_PER_CLASS} images/class -> {len(df)} total "
              f"(set config.USE_FULL_DATASET = True to train on everything instead)")
    else:
        print(f"USE_FULL_DATASET is True -- training on all {len(df)} images. This will take a while on CPU.")

    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=config.SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=config.SEED)
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        norm,
    ])
    eval_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        norm,
    ])

    loader_generator = torch.Generator()
    loader_generator.manual_seed(config.SEED)

    train_loader = DataLoader(ShoeCategoryDataset(train_df, train_tf), batch_size=config.CNN_BATCH,
                               shuffle=True, num_workers=config.NUM_WORKERS,
                               persistent_workers=config.NUM_WORKERS > 0,
                               worker_init_fn=_worker_init_fn, generator=loader_generator)
    val_loader = DataLoader(ShoeCategoryDataset(val_df, eval_tf), batch_size=config.CNN_BATCH,
                             shuffle=False, num_workers=config.NUM_WORKERS,
                             persistent_workers=config.NUM_WORKERS > 0,
                             worker_init_fn=_worker_init_fn)
    test_loader = DataLoader(ShoeCategoryDataset(test_df, eval_tf), batch_size=config.CNN_BATCH,
                              shuffle=False, num_workers=config.NUM_WORKERS,
                              persistent_workers=config.NUM_WORKERS > 0,
                              worker_init_fn=_worker_init_fn)

    model = build_cnn(len(classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=config.CNN_LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    ckpt_path = os.path.join(config.MODELS_DIR, "cnn_category_classifier.pt")
    epoch_times = []
    for epoch in range(config.CNN_EPOCHS):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, train=True)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer, train=False)
        scheduler.step()
        history["train_loss"].append(tr_loss); history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc); history["val_acc"].append(va_acc)

        # Save after EVERY epoch -- if you stop the run (Ctrl+C) or it crashes,
        # this file is already a usable, trained model, not a wasted run.
        torch.save(model.state_dict(), ckpt_path)

        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        avg = sum(epoch_times) / len(epoch_times)
        remaining = avg * (config.CNN_EPOCHS - epoch - 1)
        print(f"epoch {epoch+1:02d}/{config.CNN_EPOCHS}  "
              f"train_loss={tr_loss:.3f} train_acc={tr_acc:.3f}  "
              f"val_loss={va_loss:.3f} val_acc={va_acc:.3f}  "
              f"[{elapsed:.0f}s/epoch, ~{remaining/60:.1f} min left]  "
              f"(checkpoint saved)")

    # curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="train"); axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("CNN loss"); axes[0].legend()
    axes[1].plot(history["train_acc"], label="train"); axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("CNN accuracy"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "cnn_training_curves.png"), dpi=150)
    plt.close()

    # test set report
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            out = model(imgs.to(DEVICE))
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    report = classification_report(all_labels, all_preds, target_names=classes)
    print(report)
    with open(os.path.join(config.OUTPUT_DIR, "cnn_test_report.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
    plt.yticks(range(len(classes)), classes)
    plt.xlabel("predicted"); plt.ylabel("actual"); plt.title("CNN confusion matrix")
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "cnn_confusion_matrix.png"), dpi=150)
    plt.close()

    torch.save(model.state_dict(), os.path.join(config.MODELS_DIR, "cnn_category_classifier.pt"))
    with open(os.path.join(config.OUTPUT_DIR, "cnn_classes.json"), "w") as f:
        json.dump(classes, f)

    print(f"\nSaved model  -> {config.MODELS_DIR}/cnn_category_classifier.pt")
    print(f"Saved plots  -> {config.OUTPUT_DIR}/cnn_training_curves.png, cnn_confusion_matrix.png")
    print(f"Saved report -> {config.OUTPUT_DIR}/cnn_test_report.txt")


if __name__ == "__main__":
    main()