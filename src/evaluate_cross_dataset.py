"""
Runs BOTH trained models against `shoe-dataset` (never used for training --
only Zappos50K was used for the CNN, and a mixed pseudo-labeled subset was
used for YOLO but this scores fresh held-out images either way).

Run:
    python src/evaluate_cross_dataset.py

Produces:
    outputs/cross_dataset_samples.png   (grid of images with YOLO box + CNN label)
    outputs/cross_dataset_report.txt    (accuracy if shoe-dataset's folder names map onto CLASSES)
"""
import os
import sys
import json

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models import resnet18
import torch.nn as nn
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_utils import find_images_with_category, list_images_flat, safe_imread

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# EDIT if shoe-dataset's folder names differ from Zappos50K's category names.
# Confirmed from your folder listing: shoeTypeClassifierDataset uses lowercase
# names like "boots" under a "training" wrapper (already handled automatically
# in data_utils.py). Map them onto your CNN's actual class names below --
# check outputs/cnn_classes.json for the exact spelling/capitalization Zappos
# produced, then fill in every category you want scored, e.g.:
# CATEGORY_MAP = {"boots": "Boots", "sandals": "Sandals", "shoes": "Shoes", "slippers": "Slippers"}
CATEGORY_MAP = {}   # left empty on purpose -- fill in after checking outputs/cnn_classes.json


def load_cnn():
    classes = json.load(open(os.path.join(config.OUTPUT_DIR, "cnn_classes.json")))
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load(
        os.path.join(config.MODELS_DIR, "cnn_category_classifier.pt"), map_location=DEVICE))
    model.to(DEVICE).eval()
    return model, classes


def load_yolo():
    from ultralytics import YOLO
    weights = os.path.join(config.MODELS_DIR, "yolo_run", "weights", "best.pt")
    if not os.path.exists(weights):
        print("!! YOLO weights not found -- run `python src/train_yolo.py` first.")
        sys.exit(1)
    return YOLO(weights)


def predict_category(model, classes, img_bgr):
    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = tf(img_rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(x).argmax(1).item()
    return classes[pred]


def main():
    cnn_model, classes = load_cnn()
    yolo_model = load_yolo()

    df = find_images_with_category(config.SHOE_DATASET_ROOT)
    if df.empty:
        flat = list_images_flat(config.SHOE_DATASET_ROOT)
        print(f"No category subfolders found under SHOE_DATASET_ROOT -- "
              f"treating as {len(flat)} unlabeled images (qualitative check only).")
        df = None
        sample_paths = flat[:8]
    else:
        print(df["category"].value_counts())
        sample_paths = df["path"].tolist()[:8]

    # --- Qualitative grid: YOLO box + CNN label on sample images ---
    n = len(sample_paths)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, path in zip(axes, sample_paths):
        img_bgr = safe_imread(path)
        if img_bgr is None:
            ax.set_title("(unreadable file)", fontsize=10)
            ax.axis("off")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        result = yolo_model.predict(path, verbose=False)[0]
        category = predict_category(cnn_model, classes, img_bgr)
        ax.imshow(img_rgb)
        if len(result.boxes):
            best = result.boxes.conf.argmax()
            x0, y0, x1, y1 = result.boxes.xyxy[best].cpu().numpy()
            conf = result.boxes.conf[best].item()
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                        edgecolor="lime", linewidth=2))
            ax.set_title(f"{category}\nYOLO conf={conf:.2f}", fontsize=10)
        else:
            ax.set_title(f"{category}\n(no YOLO detection)", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    out_path = os.path.join(config.OUTPUT_DIR, "cross_dataset_samples.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved qualitative grid -> {out_path}")

    # --- Quantitative CNN accuracy, only if category names can be mapped ---
    report_lines = []
    if df is not None and (set(df["category"].unique()) & set(CATEGORY_MAP)):
        eval_rows = df[df["category"].isin(CATEGORY_MAP)].copy()
        eval_rows["mapped_label"] = eval_rows["category"].map(CATEGORY_MAP)
        eval_rows["img"] = eval_rows["path"].map(safe_imread)
        eval_rows = eval_rows[eval_rows["img"].notna()]
        preds = [predict_category(cnn_model, classes, img) for img in eval_rows["img"]]
        eval_rows["cnn_pred"] = preds
        acc = (eval_rows["cnn_pred"] == eval_rows["mapped_label"]).mean()
        report = classification_report(eval_rows["mapped_label"], eval_rows["cnn_pred"])
        report_lines.append(f"Cross-dataset CNN accuracy: {acc:.3f}  (n={len(eval_rows)})")
        report_lines.append(report)
        print("\n".join(report_lines))
    else:
        report_lines.append(
            "CATEGORY_MAP is empty or shoe-dataset's folder names don't match Zappos "
            "categories -- fill in CATEGORY_MAP at the top of this script once you've "
            "looked at the real folder names (run src/data_utils.py to see them). "
            "Until then this is a qualitative check only (see cross_dataset_samples.png)."
        )
        print(report_lines[-1])

    with open(os.path.join(config.OUTPUT_DIR, "cross_dataset_report.txt"), "w") as f:
        f.write("\n".join(report_lines))


if __name__ == "__main__":
    main()
