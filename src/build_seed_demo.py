"""
Run this ONCE, before presenting, after both models are fully trained.
Picks a FIXED (seeded) set of sample images, runs both models on each one
exactly once, and saves every result -- numbers and annotated images -- to
disk. For the panel, you just OPEN these saved files. Nothing gets
recomputed live, so there's no risk of a different number showing up on a
rerun, and no waiting on inference during the actual presentation.

Run:
    python src/build_seed_demo.py

Produces:
    outputs/seed_demo/results.json          <- all numbers in one file
    outputs/seed_demo/sample_XX_<cat>.png    <- one annotated image per sample
    outputs/seed_demo/summary.png            <- a single grid image of all samples together
"""
import os
import sys
import json

import cv2
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

config.seed_everything()

from src.data_utils import find_images_with_category, safe_imread
from src.demo_pipeline import (
    load_models, predict_category, estimate_foot_length, box_dimensions_mm,
    detect_shoe_box,
)

# How many sample images to pull from each dataset/category for the fixed set.
N_PER_CATEGORY = 2

# A representative outsole_mm per category, standing in for what your A4/SAM2
# pipeline would measure -- since these particular sample images don't have
# a real A4 sheet in frame. Say this plainly if asked: for photos that DO
# have a real A4 sheet, plug the real measured outsole_mm in instead (see
# demo_pipeline.py --outsole_mm for that case).
NOMINAL_OUTSOLE_MM = {"Boots": 290.0, "Sandals": 250.0, "Shoes": 265.0, "Slippers": 255.0}


def pick_seed_samples():
    """Deterministically picks the same N images per category every time
    (sorted + fixed seed), so the sample set itself doesn't change between
    runs either."""
    df = find_images_with_category(config.SHOE_DATASET_ROOT)
    if df.empty:
        print("!! SHOE_DATASET_ROOT scan is empty -- check config.py")
        return []
    samples = []
    for cat, group in df.groupby("category"):
        chosen = group.sort_values("path").head(N_PER_CATEGORY)
        for _, row in chosen.iterrows():
            samples.append({"path": row["path"], "true_category_folder": cat})
    return samples


def main():
    out_dir = os.path.join(config.OUTPUT_DIR, "seed_demo")
    os.makedirs(out_dir, exist_ok=True)

    cnn, classes, yolo = load_models()
    samples = pick_seed_samples()
    if not samples:
        return
    print(f"Running the fixed set of {len(samples)} seed samples ...")

    results = []
    fig, axes = plt.subplots(1, len(samples), figsize=(4 * len(samples), 4))
    if len(samples) == 1:
        axes = [axes]

    for i, (sample, ax) in enumerate(zip(samples, axes)):
        path = sample["path"]
        img_bgr = safe_imread(path)
        if img_bgr is None:
            print(f"  skipping unreadable file: {path}")
            continue

        category = predict_category(cnn, classes, img_bgr)

        # Same shared detector as demo_pipeline.py / app.py, so it gets the
        # same lower inference-time threshold and contour fallback -- these
        # seed samples come from a plain-background dataset so YOLO should
        # find them directly, but staying on one code path avoids drift.
        box, conf, box_method, box_warning = detect_shoe_box(yolo, path)
        if box_warning:
            print(f"  !! {box_warning}")

        outsole_mm = NOMINAL_OUTSOLE_MM.get(category, 260.0)
        insole = foot = eu_size = width_mm = height_mm = None
        if box is not None:
            insole, foot = estimate_foot_length(outsole_mm, category)
            eu_size = 1.5 * (foot / 10.0 + 1.5)
            _, width_mm = box_dimensions_mm(box, outsole_mm)
            height_mm = config.CATEGORY_NOMINAL_HEIGHT_MM.get(category, 80.0)

        result = {
            "image": path,
            "true_category_folder": sample["true_category_folder"],
            "predicted_category": category,
            "yolo_box_px": box,
            "yolo_confidence": conf,
            "nominal_outsole_mm": outsole_mm,
            "est_foot_length_mm": foot,
            "est_eu_size": eu_size,
            "est_width_mm": width_mm,
            "est_height_mm_category_estimate": height_mm,
        }
        results.append(result)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        if box:
            x0, y0, x1, y1 = box
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                        edgecolor="lime", linewidth=2))
        title = f"{category}"
        if foot:
            title += f"\nFoot~{foot:.0f}mm  W~{width_mm:.0f}mm"
        ax.set_title(title, fontsize=9)
        ax.axis("off")

        # individual annotated image too, for showing one at a time
        single_fig, single_ax = plt.subplots(figsize=(5, 5))
        single_ax.imshow(img_rgb)
        if box:
            x0, y0, x1, y1 = box
            single_ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                               edgecolor="lime", linewidth=3))
        single_ax.set_title(title, fontsize=11)
        single_ax.axis("off")
        single_path = os.path.join(out_dir, f"sample_{i:02d}_{category}.png")
        single_fig.tight_layout()
        single_fig.savefig(single_path, dpi=150)
        plt.close(single_fig)
        print(f"  [{i+1}/{len(samples)}] {os.path.basename(path)} -> {category}  saved {single_path}")

    fig.tight_layout()
    summary_path = os.path.join(out_dir, "summary.png")
    fig.savefig(summary_path, dpi=150)
    plt.close(fig)

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. Fixed, pre-computed results saved to {out_dir}/")
    print(f"  - summary.png       <- show this one grid image to the panel")
    print(f"  - sample_XX_*.png   <- individual images if you want to show one at a time")
    print(f"  - results.json      <- exact numbers, in case you're asked for specifics")
    print(f"\nThese files do NOT change unless you re-run this script -- safe to open "
          f"and show repeatedly without any risk of a different number appearing.")


if __name__ == "__main__":
    main()
