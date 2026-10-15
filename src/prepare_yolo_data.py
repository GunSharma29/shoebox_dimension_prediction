"""
Builds a YOLO-format detection dataset with NO manual bounding boxes needed.
Neither Kaggle dataset ships bounding boxes/dimensions, so this generates
pseudo-labels with a background-threshold heuristic (auto_bbox) -- this is
the direct answer to "dimensions are not specified in dataset": we derive a
box automatically instead of requiring annotated ones, which is exactly what
YOLO needs to train (a normalized x_center,y_center,w,h per image).

Run:
    python src/prepare_yolo_data.py

Produces:
    outputs/yolo_dataset/images/{train,val}/*.jpg
    outputs/yolo_dataset/labels/{train,val}/*.txt
    outputs/yolo_dataset/data.yaml
"""
import os
import sys
import shutil
import random

import cv2
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_utils import find_images_with_category, list_images_flat, auto_bbox, safe_imread

config.seed_everything()


def gather_all_images():
    paths = []
    z = find_images_with_category(config.ZAPPOS_ROOT)
    if not z.empty:
        paths.extend(z["path"].tolist())
    s = find_images_with_category(config.SHOE_DATASET_ROOT)
    if not s.empty:
        paths.extend(s["path"].tolist())
    else:
        paths.extend(list_images_flat(config.SHOE_DATASET_ROOT))
    return paths


def write_yolo_label(txt_path, box, img_w, img_h):
    x0, y0, x1, y1 = box
    xc = (x0 + x1) / 2 / img_w
    yc = (y0 + y1) / 2 / img_h
    w = (x1 - x0) / img_w
    h = (y1 - y0) / img_h
    with open(txt_path, "w") as f:
        f.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")   # class 0 = shoe


def main():
    all_paths = gather_all_images()
    print(f"Found {len(all_paths)} total images across both datasets.")
    if not all_paths:
        print("!! No images found -- fix config.py paths first "
              "(run `python src/data_utils.py` to inspect folders).")
        return

    random.shuffle(all_paths)
    subset = all_paths if config.DET_SUBSET_N is None else all_paths[:config.DET_SUBSET_N]
    print(f"Using {len(subset)} images for detection training "
          f"(edit config.DET_SUBSET_N, or set config.USE_FULL_DATASET=True, to change this).")

    train_paths, val_paths = train_test_split(subset, test_size=0.15, random_state=config.SEED)

    for split, paths in [("train", train_paths), ("val", val_paths)]:
        img_dir = os.path.join(config.YOLO_DATASET_DIR, "images", split)
        lbl_dir = os.path.join(config.YOLO_DATASET_DIR, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        kept = 0
        skipped = 0
        for i, p in enumerate(paths):
            img = safe_imread(p)
            if img is None:
                skipped += 1
                continue
            h, w = img.shape[:2]
            box = auto_bbox(img)
            stem = f"{split}_{i:06d}"
            dst_img = os.path.join(img_dir, stem + ".jpg")
            dst_lbl = os.path.join(lbl_dir, stem + ".txt")
            cv2.imwrite(dst_img, img)
            write_yolo_label(dst_lbl, box, w, h)
            kept += 1
            if (i + 1) % 200 == 0:
                print(f"  [{split}] {i+1}/{len(paths)}")
        print(f"{split}: wrote {kept} image/label pairs ({skipped} unreadable files skipped)")

    yaml_path = os.path.join(config.YOLO_DATASET_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"""path: {config.YOLO_DATASET_DIR}
train: images/train
val: images/val
names:
  0: shoe
""")
    print(f"\nWrote {yaml_path}")
    print("Next: python src/train_yolo.py")


if __name__ == "__main__":
    main()
