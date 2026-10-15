"""
Shared helpers for walking the dataset folders and generating pseudo
bounding boxes. Nothing here trains anything -- see train_cnn.py /
train_yolo.py for that.
"""
import os
import sys
import cv2
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def list_tree(root, depth=3):
    """Print the folder structure so you can SEE the real layout before
    trusting any assumption about it. Always run this first on a new dataset."""
    if not os.path.exists(root):
        print(f"!! Path does not exist: {root}  (fix config.py)")
        return
    root = os.path.normpath(root)
    for dirpath, _, filenames in os.walk(root):
        level = dirpath.replace(root, "").count(os.sep)
        if level > depth:
            continue
        indent = "  " * level
        print(f"{indent}{os.path.basename(dirpath) or dirpath}/  ({len(filenames)} files)")


_WRAPPER_NAMES = {"train", "training", "val", "validation", "valid", "test", "testing", "data", "images"}


def find_images_with_category(root):
    """Walk `root`; label = the first REAL category folder below `root`,
    skipping generic wrapper folders like training/val/test if present.
    Handles both:
      - Zappos50K's Category/Sub-category/Brand/*.jpg  (no wrapper -> uses top folder)
      - training/Category/*.jpg                         (skips 'training' -> uses next folder)
    If your layout puts the category somewhere else entirely, check
    list_tree() output and adjust the logic below.

    Caches the result to CSV (outputs/cache/) since walking a 50k-image
    folder from scratch on every run is a major, easily avoidable source of
    slowness -- especially on a OneDrive-synced path. Delete the matching
    CSV in outputs/cache/ if the dataset contents change."""
    root = os.path.normpath(root)
    cache_path = os.path.join(config.CACHE_DIR, _cache_name(root) + ".csv")

    if config.USE_SCAN_CACHE and os.path.exists(cache_path):
        print(f"[cache] loading file list from {cache_path} (delete this file to force a rescan)")
        return pd.read_csv(cache_path)

    print(f"Scanning {root} ... (first run only -- result gets cached)")
    rows = []
    if not os.path.exists(root):
        return pd.DataFrame(columns=["path", "category"])

    # Kaggle zips commonly extract with a duplicated folder, e.g.
    # ut-zap50k-images-square/ut-zap50k-images-square/Boots/... -- treat the
    # root's own folder name as an implicit wrapper too, not just train/val/test.
    root_basename = os.path.basename(root).lower()
    wrapper_names = _WRAPPER_NAMES | {root_basename}

    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() not in config.IMG_EXT:
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root)
            parts = rel.split(os.sep)
            if len(parts) < 2:
                continue
            idx = 0
            while idx < len(parts) - 1 and parts[idx].lower() in wrapper_names:
                idx += 1
            category = parts[idx]
            rows.append({"path": full, "category": category})
    df = pd.DataFrame(rows)

    if config.USE_SCAN_CACHE:
        df.to_csv(cache_path, index=False)
        print(f"[cache] saved {len(df)} entries -> {cache_path}")
    return df


def _cache_name(root):
    import hashlib
    return "scan_" + hashlib.md5(root.encode("utf-8")).hexdigest()[:10]


def list_images_flat(root):
    """Fallback for datasets with no category subfolders -- just every image."""
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in config.IMG_EXT:
                out.append(os.path.join(dirpath, fn))
    return out


def safe_imread(path):
    """cv2.imread can silently return None on Windows for paths with
    special/non-ASCII characters, or for a genuinely corrupt file. This
    routes through numpy + imdecode (handles unicode paths reliably) and
    returns None (never raises) so callers can skip bad files instead of
    crashing mid-training."""
    try:
        import numpy as np
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def auto_bbox(img_bgr, pad_frac=0.03):
    """Derive a bounding box from a clean/near-uniform background via
    Otsu threshold + largest contour. This is a labeling SHORTCUT for
    datasets with no ground-truth boxes -- it only works because product
    photos usually sit on a plain background. Not used at inference time."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, __import__("numpy").ones((7, 7), "uint8"))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    if not contours:
        return [0, 0, w, h]
    c = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(c)
    padx, pady = int(cw * pad_frac), int(ch * pad_frac)
    x0, y0 = max(0, x - padx), max(0, y - pady)
    x1, y1 = min(w, x + cw + padx), min(h, y + ch + pady)
    return [x0, y0, x1, y1]


if __name__ == "__main__":
    print("=== ZAPPOS_ROOT ===")
    list_tree(config.ZAPPOS_ROOT)
    print("\n=== SHOE_DATASET_ROOT ===")
    list_tree(config.SHOE_DATASET_ROOT)
