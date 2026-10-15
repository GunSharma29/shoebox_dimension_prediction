"""
Central config. Edit the two paths below to wherever you extracted the
Kaggle downloads, then everything else can be run as-is.

*** IMPORTANT if your dataset is under a OneDrive-synced folder (e.g.
C:\\Users\\...\\OneDrive\\Desktop\\dataset): OneDrive live-syncs every file
read on thousands of small images and this alone can turn a 5-minute job
into hours. Before training today, either:
  (a) right-click the "dataset" folder in File Explorer -> "Always keep on
      this device" (stops OneDrive from re-checking each file), or
  (b) better: copy the dataset folder to a plain local path like
      C:\\ShoeboxData and point the paths below at that instead.
This one change is usually a bigger speedup than anything below. ***
"""
import os

# ---- EDIT THESE TWO ----
ZAPPOS_ROOT = r"C:\Updated_Project\dataset\ut-zap50k-images-square"
SHOE_DATASET_ROOT = r"C:\Updated_project\dataset\shoeTypeClassifierDataset"
# -------------------------

# Where this project writes everything it produces (created automatically)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
YOLO_DATASET_DIR = os.path.join(OUTPUT_DIR, "yolo_dataset")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(YOLO_DATASET_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

IMG_EXT = {".jpg", ".jpeg", ".png"}

# Folder scan results are cached to CSV after the first run (see data_utils.py)
# so you don't re-walk a 50k-image folder every single time you run a script.
# Delete the files in outputs/cache/ if you ever change the dataset contents.
USE_SCAN_CACHE = True

SEED = 42

def seed_everything():
    """Call this at the start of every training script. Fixes every source
    of randomness we control so re-running training on the same data
    produces the same result every time -- no more "different accuracy on
    every rerun." (Full bit-for-bit determinism on CPU across different
    machines/thread counts still isn't 100% guaranteed by PyTorch itself,
    but this removes every source of randomness that was previously
    unseeded, which is what was actually causing the visible swings.)"""
    import random as _random
    import numpy as _np
    import torch as _torch
    _random.seed(SEED)
    _np.random.seed(SEED)
    _torch.manual_seed(SEED)
    _torch.cuda.manual_seed_all(SEED)
    _torch.backends.cudnn.deterministic = True
    _torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(SEED)


# --- Dataset size: how much of the data to actually train on ---
# mam asked for the WHOLE dataset, not a small capped subset. Two honest
# options, pick one:
#
#   USE_FULL_DATASET = True   -> trains on every image. This is the real
#       "whole dataset" run mam asked for. On CPU this is genuinely slow --
#       expect this to run for HOURS (Shoes alone has 30k+ images). Start
#       this well before you need results (e.g. run it overnight), not
#       right before presenting.
#
#   USE_FULL_DATASET = False  -> caps each category at MAX_IMAGES_PER_CLASS
#       below. This is still a LARGE, honest sample (2000/class covers
#       Slippers and Sandals completely, since they have fewer images than
#       that), just bounded so it finishes in a predictable time on CPU.
#       This is the practical choice if you're training today, not a
#       "fake demo" -- say plainly to the panel that it's a representative
#       sample of the full dataset, sized to fit the time available.
USE_FULL_DATASET = False
MAX_IMAGES_PER_CLASS = 2000     # only used if USE_FULL_DATASET is False
DET_SUBSET_N = 5000             # YOLO pseudo-label set size, same tradeoff

# CNN
IMG_SIZE = 224
CNN_BATCH = 32
CNN_EPOCHS = 12
CNN_LR = 1e-3
NUM_WORKERS = 4   # parallel image loading

# YOLO
YOLO_MODEL = "yolov8n.pt"   # nano: fastest to train
YOLO_EPOCHS = 25
YOLO_IMG_SIZE = 640
YOLO_BATCH = 16

if USE_FULL_DATASET:
    MAX_IMAGES_PER_CLASS = None   # no cap -- every image
    DET_SUBSET_N = None           # no cap -- every image

# Category-specific outsole -> foot-length offsets (mm).
# PLACEHOLDER values -- replace with measured medians if you get time;
# say so explicitly to the review panel rather than presenting them as measured.
CATEGORY_OFFSETS_MM = {
    "Boots":    {"outsole_to_insole": -14.0, "insole_to_foot": -9.0},
    "Sandals":  {"outsole_to_insole": -6.0,  "insole_to_foot": -4.0},
    "Shoes":    {"outsole_to_insole": -12.0, "insole_to_foot": -8.0},
    "Slippers": {"outsole_to_insole": -10.0, "insole_to_foot": -6.0},
}

# Nominal shoe HEIGHT (mm) per category -- NOT measured from the photo.
# A single top-down photo genuinely cannot capture height/depth without a
# second reference (a side-view photo, or a known-height object in frame).
# These are rough category typical values so the shoebox estimate has a
# height dimension at all; say plainly to the panel that this is a category
# average, not something the model measured from this specific shoe.
CATEGORY_NOMINAL_HEIGHT_MM = {
    "Boots": 120.0,
    "Sandals": 45.0,
    "Shoes": 90.0,
    "Slippers": 55.0,
}


NOMINAL_OUTSOLE_MM = {"Boots": 290.0, "Sandals": 250.0, "Shoes": 265.0, "Slippers": 255.0}

# Extra clearance added around the shoe's own dimensions to size the box itself.
SHOEBOX_CLEARANCE_MM = {"length": 20.0, "width": 15.0, "height": 15.0}

# Wall thickness (mm) used by src/generate_stl_box.py when exporting the
# recommended shoebox as a 3D-printable STL. 2.4mm = 6 perimeters at a
# standard 0.4mm nozzle -- sturdy enough for a small open-top tray without
# needing infill. Override per-run with `--wall` on generate_stl_box.py.
STL_WALL_THICKNESS_MM = 2.4

# YOLO confidence threshold used at INFERENCE time (demo_pipeline.py), not
# training. Lower than Ultralytics' 0.25 default because the detector is
# trained only on plain-background product photos (see prepare_yolo_data.py
# / auto_bbox) -- real-world photos with backgrounds/clutter it never saw
# in training tend to score lower even when the box is basically right.
YOLO_INFERENCE_CONF = 0.15

