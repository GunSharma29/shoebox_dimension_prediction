# Shoebox Dimension Prediction — CNN + YOLO (VS Code project)

## Read this first — what changed based on panel feedback

1. **Reproducibility fixed.** Every training script now calls
   `config.seed_everything()` and the DataLoader's shuffling/worker
   processes are explicitly seeded too (they weren't before — that was the
   actual cause of getting a different accuracy on every rerun). Training
   on the same data with the same config now gives the same result each
   time.
2. **Full dataset, not a tiny capped subset.** `config.py` now has
   `USE_FULL_DATASET` — set `True` to train on every image (slow on CPU,
   run it ahead of time, not right before presenting), or leave `False` to
   use a large-but-bounded sample (`MAX_IMAGES_PER_CLASS = 2000` by
   default) that still finishes in a predictable time. Either way, no more
   40-images-per-class demo cap.
3. **No live retraining in front of the panel.** `src/build_seed_demo.py`
   is new: run it once, ahead of time, and it picks a fixed set of sample
   images, runs both trained models on them exactly once, and saves the
   results (numbers + annotated images) to `outputs/seed_demo/`. For the
   panel, you open those saved files — nothing gets recomputed live, so
   there's no risk of a different number appearing on a rerun.
4. **Real dimensions, not just pixel box coordinates.** `demo_pipeline.py`
   now converts the YOLO box into actual length/width in mm (using the box
   aspect ratio against your outsole measurement), plus a category-typical
   height estimate and a full recommended shoebox size with clearance. See
   the honesty note on height/depth below — read this before a panel
   member asks about it.

### ⚡ Still worried about training speed?

If your dataset is inside a OneDrive-synced folder, fix that first — it
alone can turn a 5-minute job into hours. Right-click the `dataset` folder
→ **"Always keep on this device"**, or move it to a plain local path like
`C:\ShoeboxData`. The scan is also cached after the first run
(`outputs/cache/`), and `train_cnn.py` checkpoints every epoch so an
interrupted run isn't a lost one.

---

Two trained models layered on top of your existing A4-reference geometric
measurement pipeline:

1. **CNN (ResNet18)** — classifies shoe **category** (Boots/Sandals/Shoes/Slippers
   from Zappos50K), used to pick a category-specific outsole→foot offset
   instead of one fixed number for every shoe.
2. **YOLOv8** — detects the shoe as a bounding box automatically, replacing
   the manual segmentation-prompt step. Because **neither Kaggle dataset
   ships bounding boxes or dimensions**, the boxes used to train YOLO are
   generated automatically with a background-threshold heuristic
   (`auto_bbox` in `src/data_utils.py`) — this is what your professor meant
   about the dataset not specifying dimensions, and it's exactly why YOLO
   (which just needs *a* box per image, not measured ones) is a good fit
   here: **be upfront about this with the panel** — the boxes are
   pseudo-labels, not measured ground truth, and this is normal practice
   when no annotation exists yet.

## 0. Setup (5 min)

```bash
cd shoebox_vscode
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

Open the folder in VS Code (`code .`), install the Python extension if you
haven't, and select the `venv` interpreter (bottom-right corner or
Ctrl+Shift+P → "Python: Select Interpreter").

## 1. Edit `config.py`

Set the two paths to wherever you extracted the Kaggle zips, and decide on
dataset size:
```python
ZAPPOS_ROOT = r"C:\path\to\zappos50k"
SHOE_DATASET_ROOT = r"C:\path\to\shoe_dataset"

USE_FULL_DATASET = False        # True = every image (slow, run ahead of time)
MAX_IMAGES_PER_CLASS = 2000     # only used if USE_FULL_DATASET is False
```

## 2. Look at the real folder structure first

```bash
python src/data_utils.py
```
This just prints the folder tree for both datasets. Compare it against the
assumption in `find_images_with_category()` (category = first real folder
under the root, auto-skipping generic wrapper folders like `train`/`test`)
— if it doesn't match, that function is the only place to edit.

## 3. Train the CNN (Model 1)

```bash
python src/train_cnn.py
```
Produces `outputs/cnn_training_curves.png`, `outputs/cnn_confusion_matrix.png`,
`outputs/cnn_test_report.txt`, and the saved model. Checkpoints every
epoch, so Ctrl+C at any point still leaves a usable model. Re-running this
with the same config now gives the same accuracy every time.

## 4. Build pseudo-labels and train YOLO (Model 2)

```bash
python src/prepare_yolo_data.py
python src/train_yolo.py
```
`prepare_yolo_data.py` auto-generates the bounding box labels (Section
above explains why). `train_yolo.py` trains YOLOv8-nano — ultralytics
prints mAP50 / mAP50-95 at the end and auto-saves a confusion matrix and
PR curve under `outputs/models/yolo_run/`. **Have that folder open** —
those plots are ultralytics defaults and look good on a slide.

## 5. Cross-dataset check

```bash
python src/evaluate_cross_dataset.py
```
Runs both trained models on `shoe-dataset`, which neither model was
deliberately trained on the same way as Zappos50K. Produces
`outputs/cross_dataset_samples.png` — a visual grid you can show directly.
If `shoe-dataset`'s folder names correspond to your CNN's categories, fill
in `CATEGORY_MAP` near the top of the script to also get a real accuracy
number; otherwise this stays a qualitative check, which is still a fair
thing to present as long as you say so.

## 6. Build the fixed panel-ready demo (run ONCE, ahead of time)

```bash
python src/build_seed_demo.py
```
Picks a deterministic, fixed set of sample images and runs both models on
each one exactly once, saving everything to `outputs/seed_demo/`:
- `summary.png` — one grid image of all samples, good for a slide
- `sample_XX_<category>.png` — individual annotated images
- `results.json` — every number, in case you're asked for specifics

**For the panel, just open these files. Do not re-run this script live** —
that's the whole point of it existing.

## 7. (Optional) Live single-image demo

```bash
python src/demo_pipeline.py --image path\to\some_shoe.jpg --outsole_mm 255.6
```
Prints CNN category, YOLO box + pixel size, and — if you pass
`--outsole_mm` from your existing A4 pipeline — the estimated foot length,
shoe width, a category-typical height estimate, and a full recommended
shoebox size with clearance. Only use this live if you've already tested
the exact photo beforehand; otherwise prefer the pre-built seed demo above.

## On "dimensions" — what's measured vs. estimated

- **Length** comes from your A4/SAM2 pipeline's real geometric measurement
  (`--outsole_mm`) — this is the one true measured value in the whole chain.
- **Width** is derived from the YOLO box's aspect ratio scaled against that
  same measurement — a reasonable estimate, not an independently calibrated
  measurement.
- **Height** is a **category-typical average** (`config.CATEGORY_NOMINAL_HEIGHT_MM`),
  not measured from the photo at all. A single top-down photo genuinely
  cannot capture height/depth without a second reference — a side-view
  photo, or a known-height object in frame. **Say this plainly if asked** —
  it's a legitimate limitation to name, not something to gloss over.

## What to tell the panel, honestly

- The **geometric core (A4 sheet + homography) still does the actual
  scaling** — neither network is asked to guess millimetres from pixels
  directly, which is why the system doesn't inherit the ~5-15% error of
  pure photo-to-size regression.
- The **CNN's job** is category classification, which feeds a lookup table.
- The **YOLO's job** is localizing the shoe, replacing a manual click.
- The **bounding boxes YOLO trained on are self-generated pseudo-labels**
  (background threshold), not measured annotations — this is a normal and
  defensible approach when a dataset lacks them, but it should be named as
  such, not presented as ground truth.
- The **category-specific offset table (`config.py`) is currently
  placeholder values**, not measured medians. If asked "how do you know
  these offsets are right," the honest answer is: they aren't calibrated
  yet — the next step is building a small labeled set (photo + verified
  foot length) to replace them with real numbers.
- **Height/depth is a category average, not measured** — see above.

## File map

```
config.py                       <- edit dataset paths + hyperparams here
app.py                           <- Flask web UI backend (new)
frontend/index.html              <- neo-brutalist web UI (new)
src/data_utils.py                <- folder inspection + pseudo-bbox generation
src/train_cnn.py                 <- Model 1: category classifier
src/prepare_yolo_data.py         <- builds YOLO-format pseudo-labeled dataset
src/train_yolo.py                <- Model 2: shoe detector
src/evaluate_cross_dataset.py    <- runs both models on the held-out dataset
src/demo_pipeline.py             <- single-image live demo (CLI) + run_analysis() shared by app.py
src/build_seed_demo.py           <- run ONCE ahead of time -- fixed, pre-computed panel demo
src/generate_stl_box.py          <- 3D-printable STL export for the recommended box
outputs/                         <- everything the scripts produce lands here
outputs/seed_demo/                <- what to actually show the panel
```

---

## Recent fixes (read this if you hit either bug below)

### 1. `AttributeError: module 'config' has no attribute 'STL_WALL_THICKNESS_MM'`

`config.py` was missing this setting entirely, so `--stl` crashed every time.
Fixed: added `STL_WALL_THICKNESS_MM = 2.4` (mm) to `config.py`. Override per
run with `python src/generate_stl_box.py --wall <mm> ...` if you want a
thicker/thinner wall.

### 2. No 2D bounding box on your own uploaded photos

**Root cause:** `src/train_yolo.py` trains only on pseudo-labels generated
by `auto_bbox()` (`src/data_utils.py`) from the Zappos/shoe **catalog**
datasets -- plain background, single shoe, centered. It never saw a real
photo with a floor, shadows, clutter, or a hand in frame. On those photos
it often scores every candidate box below Ultralytics' default 0.25
confidence, so `result.boxes` comes back empty and the old code printed
"no shoe detected" and gave up.

**Fix**, in `src/demo_pipeline.py`'s `detect_shoe_box()`:
1. Inference now uses a lower confidence threshold
   (`config.YOLO_INFERENCE_CONF = 0.15`) so weaker-but-plausible real YOLO
   detections aren't thrown away.
2. If YOLO still finds *nothing*, the pipeline falls back to the same
   contour/Otsu heuristic used to generate the training labels in the
   first place, so you always get **some** box on a real photo instead of
   a hard failure. This fallback is clearly labeled `box_method:
   "fallback_contour"` (vs `"yolo"`) everywhere -- CLI output, the JSON
   API, and the web UI (drawn in **yellow** instead of green) -- and comes
   with an explicit warning. It is a rough heuristic, not a trained
   detector, so treat it as a starting point, not a measurement.

**The real long-term fix** (didn't do this here, since it needs your
dataset/GPU time, not just code): add real-world, cluttered-background
photos to YOLO's training set, or apply background-randomization
augmentation to the existing catalog photos before training, so the
detector actually learns to ignore backgrounds instead of implicitly
depending on them being plain.

Also fixed in passing: `src/build_seed_demo.py` imported
`estimate_foot_length` / `box_dimensions_mm` from `demo_pipeline.py`, which
didn't exist there -- this would have crashed the first time you ran it.
Added as back-compat helpers, and switched its detection call to the same
shared `detect_shoe_box()` so it benefits from the fallback too.

### 3. New: a web UI (neo-brutalism)

`app.py` + `frontend/index.html` add a browser-based front end on top of
the exact same pipeline the CLI uses (`demo_pipeline.run_analysis()` --
one code path, so the CLI and the web UI can never disagree on the same
photo).

Run it:
```bash
pip install -r requirements.txt
python app.py
```
Then open **http://127.0.0.1:5000** and drag a photo in. It shows the
category, the 2D box (green = real YOLO detection, yellow = fallback),
any scale reference found in the photo (blue = A4, orange = card, red =
coin), the measured/assumed dimensions with their honesty tags, the
recommended shoebox size, and a **Download STL** button. Everything runs
locally -- no photo or data leaves your machine.

This requires the models to already be trained (`outputs/models/...`
must exist, same as the CLI) -- if they aren't, `/api/analyze` returns a
clear error telling you to run `train_cnn.py` / `train_yolo.py` first,
instead of crashing the server.

### 4. Scale reference dropdown — and a real bug this exposed

The web UI now has a **"What's in the photo?"** dropdown: Auto-detect, A4
sheet, ID/credit card, a coin (with a preset list -- Indian ₹1/₹2/₹5/₹10,
US coins, Euro coins -- or type your own diameter), or "nothing, I'll type
the shoe's length myself." The CLI has the same choice via
`--reference {auto,a4,card,coin,manual}`.

Building this exposed a real bug in `src/scale_reference.py`: it only ever
had `detect_a4()`, which matched *any* rectangle with roughly A4's aspect
ratio (1.25-1.60) above a size threshold -- but a standard ID/credit
card's ratio (85.60/53.98 = **1.586**) falls inside that exact same
window. A card in frame would get silently detected and scaled as if it
were a full A4 sheet (~3.5x larger), giving a wrong measurement with no
warning at all, even though the old warning text already said "less
reliable than A4/card" as if card detection existed. It didn't.

Fixed: `detect_a4()` and a new, separate `detect_card()` now use different
area/ratio tolerances tuned to each object's real size, so a card can
never be mistaken for A4 or vice versa (verified with a test that draws
both shapes and confirms each detector only fires on its own shape).

### Why the manual shoe-length field exists at all

If a photo has no A4 sheet, card, or coin in it, the pipeline has no way
to know how many real millimetres one pixel represents -- it can only
guess a category-typical length (e.g. "shoes are usually about 260mm"),
labeled **ASSUMED** in the output, which is *not* a measurement of your
actual shoe. Typing in the outsole length once (measured by hand with a
ruler) gives the pipeline a real number to scale everything else (width,
height) from, without needing a physical reference object in the photo at
all. It's optional -- skip it and you still get a result, just an assumed
one.

### 5. 3D box preview

`/api/analyze` now also renders a 3D preview image of the recommended
shoebox (reusing `generate_stl_box.py`'s box geometry, via a new headless
`render_box_preview_png()` that returns PNG bytes instead of popping up a
window) and returns it as base64 -- the web UI shows it directly under the
dimensions, no download needed. The STL download button still works
exactly as before for actually printing it.

### 6. Live webcam capture (fastest option — no file transfer at all)

The web UI now has a **"📷 Use camera"** toggle next to the upload box. It
uses the browser's own camera API (`getUserMedia`) to show a live feed
from any camera your PC recognizes (built-in laptop webcam or a plugged-in
USB webcam) and a **"Capture & Analyze"** button that grabs the current
frame and runs it through the exact same `/api/analyze` pipeline as a
file upload -- no SD card, no folder-watching, no file at all. This is the
fastest option for a live demo: point the camera at the shoe, click once,
done. Works out of the box at `http://127.0.0.1:5000` (getUserMedia is
allowed on `localhost` without HTTPS); if you ever want to drive it from
a *different* device's browser over WiFi instead of the demo PC's own
camera, that would need HTTPS, which isn't set up here.

### 7. Live camera feed over Bluetooth (for camera hardware with no live/webcam mode)

Use section 6 instead if your camera can act as a live webcam -- it's
faster and needs no extra script. This section is for a standalone
digicam/camera with no wireless of its own, where the "transfer" step is
really just copying the file off an SD card or over USB. `bluetooth_watcher.py`
doesn't care how the photo lands in its watched folder -- point it at
wherever your photos actually land (a plain folder you copy an SD card's
photos into is fine, not just a genuine Bluetooth-receive folder) and it
still auto-submits each new one the moment it appears:

```
[Phone/camera] --Bluetooth--> [This PC] --bluetooth_watcher.py--> [app.py] --polling--> [Browser]
```

1. Pair your phone (or any Bluetooth-capable camera) with this PC the
   normal Windows way (Settings → Bluetooth & devices → Add device).
2. Find/set where Windows saves Bluetooth-received files: Settings →
   Bluetooth & devices → Devices → "Send or receive files via Bluetooth" →
   Receive files. Put that exact path into `WATCH_FOLDER` at the top of
   `bluetooth_watcher.py`.
3. Run both, in two terminals:
   ```bash
   python app.py                 # terminal 1
   python bluetooth_watcher.py   # terminal 2
   ```
4. Send a photo from your phone to the PC over Bluetooth. The watcher
   detects the new file once Windows has finished writing it, POSTs it to
   the same `/api/analyze` endpoint the browser uses, and the web page's
   **"Live camera feed"** toggle (bottom of the page) picks it up
   automatically within a few seconds -- no manual upload.

This script only talks to a local folder and to `app.py`'s HTTP API -- it
does not touch Bluetooth hardware directly, which is what keeps it simple
(Windows' own Bluetooth stack does the actual pairing/transfer). Install
its one extra dependency with `pip install -r requirements.txt` (adds
`requests`, used only by this script).

