"""
Web front end for the shoebox pipeline.

Run:
    python app.py

Then open http://127.0.0.1:5000 in a browser, upload a photo, and it will
run the exact same CNN + YOLO + scale-reference pipeline as
`src/demo_pipeline.py` (they share the `run_analysis()` function, so the
CLI and the web UI can never give different answers on the same photo) and
return the category, 2D bounding box, dimensions and recommended shoebox
size as JSON for the page to draw. A second endpoint exports the
recommended box as a downloadable STL.

This does NOT replace demo_pipeline.py -- it's an additional way to run the
same pipeline from a browser instead of the command line.
"""
import os
import sys
import json
import time
import base64
import tempfile
import traceback

import matplotlib
matplotlib.use("Agg")  # MUST happen before anything (including demo_pipeline)
                        # imports matplotlib.pyplot -- a Flask server has no
                        # display, so the default interactive backend would
                        # hang/crash the first request that tries to plot.

import numpy as np
from flask import Flask, request, jsonify, send_from_directory, send_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.data_utils import safe_imread
from src.demo_pipeline import load_models, run_analysis
from src.generate_stl_box import write_box_stl, render_box_preview_png
from src.scale_reference import COMMON_COINS_MM, REFERENCE_SIZES_MM

app = Flask(__name__, static_folder="frontend", static_url_path="")

MAX_UPLOAD_MB = 20
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

_MODELS = {"cnn": None, "classes": None, "yolo": None, "error": None}

# Holds the most recent analysis result so the frontend can poll for it --
# this is what makes the "Live camera feed" toggle work: an external
# process (bluetooth_watcher.py) posts a photo to /api/analyze exactly like
# the browser upload does, and the browser just polls /api/latest to pick
# up whatever came in most recently, from either source.
_LAST_RESULT = {"result": None, "ts": 0, "source": None}


def get_models():
    """Loads the CNN + YOLO models once and caches them. Training must have
    already been run (see README) so outputs/models/ contains the weight
    files -- if it hasn't, every /api/analyze call will return a clear
    error instead of the server crashing at startup."""
    if _MODELS["cnn"] is None and _MODELS["error"] is None:
        try:
            cnn, classes, yolo = load_models()
            _MODELS.update(cnn=cnn, classes=classes, yolo=yolo)
        except Exception as e:
            _MODELS["error"] = (
                f"Could not load models ({e}). Have you run "
                "src/train_cnn.py and src/train_yolo.py yet? "
                "See README.md."
            )
    return _MODELS


def to_jsonable(obj):
    """Recursively converts numpy types (which the CV/YOLO/scale-reference
    code returns all over the place) into plain JSON-safe Python types."""
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/health")
def health():
    m = get_models()
    return jsonify({"ok": m["error"] is None, "error": m["error"]})


@app.route("/api/reference_options")
def reference_options():
    """Lets the frontend build its 'scale reference' dropdown from the
    same source of truth the backend uses, instead of hardcoding coin
    names/sizes twice."""
    return jsonify({
        "coins": COMMON_COINS_MM,
        "sizes_mm": {k: {"short": v[0], "long": v[1]} for k, v in REFERENCE_SIZES_MM.items()},
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    m = get_models()
    if m["error"] is not None:
        return jsonify({"error": m["error"]}), 503

    if "image" not in request.files:
        return jsonify({"error": "No 'image' file in request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    def _float_or_none(name):
        v = request.form.get(name)
        if v is None or v == "":
            return None
        try:
            return float(v)
        except ValueError:
            return None

    coin_mm = _float_or_none("coin_mm")
    outsole_mm = _float_or_none("outsole_mm")
    reference_type = request.form.get("reference_type", "auto") or "auto"
    if reference_type not in ("auto", "a4", "card", "coin", "manual"):
        reference_type = "auto"
    # "source" lets us tell in the UI whether a result came from a manual
    # browser upload or from bluetooth_watcher.py -- purely informational.
    source = request.form.get("source", "upload")

    suffix = os.path.splitext(file.filename)[1] or ".jpg"
    tmp_dir = tempfile.mkdtemp(prefix="shoebox_upload_")
    tmp_path = os.path.join(tmp_dir, "upload" + suffix)
    file.save(tmp_path)

    try:
        img = safe_imread(tmp_path)
        if img is None:
            return jsonify({"error": "Could not decode that image file."}), 400
        img_h, img_w = img.shape[:2]

        result = run_analysis(
            m["cnn"], m["classes"], m["yolo"], tmp_path,
            coin_mm=coin_mm, outsole_mm=outsole_mm,
            reference_type=reference_type,
        )
        result["image_width"] = img_w
        result["image_height"] = img_h

        # 3D preview image of the recommended box, as a base64 PNG the
        # frontend can drop straight into an <img src="data:image/png..">
        # -- no extra request/round-trip needed.
        if result.get("shoebox"):
            try:
                sb = result["shoebox"]
                png_bytes = render_box_preview_png(sb["height"], sb["width"], sb["depth"])
                result["preview_3d_png_base64"] = base64.b64encode(png_bytes).decode("ascii")
            except Exception as e:
                # Never let a preview-rendering hiccup break the whole
                # analysis result -- the numbers/box are still valid.
                traceback.print_exc()
                result["preview_3d_error"] = str(e)

        result_json = to_jsonable(result)
        _LAST_RESULT.update(result=result_json, ts=time.time(), source=source)
        return jsonify(result_json)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Pipeline failed: {e}"}), 500
    finally:
        try:
            os.remove(tmp_path)
            os.rmdir(tmp_dir)
        except OSError:
            pass


@app.route("/api/latest")
def latest():
    """Polled by the frontend's 'Live camera feed' toggle. `since` (a
    timestamp the client already has) lets it ask for "only tell me if
    something newer has come in" instead of re-rendering on every poll."""
    since = request.args.get("since", type=float, default=0)
    if _LAST_RESULT["result"] is None or _LAST_RESULT["ts"] <= since:
        return jsonify({"new": False})
    return jsonify({
        "new": True,
        "ts": _LAST_RESULT["ts"],
        "source": _LAST_RESULT["source"],
        "result": _LAST_RESULT["result"],
    })


@app.route("/api/stl", methods=["POST"])
def stl():
    data = request.get_json(force=True, silent=True) or {}
    try:
        height = float(data["height"])
        width = float(data["width"])
        depth = float(data["depth"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "height, width and depth (mm) are required"}), 400

    wall = data.get("wall")
    wall = float(wall) if wall else None

    tmp_dir = tempfile.mkdtemp(prefix="shoebox_stl_")
    out_path = os.path.join(tmp_dir, "shoebox.stl")
    try:
        write_box_stl(out_path, height, width, depth, wall_mm=wall)
        return send_file(
            out_path,
            as_attachment=True,
            download_name="shoebox.stl",
            mimetype="model/stl",
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    print("Loading models (this can take a few seconds) ...")
    m = get_models()
    if m["error"]:
        print(f"!! {m['error']}")
        print("The server will still start, but /api/analyze will return an error until you fix this.")
    print(f"\nOpen http://127.0.0.1:5000 in your browser.")
    app.run(host="127.0.0.1", port=5000, debug=False)
