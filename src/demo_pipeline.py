
"""
THE DEMO SCRIPT -- run this for the review panel on a single photo, dataset
or user-supplied.

Pipeline: YOLO detects the shoe -> CNN classifies its category -> a scale
(mm-per-pixel) is resolved from the best available source -> Height/Width/
Depth are computed -> a recommended shoebox size (with clearance) is
printed, drawn, and optionally exported as a 3D-printable STL file.

--- Scale resolution priority ---
1. Saved camera calibration (production-line setup)
2. A4 sheet detected in THIS photo
3. Credit/debit card detected in THIS photo
4. Coin detected in THIS photo (--coin_mm)
5. Manual override (--outsole_mm)
6. Category-typical ASSUMED length (last resort, always labeled as such)

--- Height priority ---
1. Real sensor reading (--height_sensor_mm)
2. Second side-view photo (--side_image, with its own reference)
3. Category-typical ASSUMED height (last resort, labeled as such)

Usage examples:
    python src/demo_pipeline.py --image photo.jpg
    python src/demo_pipeline.py --image photo.jpg --coin_mm 24.26
    python src/demo_pipeline.py --image photo.jpg --outsole_mm 265
    python src/demo_pipeline.py --image photo.jpg --stl
"""

import os
import sys
import json
import argparse

import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models import resnet18

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import config
from src.data_utils import safe_imread
from src import scale_reference


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PLAUSIBLE_SHOE_LENGTH_MM = (100, 500)


def load_models():
    classes_path = os.path.join(
        config.OUTPUT_DIR,
        "cnn_classes.json"
    )

    with open(classes_path, "r") as f:
        classes = json.load(f)

    cnn = resnet18(weights=None)

    cnn.fc = nn.Linear(
        cnn.fc.in_features,
        len(classes)
    )

    cnn.load_state_dict(
        torch.load(
            os.path.join(
                config.MODELS_DIR,
                "cnn_category_classifier.pt"
            ),
            map_location=DEVICE
        )
    )

    cnn.to(DEVICE)
    cnn.eval()

    from ultralytics import YOLO

    yolo = YOLO(
        os.path.join(
            config.MODELS_DIR,
            "yolo_run",
            "weights",
            "best.pt"
        )
    )

    return cnn, classes, yolo


def predict_category(cnn, classes, img_bgr):
    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(
            (config.IMG_SIZE, config.IMG_SIZE)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    img_rgb = cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2RGB
    )

    x = tf(img_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = cnn(x).argmax(1).item()

    return classes[pred]


def detect_shoe_box(yolo, image_path, conf_threshold=None):
    """Returns (box, confidence, method, warning_or_None).

    method is "yolo" for a real detection, or "fallback_contour" when YOLO
    found nothing and we fell back to a plain contour heuristic.

    Why external/uploaded photos were coming back with NO box at all:
    YOLO here is trained purely on pseudo-labels from plain-background
    catalog photos (auto_bbox() in data_utils.py, used by
    prepare_yolo_data.py) -- it never saw real backgrounds, clutter, floors,
    hands, shadows, etc. On those photos it often scores every candidate box
    below Ultralytics' default 0.25 confidence cutoff, so `result.boxes` is
    empty and the whole pipeline used to bail out with "no shoe detected."
    This function now (1) uses a lower inference-time confidence threshold
    (config.YOLO_INFERENCE_CONF) so weaker-but-plausible detections aren't
    thrown away, and (2) if YOLO still finds nothing, falls back to the same
    contour/Otsu heuristic used to LABEL the training data in the first
    place -- so the pipeline always returns *some* box for a real photo
    instead of silently giving up. The fallback is clearly flagged as such
    (method + warning) so it's never confused with a real detection.
    """
    if conf_threshold is None:
        conf_threshold = getattr(config, "YOLO_INFERENCE_CONF", 0.15)

    result = yolo.predict(
        image_path,
        verbose=False,
        conf=conf_threshold
    )[0]

    if len(result.boxes):
        best = result.boxes.conf.argmax()

        box = (
            result.boxes.xyxy[best]
            .cpu()
            .numpy()
            .tolist()
        )

        conf = float(
            result.boxes.conf[best].item()
        )

        return box, conf, "yolo", None

    # ---- Fallback: no YOLO detection above threshold ----
    from src.data_utils import safe_imread, auto_bbox

    img = safe_imread(image_path)
    if img is None:
        return None, None, None, None

    box = auto_bbox(img)
    warning = (
        "YOLO found no shoe above the confidence threshold on this photo "
        "-- likely because it was trained only on plain-background catalog "
        "photos (see prepare_yolo_data.py). Falling back to a simple "
        "contour-based box instead of giving up. This is LESS RELIABLE "
        "than a real detection, especially on cluttered backgrounds -- "
        "check outputs/demo_result.png before trusting it."
    )
    return box, None, "fallback_contour", warning


def box_long_short_px(box):
    x0, y0, x1, y1 = box

    w_px = abs(x1 - x0)
    h_px = abs(y1 - y0)

    return (
        max(w_px, h_px),
        min(w_px, h_px)
    )


def load_camera_calibration():
    calib_path = os.path.join(
        config.OUTPUT_DIR,
        "camera_calibration.json"
    )

    if not os.path.exists(calib_path):
        return None

    with open(calib_path, "r") as f:
        return json.load(f)


def resolve_length_width_mm(
    img_bgr,
    box,
    category,
    args
):
    """Returns (length_mm, width_mm, method_str, scale_viz, warning_or_None)."""
    long_px, short_px = box_long_short_px(box)

    # ---------------------------------------------------------
    # 1. Fixed camera calibration
    # ---------------------------------------------------------
    calibration = load_camera_calibration()

    if calibration is not None:
        mm_per_px = calibration["mm_per_px"]

        return (
            long_px * mm_per_px,
            short_px * mm_per_px,
            (
                "MEASURED "
                f"(fixed camera calibration, "
                f"{calibration['method']} reference at setup)"
            ),
            None,
            None
        )

    # ---------------------------------------------------------
    # 2. A4 / Card / Coin reference detection
    #
    # reference_type (default "auto") lets the caller say exactly what's
    # in the photo -- "a4", "card", "coin", or "manual" (skip detection
    # entirely, go straight to a manual/assumed length below) -- instead
    # of relying on ambiguous auto-detection. See scale_reference.get_scale().
    # ---------------------------------------------------------
    reference_type = getattr(args, "reference_type", "auto") or "auto"

    scale_info = None
    if reference_type != "manual":
        scale_info = scale_reference.get_scale(
            img_bgr,
            coin_diameter_mm=args.coin_mm,
            reference_type=reference_type,
        )

    if scale_info is not None:
        mm_per_px = scale_info["mm_per_px"]

        method = (
            "MEASURED "
            f"({scale_info['method']} reference detected "
            "in this photo)"
        )

        warning = None
        if scale_info["method"] == "coin":
            cx, cy, r = scale_info["viz"]["circle"]

            warning = (
                f"Using a COIN reference at pixel "
                f"({cx},{cy}), radius {r}px -- "
                "less reliable than A4/card. "
                "Check outputs/demo_result.png."
            )
            print(f"\n!! {warning}")

        return (
            long_px * mm_per_px,
            short_px * mm_per_px,
            method,
            scale_info,
            warning
        )

    not_found_warning = None
    if reference_type != "auto" and reference_type != "manual":
        # The person explicitly said e.g. "there's a card in this photo"
        # but we couldn't find one -- worth telling them that specifically,
        # since it's different from "no reference was ever expected."
        not_found_warning = (
            f"You selected '{reference_type}' as the scale reference, but "
            f"none was found in the photo -- check that it's fully visible, "
            f"flat, and well-lit, then try again."
        )
        print(f"\n!! {not_found_warning}")

    # ---------------------------------------------------------
    # 3. Manual outsole length
    # ---------------------------------------------------------
    if args.outsole_mm is not None and long_px:
        mm_per_px = args.outsole_mm / long_px

        return (
            args.outsole_mm,
            short_px * mm_per_px,
            (
                "MANUAL OVERRIDE "
                "(shoe length entered by hand, no reference detected in photo)"
            ),
            None,
            not_found_warning
        )

    # ---------------------------------------------------------
    # 4. Category-typical assumed length
    # ---------------------------------------------------------
    nominal_mm = config.NOMINAL_OUTSOLE_MM.get(
        category,
        260.0
    )

    if long_px:
        mm_per_px = nominal_mm / long_px

        warning = (
            (f"{not_found_warning} " if not_found_warning else "")
            + f"No real scale reference found -- "
            f"using an ASSUMED category-typical length "
            f"({nominal_mm:.0f}mm for '{category}'). "
            "NOT a measurement of this shoe."
        )
        print(f"\n!! {warning}")

        return (
            nominal_mm,
            short_px * mm_per_px,
            "ASSUMED (category-typical, not measured)",
            None,
            warning
        )

    return (
        None,
        None,
        "FAILED (degenerate box)",
        None,
        "The detected box was degenerate (zero-width/height) -- can't derive a scale from it."
    )


def resolve_height_mm(
    yolo,
    category,
    args
):
    # ---------------------------------------------------------
    # 1. Real height sensor
    # ---------------------------------------------------------
    if args.height_sensor_mm is not None:
        return (
            args.height_sensor_mm,
            "MEASURED (height sensor)",
            None
        )

    # ---------------------------------------------------------
    # 2. Side-view image
    # ---------------------------------------------------------
    if args.side_image is not None:

        img_bgr = safe_imread(
            args.side_image
        )

        if img_bgr is None:
            box = None
        else:
            box, _, _, _ = detect_shoe_box(
                yolo,
                args.side_image
            )

        if img_bgr is not None and box is not None:

            # IMPORTANT:
            # Current get_scale() does not accept shoe_box.
            scale_info = scale_reference.get_scale(
                img_bgr,
                coin_diameter_mm=args.side_coin_mm
            )

            if scale_info is not None:

                x0, y0, x1, y1 = box

                height_px = abs(y1 - y0)

                return (
                    height_px *
                    scale_info["mm_per_px"],
                    (
                        "MEASURED "
                        f"({scale_info['method']} reference "
                        "in side photo)"
                    ),
                    None
                )

        warning = (
            "Side image given but height could not "
            "be measured -- falling back to category estimate"
        )
        print(f"!! {warning}")
    else:
        warning = None

    # ---------------------------------------------------------
    # 3. Category-typical assumed height
    # ---------------------------------------------------------
    return (
        config.CATEGORY_NOMINAL_HEIGHT_MM.get(
            category,
            80.0
        ),
        "ASSUMED (category-typical, not measured)",
        warning
    )


def estimate_foot_size(
    shoe_length_mm,
    category
):
    off = config.CATEGORY_OFFSETS_MM.get(
        category,
        {
            "outsole_to_insole": -12.0,
            "insole_to_foot": -8.0
        }
    )

    insole = (
        shoe_length_mm +
        off["outsole_to_insole"]
    )

    foot = (
        insole +
        off["insole_to_foot"]
    )

    eu_size = 1.5 * (
        foot / 10.0 + 1.5
    )

    return foot, eu_size


def estimate_foot_length(outsole_mm, category):
    """Back-compat helper for src/build_seed_demo.py, which imports this
    name directly. Same outsole -> insole -> foot chain as
    estimate_foot_size(), just returning the intermediate insole length too
    (build_seed_demo.py's `insole, foot = estimate_foot_length(...)`)."""
    off = config.CATEGORY_OFFSETS_MM.get(
        category,
        {"outsole_to_insole": -12.0, "insole_to_foot": -8.0}
    )
    insole = outsole_mm + off["outsole_to_insole"]
    foot = insole + off["insole_to_foot"]
    return insole, foot


def box_dimensions_mm(box, outsole_mm):
    """Back-compat helper for src/build_seed_demo.py, which imports this
    name directly. Converts a YOLO box's pixel aspect ratio into
    (length_mm, width_mm), using outsole_mm as the real-world length for
    the box's long side -- same logic resolve_length_width_mm() uses for
    its manual-override branch."""
    long_px, short_px = box_long_short_px(box)
    if not long_px:
        return outsole_mm, 0.0
    mm_per_px = outsole_mm / long_px
    return outsole_mm, short_px * mm_per_px


def run_analysis(
    cnn,
    classes,
    yolo,
    image_path,
    coin_mm=None,
    outsole_mm=None,
    side_image=None,
    side_coin_mm=None,
    height_sensor_mm=None,
    reference_type="auto",
):
    """Runs the full pipeline on one image and returns a plain dict of
    results -- no argparse, no printing, no plotting, no file I/O. This is
    the function both the CLI (main(), below) and the Flask web app
    (app.py) call, so the two front ends can never drift out of sync with
    each other.

    Returns a dict with (at minimum):
        category, box, box_confidence, box_method, box_warning,
        length_mm, width_mm, length_method, scale_viz, length_warning,
        height_mm, height_method, height_warning,
        shoe_length_plausible (bool),
        shoebox (dict with height/width/depth) or None,
        foot_length_mm, eu_size,
        warnings (list[str]) -- every non-fatal warning collected above.
    """
    args = argparse.Namespace(
        coin_mm=coin_mm,
        outsole_mm=outsole_mm,
        side_image=side_image,
        side_coin_mm=side_coin_mm,
        height_sensor_mm=height_sensor_mm,
        reference_type=reference_type,
    )

    warnings = []

    img_bgr = safe_imread(image_path)
    if img_bgr is None:
        return {"error": f"Could not read image: {image_path}"}

    box, box_conf, box_method, box_warning = detect_shoe_box(yolo, image_path)
    if box_warning:
        warnings.append(box_warning)

    category = predict_category(cnn, classes, img_bgr)

    result = {
        "category": category,
        "box": box,
        "box_confidence": box_conf,
        "box_method": box_method,
        "box_warning": box_warning,
    }

    if box is None:
        result["error"] = "No shoe detected (YOLO found nothing and the fallback contour also failed)."
        result["warnings"] = warnings
        return result

    length_mm, width_mm, length_method, scale_viz, length_warning = resolve_length_width_mm(
        img_bgr, box, category, args
    )
    if length_warning:
        warnings.append(length_warning)

    height_mm, height_method, height_warning = resolve_height_mm(yolo, category, args)
    if height_warning:
        warnings.append(height_warning)

    shoe_length_plausible = True
    if length_mm is not None and not (
        PLAUSIBLE_SHOE_LENGTH_MM[0] <= length_mm <= PLAUSIBLE_SHOE_LENGTH_MM[1]
    ):
        shoe_length_plausible = False
        warnings.append(
            f"WARNING: measured length ({length_mm:.0f}mm) is outside the "
            "plausible range for a shoe. Don't trust these numbers -- "
            "check outputs/demo_result.png."
        )

    shoebox = None
    foot_mm = eu_size = None

    if length_mm is not None:
        clr = config.SHOEBOX_CLEARANCE_MM
        foot_mm, eu_size = estimate_foot_size(length_mm, category)
        shoebox = {
            "height": height_mm + clr["height"],
            "width": width_mm + clr["width"],
            "depth": length_mm + clr["length"],
        }

    result.update({
        "length_mm": length_mm,
        "width_mm": width_mm,
        "length_method": length_method,
        "scale_viz": scale_viz,
        "height_mm": height_mm,
        "height_method": height_method,
        "shoe_length_plausible": shoe_length_plausible,
        "shoebox": shoebox,
        "foot_length_mm": foot_mm,
        "eu_size": eu_size,
        "warnings": warnings,
    })
    return result


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--image",
        required=True,
        help="Path to a shoe photo"
    )

    ap.add_argument(
        "--coin_mm",
        type=float,
        default=None
    )

    ap.add_argument(
        "--outsole_mm",
        type=float,
        default=None,
        help=(
            "Manually measured shoe outsole (heel-to-toe) length in mm. "
            "Use this ONLY if you don't have an A4 sheet, card, or coin to "
            "put in the photo -- it lets you get an accurate measurement "
            "just by measuring the shoe once with a ruler and typing the "
            "number in, instead of the pipeline having to guess a "
            "category-typical length."
        )
    )

    ap.add_argument(
        "--reference",
        dest="reference_type",
        choices=["auto", "a4", "card", "coin", "manual"],
        default="auto",
        help=(
            "Which scale reference is in the photo. 'auto' (default) tries "
            "A4, then card, then coin (if --coin_mm given). Set this "
            "explicitly if you know what's in frame -- e.g. --reference "
            "card -- for a faster, less ambiguous result. 'manual' skips "
            "photo-based detection entirely and uses --outsole_mm instead."
        )
    )

    ap.add_argument(
        "--side_image",
        default=None
    )

    ap.add_argument(
        "--side_coin_mm",
        type=float,
        default=None
    )

    ap.add_argument(
        "--height_sensor_mm",
        type=float,
        default=None
    )

    ap.add_argument(
        "--stl",
        action="store_true",
        help=(
            "Also export a 3D-printable STL "
            "of the recommended shoebox"
        )
    )

    args = ap.parse_args()

    # ---------------------------------------------------------
    # Load models + run the shared pipeline
    # ---------------------------------------------------------
    cnn, classes, yolo = load_models()

    print(f"Image:          {args.image}")

    r = run_analysis(
        cnn, classes, yolo,
        args.image,
        coin_mm=args.coin_mm,
        outsole_mm=args.outsole_mm,
        side_image=args.side_image,
        side_coin_mm=args.side_coin_mm,
        height_sensor_mm=args.height_sensor_mm,
        reference_type=args.reference_type,
    )

    if "error" in r and "category" not in r:
        # Image itself couldn't even be read -- nothing else to report.
        print(f"!! {r['error']}")
        return

    print(f"CNN category:   {r['category']}")

    if r.get("box") is None:
        print(f"!! {r['error']}")
        for w in r.get("warnings", []):
            print(f"\n!! {w}")
        return

    if r["box_method"] == "yolo":
        print(
            "YOLO detection: "
            f"box={[round(v, 1) for v in r['box']]}  "
            f"confidence={r['box_confidence']:.2f}"
        )
    else:
        print(
            "YOLO detection: none above threshold -- "
            f"box={[round(v, 1) for v in r['box']]} "
            "(fallback contour box, see warning below)"
        )

    for w in r["warnings"]:
        print(f"\n!! {w}")

    box_h = box_w = box_d = None
    length_mm = r.get("length_mm")
    width_mm = r.get("width_mm")
    height_mm = r.get("height_mm")
    length_method = r.get("length_method")
    height_method = r.get("height_method")
    scale_viz = r.get("scale_viz")
    box = r["box"]
    category = r["category"]

    if length_mm is not None:
        shoebox = r["shoebox"]
        box_h, box_w, box_d = shoebox["height"], shoebox["width"], shoebox["depth"]

        print("\n--- Shoe (outer) dimensions ---")
        print(f"Height: {height_mm:.1f} mm   [{height_method}]")
        print(f"Width:  {width_mm:.1f} mm   [{length_method}]")
        print(f"Depth:  {length_mm:.1f} mm   [{length_method}]   (heel-to-toe)")

        print("\n--- Recommended shoebox (shoe size + clearance) ---")
        print(f"Height: {box_h:.0f} mm")
        print(f"Width:  {box_w:.0f} mm")
        print(f"Depth:  {box_d:.0f} mm")

        print("\n--- Informational only (not used to size the box) ---")
        print(f"Est. foot length: {r['foot_length_mm']:.1f} mm   Est. EU size: {r['eu_size']:.1f}")

        # -----------------------------------------------------
        # STL export
        # -----------------------------------------------------
        if args.stl:
            from src.generate_stl_box import write_box_stl, preview_box_3d

            stl_path = os.path.join(config.OUTPUT_DIR, "shoebox.stl")
            write_box_stl(stl_path, box_h, box_w, box_d)
            print(f"\n3D-printable box STL saved -> {stl_path}")

            preview_box_3d(
                box_h, box_w, box_d,
                save_path=os.path.join(config.OUTPUT_DIR, "shoebox_preview.png")
            )

    # ---------------------------------------------------------
    # Create annotated image
    # ---------------------------------------------------------
    img_bgr = safe_imread(args.image)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img_rgb)

    x0, y0, x1, y1 = box
    box_color = "lime" if r["box_method"] == "yolo" else "yellow"
    ax.add_patch(
        plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=box_color, linewidth=3)
    )

    # ---------------------------------------------------------
    # Draw scale reference
    # ---------------------------------------------------------
    if scale_viz is not None:
        if scale_viz["method"] in ("A4", "card"):
            corners = scale_viz["viz"]["corners"]
            color = "deepskyblue" if scale_viz["method"] == "A4" else "orange"
            ax.add_patch(plt.Polygon(corners, fill=False, edgecolor=color, linewidth=2))
        elif scale_viz["method"] == "coin":
            cx, cy, rr = scale_viz["viz"]["circle"]
            ax.add_patch(plt.Circle((cx, cy), rr, fill=False, edgecolor="red", linewidth=2))

    # ---------------------------------------------------------
    # Title
    # ---------------------------------------------------------
    title = f"Category: {category}"
    if r["box_method"] != "yolo":
        title += "  (box: fallback, not YOLO)"

    if length_mm is not None:
        is_measured = "MEASURED" in length_method
        tag = "" if is_measured else "  (contains an ASSUMED value)"
        title += f"\nShoe: {height_mm:.0f}x{width_mm:.0f}x{length_mm:.0f}mm (HxWxD){tag}"
        title += f"\nShoebox: {box_h:.0f}x{box_w:.0f}x{box_d:.0f}mm"

    ax.set_title(title, fontsize=10)
    ax.axis("off")

    # ---------------------------------------------------------
    # Save result
    # ---------------------------------------------------------
    out_path = os.path.join(config.OUTPUT_DIR, "demo_result.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved annotated result -> {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
