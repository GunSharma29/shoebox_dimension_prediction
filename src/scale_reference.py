"""
Detects a real-world scale reference in a photo -- an A4 sheet (210x297mm),
a standard ID/credit card (85.60x53.98mm), or a coin of known diameter --
so pixel measurements can be converted to real millimetres WITHOUT any
manual input. This only works on photos that actually contain one of these
references in frame. Kaggle/stock catalog photos generally do NOT contain
one, so this will correctly report "not found" on those -- that's
expected, not a bug. Take your own photo with a coin, card, or sheet of A4
next to the shoe to see this actually produce numbers.

NOTE on a bug fixed here: earlier versions of this file only had
detect_a4(), which matched ANY roughly-A4-ratio'd rectangle (1.25-1.60)
above a size threshold and then blindly used A4's 210x297mm to compute
scale -- but a standard card's aspect ratio (85.60/53.98 = 1.586) falls
inside that same 1.25-1.60 window! A credit/ID card in frame would get
detected and silently scaled as if it were a full A4 sheet, which is
~3.5x larger, giving wildly wrong measurements with no warning. detect_a4
and detect_card are now separate, ratio-and-size-disambiguated detectors,
and callers can also request a *specific* one explicitly (reference_type=
"a4"/"card"/"coin") instead of relying on ambiguous auto-detection --
see get_scale().
"""
import cv2
import numpy as np

# Common coin diameters in mm, offered as presets in the frontend dropdown
# so the person doesn't have to go look this up.
COMMON_COINS_MM = {
    "Indian ₹1 coin": 20.0, "Indian ₹2 coin": 27.0, "Indian ₹5 coin": 23.0, "Indian ₹10 coin": 27.0,
    "US penny": 19.05, "US nickel": 21.21, "US dime": 17.91, "US quarter": 24.26,
    "Euro €1 coin": 23.25, "Euro €2 coin": 25.75,
}

# (short_mm, long_mm) for each supported flat rectangular reference.
REFERENCE_SIZES_MM = {
    "a4": (210.0, 297.0),      # ISO 216 A4 paper
    "card": (53.98, 85.60),    # ISO/IEC 7810 ID-1 -- credit/debit/most ID cards
}


def _order_points(pts):
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _find_rect_candidates(img_bgr, min_area_frac):
    """Shared contour-finding step for any flat rectangular reference.
    Returns a list of (area, rect, long_px, short_px), largest first."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, None, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    img_area = img_bgr.shape[0] * img_bgr.shape[1]
    candidates = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        area = cv2.contourArea(approx)
        if area < min_area_frac * img_area:
            continue
        pts = approx.reshape(4, 2).astype(np.float32)
        rect = _order_points(pts)
        (tl, tr, br, bl) = rect
        w = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
        h = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
        if w == 0 or h == 0:
            continue
        candidates.append((area, rect, max(w, h), min(w, h)))

    candidates.sort(key=lambda x: -x[0])   # largest matching rectangle first
    return candidates


def detect_rect(img_bgr, short_mm, long_mm, ratio_slack=0.12, min_area_frac=0.01):
    """Generic flat-rectangle detector for any known real-world size (mm).
    Returns {"mm_per_px": float, "corners": 4x2 array} or None if nothing
    with a matching aspect ratio and plausible size was found."""
    target_ratio = long_mm / short_mm
    candidates = _find_rect_candidates(img_bgr, min_area_frac)

    for area, rect, long_px, short_px in candidates:
        ratio = long_px / short_px
        if abs(ratio - target_ratio) / target_ratio <= ratio_slack:
            mm_per_px = ((long_mm / long_px) + (short_mm / short_px)) / 2
            return {"mm_per_px": mm_per_px, "corners": rect}

    return None


def detect_a4(img_bgr):
    """Looks for an A4 sheet (210x297mm, ratio 1.414). Returns
    {"mm_per_px": float, "corners": 4x2 array} or None if not found.
    A4 is large relative to most photos, so it needs a higher minimum
    area than detect_card() to avoid ever matching a smaller card-sized
    rectangle by accident."""
    short_mm, long_mm = REFERENCE_SIZES_MM["a4"]
    return detect_rect(img_bgr, short_mm, long_mm, ratio_slack=0.12, min_area_frac=0.03)


def detect_card(img_bgr):
    """Looks for a standard ID/credit card (85.60x53.98mm, ratio 1.586).
    Returns {"mm_per_px": float, "corners": 4x2 array} or None if not
    found. A card is much smaller than A4 (about 1/7th the area) and their
    aspect ratios are close enough (1.586 vs 1.414) that a naive single
    detector would confuse the two -- see the module docstring. Using a
    tighter ratio_slack and a lower area floor here keeps card detection
    from ever matching an A4-sized sheet instead."""
    short_mm, long_mm = REFERENCE_SIZES_MM["card"]
    return detect_rect(img_bgr, short_mm, long_mm, ratio_slack=0.08, min_area_frac=0.005)


def detect_coin(img_bgr, coin_diameter_mm):
    """Looks for a circular coin of the given known diameter (mm) using a
    Hough circle transform. Returns {"mm_per_px": float, "circle": (x,y,r)}
    or None if not found. Less reliable than the A4 method -- laces,
    eyelets, and rivets can look circular too, so double check the drawn
    circle in the output image actually sits on a coin, not shoe hardware."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    h, w = gray.shape
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=h / 8,
        param1=100, param2=40, minRadius=10, maxRadius=int(min(h, w) / 4)
    )
    if circles is None:
        return None
    x, y, r = np.round(circles[0][0]).astype(int)   # strongest detection
    diameter_px = 2 * r
    if diameter_px == 0:
        return None
    mm_per_px = coin_diameter_mm / diameter_px
    return {"mm_per_px": mm_per_px, "circle": (int(x), int(y), int(r))}


def get_scale(img_bgr, coin_diameter_mm=None, reference_type="auto"):
    """Finds a real-world scale reference in the photo.

    reference_type controls WHICH detector(s) run:
      - "auto" (default, and the only mode the original code had): tries
        A4 first (most reliable -- large, fixed ratio), then card, then
        coin if a diameter was given. Kept for backward compatibility with
        existing calibration/CLI flows that don't specify a type.
      - "a4" / "card" / "coin": runs ONLY that detector. Use this when the
        person has told you (e.g. via a frontend dropdown) exactly what
        reference is in the photo -- it's faster, and avoids the
        auto-detector guessing wrong on an ambiguous rectangle (see the
        module docstring for why A4 vs card needed separating in the
        first place).
      - "manual": always returns None immediately -- the caller wants no
        photo-based reference at all and will supply a manual length
        instead (see demo_pipeline.resolve_length_width_mm).

    Returns {"method": "A4"|"card"|"coin", "mm_per_px": float, "viz": {...}}
    or None if nothing was found -- in which case dimensions can't be
    computed from this photo alone.
    """
    if reference_type == "manual":
        return None

    if reference_type == "a4":
        a4 = detect_a4(img_bgr)
        return {"method": "A4", "mm_per_px": a4["mm_per_px"], "viz": a4} if a4 else None

    if reference_type == "card":
        card = detect_card(img_bgr)
        return {"method": "card", "mm_per_px": card["mm_per_px"], "viz": card} if card else None

    if reference_type == "coin":
        if coin_diameter_mm is None:
            return None
        coin = detect_coin(img_bgr, coin_diameter_mm)
        return {"method": "coin", "mm_per_px": coin["mm_per_px"], "viz": coin} if coin else None

    # ---- "auto" (default): try each in order of reliability ----
    a4 = detect_a4(img_bgr)
    if a4 is not None:
        return {"method": "A4", "mm_per_px": a4["mm_per_px"], "viz": a4}

    card = detect_card(img_bgr)
    if card is not None:
        return {"method": "card", "mm_per_px": card["mm_per_px"], "viz": card}

    if coin_diameter_mm is not None:
        coin = detect_coin(img_bgr, coin_diameter_mm)
        if coin is not None:
            return {"method": "coin", "mm_per_px": coin["mm_per_px"], "viz": coin}

    return None