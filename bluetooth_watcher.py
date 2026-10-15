"""
Bridges an external camera to the web app over Bluetooth.

THE ARCHITECTURE (how this actually works end to end):

    [Camera / phone]  --Bluetooth-->  [This Windows PC]  --this script-->  [Flask app.py]  --polling-->  [Browser]

1. Your camera device (a phone is the easiest "external camera" -- most
   phones can send a just-taken photo via Bluetooth) pairs with this
   Windows PC over Bluetooth, the normal OS-level way (Settings ->
   Bluetooth & devices -> Add device). No code needed for this part --
   it's a one-time OS pairing.

2. When you send a photo from the phone to the PC over Bluetooth, Windows
   itself receives it (via "Bluetooth File Transfer" / OBEX) and drops it,
   as a plain image file, into a folder YOU configured. Find/set that
   folder at: Settings -> Bluetooth & devices -> Devices -> "Send or
   receive files via Bluetooth" -> Receive files -> it shows/lets you
   change the destination folder. Set WATCH_FOLDER below to that exact
   path. This step is entirely Windows' own Bluetooth stack -- this
   script does NOT talk to Bluetooth hardware directly, which keeps it
   simple and avoids needing a separate Bluetooth pairing library.

3. This script polls that folder. Whenever a new, fully-written image file
   shows up, it POSTs it to the already-running app.py's /api/analyze
   endpoint -- the exact same endpoint the browser's "Analyze photo"
   button calls -- with source=bluetooth so the UI can tell them apart.

4. app.py remembers the most recent result. The browser page's "Live
   camera feed" toggle polls /api/latest every few seconds and renders
   whatever comes in automatically -- so the panel/demo flow is: take a
   photo on the phone -> send over Bluetooth -> it appears on the website
   within a few seconds, no manual upload.

Requirements: `pip install requests` (this script only, app.py doesn't
need it). Run this ALONGSIDE app.py (in a second terminal), not instead of
it:
    # terminal 1
    python app.py
    # terminal 2
    python bluetooth_watcher.py
"""
import os
import sys
import time
import argparse

import requests

# ---- EDIT THIS to match your PC's actual Bluetooth-received-files folder
# (Settings -> Bluetooth & devices -> Devices -> "Send or receive files via
# Bluetooth" -> Receive files). This default is a common one but Windows
# lets you change it, so don't assume -- go check. ----
WATCH_FOLDER = os.path.expanduser(r"~\Documents\Bluetooth Exchange Folder")

APP_URL = "http://127.0.0.1:5000/api/analyze"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
POLL_SECONDS = 2
STABLE_CHECKS = 2   # how many consecutive same-size polls before we trust
                     # a file is fully written and safe to upload


def _is_image(path):
    return os.path.splitext(path)[1].lower() in IMG_EXT


def watch(folder, coin_mm=None, outsole_mm=None):
    if not os.path.isdir(folder):
        print(f"!! Watch folder does not exist: {folder}")
        print("   Edit WATCH_FOLDER at the top of this script (or pass --folder) "
              "to match your actual Bluetooth receive folder -- see the module "
              "docstring for where to find it in Windows Settings.")
        sys.exit(1)

    print(f"Watching for new photos in: {folder}")
    print(f"Will POST each new photo to: {APP_URL}")
    print("(Ctrl+C to stop)\n")

    seen = set(os.listdir(folder))          # ignore files already there at startup
    pending = {}                            # path -> (last_size, stable_count)

    while True:
        try:
            current = set(os.listdir(folder))
        except OSError as e:
            print(f"!! Could not list watch folder: {e}")
            time.sleep(POLL_SECONDS)
            continue

        new_files = [f for f in current - seen if _is_image(f)]

        for name in new_files:
            path = os.path.join(folder, name)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue

            last_size, stable_count = pending.get(path, (None, 0))
            if last_size == size:
                stable_count += 1
            else:
                stable_count = 0
            pending[path] = (size, stable_count)

            if stable_count >= STABLE_CHECKS:
                # File size hasn't changed for a couple of polls -- Bluetooth
                # transfer is done, safe to read/upload.
                _submit(path, coin_mm, outsole_mm)
                seen.add(name)
                pending.pop(path, None)

        time.sleep(POLL_SECONDS)


def _submit(path, coin_mm, outsole_mm):
    print(f"New photo detected: {path}")
    try:
        with open(path, "rb") as f:
            files = {"image": (os.path.basename(path), f)}
            data = {"source": "bluetooth"}
            if coin_mm is not None:
                data["coin_mm"] = coin_mm
            if outsole_mm is not None:
                data["outsole_mm"] = outsole_mm
            resp = requests.post(APP_URL, files=files, data=data, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        if "error" in result and result.get("box") is None:
            print(f"  !! Pipeline error: {result['error']}")
            return
        print(f"  -> category: {result.get('category')}  "
              f"box_method: {result.get('box_method')}")
        if result.get("shoebox"):
            sb = result["shoebox"]
            print(f"  -> recommended shoebox: "
                  f"{sb['height']:.0f} x {sb['width']:.0f} x {sb['depth']:.0f} mm (HxWxD)")
        print("  -> now visible on the website (Live camera feed toggle)\n")
    except requests.exceptions.ConnectionError:
        print("  !! Could not reach app.py -- is it running? (python app.py, in another terminal)")
    except Exception as e:
        print(f"  !! Upload failed: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=WATCH_FOLDER,
                     help="Folder to watch (default: WATCH_FOLDER set at the top of this file)")
    ap.add_argument("--coin_mm", type=float, default=None,
                     help="Pass through to every analysis, same as demo_pipeline.py --coin_mm")
    ap.add_argument("--outsole_mm", type=float, default=None)
    args = ap.parse_args()
    watch(args.folder, coin_mm=args.coin_mm, outsole_mm=args.outsole_mm)
