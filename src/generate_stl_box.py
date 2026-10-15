"""
Generates a 3D-printable STL file of an open-top box (5 sides: base + 4
walls) sized to the given outer Height x Width x Depth, with a wall
thickness from config.STL_WALL_THICKNESS_MM. Pure Python, writes ASCII
STL directly -- no extra pip packages needed, so it can't break the night
before a review over a missing dependency.

Called automatically by demo_pipeline.py --stl, or run standalone:
    python src/generate_stl_box.py --height 195 --width 130 --depth 310 --out my_box.stl
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _tri(f, v1, v2, v3):
    """Write one triangle facet in ASCII STL format. Normal is left as
    (0,0,0) -- most slicers/viewers recompute it automatically, which
    keeps this generator simple."""
    f.write("  facet normal 0 0 0\n")
    f.write("    outer loop\n")
    for v in (v1, v2, v3):
        f.write(f"      vertex {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
    f.write("    endloop\n")
    f.write("  endfacet\n")


def _quad(f, a, b, c, d):
    """Write a quad as two triangles. Vertices should be in consistent
    winding order (counter-clockwise viewed from outside the solid)."""
    _tri(f, a, b, c)
    _tri(f, a, c, d)


def write_box_stl(out_path, height_mm, width_mm, depth_mm, wall_mm=None, name="shoebox"):
    """Writes an open-top rectangular box (tray): a solid base plus four
    walls, with an inner cavity hollowed out by `wall_mm`. Units are mm,
    matching standard STL/3D-printing convention (1 STL unit = 1mm)."""
    if wall_mm is None:
        wall_mm = config.STL_WALL_THICKNESS_MM

    H, W, D = height_mm, width_mm, depth_mm
    t = wall_mm

    if min(H, W, D) <= 2 * t:
        raise ValueError(f"Box too small for wall thickness {t}mm given dimensions {H}x{W}x{D}mm")

    with open(out_path, "w") as f:
        f.write(f"solid {name}\n")

        o = [
            (0, 0, 0), (W, 0, 0), (W, D, 0), (0, D, 0),
            (0, 0, H), (W, 0, H), (W, D, H), (0, D, H),
        ]
        i = [
            (t, t, t), (W - t, t, t), (W - t, D - t, t), (t, D - t, t),
            (t, t, H), (W - t, t, H), (W - t, D - t, H), (t, D - t, H),
        ]

        _quad(f, o[0], o[1], o[5], o[4])
        _quad(f, o[1], o[2], o[6], o[5])
        _quad(f, o[2], o[3], o[7], o[6])
        _quad(f, o[3], o[0], o[4], o[7])
        _quad(f, o[0], o[3], o[2], o[1])

        _quad(f, i[0], i[4], i[5], i[1])
        _quad(f, i[1], i[5], i[6], i[2])
        _quad(f, i[2], i[6], i[7], i[3])
        _quad(f, i[3], i[7], i[4], i[0])
        _quad(f, i[0], i[1], i[2], i[3])

        _quad(f, o[4], o[5], i[5], i[4])
        _quad(f, o[5], o[6], i[6], i[5])
        _quad(f, o[6], o[7], i[7], i[6])
        _quad(f, o[7], o[4], i[4], i[7])

        f.write(f"endsolid {name}\n")

    return out_path


def _build_box_figure(height_mm, width_mm, depth_mm):
    """Shared figure-building code for the 3D preview -- used by both the
    CLI (preview_box_3d, which pops up a window) and the web app
    (render_box_preview_png, which never opens a window and just returns
    image bytes -- opening a matplotlib window from inside a Flask server
    process would hang/crash the request)."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    H, W, D = height_mm, width_mm, depth_mm
    corners = [
        (0, 0, 0), (W, 0, 0), (W, D, 0), (0, D, 0),
        (0, 0, H), (W, 0, H), (W, D, H), (0, D, H),
    ]
    faces = [
        [corners[0], corners[1], corners[5], corners[4]],
        [corners[1], corners[2], corners[6], corners[5]],
        [corners[2], corners[3], corners[7], corners[6]],
        [corners[3], corners[0], corners[4], corners[7]],
        [corners[0], corners[1], corners[2], corners[3]],
        [corners[4], corners[5], corners[6], corners[7]],
    ]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    poly = Poly3DCollection(faces, facecolor="#FFE338", edgecolor="black", linewidths=1.5, alpha=0.85)
    ax.add_collection3d(poly)

    ax.set_xlim(0, max(W, D, H))
    ax.set_ylim(0, max(W, D, H))
    ax.set_zlim(0, max(W, D, H))
    ax.set_xlabel("Width (mm)")
    ax.set_ylabel("Depth (mm)")
    ax.set_zlabel("Height (mm)")
    ax.set_title(f"Shoebox: {H:.0f} x {W:.0f} x {D:.0f} mm (HxWxD)")
    return fig


def preview_box_3d(height_mm, width_mm, depth_mm, save_path=None):
    """Pops up an actual 3D view of the outer box shape -- write_box_stl()
    only writes a file to disk with no visual feedback. This shows what
    was actually built, the same way demo_pipeline.py's 2D result already
    pops up a window. CLI use only -- see render_box_preview_png() for the
    web app's non-interactive equivalent."""
    import matplotlib.pyplot as plt

    fig = _build_box_figure(height_mm, width_mm, depth_mm)
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved 3D preview image -> {save_path}")
    plt.show()


def render_box_preview_png(height_mm, width_mm, depth_mm):
    """Headless equivalent of preview_box_3d() for the Flask app: builds
    the same 3D figure but returns raw PNG bytes instead of opening a
    window. Never calls plt.show() -- doing so on a server with no display
    would hang the request. Requires matplotlib's Agg backend, which
    app.py sets before anything else is imported."""
    import io
    import matplotlib.pyplot as plt

    fig = _build_box_figure(height_mm, width_mm, depth_mm)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=float, required=True, help="Box height in mm")
    ap.add_argument("--width", type=float, required=True, help="Box width in mm")
    ap.add_argument("--depth", type=float, required=True, help="Box depth in mm")
    ap.add_argument("--wall", type=float, default=None, help="Wall thickness in mm (default from config)")
    ap.add_argument("--out", default=None, help="Output STL path (default: outputs/shoebox.stl)")
    ap.add_argument("--no_preview", action="store_true", help="Skip the 3D preview window")
    args = ap.parse_args()

    out_path = args.out or os.path.join(config.OUTPUT_DIR, "shoebox.stl")
    write_box_stl(out_path, args.height, args.width, args.depth, wall_mm=args.wall)
    print(f"STL written -> {out_path}")
    print(f"Dimensions: {args.height:.0f} x {args.width:.0f} x {args.depth:.0f} mm (HxWxD)")
    print(f"Open this file in any slicer (Cura, PrusaSlicer) or 3D viewer to confirm it looks right "
          f"before printing.")

    if not args.no_preview:
        preview_box_3d(args.height, args.width, args.depth,
                        save_path=os.path.join(config.OUTPUT_DIR, "shoebox_preview.png"))


if __name__ == "__main__":
    main()