"""Measure a crack's width and length in millimetres from a photo with a reference object of known size.

Spec: spec/features/measurement.md. Usage, from the repository root:

    python src/measure/measure_crack.py --demo                          # synthetic photo with a known answer
    python src/measure/measure_crack.py P0001.jpg                       # bank card found automatically
    python src/measure/measure_crack.py P0001.jpg --roi "900,300,1500,1300"   # search box around the crack
    python src/measure/measure_crack.py P0001.jpg --click               # click the card corners and the box
    python src/measure/measure_crack.py P0001.jpg --ref bar50 --ref-points "812,1404;1630,1398" --roi full
    python src/measure/measure_crack.py P0001.jpg --mask P0001_mask.png # mask from a U-Net or drawn by hand

The app (upload a photo in the browser): streamlit run src/measure/app.py
Reference objects: card (bank card, default: found automatically, with a tilt check), a4, a6, square50 (4 corners),
bar50 (50 mm scale bar, 2 ends), euro1 (coin, 2 points across), or your own with --ref-size WxH or --ref-length L.
Writes <photo>_measurement.json and <photo>_overlay.png to data/measurements/ (ignored by git: photos are never
committed, AGENTS.md rule 9). The result is a measurement, never a safety statement.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from measure.analyse import VERSION, analyse
from measure.crack import BACKGROUND_PX, measure, segment
from measure.overlay import draw_overlay
from measure.reference import PRESETS, from_length, from_preset, from_rectangle
from measure.synthetic import make_demo

ROOT = Path(__file__).resolve().parents[2]
ZOOM_PX = 60  # half-size of the zoomed window used to place each reference point


def load_photo(path) -> np.ndarray:
    """RGB array with the phone's orientation applied (so coordinates match what you see)."""
    with Image.open(path) as img:
        return np.asarray(ImageOps.exif_transpose(img).convert("RGB"))


def parse_points(text: str) -> list[tuple[float, float]]:
    return [tuple(float(v) for v in p.split(",")) for p in text.replace(" ", "").split(";") if p]


def load_mask(path, shape) -> np.ndarray:
    with Image.open(path) as m:
        mask = np.asarray(ImageOps.exif_transpose(m).convert("L")) > 127
    if mask.shape != shape[:2]:
        sys.exit(f"mask is {mask.shape[::-1]}, photo is {shape[1::-1]}: they must match")
    return mask


# ---------- clicking ----------

def click(rgb, n, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(rgb)
    ax.set_title(title + "\n(left click = point, right click = undo; zoom: use the magnifier, then switch it off)",
                 fontsize=10)
    ax.set_axis_off()
    fig.tight_layout()
    pts = plt.ginput(n, timeout=0)
    plt.close(fig)
    if len(pts) < n:
        sys.exit("Cancelled: not enough points clicked.")
    return pts


def refine(rgb, rough, label):
    """Show a zoomed window around each rough point; one click there places it precisely."""
    import matplotlib.pyplot as plt
    out = []
    h, w = rgb.shape[:2]
    for i, (x, y) in enumerate(rough, 1):
        x0, y0 = int(max(0, x - ZOOM_PX)), int(max(0, y - ZOOM_PX))
        x1, y1 = int(min(w, x + ZOOM_PX)), int(min(h, y + ZOOM_PX))
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(rgb[y0:y1, x0:x1], extent=(x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5), interpolation="nearest")
        ax.set_title(f"{label} {i} of {len(rough)}: click exactly on the point", fontsize=10)
        p = plt.ginput(1, timeout=0)
        plt.close(fig)
        out.append(p[0] if p else (x, y))
    return out


def click_reference(rgb, n, zoom=True):
    what = "the 4 corners of the reference, in order around it" if n == 4 else "the 2 ends of the reference length"
    pts = click(rgb, n, f"Click {what}")
    return refine(rgb, pts, "Reference point") if zoom else pts


def click_roi(rgb):
    (ax, ay), (bx, by) = click(rgb, 2, "Click two opposite corners of a box around the crack")
    return min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)


# ---------- main ----------

def _other_reference(args, rgb, roi, mask):
    """Scale card bar, coin, A4 ...: points clicked or given, no tilt check."""
    from measure.analyse import reference_mask
    if args.ref_size:
        kind_n = 4
    elif args.ref_length:
        kind_n = 2
    else:
        kind_n = 4 if PRESETS[args.ref][0] == "rect" else 2
    pts = parse_points(args.ref_points) if args.ref_points else click_reference(rgb, kind_n, not args.no_zoom)
    if len(pts) != kind_n:
        sys.exit(f"--ref-points needs {kind_n} points for this reference")
    if args.ref_size:
        w, h = (float(v) for v in args.ref_size.lower().split("x"))
        scale = from_rectangle(pts, (w, h), f"rectangle {w:g} x {h:g} mm")
    elif args.ref_length:
        scale = from_length(pts[0], pts[1], args.ref_length, f"length {args.ref_length:g} mm")
    else:
        scale = from_preset(args.ref, pts)
    grey = np.asarray(Image.fromarray(rgb).convert("L"), dtype=float)
    if mask is None:
        excl = reference_mask(grey.shape, scale.points_px, args.background_px) if scale.kind == "rect" else None
        mask = segment(grey, roi, excl, args.background_px, args.threshold)
    centre = scale.to_mm(np.mean(scale.points_px, axis=0))[0]
    result = measure(grey, mask, scale.to_mm, scale.rel_uncertainty, centre)
    report = {"version": VERSION, "reference": scale.to_dict(), "roi_px": [round(float(v), 1) for v in roi],
              "crack": result.to_dict(), "matrix_input": result.matrix_input(),
              "note": "Measurement only, never a safety statement. No tilt check with this reference."}
    return report, result, scale.points_px, list(result.flags)


def run(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("photo", nargs="?", type=Path)
    p.add_argument("--ref", choices=sorted(PRESETS), default="card")
    p.add_argument("--ref-size", help="own rectangle in mm, e.g. 100x70 (4 corners)")
    p.add_argument("--ref-length", type=float, help="own length in mm (2 points)")
    p.add_argument("--ref-points", help='"x,y;x,y;..." in pixels (for a card: skips the automatic search)')
    p.add_argument("--roi", help='search box "x0,y0,x1,y1" in pixels, or "full" (default for the card)')
    p.add_argument("--click", action="store_true", help="click the card corners and the search box")
    p.add_argument("--mask", type=Path, help="crack mask image (white = crack) instead of the built-in finder")
    p.add_argument("--all-pieces", action="store_true", help="keep every crack piece, not only the longest")
    p.add_argument("--background-px", type=int, default=BACKGROUND_PX, help="wider than the crack in pixels")
    p.add_argument("--threshold", type=float, help="fixed black-hat threshold instead of Otsu")
    p.add_argument("--no-zoom", action="store_true", help="skip the zoomed second click per reference point")
    p.add_argument("--no-show", action="store_true", help="do not open windows")
    p.add_argument("--demo", action="store_true", help="synthetic photo with a known answer")
    p.add_argument("--demo-width", type=float, default=1.0, help="crack width of the demo photo in mm")
    p.add_argument("--demo-tilt", type=float, default=12.0, help="left-right tilt of the demo camera in degrees")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "measurements")
    args = p.parse_args(argv)

    truth = None
    if args.demo:
        grey, truth = make_demo(args.demo_width, tilt_lr_deg=args.demo_tilt)
        rgb = np.repeat(grey.astype(np.uint8)[..., None], 3, axis=2)
        stem = f"demo_{args.demo_width:g}mm"
        args.roi = args.roi or ",".join(map(str, truth["roi_px"]))
    elif args.photo is None:
        p.error("give a photo, or --demo")
    else:
        rgb = load_photo(args.photo)
        stem = args.photo.stem

    h, w = rgb.shape[:2]
    if args.roi and args.roi != "full":
        roi = tuple(float(v) for v in args.roi.split(","))
        if len(roi) != 4:
            p.error('--roi needs "x0,y0,x1,y1" or "full"')
    elif args.click and not args.roi:
        roi = None  # clicked below, after the reference
    else:
        roi = (0, 0, w, h)
    mask = load_mask(args.mask, rgb.shape) if args.mask else None

    is_card = args.ref == "card" and not args.ref_size and not args.ref_length
    if is_card:
        corners = parse_points(args.ref_points) if args.ref_points else None
        if corners is None and args.click:
            corners = click_reference(rgb, 4, not args.no_zoom)
        if roi is None:
            roi = click_roi(rgb)
        a = analyse(rgb, roi, corners, mask, longest_only=not args.all_pieces, background_px=args.background_px,
                    threshold=args.threshold)
        if a.card is None and not args.no_show and not args.click:
            print("Bank card not found automatically: click its 4 corners.")
            a = analyse(rgb, roi, click_reference(rgb, 4, not args.no_zoom), mask,
                        longest_only=not args.all_pieces, background_px=args.background_px, threshold=args.threshold)
        report, result = a.to_dict(), a.crack
        ref_px = None if a.card is None else a.card.corners_px
        lines = [f"Outcome: {a.outcome}"] + [f"RETAKE: {r}" for r in a.retake] + [f"! {x}" for x in a.warnings]
        if a.card is not None:
            lines.insert(1, f"Card tilt: {a.card.tilt_deg:.1f} degrees ({a.card.status})")
    else:
        if roi is None:
            roi = click_roi(rgb)
        report, result, ref_px, flags = _other_reference(args, rgb, roi, mask)
        lines = [f"! {x}" for x in flags]

    report["photo"] = str(args.photo) if args.photo else stem
    if truth:
        report["demo_truth"] = {"width_mm": truth["width_mm"], "length_mm": round(truth["length_mm"], 1),
                                "tilt_deg": round(truth["tilt_deg"], 1)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{stem}_measurement.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    draw_overlay(rgb, ref_px, roi, result, f"{stem}  ·  reference: {args.ref}", args.out / f"{stem}_overlay.png",
                 show=not args.no_show, extra_lines=lines)

    print(json.dumps(report, indent=2))
    if truth:
        print(f"demo truth: width {truth['width_mm']:g} mm, length {truth['length_mm']:.1f} mm, "
              f"tilt {truth['tilt_deg']:.1f} degrees")
    print(f"written: {args.out / (stem + '_measurement.json')}, {args.out / (stem + '_overlay.png')}")
    return report


if __name__ == "__main__":
    run()
