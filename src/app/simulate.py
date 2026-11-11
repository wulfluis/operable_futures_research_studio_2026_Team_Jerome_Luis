"""Simulated residents for the class demo (D59): drops test cases into data/inbox/, one every few seconds; the running
app picks them up, puts them on the wall (marked TEST) and writes them for Grasshopper.

Usage, from the repository root, with the app running (py src/app/launch.py):
    py src/app/simulate.py              # 10 real two-photo cases (after pick_cases.py), else the synthetic ones
    py src/app/simulate.py --synthetic  # 10 cases with a synthetic photo 1 and a bank card (D61)
    py src/app/simulate.py --n 4 --delay 2
    py src/app/simulate.py --mixed      # also real crack photos without a card, a wall without a crack, a building

Crack photos with a card are synthetic (a drawn wall, card and crack of known width; no public photo has a card).
Element photos are real, from the PHI-Net task 6 test set (CC BY-NC-SA 4.0, research and teaching only).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from app.inbox import INBOX  # noqa: E402
from measure.synthetic import make_demo  # noqa: E402

PICKS = ROOT / "data" / "inbox_cases" / "picks.json"

PHINET6 = ROOT / "data" / "raw" / "phinet" / "task6_component_type"
PHINET1 = ROOT / "data" / "raw" / "phinet" / "task1_scene_level"
OZGENEL = ROOT / "data" / "processed" / "segmentation" / "test"
# crack centre lines in the wall plane (mm, y down), clear of the card (x 15 to 101, y 60 to 114)
SHAPES = {  # they wander and stay inside the photo, as real cracks do (a straight line to the edge is a corner, D51)
    "vertical": [(125, 15), (140, 55), (133, 95), (160, 140)],
    "diagonal": [(105, 15), (122, 30), (130, 48), (150, 62), (160, 82), (182, 98), (192, 115)],  # about 45 degrees
    "horizontal": [(30, 25), (60, 33), (85, 24), (115, 34), (145, 26), (175, 33)],
    "steep": [(150, 10), (156, 50), (148, 95), (158, 140)],
}
# Ten logical cases (D61): photo 2 is a real PHI-Net task 6 test photo of a damaged element (index) on which the crack
# model finds a large crack; photo 1 is synthetic with a bank card, its crack in the same direction as on photo 2, so
# the type, the level and the 3D drawing agree. Chosen because the element model reads the element right and the
# scene check sees a whole element: the demo shows the flow, not the models' accuracy (that is in the experiments).
# (photo 1 crack shape, width mm, element, PHI-Net test index for photo 2, camera tilt left-right, up-down)
CASES = [
    ("horizontal", 1.5, "column", 119, 8, -4), ("horizontal", 0.8, "column", 115, 10, -6),
    ("vertical", 1.2, "column", 98, 6, 3), ("vertical", 2.5, "column", 84, 5, 5),
    ("steep", 0.6, "beam", 35, 9, -3), ("diagonal", 1.2, "beam", 42, 7, 6),
    ("horizontal", 1.0, "beam", 9, 11, -2), ("diagonal", 0.9, "wall", 400, 4, -7),
    ("diagonal", 2.5, "wall", 375, 10, 4), ("horizontal", 1.2, "wall", 459, 12, -5),
]
ELEMENT_CODE = {"beam": 0, "column": 1, "others": 2, "wall": 3}  # PHI-Net task 6 numbering


def _phinet(folder: Path, code: int, k: int) -> Image.Image:
    x = np.load(folder / "X_test.npy", mmap_mode="r")
    y = np.load(folder / "y_test.npy").astype(int)
    idx = np.flatnonzero(y == code)
    return Image.fromarray(np.asarray(x[idx[(k * 7) % len(idx)]])).resize((672, 672), Image.BICUBIC)


def _element_photo(index: int) -> Image.Image:
    """A real PHI-Net task 6 test photo (224 px), enlarged for the app."""
    x = np.load(PHINET6 / "X_test.npy", mmap_mode="r")
    return Image.fromarray(np.asarray(x[index])).resize((672, 672), Image.BICUBIC)


def _write(case_dir: Path, crack: Image.Image, element: Image.Image | None, meta: dict) -> None:
    tmp = case_dir.with_name(case_dir.name + ".part")
    tmp.mkdir(parents=True)
    crack.convert("RGB").save(tmp / "crack.jpg", quality=92)
    if element is not None:
        element.convert("RGB").save(tmp / "element.jpg", quality=92)
    (tmp / "case.json").write_text(json.dumps(meta), encoding="utf-8")  # written last: the case is complete
    tmp.rename(case_dir)


def card_cases(n: int):
    for k in range(n):
        shape, width, element, index, lr, ud = CASES[k % len(CASES)]
        grey, _ = make_demo(width, seed=k, tilt_lr_deg=lr, tilt_ud_deg=ud, crack_mm=SHAPES[shape])
        crack = Image.fromarray(np.repeat(grey.astype(np.uint8)[..., None], 3, axis=2))
        yield (crack, _element_photo(index),
               {"card_placed": True, "kind": f"photo 1: synthetic {shape} crack {width} mm with a card; photo 2: real "
                                             f"damaged {element}, PHI-Net test photo {index}"})


def real_cases(n: int):
    """Real two-photo cases picked by src/app/pick_cases.py (D63): photo 2 the whole photo of the damaged element,
    photo 1 the close-up cut from it around the same crack. Local photos only, never copied into the repository."""
    from PIL import ImageOps
    picks = json.loads(PICKS.read_text(encoding="utf-8"))["cases"]
    for case in picks[:n]:
        with Image.open(ROOT / case["photo"]) as im:
            full = ImageOps.exif_transpose(im).convert("RGB")
        close_up = full.crop(tuple(case["crop"]))
        whole = full.copy()
        whole.thumbnail((2400, 2400))  # under the app's 10 MB limit; the close-up keeps full resolution
        yield close_up, whole, {"card_placed": False,
                                "kind": f"REAL: {case['element']} from a 2010 survey photo, same crack in both photos "
                                        "(local only, D63)"}


def mixed_cases():
    ozg = sorted(OZGENEL.glob("images/ozg_*"))
    masks = OZGENEL / "masks"
    with_crack = [p for p in ozg if (masks / p.name).exists() and np.asarray(Image.open(masks / p.name)).any()][:3]
    without = [p for p in ozg if (masks / p.name).exists() and not np.asarray(Image.open(masks / p.name)).any()][:1]
    for p in with_crack:
        yield Image.open(p), _element_photo(400), {"card_placed": False,
                                                                          "kind": "real crack close-up, no card"}
    for p in without:
        yield Image.open(p), None, {"card_placed": False, "kind": "real wall without a crack"}
    yield _phinet(PHINET1, 2, 3), None, {"card_placed": False, "kind": "a whole building"}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=10, help="cracks with a bank card (each gets a 3D model)")
    ap.add_argument("--delay", type=float, default=3.0, help="seconds between residents")
    ap.add_argument("--mixed", action="store_true", help="also real photos without a card, no crack, a building")
    ap.add_argument("--synthetic", action="store_true", help="the synthetic photo 1 cases (D61) instead of the real ones")
    args = ap.parse_args(argv)
    INBOX.mkdir(parents=True, exist_ok=True)
    real = PICKS.exists() and not args.synthetic
    if not real and not args.synthetic:
        print(f"No real cases picked yet ({PICKS}); using the synthetic ones. To pick real ones: "
              "py src/app/pick_cases.py data/raw/haiti2010/Photos")
    cases = (list(real_cases(args.n)) if real else list(card_cases(args.n))) + (list(mixed_cases()) if args.mixed
                                                                                else [])
    print(f"{len(cases)} simulated residents into {INBOX}; the running app picks them up within a second.")
    stamp = time.strftime("%H%M%S")
    for i, (crack, element, meta) in enumerate(cases, 1):
        _write(INBOX / f"{stamp}_{i:02d}", crack, element, meta)
        print(f"  resident {i}: {meta['kind']}", flush=True)
        time.sleep(args.delay)
    print("Done. Wall: http://localhost:8501/?wall  (click Crack found or Expert needed, then a photo for its 3D model)")


if __name__ == "__main__":
    main()
