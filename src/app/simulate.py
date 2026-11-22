"""Simulated residents for the class demo (D59): drops test cases into data/inbox/, one every few seconds; the running
app picks them up, puts them on the wall (marked TEST) and writes them for Grasshopper.

Usage, from the repository root, with the app running (py src/app/launch.py):
    py src/app/simulate.py              # 10 real two-photo cases (after pick_cases.py), else the synthetic ones
    py src/app/simulate.py --synthetic  # 10 cases with a synthetic photo 1 and a bank card (D61)
    py src/app/simulate.py --n 4 --delay 2
    py src/app/simulate.py --mixed      # also real crack photos without a card, a wall without a crack, a building
    py src/app/simulate.py --by-level   # about 10 real cases per risk tab (after pick_by_level.py, D79)
    py src/app/simulate.py --no-field   # without the team's own field photos
    py src/app/simulate.py --showcase   # also the six drawn showcase cases (photo 1 only, D90, D91)
    py src/app/simulate.py --test-set   # the 74 real survey cases reviewed by eye (D94), after the field photos

Every run first sends the team's own field photo pairs with a bank card (data/field/<folder>/pair_N_crack.jpg and
pair_N_element.jpg, local only, never committed; D87), so real cards are always in the population.
Two photos are sent together only when both show the same real crack (D91): the team's field pairs, and the real survey
cases whose close-up is cut from the element photo. A drawn photo 1 is never paired with an unrelated photo 2.

Drawn crack photos (--synthetic, --showcase) come alone, without an element photo: they test the measurement on a
known width, they are not evidence about a real building (D91).
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
FIELD = ROOT / "data" / "field"  # the team's own field photos with a bank card (D87), local only
BY_LEVEL = ROOT / "data" / "inbox_cases" / "by_level.json"
TEST_SET = ROOT / "data" / "inbox_cases" / "test_set.json"  # D94: real cases reviewed by eye, local only
TOPO_TEST = ROOT / "data" / "raw" / "epfl_topo" / "data" / "test_set" / "images"  # [108]: cracked buildings, local only
# D101, reviewed by eye on 22 Nov: the close-up shows a real crack of the building in the far photo. Left out: the
# close-up cut around a roof edge, the sky or a shutter (DSC_0123, IMG_4237, IMG_6845, KB0739).
TOPO_KEEP = {"AD0021.png", "AD0043.png", "DSC_0086.png", "IMG_1400.png", "SFR_1955.png"}

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
        yield (crack, None,  # a drawn photo is never paired with an unrelated real element photo (D91)
               {"card_placed": True, "kind": f"DRAWN: {shape} crack {width} mm with a card (photo 1 only)"})


def real_cases(n: int, picks_file: Path | None = None):
    """Real two-photo cases picked by src/app/pick_cases.py (D63) or pick_by_level.py (D79): photo 2 the whole photo
    of the damaged element, photo 1 the close-up cut from it around the same crack. Local photos only, never copied
    into the repository."""
    from PIL import ImageOps
    picks = json.loads((picks_file or PICKS).read_text(encoding="utf-8"))["cases"]
    for case in picks[:n]:
        if case.get("synthetic"):  # a filled tab (D79): regenerated from its parameters, said so on the wall
            from app.pick_by_level import synthetic_case
            yield synthetic_case(case)
            continue
        with Image.open(ROOT / case["photo"]) as im:
            full = ImageOps.exif_transpose(im).convert("RGB")
        close_up = full.crop(tuple(case["crop"]))
        whole = full.copy()
        whole.thumbnail((2400, 2400))  # under the app's 10 MB limit; the close-up keeps full resolution
        yield close_up, whole, {"card_placed": False,
                                "kind": f"REAL: {case['element']} from a 2010 survey photo, same crack in both photos "
                                        "(local only, D63)"}


def all_real_cases():
    """Every real two-photo case set, each case once (D101): the test set reviewed by eye (D94), the cases per risk tab
    (D79) and the earlier picks (D63), by their photo and close-up."""
    seen, out = set(), []
    for f in (TEST_SET, BY_LEVEL, PICKS):
        if not f.exists():
            continue
        for case in json.loads(f.read_text(encoding="utf-8"))["cases"]:
            key = (case.get("photo"), tuple(case.get("crop") or ()))
            if case.get("synthetic") or key in seen:
                continue
            seen.add(key)
            out.append(case)
    return out


def topo_cases(folder: Path = TOPO_TEST):
    """The EPFL test photos of cracked buildings after earthquakes ([108], CC BY 4.0, D101): photo 2 the whole photo
    (a facade, often more than one element), photo 1 the close-up cut around the crack the crack model finds there.
    Local only; the photos are never copied into the repository."""
    from PIL import ImageOps
    if not folder.is_dir():
        return
    import torch
    from app.pick_cases import crop_box
    from element import photo2
    from segmentation.train import load_checkpoint
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_checkpoint(ROOT / "models" / "unet_crack.pt", dev)
    for path in sorted(folder.glob("*.png")):
        if path.name not in TOPO_KEEP:
            continue
        with Image.open(path) as im:
            full = ImageOps.exif_transpose(im).convert("RGB")
        small = full.copy()
        small.thumbnail((1100, 1100))
        on = photo2.read(model, small, dev)  # only to place the close-up
        if on.bbox_px is None:
            continue
        close_up = full.crop(crop_box(on.bbox_px, on.size_px, full.size))
        whole = full.copy()
        whole.thumbnail((2400, 2400))
        yield close_up, whole, {"card_placed": False, "photo": path.name,
                                "kind": "REAL: cracked building after an earthquake, EPFL test photo [108], close-up "
                                        "cut around its crack (local only, D101)"}


def field_cases(folder: Path = FIELD):
    """The team's own field photos (D87): every pair_N_crack.jpg with its pair_N_element.jpg, in any subfolder of
    data/field/. A crack photo without its element photo is sent alone. Local only, never copied into the repository
    (rule 9); the wall shows them as field tests, without the photographer's name."""
    from PIL import ImageOps
    if not folder.is_dir():
        return
    for crack_path in sorted(folder.glob("*/pair_*_crack.jpg")):
        element_path = crack_path.with_name(crack_path.name.replace("_crack.jpg", "_element.jpg"))
        with Image.open(crack_path) as im:
            crack = ImageOps.exif_transpose(im).convert("RGB")
        element = None
        if element_path.exists():
            with Image.open(element_path) as im:
                element = ImageOps.exif_transpose(im).convert("RGB")
        pair = crack_path.name.split("_")[1]
        yield crack, element, {"card_placed": True,
                               "kind": f"FIELD: team photo pair {pair} with a bank card (local only, D87)"}


def showcase_cases():
    """The drawn showcase cases (D90), each with a real PHI-Net element photo as photo 2 when the data is here."""
    from app import showcase
    for crack, _element, _index, name in showcase.cases():
        yield (Image.fromarray(crack), None,  # never paired with an unrelated real element photo (D91)
               {"card_placed": True, "kind": f"DRAWN: {name} (photo 1 only, with a card)"})


def mixed_cases():
    ozg = sorted(OZGENEL.glob("images/ozg_*"))
    masks = OZGENEL / "masks"
    with_crack = [p for p in ozg if (masks / p.name).exists() and np.asarray(Image.open(masks / p.name)).any()][:3]
    without = [p for p in ozg if (masks / p.name).exists() and not np.asarray(Image.open(masks / p.name)).any()][:1]
    for p in with_crack:
        yield Image.open(p), None, {"card_placed": False, "kind": "real crack close-up, no card (photo 1 only)"}
    for p in without:
        yield Image.open(p), None, {"card_placed": False, "kind": "real wall without a crack"}
    yield _phinet(PHINET1, 2, 3), None, {"card_placed": False, "kind": "a whole building"}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=10, help="cracks with a bank card (each gets a 3D model)")
    ap.add_argument("--delay", type=float, default=3.0, help="seconds between residents")
    ap.add_argument("--mixed", action="store_true", help="also real photos without a card, no crack, a building")
    ap.add_argument("--synthetic", action="store_true", help="the synthetic photo 1 cases (D61) instead of the real ones")
    ap.add_argument("--by-level", action="store_true", help="the real cases per risk tab from pick_by_level.py (D79)")
    ap.add_argument("--no-field", action="store_true", help="leave out the team's own field photos (D87)")
    ap.add_argument("--test-set", action="store_true", help="the real test set reviewed by eye (D94)")
    ap.add_argument("--showcase", action="store_true", help="also the six drawn showcase cases, photo 1 only (D90, D91)")
    ap.add_argument("--everything", action="store_true",
                    help="every real case once: field photos, test set, risk tabs, picks, EPFL buildings, and the "
                         "photos without a card, without a crack and of a whole building (D101)")
    args = ap.parse_args(argv)
    INBOX.mkdir(parents=True, exist_ok=True)
    field = ([] if args.no_field else list(field_cases())) + (list(showcase_cases()) if args.showcase else [])
    # D102: every real case is the default population; the older, smaller sets only when asked for by name
    if args.everything or not (args.test_set or args.by_level or args.synthetic or args.mixed):
        tmp = ROOT / "data" / "inbox_cases" / "_everything.json"  # local, rebuilt each run
        tmp.write_text(json.dumps({"cases": all_real_cases()}), encoding="utf-8")
        cases = field + list(real_cases(10_000, tmp)) + list(topo_cases()) + list(mixed_cases())
        print(f"{len(cases)} real cases into {INBOX}: field photos, survey photos, EPFL buildings, and photos "
              "without a card, without a crack and of a whole building.")
        stamp = time.strftime("%H%M%S")
        for i, (crack, element, meta) in enumerate(cases, 1):
            _write(INBOX / f"{stamp}_{i:03d}", crack, element, meta)
            print(f"  resident {i}: {meta['kind']}", flush=True)
            time.sleep(args.delay)
        print("Done. Wall: http://localhost:8501/?wall")
        return
    if args.test_set:
        if not TEST_SET.exists():
            sys.exit(f"No test set yet ({TEST_SET}): py src/app/pick_test_set.py data/raw/haiti2010/Photos")
        cases = field + list(real_cases(10_000, TEST_SET))
        print(f"{len(cases)} real cases (the field photos and the test set) into {INBOX}.")
        stamp = time.strftime("%H%M%S")
        for i, (crack, element, meta) in enumerate(cases, 1):
            _write(INBOX / f"{stamp}_{i:02d}", crack, element, meta)
            time.sleep(args.delay)
        print("Done. Wall: http://localhost:8501/?wall")
        return
    if args.by_level:
        if not BY_LEVEL.exists():
            sys.exit(f"No cases per risk tab yet ({BY_LEVEL}): py src/app/pick_by_level.py data/raw/haiti2010/Photos")
        cases = field + list(real_cases(10_000, BY_LEVEL))
        print(f"{len(cases)} real cases, about 10 per risk tab, into {INBOX}.")
        stamp = time.strftime("%H%M%S")
        for i, (crack, element, meta) in enumerate(cases, 1):
            _write(INBOX / f"{stamp}_{i:02d}", crack, element, meta)
            time.sleep(args.delay)
        print("Done. Wall: http://localhost:8501/?wall")
        return
    real = PICKS.exists() and not args.synthetic
    if not real and not args.synthetic:
        print(f"No real cases picked yet ({PICKS}); using the synthetic ones. To pick real ones: "
              "py src/app/pick_cases.py data/raw/haiti2010/Photos")
    cases = field + (list(real_cases(args.n)) if real else list(card_cases(args.n))) + (
        list(mixed_cases()) if args.mixed else [])
    print(f"{len(cases)} simulated residents into {INBOX}; the running app picks them up within a second.")
    stamp = time.strftime("%H%M%S")
    for i, (crack, element, meta) in enumerate(cases, 1):
        _write(INBOX / f"{stamp}_{i:02d}", crack, element, meta)
        print(f"  resident {i}: {meta['kind']}", flush=True)
        time.sleep(args.delay)
    print("Done. Wall: http://localhost:8501/?wall  (click Crack found or Expert needed, then a photo for its 3D model)")


if __name__ == "__main__":
    main()
