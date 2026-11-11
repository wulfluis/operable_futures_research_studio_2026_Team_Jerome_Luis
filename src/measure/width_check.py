"""Compare the app's crack width in mm with a known width or a by-eye reading on the gauge.
Spec: spec/features/measurement.md ("Width check").

Usage, from the repository root:
    py src/measure/width_check.py <folder with the photos> <truth.csv>

truth.csv, one row per photo. Columns: photo, kind, true_mm, nominal_mm, gauge_mm, roi, group, note.
  kind "gauge": a photo of the whole printed gauge (docs/width_gauge) with a bank card in its frame, the sheet upright
                in the photo (title at the top) and the card's long side horizontal. Every line is found from the
                card's position and measured: one row gives up to eight measurements. The card must be darker than the
                paper (a white card cannot be told from the empty printed frame).
  kind "line":  one line of the gauge, searched by hand. Give nominal_mm (0.5, 1, 2 ...; the width the printer really
                lays down is used) or true_mm. roi "x0,y0,x1,y1" in photo pixels, a box around that one line.
  kind "crack": a real crack, measured the way the app does it (U-Net and bank card). gauge_mm is the width read by
                eye on the gauge at the widest point; group names the crack when it is photographed several times.
Writes results.csv and report.md to data/results/width_check/<date and time>/ (ignored by git: photos are never
committed, AGENTS.md rule 9). The report says what the check does and does not show; it sets no pass mark.
"""

import argparse
import csv
import statistics
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT / "src"))
from measure import card as card_mod
from measure import gauge_sheet
from measure.analyse import analyse
from measure.crack import BACKGROUND_PX
from measure.reference import from_rectangle
from measure.gauge_sheet import printed_width_mm

KINDS = ("gauge", "line", "crack")


def parse_roi(text: str | None) -> tuple[float, float, float, float] | None:
    if not text or not text.strip():
        return None
    parts = [float(v) for v in text.replace(";", ",").split(",")]
    if len(parts) != 4 or parts[2] <= parts[0] or parts[3] <= parts[1]:
        raise ValueError(f"roi must be 'x0,y0,x1,y1' with x1 > x0 and y1 > y0, got {text!r}")
    return tuple(parts)  # type: ignore[return-value]


def _num(text: str | None) -> float | None:
    return float(text) if text is not None and text.strip() != "" else None


def read_rows(csv_path: Path) -> list[dict]:
    """The truth table. A bad row stops the run with its line number, never a silent skip."""
    rows = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        for n, r in enumerate(csv.DictReader(f), start=2):
            try:
                kind = (r.get("kind") or "").strip()
                if kind not in KINDS:
                    raise ValueError(f"kind must be one of {KINDS}")
                if not (r.get("photo") or "").strip():
                    raise ValueError("photo is empty")
                row = {"photo": r["photo"].strip(), "kind": kind, "group": (r.get("group") or "").strip(),
                       "note": (r.get("note") or "").strip(), "roi": parse_roi(r.get("roi"))}
                if kind == "gauge":
                    row["expected_mm"] = None  # one expected width per line of the sheet, found later
                elif kind == "line":
                    true, nominal = _num(r.get("true_mm")), _num(r.get("nominal_mm"))
                    if true is None and nominal is None:
                        raise ValueError("a line needs nominal_mm or true_mm")
                    row["expected_mm"] = true if true is not None else printed_width_mm(nominal)
                else:
                    row["expected_mm"] = _num(r.get("gauge_mm"))  # may be empty: repeatability still counts
                rows.append(row)
            except ValueError as e:
                raise ValueError(f"{csv_path.name} line {n}: {e}") from e
    return rows


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(ImageOps.exif_transpose(img).convert("RGB"))


def _empty(row: dict, outcome: str) -> dict:
    return {**{k: row[k] for k in ("photo", "kind", "group", "expected_mm", "note")}, "median_mm": None, "max_mm": None,
            "uncertainty_mm": None, "px_across": None, "tilt_deg": None, "scale_ratio": None, "outcome": outcome}


def analyse_line(row: dict, rgb: np.ndarray, background_px: int = BACKGROUND_PX, card_corners=None) -> dict:
    a = analyse(rgb, roi=row["roi"], background_px=background_px, card_corners=card_corners)
    out = _empty(row, "")
    out["tilt_deg"] = None if a.card is None else round(a.card.tilt_deg, 1)
    c = a.crack
    if a.retake:
        out["outcome"] = "retake: " + a.retake[0][:90]
    elif c is None or not c.found or c.width_max_mm is None:
        out["outcome"] = "not measured"
    else:
        out.update(median_mm=c.width_median_mm, max_mm=c.width_max_mm, uncertainty_mm=c.width_uncertainty_mm,
                   px_across=c.px_across, outcome="measured")
    return out


def find_sheet(rgb: np.ndarray) -> np.ndarray | None:
    """Homography from the gauge's own millimetres to photo pixels, from its four corner markers (ArUco).
    None unless all four are found."""
    grey = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(gauge_sheet.DICTIONARY),
                                       cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(grey)
    if ids is None:
        return None
    ref = gauge_sheet.marker_corners_mm()
    src, dst, seen = [], [], set()
    for c, i in zip(corners, ids.ravel()):
        i = int(i)
        if i in ref and i not in seen:  # a repeated id is a stray marker: keep the first
            seen.add(i)
            src += ref[i]
            dst += [tuple(pt) for pt in c.reshape(4, 2)]
    if len(seen) < 4:
        return None
    H, _ = cv2.findHomography(np.float32(src), np.float32(dst), 0)
    return H


def _project(H: np.ndarray, pts) -> np.ndarray:
    return cv2.perspectiveTransform(np.float32([list(pts)]), np.asarray(H, dtype=np.float64))[0]


def gauge_boxes(H: np.ndarray, margin_mm: float = 3.0, inset_mm: float = 5.0):
    """[(nominal_mm, (x0, y0, x1, y1))]: a search box around each line of the gauge, in photo pixels."""
    x0, x1 = gauge_sheet.LINE_X_MM + inset_mm, gauge_sheet.LINE_X_MM + gauge_sheet.LINE_LEN_MM - inset_mm
    boxes = []
    for i, nominal in enumerate(gauge_sheet.LINES_MM):
        y = gauge_sheet.FIRST_LINE_Y_MM + i * gauge_sheet.LINE_STEP_MM
        pts = _project(H, [(x0, y - margin_mm), (x1, y - margin_mm), (x1, y + margin_mm), (x0, y + margin_mm)])
        boxes.append((nominal, (float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max()))))
    return boxes


def _frame_crop(H: np.ndarray, shape, pad_mm: float = 12.0) -> tuple[int, int, int, int]:
    """The photo around the card's frame, so the card finder sees the card and not the sheet's edges."""
    fx0, fy0, fx1, fy1 = gauge_sheet.CARD_FRAME_MM
    q = _project(H, [(fx0 - pad_mm, fy0 - pad_mm), (fx1 + pad_mm, fy0 - pad_mm), (fx1 + pad_mm, fy1 + pad_mm), (fx0 - pad_mm, fy1 + pad_mm)])
    h, w = shape[:2]
    return (max(0, int(np.floor(q[:, 0].min()))), max(0, int(np.floor(q[:, 1].min()))),
            min(w, int(np.ceil(q[:, 0].max()))), min(h, int(np.ceil(q[:, 1].max()))))


def _card_is_solid(rgb: np.ndarray, corners_px, frame_view: np.ndarray, min_gap: float = 30.0) -> bool:
    """A card, not the printed dashed frame: the inside of the quad is clearly darker than the paper around it
    (min_gap grey levels, illustrative). A white card on white paper cannot be told from an empty frame, so it is refused."""
    grey = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)
    q = np.asarray(corners_px, dtype=np.float32)
    inner = (q.mean(axis=0) + 0.7 * (q - q.mean(axis=0))).round().astype(np.int32)  # the middle of the card, away from its edge
    mask = np.zeros(grey.shape, np.uint8)
    cv2.fillConvexPoly(mask, inner, 1)
    if not mask.any():
        return False
    paper = float(np.percentile(cv2.cvtColor(np.ascontiguousarray(frame_view), cv2.COLOR_RGB2GRAY), 90))
    return float(grey[mask > 0].mean()) < paper - min_gap


def measure_gauge(row: dict, rgb: np.ndarray) -> list[dict]:
    """Every line of a photographed gauge, each against the width the printer really laid down. The lines are found
    from the sheet's markers; the width is measured with the bank card's scale, as the app does; and the card's scale
    is compared with the sheet's own scale (scale_ratio, 1.00 = they agree)."""
    H = find_sheet(rgb)
    if H is None:
        return [_empty({**row, "expected_mm": None}, "gauge markers not found: all four corner markers must be in the photo")]
    cx0, cy0, cx1, cy1 = _frame_crop(H, rgb.shape)
    # The card finder checks its candidates with the camera model of the whole photo, so it gets the whole photo with
    # everything outside the card's frame painted white (a crop would change the model, and the sheet's edges and lines
    # would look like card-shaped boxes).
    view = np.full_like(rgb, 255)
    view[cy0:cy1, cx0:cx1] = rgb[cy0:cy1, cx0:cx1]
    card = card_mod.detect_card(view) if cx1 - cx0 > 20 and cy1 - cy0 > 20 else None
    if card is None:
        return [_empty({**row, "expected_mm": None}, "no bank card found in the frame: use a dark or coloured card lying flat in it")]
    if not _card_is_solid(rgb, card.corners_px, rgb[cy0:cy1, cx0:cx1]):
        return [_empty({**row, "expected_mm": None}, "no bank card found in the frame: it is empty or the card is as light as the paper")]
    scale = from_rectangle(card.corners_px, card_mod.CARD_MM, "bank card", long_first=card.long_first)
    Hinv = np.linalg.inv(H)
    out = []
    for nominal, roi in gauge_boxes(H):
        expected = printed_width_mm(nominal)
        centre = np.array([(roi[0] + roi[2]) / 2, (roi[1] + roi[3]) / 2])
        step = np.array([20.0, 0.0])
        mm_sheet = float(np.linalg.norm(np.diff(_project(Hinv, [tuple(centre), tuple(centre + step)]), axis=0))) / 20.0
        ratio = scale.mm_per_px(centre, (1.0, 0.0)) / mm_sheet
        px_per_mm = 1.0 / mm_sheet
        wide = max(BACKGROUND_PX, int(1.6 * expected * px_per_mm) + 10)  # the finder's window must be wider than the line
        line = {**row, "kind": "line", "roi": roi, "expected_mm": expected, "note": f"gauge line {nominal:g} mm"}
        result = analyse_line(line, rgb, wide, card.corners_px)
        result["scale_ratio"] = round(ratio, 4)
        out.append(result)
    return out


def measure_crack(row: dict, photo: Image.Image, model, device: str) -> dict:
    from app.core import run_measured  # the very pipeline the app uses
    r = run_measured(photo, model, device, card_placed=True)  # every width-check photo holds the card (D56)
    out = _empty(row, "")
    if r.card is None:
        out["outcome"] = "no bank card found"
        return out
    out["tilt_deg"] = round(r.card.tilt_deg, 1)
    if r.measured:
        c = r.crack
        out.update(median_mm=c.width_median_mm, max_mm=c.width_max_mm, uncertainty_mm=c.width_uncertainty_mm,
                   px_across=c.px_across, outcome="measured")
    else:
        out["outcome"] = "not measured: " + (r.measure_note or "no crack found")[:90]
    return out


def check(photos: Path, rows: list[dict], load_model=None, device: str = "cpu") -> list[dict]:
    model = None
    results = []
    for row in rows:
        path = photos / row["photo"]
        if not path.is_file():
            raise FileNotFoundError(f"{row['photo']} is not in {photos}")
        if row["kind"] == "gauge":
            results.extend(measure_gauge(row, load_rgb(path)))
        elif row["kind"] == "line":
            results.append(analyse_line(row, load_rgb(path)))
        else:
            if model is None:
                model = load_model()
            with Image.open(path) as img:
                results.append(measure_crack(row, img.copy(), model, device))
    return results


# ---------- numbers ----------

def line_stats(results: list[dict]) -> dict | None:
    """Measured width (the median along the line) against the printed width."""
    done = [r for r in results if r["kind"] == "line" and r["outcome"] == "measured"]
    if not done:
        return None
    err = [r["median_mm"] - r["expected_mm"] for r in done]
    by = {}
    for r, e in zip(done, err):
        by.setdefault(round(r["expected_mm"], 3), []).append(e)
    return {"n": len(done), "bias_mm": statistics.fmean(err), "sd_mm": statistics.stdev(err) if len(err) > 1 else None,
            "mean_abs_mm": statistics.fmean(abs(e) for e in err),
            "within_uncertainty": sum(abs(e) <= r["uncertainty_mm"] for r, e in zip(done, err)),
            "by_width": [(w, len(v), statistics.fmean(v)) for w, v in sorted(by.items())]}


def repeatability(results: list[dict]) -> dict | None:
    """Spread of the widest-point width over repeated photos of the same crack (needs no truth)."""
    groups: dict[str, list[float]] = {}
    for r in results:
        if r["kind"] == "crack" and r["outcome"] == "measured" and r["group"]:
            groups.setdefault(r["group"], []).append(r["max_mm"])
    rep = [(g, len(v), statistics.fmean(v), statistics.stdev(v)) for g, v in sorted(groups.items()) if len(v) > 1]
    return {"groups": rep, "mean_sd_mm": statistics.fmean(sd for *_, sd in rep)} if rep else None


def gauge_agreement(results: list[dict]) -> dict | None:
    """The app's widest-point width minus the by-eye reading on the gauge."""
    done = [r for r in results if r["kind"] == "crack" and r["outcome"] == "measured" and r["expected_mm"] is not None]
    if not done:
        return None
    d = [r["max_mm"] - r["expected_mm"] for r in done]
    return {"n": len(done), "mean_diff_mm": statistics.fmean(d), "mean_abs_mm": statistics.fmean(abs(x) for x in d)}


def scale_agreement(results: list[dict]) -> dict | None:
    """The bank card's scale divided by the printed sheet's own scale, at each measured line (1.00 = they agree)."""
    r = [x["scale_ratio"] for x in results if x.get("scale_ratio") is not None]
    if not r:
        return None
    return {"n": len(r), "mean": statistics.fmean(r), "min": min(r), "max": max(r)}


def report(results: list[dict], when: str) -> str:
    n_line = sum(r["kind"] == "line" for r in results)
    n_crack = len(results) - n_line
    lines = [f"# Width check, {when}", "",
             f"{len(results)} measurements from {len({r['photo'] for r in results})} photos: {n_line} of printed lines, "
             f"{n_crack} of real cracks.", ""]
    outcomes: dict[str, int] = {}
    for r in results:
        key = r["outcome"].split(":")[0]
        outcomes[key] = outcomes.get(key, 0) + 1
    lines += ["Outcomes: " + ", ".join(f"{k} {v}" for k, v in sorted(outcomes.items())) + ".", ""]
    ls = line_stats(results)
    lines += ["## Printed lines (known width)", ""]
    if ls:
        sd = "n/a" if ls["sd_mm"] is None else f"{ls['sd_mm']:.3f} mm"
        lines += [f"{ls['n']} measured. Bias (measured minus printed) {ls['bias_mm']:+.3f} mm, spread {sd}, mean absolute "
                  f"error {ls['mean_abs_mm']:.3f} mm. Within its own reported uncertainty: {ls['within_uncertainty']} of {ls['n']}.", "",
                  "| printed width (mm) | photos | mean error (mm) |", "|---|---|---|"]
        lines += [f"| {w:.3f} | {n} | {e:+.3f} |" for w, n, e in ls["by_width"]]
    else:
        lines.append("No line was measured.")
    sa = scale_agreement(results)
    if sa:
        lines += ["", f"Scale of the bank card against the printed sheet's own scale ({sa['n']} places): mean ratio "
                      f"{sa['mean']:.3f}, from {sa['min']:.3f} to {sa['max']:.3f} (1.000 = they agree)."]
    lines += ["", "## Real cracks", ""]
    ga, rp = gauge_agreement(results), repeatability(results)
    if ga:
        lines.append(f"Against the by-eye reading on the gauge ({ga['n']} photos): the app's widest point minus the "
                     f"reading is {ga['mean_diff_mm']:+.2f} mm on average, {ga['mean_abs_mm']:.2f} mm in absolute value.")
    if rp:
        lines += ["", f"Repeatability over repeated photos of the same crack: mean spread {rp['mean_sd_mm']:.3f} mm.", "",
                  "| crack | photos | mean width (mm) | spread (mm) |", "|---|---|---|---|"]
        lines += [f"| {g} | {n} | {m:.2f} | {sd:.3f} |" for g, n, m, sd in rp["groups"]]
    if not ga and not rp:
        lines.append("No real crack was measured.")
    lines += ["", "## What this shows and what it does not", "",
              "- **Printed lines** test the card scale, the camera tilt handling and the width method, with a width we know "
              "to within the printer's ink spread (assumption: a few hundredths of a mm, not measured). They do not test "
              "the U-Net: a line on white paper is not a crack on a wall.",
              "- **Real cracks** have no independent truth here. The gauge reading is made by eye, and neighbouring gauge "
              "lines are far apart (for example 0.3, 0.5 and 1 mm), so differences of that size are expected. "
              "Repeatability is the cleaner number: it shows how much the answer moves between photos of one crack.",
              "- This is not a calibration against a crack microscope or a feeler gauge. What error is acceptable for the "
              "risk matrix is to be agreed with the structural engineer; no pass mark is set here.", "",
              "## Per photo", "", "| photo | kind | expected (mm) | median (mm) | widest (mm) | ± (mm) | px across | tilt | outcome |",
              "|---|---|---|---|---|---|---|---|---|"]
    f = lambda v, d=2: "" if v is None else f"{v:.{d}f}"  # noqa: E731
    lines += [f"| {r['photo']} | {r['kind']} | {f(r['expected_mm'], 3)} | {f(r['median_mm'], 3)} | {f(r['max_mm'], 3)} | "
              f"{f(r['uncertainty_mm'], 3)} | {f(r['px_across'], 1)} | {f(r['tilt_deg'], 1)} | {r['outcome']} |" for r in results]
    return "\n".join(lines) + "\n"


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("photos", type=Path)
    ap.add_argument("truth", type=Path)
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "unet_crack.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "results" / "width_check")
    args = ap.parse_args(argv)

    def load_model():
        import torch
        from segmentation.train import load_checkpoint
        return load_checkpoint(args.model, "cuda" if torch.cuda.is_available() else "cpu")

    device = "cpu"
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        pass
    results = check(args.photos, read_rows(args.truth), load_model, device)
    when = datetime.now().strftime("%Y-%m-%d %H:%M")
    out = args.out / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    cols = ["photo", "kind", "group", "expected_mm", "median_mm", "max_mm", "uncertainty_mm", "px_across", "tilt_deg",
            "scale_ratio", "outcome", "note"]
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows({k: r[k] for k in cols} for r in results)
    (out / "report.md").write_text(report(results, when), encoding="utf-8")
    print(f"written {out / 'report.md'}")
    return out


if __name__ == "__main__":
    main()
