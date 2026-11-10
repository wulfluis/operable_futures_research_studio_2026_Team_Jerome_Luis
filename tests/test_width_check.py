"""The width check and the printable gauge (spec/features/measurement.md, "Width check")."""

import csv

import numpy as np
import pytest
import torch
from PIL import Image

from measure import gauge_sheet
from measure.synthetic import make_demo
from measure.width_check import (check, gauge_agreement, line_stats, main, parse_roi, read_rows, repeatability, report)


class FakeModel(torch.nn.Module):
    """Stands in for the U-Net: 'sees' a crack wherever the photo is dark (as in test_app_measure.py)."""

    def forward(self, x):
        return (x.mean(1, keepdim=True) < -0.4).float() * 20 - 10


def _write_csv(path, rows):
    cols = ["photo", "kind", "true_mm", "nominal_mm", "gauge_mm", "roi", "group", "note"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows({c: r.get(c, "") for c in cols} for r in rows)


def _demo_png(folder, name, width_mm, seed=0):
    grey, truth = make_demo(width_mm, seed=seed)
    Image.fromarray(grey.astype(np.uint8)).save(folder / name)
    return truth


# ---------- the gauge ----------

def test_printed_widths_are_whole_printer_pixels():
    assert gauge_sheet.printed_px(1.0) == 24 and gauge_sheet.printed_width_mm(1.0) == pytest.approx(1.016)
    assert gauge_sheet.printed_width_mm(0.1) == pytest.approx(2 * 25.4 / 600)  # 2 pixels: really 0.085 mm
    assert gauge_sheet.printed_px(0.001) == 1  # never zero


def test_the_sheet_has_the_lines_at_their_printed_width_and_the_100_mm_bar():
    dpi = 300
    img = np.asarray(gauge_sheet.render(dpi))
    k = dpi / 25.4
    assert img.shape == (round(297 * k), round(210 * k))
    xmid = round((gauge_sheet.LINE_X_MM + 40) * k)
    for i, nominal in enumerate(gauge_sheet.LINES_MM):
        y = round((gauge_sheet.FIRST_LINE_Y_MM + i * gauge_sheet.LINE_STEP_MM) * k)
        column = img[y - 60:y + 61, xmid] < 128  # the lines are 17 mm (200 px) apart: no neighbour in the window
        assert column.sum() == gauge_sheet.printed_px(nominal, dpi), nominal
    bar_row = img[round((gauge_sheet.BAR_Y_MM + 1.2) * k)] < 128
    xs = np.flatnonzero(bar_row)
    assert (xs.max() - xs.min() + 1) / k == pytest.approx(100.0, abs=0.5)


def test_the_truth_file_matches_the_printed_widths(tmp_path):
    import json
    png, pdf, js = gauge_sheet.save(tmp_path, dpi=150)
    t = json.loads(js.read_text(encoding="utf-8"))
    assert t["lines"]["1"] == pytest.approx(gauge_sheet.printed_width_mm(1.0, 150), abs=1e-4) and t["bar_mm"] == 100
    assert png.exists() and pdf.exists()


# ---------- reading the table ----------

def test_a_line_row_takes_the_printed_width_from_the_nominal_one(tmp_path):
    p = tmp_path / "t.csv"
    _write_csv(p, [{"photo": "a.jpg", "kind": "line", "nominal_mm": "0.5", "roi": "10,20,300,60"},
                   {"photo": "b.jpg", "kind": "line", "true_mm": "1.2"},
                   {"photo": "c.jpg", "kind": "crack", "gauge_mm": "0.5", "group": "c1"},
                   {"photo": "d.jpg", "kind": "crack"}])
    a, b, c, d = read_rows(p)
    assert a["expected_mm"] == pytest.approx(gauge_sheet.printed_width_mm(0.5)) and a["roi"] == (10.0, 20.0, 300.0, 60.0)
    assert b["expected_mm"] == 1.2 and c["expected_mm"] == 0.5 and c["group"] == "c1" and d["expected_mm"] is None


@pytest.mark.parametrize("row,fragment", [({"photo": "a.jpg", "kind": "wall"}, "kind"), ({"photo": "", "kind": "crack"}, "photo"),
                                          ({"photo": "a.jpg", "kind": "line"}, "nominal_mm"),
                                          ({"photo": "a.jpg", "kind": "line", "nominal_mm": "x"}, "line 2"),
                                          ({"photo": "a.jpg", "kind": "line", "nominal_mm": "1", "roi": "5,5,1,1"}, "roi")])
def test_a_bad_row_stops_the_run_and_names_the_line(tmp_path, row, fragment):
    p = tmp_path / "t.csv"
    _write_csv(p, [row])
    with pytest.raises(ValueError, match=fragment):
        read_rows(p)


def test_roi_parsing():
    assert parse_roi("1,2,3,4") == (1.0, 2.0, 3.0, 4.0) and parse_roi("1;2;3;4") == (1.0, 2.0, 3.0, 4.0)
    assert parse_roi("") is None and parse_roi(None) is None
    with pytest.raises(ValueError):
        parse_roi("1,2,3")


# ---------- the numbers ----------

def _r(kind, expected, median=None, max_=None, unc=0.1, group="", outcome="measured"):
    return {"photo": "p.jpg", "kind": kind, "group": group, "expected_mm": expected, "median_mm": median, "max_mm": max_,
            "uncertainty_mm": unc, "px_across": 8.0, "tilt_deg": 3.0, "outcome": outcome, "note": ""}


def test_line_statistics():
    rows = [_r("line", 1.0, 1.1, unc=0.15), _r("line", 1.0, 0.9, unc=0.15), _r("line", 2.0, 2.3, unc=0.15),
            _r("line", 2.0, outcome="not measured: under 3 px"), _r("crack", None, 1.0, 1.0)]
    s = line_stats(rows)
    assert s["n"] == 3 and s["bias_mm"] == pytest.approx(0.1) and s["mean_abs_mm"] == pytest.approx(0.1667, abs=1e-3)
    assert s["within_uncertainty"] == 2  # the 0.3 mm error is outside its +-0.1 mm
    assert s["by_width"][0][0] == 1.0 and s["by_width"][0][1] == 2 and s["by_width"][0][2] == pytest.approx(0.0)
    assert line_stats([_r("crack", 1.0, 1.0, 1.0)]) is None


def test_repeatability_and_gauge_agreement():
    rows = [_r("crack", 0.5, 0.6, 0.6, group="c1"), _r("crack", 0.5, 0.8, 0.8, group="c1"), _r("crack", 0.5, 0.7, 0.7, group="c1"),
            _r("crack", 1.0, 1.0, 1.0, group="c2"), _r("crack", None, 0.4, 0.4, group="c3"),
            _r("crack", 2.0, outcome="no bank card found", group="c4")]
    rp = repeatability(rows)
    assert [g[0] for g in rp["groups"]] == ["c1"] and rp["groups"][0][3] == pytest.approx(0.1) and rp["mean_sd_mm"] == pytest.approx(0.1)
    ga = gauge_agreement(rows)
    assert ga["n"] == 4 and ga["mean_diff_mm"] == pytest.approx((0.1 + 0.3 + 0.2 + 0.0) / 4)
    assert repeatability([_r("crack", 1.0, 1.0, 1.0, group="solo")]) is None and gauge_agreement([]) is None


def test_the_report_says_what_it_does_not_show_and_sets_no_pass_mark():
    rows = [_r("line", 1.0, 1.1), _r("crack", 0.5, 0.6, 0.6, group="c1"), _r("crack", 0.5, 0.7, 0.7, group="c1")]
    text = report(rows, "2026-11-10 12:00")
    assert "They do not test the U-Net" in text
    assert "not a calibration" in text.lower() and "no pass mark" in text
    lowered = text.lower()
    for word in ("pass:", "passed", "safe", "clearance", "permit"):
        assert word not in lowered


# ---------- end to end on synthetic photos with a known answer ----------

def test_lines_are_measured_against_their_known_width(tmp_path):
    rows = []
    for name, width in (("l1.png", 0.8), ("l2.png", 1.5)):
        truth = _demo_png(tmp_path, name, width)
        rows.append({"photo": name, "kind": "line", "expected_mm": width, "group": "", "note": "",
                     "roi": tuple(truth["roi_px"])})
    results = check(tmp_path, rows)
    assert [r["outcome"] for r in results] == ["measured", "measured"]
    for r, width in zip(results, (0.8, 1.5)):
        assert abs(r["median_mm"] - width) < 0.3
    assert line_stats(results)["n"] == 2


def test_cracks_go_through_the_app_pipeline_and_repeat_photos_give_a_spread(tmp_path):
    rows = []
    for i in range(3):
        _demo_png(tmp_path, f"c{i}.png", 1.0, seed=i)
        rows.append({"photo": f"c{i}.png", "kind": "crack", "expected_mm": 1.0, "group": "crack-A", "note": "", "roi": None})
    results = check(tmp_path, rows, load_model=lambda: FakeModel())
    assert all(r["outcome"] == "measured" for r in results)
    assert all(abs(r["max_mm"] - 1.0) < 0.35 for r in results)
    assert repeatability(results)["groups"][0][1] == 3


def test_a_missing_photo_stops_the_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope.png"):
        check(tmp_path, [{"photo": "nope.png", "kind": "line", "expected_mm": 1.0, "group": "", "note": "", "roi": None}])


def test_main_writes_the_csv_and_the_report(tmp_path):
    truth = _demo_png(tmp_path, "l1.png", 1.0)
    _write_csv(tmp_path / "t.csv", [{"photo": "l1.png", "kind": "line", "true_mm": "1.0",
                                     "roi": ",".join(f"{v:.0f}" for v in truth["roi_px"])}])
    out = main([str(tmp_path), str(tmp_path / "t.csv"), "--out", str(tmp_path / "res")])
    assert (out / "results.csv").is_file() and (out / "report.md").is_file()
    rows = list(csv.DictReader(open(out / "results.csv", encoding="utf-8")))
    assert rows[0]["outcome"] == "measured" and abs(float(rows[0]["median_mm"]) - 1.0) < 0.3


# ---------- the whole gauge in one photo ----------

def _homography(px_per_mm=10.0, origin=(100.0, 200.0)):
    return np.array([[px_per_mm, 0.0, origin[0]], [0.0, px_per_mm, origin[1]], [0.0, 0.0, 1.0]])


def test_the_line_boxes_follow_the_sheet():
    from measure.width_check import gauge_boxes
    boxes = gauge_boxes(_homography())
    assert [n for n, _ in boxes] == list(gauge_sheet.LINES_MM)
    nominal, (x0, y0, x1, y1) = boxes[0]
    assert x0 == pytest.approx(100 + (gauge_sheet.LINE_X_MM + 5) * 10)
    assert x1 - x0 == pytest.approx((gauge_sheet.LINE_LEN_MM - 10) * 10) and y1 - y0 == pytest.approx(60)
    assert (y0 + y1) / 2 == pytest.approx(200 + gauge_sheet.FIRST_LINE_Y_MM * 10)
    assert boxes[1][1][1] - boxes[0][1][1] == pytest.approx(gauge_sheet.LINE_STEP_MM * 10)


def _sheet_photo(path, px_per_mm=8.0, tilt=(6.0, 4.0), card=True, markers=True):
    """The real printed sheet seen by a tilted camera, with a dark card lying in its frame. Returns the homography."""
    import cv2
    from measure.synthetic import camera_homography
    sheet = np.asarray(gauge_sheet.render(dpi=150))
    size = (1800, 2500)
    H, _ = camera_homography(size, tilt[0], tilt[1], px_per_mm, centre_mm=(105.0, 148.5))
    mm_per_sheet_px = 25.4 / 150
    photo = cv2.warpPerspective(sheet, H @ np.diag([mm_per_sheet_px, mm_per_sheet_px, 1.0]), size, flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=90)
    rgb = np.repeat(photo[..., None], 3, axis=2).astype(np.uint8)
    x0, y0, x1, y1 = gauge_sheet.CARD_FRAME_MM
    if card:
        quad = cv2.perspectiveTransform(np.float32([[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]]), H)[0]
        cv2.fillConvexPoly(rgb, np.round(quad).astype(np.int32), (40, 60, 120))
    if not markers:  # hide the top left marker: only three are visible
        mx, my = gauge_sheet.MARKER_ORIGINS_MM[0]
        quad = cv2.perspectiveTransform(np.float32([[(mx - 3, my - 3), (mx + 20, my - 3), (mx + 20, my + 20), (mx - 3, my + 20)]]), H)[0]
        cv2.fillConvexPoly(rgb, np.round(quad).astype(np.int32), (255, 255, 255))
    rgb = cv2.GaussianBlur(rgb, (0, 0), 0.8)
    rgb = np.clip(rgb + np.random.default_rng(0).normal(0, 2.0, rgb.shape), 0, 255).astype(np.uint8)
    Image.fromarray(rgb).save(path)
    return H


def test_the_markers_give_the_sheet_back_to_within_a_pixel(tmp_path):
    from measure.width_check import find_sheet, load_rgb
    H_true = _sheet_photo(tmp_path / "s.png")
    H = find_sheet(load_rgb(tmp_path / "s.png"))
    assert H is not None
    probe = np.float32([[[20, 78], [110, 197], [35, 215], [120, 269]]])
    import cv2
    err = np.linalg.norm(cv2.perspectiveTransform(probe, H)[0] - cv2.perspectiveTransform(probe, H_true)[0], axis=1)
    assert err.max() < 3.0  # pixels: 0.4 mm at 8 px/mm, far inside the 3 mm margin of the search boxes


def test_one_photo_of_the_gauge_measures_every_line_it_can_resolve(tmp_path):
    _sheet_photo(tmp_path / "sheet.png")
    rows = [{"photo": "sheet.png", "kind": "gauge", "expected_mm": None, "group": "30cm", "note": "", "roi": None}]
    results = check(tmp_path, rows)
    assert len(results) == len(gauge_sheet.LINES_MM) and all(r["kind"] == "line" and r["group"] == "30cm" for r in results)
    by = {r["note"]: r for r in results}
    for nominal in (1, 2, 3, 5):  # 8 to 40 px across: clearly resolved
        r = by[f"gauge line {nominal:g} mm"]
        assert r["outcome"] == "measured", (nominal, r["outcome"])
        assert abs(r["median_mm"] - r["expected_mm"]) < 0.35, (nominal, r["median_mm"], r["expected_mm"])
    for nominal in (0.1, 0.2):  # under 1 or 2 pixels: the app must say so, not guess
        assert by[f"gauge line {nominal:g} mm"]["outcome"] != "measured"
    assert line_stats(results)["n"] >= 4
    # the bank card's scale and the sheet's own scale describe the same plane
    assert all(abs(r["scale_ratio"] - 1.0) < 0.03 for r in results), [r["scale_ratio"] for r in results]
    assert "Scale of the bank card against the printed sheet" in report(results, "now")


def test_a_gauge_photo_without_all_four_markers_says_so(tmp_path):
    _sheet_photo(tmp_path / "three.png", markers=False)
    results = check(tmp_path, [{"photo": "three.png", "kind": "gauge", "expected_mm": None, "group": "", "note": "", "roi": None}])
    assert len(results) == 1 and results[0]["outcome"].startswith("gauge markers not found") and results[0]["median_mm"] is None


def test_a_gauge_photo_without_a_card_says_so(tmp_path):
    _sheet_photo(tmp_path / "nocard.png", card=False)
    results = check(tmp_path, [{"photo": "nocard.png", "kind": "gauge", "expected_mm": None, "group": "", "note": "", "roi": None}])
    assert len(results) == 1 and results[0]["outcome"].startswith("no bank card found") and results[0]["median_mm"] is None
