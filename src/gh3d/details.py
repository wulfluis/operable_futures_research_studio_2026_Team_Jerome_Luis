"""The detailed drawing of a measured crack (D81), beside its element in the 3D model: plain Python, no Rhino.

On the element's face, at each station along the crack (every 50 mm): a section mark (A, B, C...) across the crack
and a pair of arrows for the direction the concrete was pulled apart (perpendicular to the crack: direction only,
no stress value; a stress map needs a structural model with loads and materials, D81).
Beside the element, a drawing sheet: one enlarged section across the crack per station, with its measured width as a
dimension and its depth drawn dashed as "not measured" (a photo shows no depth), and a chart of the width along the
crack. All measured values come from photo 1 with a bank card; nothing on the sheet is assumed except the sheet's
own layout.
"""

from __future__ import annotations

import math

DETAIL_SCALE = 40  # the sections are drawn this many times life size
SKETCH_DEPTH = 40.0  # mm on the sheet: how far the dashed "not measured" faces go down (a drawing length, not a depth)
CELL_W, CELL_H = 190.0, 150.0  # one section's cell on the sheet, mm
COLUMNS = 2
CHART_H = 140.0  # the chart's width follows the sheet's
TEXT_MM = 13.0  # text height on the sheet
INK, DIM, TENSION = "#ffffff", "#d9d9d9", "#4ea1ff"  # Rhino's dark preview; the wall's viewer darkens white and grey


def _dashed(a, b, dash=4.0, gap=3.0):
    ax, ay = a
    bx, by = b
    length = math.hypot(bx - ax, by - ay) or 1.0
    ux, uy = (bx - ax) / length, (by - ay) / length
    out, t = [], 0.0
    while t < length:
        e = min(t + dash, length)
        out.append([[ax + ux * t, ay + uy * t], [ax + ux * e, ay + uy * e]])
        t = e + gap
    return out


def _text(name, xy, text, height=TEXT_MM, colour=INK):
    return {"name": name, "kind": "text", "color": colour, "point": list(xy), "text": text, "height": height}


def _arrow(a, b, head):
    """Segments of an arrow from a to b (2D)."""
    ax, ay = a
    bx, by = b
    length = math.hypot(bx - ax, by - ay) or 1.0
    ux, uy = (bx - ax) / length, (by - ay) / length
    h1 = [bx - ux * head - uy * head * 0.5, by - uy * head + ux * head * 0.5]
    h2 = [bx - ux * head + uy * head * 0.5, by - uy * head - ux * head * 0.5]
    return [[list(a), list(b)], [list(b), h1], [list(b), h2]]


def on_face(stations, to_face) -> list[dict]:
    """Section marks and tension arrows on the element's face; to_face maps a wall-plane mm point to 3D."""
    marks, arrows, objs = [], [], []
    for s in stations:
        (px, py), (nx, ny) = s["point_mm"], s["across"]
        a, b = [px - nx * 18, py - ny * 18], [px + nx * 18, py + ny * 18]
        marks.append([a, b])
        objs.append(_text("station_label", [px + nx * 26, py + ny * 26], s["name"], 12.0))
        for sign in (1, -1):  # pulled apart: one arrow each way, starting clear of the crack
            start = [px + sign * nx * 6, py + sign * ny * 6]
            end = [px + sign * nx * 30, py + sign * ny * 30]
            arrows += _arrow(start, end, 5.0)
    return ([{"name": "stations", "kind": "segments", "color": INK, "segments": marks},
             {"name": "tension", "kind": "segments", "color": TENSION, "segments": arrows}] + objs)


def sheet(stations, width_along, path_mm, length_mm) -> list[dict]:
    """The drawing sheet in its own 2D coordinates (mm, origin top left, y up so rows go negative)."""
    segs, dims, dashes, objs = [], [], [], []
    rows = max(1, math.ceil(len(stations) / COLUMNS))
    sheet_w = COLUMNS * CELL_W
    sheet_h = rows * CELL_H + CHART_H + 120
    segs += [[[0, 0], [sheet_w, 0]], [[sheet_w, 0], [sheet_w, -sheet_h]], [[sheet_w, -sheet_h], [0, -sheet_h]],
             [[0, -sheet_h], [0, 0]]]
    objs.append(_text("sheet_title", [10, -16], f"SECTIONS ACROSS THE CRACK  x{DETAIL_SCALE}", 15.0))
    objs.append(_text("sheet_note", [10, -32], "width measured (bank card) · depth not measured: a photo shows none",
                      9.0, DIM))
    for i, s in enumerate(stations):
        cx = (i % COLUMNS) * CELL_W + CELL_W / 2
        top = -60 - (i // COLUMNS) * CELL_H
        gap = s["width_mm"] * DETAIL_SCALE
        y = top - 40  # the wall's face in this cell
        segs += [[[cx - 70, y], [cx - gap / 2, y]], [[cx + gap / 2, y], [cx + 70, y]]]  # the face, cut by the crack
        for k in range(1, 6):  # concrete hatching under the face
            segs.append([[cx - 70 + k * 10, y - 4], [cx - 70 + k * 10 - 6, y - 10]])
            segs.append([[cx + 70 - k * 10, y - 4], [cx + 70 - k * 10 - 6, y - 10]])
        dashes += _dashed([cx - gap / 2, y], [cx, y - SKETCH_DEPTH]) + _dashed([cx + gap / 2, y], [cx, y - SKETCH_DEPTH])
        objs.append(_text("depth_note", [cx - 34, y - SKETCH_DEPTH - 12], "depth ?", 10.0, DIM))
        dy = y + 16  # the width as a dimension above the face
        dims += [[[cx - gap / 2, y + 3], [cx - gap / 2, dy + 4]], [[cx + gap / 2, y + 3], [cx + gap / 2, dy + 4]]]
        dims += _arrow([cx - gap / 2 - 22, dy], [cx - gap / 2, dy], 4.0) + _arrow([cx + gap / 2 + 22, dy],
                                                                                  [cx + gap / 2, dy], 4.0)
        objs.append(_text("width_dim", [cx - 30, dy + 8], f"{s['width_mm']:.2f} mm", TEXT_MM))
        objs.append(_text("section_title", [cx - 70, top + 10], f"{s['name']}-{s['name']}  at {s['s_mm']:.0f} mm",
                          11.0))
    # the width along the crack, as a chart under the sections
    ox, oy = 50.0, -(rows * CELL_H + 60) - CHART_H
    CHART_W = sheet_w - ox - 20
    pts = []
    if path_mm and width_along:
        s_acc = [0.0]
        for (x0, y0), (x1, y1) in zip(path_mm, path_mm[1:]):
            s_acc.append(s_acc[-1] + math.hypot(x1 - x0, y1 - y0))
        wmax = max(w for w in width_along if w is not None) if any(w is not None for w in width_along) else 1.0
        top_w = wmax * 1.25
        for s_mm, w in zip(s_acc, width_along):
            if w is not None:
                pts.append([ox + s_mm / max(s_acc[-1], 1e-9) * CHART_W, oy + w / top_w * CHART_H])
        segs += [[[ox, oy], [ox + CHART_W, oy]], [[ox, oy], [ox, oy + CHART_H]]]
        for k in range(0, 5):
            v = top_w * k / 4
            segs.append([[ox - 4, oy + CHART_H * k / 4], [ox, oy + CHART_H * k / 4]])
            objs.append(_text("chart_tick", [ox - 36, oy + CHART_H * k / 4 - 3], f"{v:.2f}", 9.0, DIM))
        for k in range(0, 5):
            segs.append([[ox + CHART_W * k / 4, oy], [ox + CHART_W * k / 4, oy - 4]])
            objs.append(_text("chart_tick", [ox + CHART_W * k / 4 - 10, oy - 14],
                              f"{length_mm * k / 4:.0f}", 9.0, DIM))
        for s in stations:
            x = ox + s["s_mm"] / max(s_acc[-1], 1e-9) * CHART_W
            dashes += _dashed([x, oy], [x, oy + CHART_H], 2.0, 3.0)
            objs.append(_text("chart_station", [x - 3, oy + CHART_H + 6], s["name"], 9.0, DIM))
        objs.append(_text("chart_title", [ox, oy + CHART_H + 20], "WIDTH ALONG THE CRACK (mm)  vs  distance (mm)", 11.0))
    out = [{"name": "sheet", "kind": "segments", "color": DIM, "segments": segs},
           {"name": "dimensions", "kind": "segments", "color": INK, "segments": dims},
           {"name": "not_measured", "kind": "segments", "color": DIM, "segments": dashes}]
    if len(pts) >= 2:
        out.append({"name": "width_chart", "kind": "polyline", "color": "#d62728", "points": pts})
    return out + objs
