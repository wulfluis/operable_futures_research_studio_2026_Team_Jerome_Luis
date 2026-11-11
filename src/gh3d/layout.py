"""The schematic 3D model of one crack on its element (D57). Spec: spec/features/rhino_3d.md.

Plain Python, no Rhino and no numpy, so the same code runs inside the Grasshopper Python 3 component and in the tests.
Input: one result's numbers (the NNN.json the app writes). Output: a dict of objects in mm, Z up, that Grasshopper turns
into Rhino meshes and writes as NNN.mesh.json for the wall's 3D view.

Measured: the crack line, its widths and length, the card (bank card scale, wall plane). Assumed and labelled as such:
the element's size and the crack's position on it.
"""

from __future__ import annotations

import math

VERSION = "gh3d-0.1"
# Typical element sizes, mm (assumption, illustrative): (along X, along Y = depth, along Z), and the Z of its base
SIZES = {"column": ((300, 300, 3000), 0), "beam": ((4000, 250, 500), 2500), "wall": ((3000, 200, 3000), 0),
         "slab": ((4000, 4000, 150), 2850)}
MATRIX_TO_SHAPE = {"column": "column", "beam": "beam", "slab": "slab", "load_bearing_wall": "wall",
                   "infill_wall": "wall"}
NAMES = {"column": "Column", "beam": "Beam", "slab": "Slab", "wall": "Wall"}
COLOURS = {"element": "#5a5a5a", "edges": "#d9d9d9", "crack": "#d62728", "marker": "#ffffff", "card": "#ffffff"}
OFFSET_MM = 1.0  # the crack and the card sit this far in front of the face, so they are never hidden in it
RING_MM = 20.0  # radius of the ring at the widest point
SCHEMATIC = "Schematic: element size and crack position assumed; crack shape and width measured."
ZONE_NOTE = "Red frame: where the crack is (it marks the place only)."
PLACED_NOTE = ("Crack placed and sized from photo 2, relative to the element; width from photo 1 (bank card); "
               "element size assumed.")
ZONE_MARGIN = 0.35  # the frame around the crack: this share of the crack's size on each side ...
ZONE_MIN_MM = 60.0  # ... and at least this much


def element_of(stats: dict) -> tuple[str, bool]:
    """The element to draw and whether it is known: the level's element, else the model's, else a wall panel."""
    t, e = stats.get("triage") or {}, stats.get("element") or {}
    key = t.get("element") or e.get("matrix_element")
    return (MATRIX_TO_SHAPE[key], True) if key in MATRIX_TO_SHAPE else ("wall", False)


def _box(x0, y0, z0, dx, dy, dz):
    v = [[x0 + i * dx, y0 + j * dy, z0 + k * dz] for k in (0, 1) for j in (0, 1) for i in (0, 1)]
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    faces = [f for a, b, c, d in quads for f in ([a, b, c], [a, c, d])]
    edges = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    return v, faces, [[v[a], v[b]] for a, b in edges]


def _ribbon(points, half):
    """A flat strip of the given half-width along a 2D polyline: vertices (2D) and triangles."""
    verts, faces = [], []
    n = len(points)
    for i, (x, y) in enumerate(points):
        a, b = points[max(i - 1, 0)], points[min(i + 1, n - 1)]
        tx, ty = b[0] - a[0], b[1] - a[1]
        length = math.hypot(tx, ty) or 1.0
        nx, ny = -ty / length, tx / length
        verts += [[x + nx * half, y + ny * half], [x - nx * half, y - ny * half]]
        if i:
            p = 2 * (i - 1)
            faces += [[p, p + 2, p + 3], [p, p + 3, p + 1]]
    return verts, faces


def _ring(cx, cy, r, k=32):
    return [[cx + r * math.cos(2 * math.pi * i / k), cy + r * math.sin(2 * math.pi * i / k)] for i in range(k + 1)]


def build(stats: dict) -> dict | None:
    """The model for one result, or None when there is no crack geometry (no bank card: nothing measured to draw)."""
    g = stats.get("geometry") or {}
    est = stats.get("estimate") or {}
    shape, known = element_of(stats)
    (dx, dy, dz), z_base = SIZES[shape]
    on = ((stats.get("photo2") or {}).get("crack_on_element") or {})
    placed = bool(on.get("found") and on.get("path_uv")) and shape != "slab"
    measured = bool(g.get("crack_path_mm")) and len(g["crack_path_mm"]) >= 2
    if not measured and not (placed and est):  # nothing measured with a card, and no test estimate (D62)
        return None
    if placed:  # D61: where photo 2 shows the crack, at its proportion on the element (face mm, centre of the face = 0)
        path = [[(u - 0.5) * dx, (v - 0.5) * dz] for u, v in on["path_uv"]]
        card = []  # the card is in photo 1, not on photo 2's element
        cx = cy = 0.0
    else:
        path = [list(map(float, p)) for p in g["crack_path_mm"]]
        card = [list(map(float, p)) for p in g.get("card_corners_mm") or []]
    xs, ys = [p[0] for p in path], [p[1] for p in path]
    if not placed:
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2  # the crack is centred on the face (assumption)

    if shape == "slab":  # seen from below: the crack lies on the underside, facing down
        def to3(p):
            return [p[0] - cx, dy / 2 + (p[1] - cy), z_base - OFFSET_MM]
        x0, y0 = -dx / 2, 0.0
    else:  # front face at Y = 0, facing -Y; the element runs to Y = depth
        def to3(p):
            return [p[0] - cx, -OFFSET_MM, z_base + dz / 2 + (p[1] - cy)]
        x0, y0 = -dx / 2, 0.0

    v, f, edges = _box(x0, y0, z_base, dx, dy, dz)
    width = g.get("width_median_mm") or g.get("width_max_mm") or est.get("width_median_mm") or 1.0
    rv, rf = _ribbon(path, float(width) / 2)
    objects = [
        {"name": "element", "kind": "mesh", "color": COLOURS["element"], "vertices": v, "faces": f},
        {"name": "element_edges", "kind": "segments", "color": COLOURS["edges"], "segments": edges},
        {"name": "crack", "kind": "mesh", "color": COLOURS["crack"], "vertices": [to3(p) for p in rv], "faces": rf},
        {"name": "crack_line", "kind": "polyline", "color": COLOURS["crack"], "points": [to3(p) for p in path]},
    ]
    if g.get("widest_point_mm") and not placed:
        wx, wy = map(float, g["widest_point_mm"])
        objects.append({"name": "widest", "kind": "polyline", "color": COLOURS["marker"],
                        "points": [to3(p) for p in _ring(wx, wy, RING_MM)]})
    if len(card) == 4:
        objects.append({"name": "card", "kind": "polyline", "color": COLOURS["card"],
                        "points": [to3(p) for p in card + card[:1]]})
    m = max(ZONE_MIN_MM, (0.12 if placed else ZONE_MARGIN) * max(max(xs) - min(xs), max(ys) - min(ys)))
    zone = [[min(xs) - m, min(ys) - m], [max(xs) + m, min(ys) - m], [max(xs) + m, max(ys) + m],
            [min(xs) - m, max(ys) + m], [min(xs) - m, min(ys) - m]]
    objects.append({"name": "crack_zone", "kind": "polyline", "color": COLOURS["crack"],
                    "points": [to3(p) for p in zone]})
    return {"version": VERSION, "number": stats.get("number"), "units": "mm", "up": "z", "element": shape,
            "element_known": known, "placed_from_photo2": placed, "label": label(stats, shape, known, placed),
            "objects": objects,
            "bounds": {"min": [x0, y0 - 50, z_base], "max": [x0 + dx, y0 + dy, z_base + dz]}}


def label(stats: dict, shape: str, known: bool, placed: bool = False) -> list[str]:
    g, t = stats.get("geometry") or {}, stats.get("triage") or {}
    head = f"#{stats.get('number', '?')} · {NAMES[shape] if known else 'Element not known (drawn as a wall)'}"
    est = stats.get("estimate") or {}
    if g.get("width_max_mm") is not None and g.get("width_median_mm") is not None and g.get("length_main_mm") is not None:
        widths = (f"Crack {g['width_max_mm']:.2f} mm at its widest, median {g['width_median_mm']:.2f} mm, "
                  f"{g['length_main_mm']:.0f} mm long")
    elif est:
        widths = (f"Marked damage about {est['width_max_mm']:g} ± {est['uncertainty_mm']:g} mm wide, crack about "
                  f"{est['length_mm']} mm long: ESTIMATED from photo 2 and the assumed element size, not measured (TEST)")
    else:
        widths = "Crack measured"
    level = f"Level: {t['level']}" if t.get("level") else "Level: send the photo of the whole element"
    if placed:
        sp = " Spalling or exposed bars seen on photo 2." if (stats.get("photo2") or {}).get("spalling_seen") else ""
        note = PLACED_NOTE if g.get("width_max_mm") is not None else ("Crack drawn from photo 2, relative to the "
                                                                       "element; element size assumed.")
        return [head, widths, level, note + sp, ZONE_NOTE]
    return [head, widths, level, SCHEMATIC, ZONE_NOTE[:-1] + "; the crack is drawn at real size."]
