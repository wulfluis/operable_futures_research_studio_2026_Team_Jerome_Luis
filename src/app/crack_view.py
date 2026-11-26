"""The crack, to explore (D83): zoom in and out, drag the photo, click the narrowest or widest point and a loupe opens
with the width measured there; drag the loupe along the crack and the width follows. The numbers live in the picture.

The width at every point comes from the measurement's profile (D81). Only millimetres are shown (D116): measured with
a bank card, or estimated (≈, with its error) from a door in photo 2. Without either, no width number is shown, only
the colours relative to the crack; pixels stay inside the app. The narrowest point leaves out the crack's two ends,
where every crack thins to nothing; it is shown only, the level uses the widest point (src/triage/matrix.py).
D118: layers to switch on and off, colours by width (stretched over this crack's own widths, with a legend) or by
angle, and one crack at a time when it is clicked. D117: every segment of every branch has its own angle, read
against the room's vertical in photo 2 when the two photos are linked (from horizontal on a beam or slab).
D119: the sender can remove a false crack and draw a missed crack or spalling; the corrections go to the app.
Plain HTML and JavaScript, no library and nothing downloaded. Spec: spec/features/crack_view.md."""

import base64
import io
import json

import numpy as np
from PIL import Image

from measure.crack import WIDTH_UNCERTAINTY_PX

PHOTO_SIDE = 2000  # px, long side of the photo sent to the viewer (zoom needs more than the wall's 1400)
MAX_LINE = 400  # points of the main crack line (ordered), for the width profile
MAX_NET = 700  # points of the whole crack (all branches), for the width under the loupe
MAX_PATHS = 1500  # points of every branch of every crack together, for drawing them all (D88)
END_SHARE = 0.08  # the narrowest point is looked for away from the ends: this share of the length at each end
SEGMENT_SHARE = 0.035  # D117: a segment of a branch is this share of the photo's long side (at least MIN_SEGMENT_PX)
MIN_SEGMENT_PX = 20
HORIZONTAL_ELEMENTS = ("beam", "slab")  # D117: their angles are read from horizontal


def _pick(n: int, most: int) -> np.ndarray:
    return np.arange(n) if n <= most else np.linspace(0, n - 1, most).round().astype(int)


def _source(result):
    """(crack measurement, unit, factor from its width numbers to the unit, scale estimate or None).
    "mm" with a bank card that measured the width; "mm" estimated from a door in photo 2 (D116); otherwise "none":
    the numbers stay in the photo's pixels for the colours, and no width is shown (D116)."""
    c = result.crack
    if result.measured:
        return c, "mm", 1.0, None
    m, k = None, 1.0
    if c is not None and c.found and c.profile_px is not None and c.mm_per_px:
        m, k = c, 1.0 / c.mm_per_px  # a card, but the width is not trusted: its pixels
    else:
        pc = getattr(result, "pixel_crack", None)
        if pc is not None and pc.found and pc.profile_px is not None:
            m = pc  # no card: measured in the photo's pixels (core.pixel_line)
    if m is None:
        return None, None, 1.0, None
    est = getattr(result, "scale_estimate", None) or None
    if est and est.get("mm_per_px") and m is getattr(result, "pixel_crack", None):
        return m, "mm", k * float(est["mm_per_px"]), est
    return m, "none", k, None


def reference(result):
    """D117: how segment angles are read. (map photo-1 pixels to the reference photo, its 'down' unit vector, read
    from horizontal, where it comes from). Linked photos with the room's vertical found: photo 2; else the photo."""
    from_horizontal = False
    t = getattr(result, "triage", None) or {}
    el = t.get("element")
    if el is None and getattr(result, "element", None) is not None:
        try:
            from element.predict import matrix_element
            el = matrix_element(result.element)
        except Exception:  # noqa: BLE001
            el = None
    if el in HORIZONTAL_ELEMENTS or (getattr(result, "parts", None) or {}).get("on") == "slab":
        from_horizontal = True
    lm, down = getattr(result, "link_map", None), getattr(result, "link_down", None)
    if lm is not None and down is not None:
        from element.locate import Location
        loc = Location(True, "unknown", "")
        loc.link_map = lm
        k = float(getattr(result, "link_k", 1.0) or 1.0)
        return (lambda pts: loc.carry(np.asarray(pts, float) * k)), np.asarray(down, float), from_horizontal, \
            "photo 2: the room's vertical lines"
    return (lambda pts: np.asarray(pts, float)), np.array([0.0, 1.0]), from_horizontal, \
        "the photo itself (phone held upright): send photo 2 for the room's vertical"


def _angle(pts: np.ndarray, to_ref, down: np.ndarray, from_horizontal: bool) -> float | None:
    """0 to 90°: the line through pts against the reference's vertical (or horizontal)."""
    if pts is None or len(pts) < 2:
        return None
    q = to_ref(pts)
    if q is None or len(q) < 2:
        return None
    q = np.asarray(q, float)
    c = q - q.mean(axis=0)
    d = np.linalg.eigh(c.T @ c)[1][:, -1] if len(q) >= 3 else q[-1] - q[0]
    nd = np.linalg.norm(d)
    if nd < 1e-9:
        return None
    a = float(np.degrees(np.arccos(min(1.0, abs(float((d / nd) @ down))))))
    return 90.0 - a if from_horizontal else a


def segments(path: np.ndarray, seg_px: float, to_ref, down, from_horizontal) -> list[list[float]]:
    """D117: a branch cut into pieces of about seg_px along it, each [i0, i1, angle] (indices into path)."""
    if len(path) < 2:
        return []
    along = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    n = max(1, int(round(along[-1] / seg_px)))
    cuts = np.searchsorted(along, np.linspace(0, along[-1], n + 1))
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        b = max(b, a + 1)
        ang = _angle(path[a:b + 1], to_ref, down, from_horizontal)
        if ang is not None:
            out.append([int(a), int(min(b, len(path) - 1)), round(ang, 1)])
    return out


def view_data(result, size: tuple[int, int] | None = None) -> dict | None:
    """The crack for the viewer, in the photo's own coordinates divided by its size (0 to 1). None: nothing to draw."""
    c, unit, k, est = _source(result)
    if c is None or c.profile_px is None or len(c.profile_px) < 3:
        return None
    photo = getattr(result, "photo", None)
    W, H = size or (photo.size if photo is not None else (None, None))
    if not W:
        return None
    n = lambda xy: [round(float(xy[0]) / W, 5), round(float(xy[1]) / H, 5)]  # noqa: E731
    prof = np.asarray(c.profile_px, float)
    wid = np.asarray(c.profile_mm, float) * k
    edges = np.asarray(c.profile_edges_px, float) if c.profile_edges_px is not None else None
    from scipy.spatial import cKDTree
    tree = cKDTree(prof)
    near_px = max(6.0, 3 * float(c.px_across or 2))
    # mm per photo pixel along the crack (lengths of branches); None without a scale
    len_k = (float(c.mm_per_px) if unit == "mm" and est is None and c.mm_per_px else
             float(est["mm_per_px"]) if est else None)

    path = np.asarray(c.main_path_px if c.main_path_px is not None else prof, float).reshape(-1, 2)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    length_px = float(seg.sum()) if len(seg) else 0.0
    line = []
    for i in _pick(len(path), MAX_LINE):
        d, j = tree.query(path[i])
        line.append(n(path[i]) + [None if d > near_px else round(float(wid[j]), 4)])

    def point(j: int) -> dict:
        out = {"x": n(prof[j])[0], "y": n(prof[j])[1], "w": round(float(wid[j]), 4)}
        if edges is not None:
            out["e"] = [n(edges[j][0]), n(edges[j][1])]
        return out

    net = []
    for j in _pick(len(prof), MAX_NET):
        p = point(j)
        net.append([p["x"], p["y"], p["w"]] + (sum(p["e"], []) if "e" in p else []))

    ref = reference(result)
    seg_px = max(MIN_SEGMENT_PX, SEGMENT_SHARE * max(W, H))
    parts = [p for p in (getattr(c, "cracks", None) or [c]) if p is not None]
    cracks = crack_paths(parts, tree, wid, near_px, n, k if unit != "none" else None, len_k, seg_px, ref)

    j_max = int(np.argmax(wid))
    keep = np.ones(len(prof), bool)
    for part in parts:  # away from both ends of every crack's main line (D88)
        pp = getattr(part, "main_path_px", None)
        if pp is None or len(pp) < 2:
            continue
        pp = np.asarray(pp, float).reshape(-1, 2)
        lp = float(np.linalg.norm(np.diff(pp, axis=0), axis=1).sum())
        away = max(END_SHARE * lp, 3 * float(c.px_across or 2))
        for end in (pp[0], pp[-1]):
            keep &= np.linalg.norm(prof - end, axis=1) > away
    keep &= wid > 0
    cand = np.flatnonzero(keep) if keep.any() else np.flatnonzero(wid > 0)
    j_min = int(cand[np.argmin(wid[cand])]) if len(cand) else int(np.argmin(wid))
    widest = point(j_max)
    if unit == "mm" and est is None and c.width_uncertainty_mm is not None:
        widest["u"] = round(float(c.width_uncertainty_mm), 3)
    elif est and est.get("crack"):
        widest["u"] = round(float(est["crack"]["uncertainty_mm"]), 3)

    card = getattr(result, "card", None)
    mm_per_px = (float(c.mm_per_px) if unit == "mm" and est is None and c.mm_per_px else
                 float(est["mm_per_px"]) if est else None)
    an = getattr(result, "anomaly", None) or None  # the autoencoder's layers (D89), shown only
    steel = getattr(result, "exposed_steel", None) or {}
    length = None
    if unit == "mm":
        length = round(float(c.length_main_mm) * k, 1) if c.length_main_mm is not None else \
            (round(length_px * len_k, 1) if len_k else None)
    sh = shape_data(c, path, unit)
    if sh is not None:  # D117: the main line's angle against the same reference as the segments
        a = _angle(path, *ref[:3])
        if a is not None:
            sh["angle"], sh["ref"], sh["from_horizontal"] = round(a, 1), ref[3], ref[2]
            sh["in_photo"] = ref[3].startswith("the photo")
    return {"unit": unit, "approx": est is not None, "line": line, "net": net, "cracks": cracks, "widest": widest,
            "narrowest": point(j_min), "median": round(float(np.median(wid)), 4), "length": length,
            "mm_per_x": None if mm_per_px is None else round(mm_per_px * W, 4),  # mm across the photo's width
            "scale": scale_data(unit, est, c),
            "camera": None if card is None else camera_angle(card),
            "shape": sh, "ref": {"source": ref[3], "from_horizontal": ref[2]},
            "defects": [{"kind": f["kind"], "label": f["label"], "c": f["centre"], "o": f["outline"]}
                        for f in an["defects"]] if an else [],
            "kin": kin_data(getattr(result, "kinematics", None), W, H, unit, mm_per_px if est else None),
            # D107: possible exposed steel: the traced lines may follow it, so no width is shown as the crack's
            "steel": bool(steel.get("seen")),
            # D111: spalling on the close-up: the crack is traced and measured only outside it
            "spall": bool(getattr(result, "spalling_close", None)),
            "causes": [x["kind"] for x in steel.get("causes") or []],  # D115: roots, pipe, shown first
            # the error of every width: one pixel ([46] ch. 5.2), in the viewer's unit
            "unc": round(WIDTH_UNCERTAINTY_PX * mm_per_px, 4) if mm_per_px else 0,
            "growth": growth_data(getattr(result, "growth", None)),
            "surface_score": None,  # not shown: not calibrated for photos at the app's scale (D89)
            "heat": ("data:image/png;base64," + base64.b64encode(an["heat_png"]).decode()) if an and an.get("heat_png") else None,
            "area": area_png(result),  # the detected crack area, clean (replaces the old annotated picture)
            "aspect": round(H / W, 5)}


AREA_SIDE = 1400  # px, long side of the detected-area layer
AREA_RGB = (255, 77, 106)  # the area's red (the app's danger colour), see-through


def area_png(result, side: int = AREA_SIDE) -> str | None:
    """The crack model's detected area alone, as a soft see-through red layer the photo's shape (review 26 Nov 2026):
    no ruler, labels or gauge drawn on it. A data URI (PNG), or None without a decided area."""
    from PIL import ImageFilter
    d = getattr(result, "decision", None)
    m = None if d is None else getattr(d, "mask", None)
    photo = getattr(result, "photo", None)
    if m is None or photo is None or not np.asarray(m).any():
        return None
    k = side / max(photo.size)
    size = (max(1, round(photo.width * k)), max(1, round(photo.height * k)))
    a = Image.fromarray(np.asarray(m, bool).astype(np.uint8) * 255).resize(size, Image.BILINEAR)
    a = a.filter(ImageFilter.GaussianBlur(1.2)).point(lambda v: int(v * 0.62))  # soft edges, see-through
    rgba = Image.new("RGBA", size, AREA_RGB + (0,))
    rgba.putalpha(a)
    buf = io.BytesIO()
    rgba.save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def scale_data(unit: str, est: dict | None, c) -> dict:
    """D116: where the millimetres come from, and how sure they are, for the viewer's scale chip."""
    if unit == "mm" and est is None:
        rel = None
        if getattr(c, "width_uncertainty_mm", None) and getattr(c, "width_max_mm", None):
            rel = round(float(c.width_uncertainty_mm) / max(float(c.width_max_mm), 1e-9), 3)
        return {"source": "card", "name": "bank card", "rel": rel, "measured": True}
    if est:
        return {"source": est.get("source"), "name": est.get("name"), "rel": est.get("rel"), "measured": False,
                "basis": est.get("basis")}
    return {"source": None, "name": None, "rel": None, "measured": False}


def growth_data(g: dict | None) -> dict | None:
    """D122: the measured growth per day between two dated photos with a card, for the illustration; None without."""
    if not g or g.get("grown") is not True or not g.get("days") or g["days"] < 1 or g.get("change_mm") is None:
        return None
    return {"per_day_mm": round(float(g["change_mm"]) / float(g["days"]), 4), "days": round(float(g["days"]), 1),
            "now_mm": g.get("width_now_mm")}


def crack_paths(parts, tree, wid, near_px: float, n, k: float | None, len_k: float | None = None,
                seg_px: float = 40.0, ref=None) -> list[dict]:
    """Every independent crack (D88) with every branch as a line of [x, y, width or None], its length and its widest
    point, in the viewer's unit. Points are thinned evenly so all cracks together stay under MAX_PATHS.
    D117: each branch also has its segments [i0, i1, angle] (indices into its thinned line), its length, widest
    width and angle, and a number counted from the crack's centre outwards."""
    if ref is None:
        ref = (lambda pts: np.asarray(pts, float), np.array([0.0, 1.0]), False, "the photo")
    lines = []
    for ci, part in enumerate(parts):
        for b in (getattr(part, "branch_paths_px", None) or [getattr(part, "main_path_px", None)]):
            if b is not None and len(b) >= 2:
                lines.append((ci, np.asarray(b, float).reshape(-1, 2)))
    total = sum(len(b) for _, b in lines) or 1
    out = [{"paths": [], "branches": [], "length": None, "max": None} for _ in parts]
    for ci, b in lines:
        idx = _pick(len(b), max(2, int(round(MAX_PATHS * len(b) / total))))
        d, j = tree.query(b[idx])
        pts = [n(b[i]) + [None if dd > near_px else round(float(wid[jj]), 4)] for i, dd, jj in zip(idx, d, j)]
        out[ci]["paths"].append(pts)
        thin = b[idx]
        L = float(np.linalg.norm(np.diff(b, axis=0), axis=1).sum())
        ws = [p[2] for p in pts if p[2] is not None]
        out[ci]["branches"].append({"segs": segments(thin, seg_px, *ref[:3]),
                                    "length": None if len_k is None else round(L * len_k, 1),
                                    "max": None if (k is None or not ws) else round(max(ws), 4),
                                    "angle": None if (a := _angle(b, *ref[:3])) is None else round(a, 1),
                                    "mid": n(b[len(b) // 2]), "_pts": b})
    for ci, part in enumerate(parts):
        br = out[ci]["branches"]
        if br:  # D117: numbered from the crack's centre outwards
            centre = np.vstack([x["_pts"] for x in br]).mean(axis=0)
            order = sorted(range(len(br)), key=lambda i: float(np.min(np.linalg.norm(br[i]["_pts"] - centre, axis=1))))
            for rank, i in enumerate(order, 1):
                br[i]["n"] = rank
        for x in br:
            x.pop("_pts", None)
        if getattr(part, "length_main_mm", None) is not None and k is not None:
            out[ci]["length"] = round(float(part.length_main_mm) * k, 1)
        if getattr(part, "width_max_mm", None) is not None and k is not None:
            out[ci]["max"] = round(float(part.width_max_mm) * k, 4)
    return [c for c in out if c["paths"]]


MAX_STATIONS = 60  # arrows of all cracks together (more hide the crack)


def kin_data(kins, W: int, H: int, unit: str = "mm", mm_per_px: float | None = None) -> list | None:
    """Opening and sliding per crack (D99) for the viewer: its summary, and its stations as
    [x, y, opening dx, opening dy, sliding dx, sliding dy] (photo shares; the arrows in photo pixels), in crack order.
    D116: no pixels are shown; amounts in pixels become ≈ mm with a door's scale, else they are left out."""
    if not kins:
        return None
    total = sum(len(k.stations) for k in kins if k is not None) or 1
    out = []
    for k in kins:
        if k is None:
            out.append(None)
            continue
        st = k.stations
        keep = _pick(len(st), max(1, int(round(MAX_STATIONS * len(st) / total)))) if st else []
        rows = []
        for i in keep:
            x = st[i]
            nn, t, d = np.asarray(x.normal), np.asarray(x.tangent), np.asarray(x.vector_px, float)
            op, sl = float(d @ nn), (None if x.sliding is None else float(d @ t))
            rows.append([round(x.point_px[0] / W, 5), round(x.point_px[1] / H, 5), round(op * nn[0], 2), round(op * nn[1], 2),
                         None if sl is None else round(sl * t[0], 2), None if sl is None else round(sl * t[1], 2)])
        summary = k.to_dict()
        summary.pop("method", None)  # D123: no names in the app
        if summary.get("unit") == "px":
            for f in ("opening_median", "opening_max", "sliding_median", "sliding_max"):
                v = summary.get(f)
                summary[f] = None if v is None or not mm_per_px else round(v * mm_per_px, 3)
            summary["unit"] = "mm" if mm_per_px else "none"
            summary["approx"] = bool(mm_per_px)
        out.append({**summary, "st": rows})
    return out if any(o is not None for o in out) else None


def shape_data(c, path: np.ndarray, unit: str) -> dict | None:
    """The crack's shape as the type rules read it (src/triage/pattern_rules.py), for the viewer's second row.
    The angle is to the photo's up-down direction (the phone held upright) unless view_data reads it against photo 2
    (D117). `lean` only tilts the little icon: +1 the top leans right, -1 left, from the main line's two ends."""
    a = getattr(c, "angle_to_vertical_deg", None)
    if a is None:
        return None
    top, bottom = (path[0], path[-1]) if path[0][1] <= path[-1][1] else (path[-1], path[0])
    r = lambda v, k=2: None if v is None else round(float(v), k)  # noqa: E731
    return {"angle": r(a, 1), "lean": 1 if top[0] >= bottom[0] else -1, "in_photo": unit != "mm",
            "straightness": r(getattr(c, "straightness", None)), "coherence": r(getattr(c, "coherence", None)),
            "loops": int(getattr(c, "n_loops", 0) or 0), "branches": r(getattr(c, "branch_ratio", None), 1)}


def camera_angle(card) -> dict:
    """The camera's angle to the wall, from the card's pose (measure/card.py): 90° is straight on. The card gives the
    tilt between the camera's axis and the wall's normal; the angle to the wall is 90° minus it."""
    return {"to_wall": round(90.0 - float(card.tilt_deg), 1), "left_right": round(abs(float(card.tilt_left_right_deg)), 1),
            "up_down": round(abs(float(card.tilt_up_down_deg)), 1), "status": card.status}


def photo_jpeg(img: Image.Image, side: int = PHOTO_SIDE) -> bytes:
    im = img.convert("RGB")
    im.thumbnail((side, side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=86)
    return buf.getvalue()


def mask_image(overlay: Image.Image) -> Image.Image:
    """The model's answer picture without its text band (the photo with the crack in red)."""
    h = overlay.info.get("photo_height")
    return overlay.crop((0, 0, overlay.width, h)) if h else overlay


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><style>
:root { --fg:#eef1f5; --muted:#8b95a3; --acc:#22d3ee; --glass:rgba(12,14,18,.86); --line:rgba(255,255,255,.12); --red:#ef4444; }
* { box-sizing:border-box; }
html,body { margin:0; height:100%; background:#050607; font:13px/1.35 -apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI",Helvetica,Arial,sans-serif; color:var(--fg); -webkit-font-smoothing:antialiased; }
.mono { font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace; }
#v { position:relative; height:100%; border-radius:18px; overflow:hidden; background:#050607; border:1px solid var(--line);
     user-select:none; -webkit-user-select:none; touch-action:none; outline:none; }
#v:focus-visible { border-color:var(--acc); }
#stage { position:absolute; inset:0; cursor:grab; overflow:hidden; }
#stage.drag { cursor:grabbing; } #stage.draw { cursor:crosshair; }
#layer { position:absolute; left:0; top:0; transform-origin:0 0; will-change:transform; }
#layer img { position:absolute; left:0; top:0; width:100%; height:100%; display:block; pointer-events:none; }
#mask, #heat { opacity:0; transition:opacity .2s; }
.df, .bn { position:absolute; transform:translate(-50%,-50%); pointer-events:none; white-space:nowrap; padding:3px 8px; border-radius:999px;
      background:var(--glass); border:1px dashed currentColor; font-size:11.5px; font-weight:600; }
.bn { border-style:solid; padding:1px 6px; font-size:11px; color:#fff; border-color:rgba(255,255,255,.5); }
#dtags.off, #btags.off { display:none; }
#lines { position:absolute; left:0; top:0; width:100%; height:100%; overflow:visible; pointer-events:none; transition:opacity .2s; }
.mk { position:absolute; transform:translate(-50%,-50%); pointer-events:auto; cursor:pointer; }
.mk .ring { width:30px; height:30px; border-radius:50%; border:2px solid currentColor; box-shadow:0 0 0 3px rgba(0,0,0,.35), 0 0 18px currentColor;
            animation:pulse 2.2s ease-in-out infinite; }
.mk .tag { position:absolute; left:22px; top:-12px; white-space:nowrap; padding:3px 8px; border-radius:999px; background:var(--glass);
           border:1px solid currentColor; font-weight:600; font-size:12px; }
.mk .tag b { color:#fff; font-weight:700; }
.mk.max .tag { top:-36px; } .mk.min .tag { top:14px; }
.mk.left .tag { left:auto; right:22px; }
#marks.lensopen .mk { opacity:.25; } #marks.lensopen .tag { display:none; }
@keyframes pulse { 50% { box-shadow:0 0 0 3px rgba(0,0,0,.35), 0 0 4px currentColor; } }
#hov { position:absolute; width:10px; height:10px; margin:-5px 0 0 -5px; border-radius:50%; background:#fff; box-shadow:0 0 0 2px var(--acc), 0 0 14px var(--acc); display:none; pointer-events:none; }
#lens { position:absolute; display:none; pointer-events:auto; cursor:move; }
#lens .glass { position:absolute; border-radius:50%; overflow:hidden; border:2px solid var(--acc); background-color:#000; background-repeat:no-repeat;
               box-shadow:0 0 0 1px rgba(0,0,0,.6), 0 20px 50px rgba(0,0,0,.6), 0 0 30px rgba(34,211,238,.35); }
#lens svg { position:absolute; left:0; top:0; overflow:visible; pointer-events:none; }
#lens .read { position:absolute; left:50%; transform:translateX(-50%); white-space:nowrap; text-align:center; padding:6px 12px; border-radius:12px;
              background:var(--glass); border:1px solid var(--line); pointer-events:none; }
#lens .read .big { font-size:20px; font-weight:700; letter-spacing:-.01em; }
#lens .read .small { font-size:11px; color:var(--muted); }
#lens .x { position:absolute; width:26px; height:26px; border-radius:50%; border:1px solid var(--line); background:var(--glass); color:var(--fg);
           display:flex; align-items:center; justify-content:center; cursor:pointer; font-size:13px; }
.hud { position:absolute; display:flex; gap:6px; flex-wrap:wrap; pointer-events:none; }
.hud > * { pointer-events:auto; }
.chip { display:inline-flex; align-items:baseline; gap:6px; padding:6px 11px; border-radius:999px; background:var(--glass); border:1px solid var(--line);
        white-space:nowrap; color:var(--fg); }
.chip .k { font:600 10px/1 ui-monospace,"SF Mono",Menlo,monospace; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); }
.chip .val { font-weight:700; font-size:14px; font-variant-numeric:tabular-nums; }
button.chip { cursor:pointer; font:inherit; transition:border-color .15s, background .15s, transform .1s; }
button.chip:hover { border-color:currentColor; }
button.chip:active { transform:scale(.96); }
button.chip.on { border-color:var(--acc); color:var(--acc); }
button.chip:disabled { opacity:.4; cursor:default; }
.hud .row { flex-basis:100%; height:0; }
.chip.sm { padding:4px 9px; } .chip.sm .val { font-size:12.5px; }
.btn { width:34px; height:34px; border-radius:50%; padding:0; justify-content:center; align-items:center; font-size:16px; font-weight:600; }
#alert { position:absolute; left:12px; right:12px; top:12px; display:none; gap:8px; align-items:center; flex-wrap:wrap; padding:9px 14px; border-radius:14px;
         background:rgba(127,29,29,.92); border:1px solid #f87171; font-weight:700; font-size:14px; pointer-events:auto; }
#alert .sub { font-weight:500; font-size:12px; color:#fecaca; }
#alert.cause { background:rgba(67,20,7,.9); border-color:#fb923c; }
#tl { left:12px; top:12px; right:56px; }
#tr { right:12px; top:12px; flex-direction:column; align-items:flex-end; }
#panel { position:absolute; right:56px; top:12px; width:270px; max-height:calc(100% - 110px); overflow:auto; display:none; padding:12px 14px; border-radius:16px;
         background:var(--glass); border:1px solid var(--line); pointer-events:auto; }
#panel h4 { margin:10px 0 6px; font:600 10px/1 ui-monospace,"SF Mono",Menlo,monospace; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); }
#panel h4:first-child { margin-top:0; }
#panel label { display:flex; align-items:center; gap:10px; padding:7px 0; cursor:pointer; border-bottom:1px solid rgba(255,255,255,.06); }
#panel label:last-of-type { border-bottom:none; }
#panel label .t { flex:1; min-width:0; font-size:13.5px; font-weight:600; line-height:1.25; }
#panel label .t small { display:block; font-weight:400; font-size:11.5px; color:var(--muted); margin-top:2px; }
#panel input[type=checkbox] { -webkit-appearance:none; appearance:none; flex:0 0 36px; width:36px; height:21px; margin:0; border-radius:999px;
  background:rgba(255,255,255,.16); position:relative; cursor:pointer; transition:background .2s; order:2; }
#panel input[type=checkbox]::after { content:""; position:absolute; left:3px; top:3px; width:15px; height:15px; border-radius:50%; background:#fff;
  box-shadow:0 1px 3px rgba(0,0,0,.4); transition:transform .2s; }
#panel input[type=checkbox]:checked { background:var(--acc); }
#panel input[type=checkbox]:checked::after { transform:translateX(15px); }
#panel input[type=checkbox]:focus-visible { outline:2px solid var(--acc); outline-offset:2px; }
#panel .info { display:flex; gap:9px; align-items:flex-start; margin:8px 0 2px; padding:9px 11px; border-radius:12px;
  background:rgba(34,211,238,.08); border:1px solid rgba(34,211,238,.22); font-size:12px; line-height:1.4; color:var(--fg); }
#panel .info svg { flex:0 0 16px; margin-top:1px; color:var(--acc); }
#panel .info b { display:block; font-size:12.5px; margin-bottom:1px; }
#panel .dot { width:10px; height:10px; border-radius:50%; display:inline-block; }
#panel .seg { display:flex; gap:4px; } #panel .seg button { flex:1; }
#panel .note { color:var(--muted); font-size:11px; margin:4px 0 0; }
#legend { position:absolute; left:12px; bottom:86px; padding:7px 11px; border-radius:12px; background:var(--glass); border:1px solid var(--line); pointer-events:auto; width:220px; max-width:calc(100% - 24px); cursor:help; }
#legend .bar { height:10px; border-radius:5px; margin:6px 0 4px; }
#legend .ticks { display:flex; justify-content:space-between; font-size:11px; font-variant-numeric:tabular-nums; }
#legend .cap { font:600 10px/1.2 ui-monospace,"SF Mono",Menlo,monospace; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); }
#legend .sub { display:none; font-size:10.5px; color:var(--muted); margin-top:3px; } #legend:hover .sub { display:block; }
#editbar { position:absolute; left:50%; bottom:86px; transform:translateX(-50%); display:none; gap:6px; flex-wrap:wrap; justify-content:center; pointer-events:auto; max-width:calc(100% - 24px); }
#bot { left:12px; right:12px; bottom:12px; align-items:flex-end; justify-content:space-between; flex-wrap:nowrap; }
#spark { flex:1 1 auto; min-width:0; max-width:560px; height:64px; padding:8px 12px 6px; border-radius:14px; background:var(--glass); border:1px solid var(--line);
         cursor:crosshair; position:relative; }
#spark svg { width:100%; height:38px; display:block; overflow:visible; }
#spark .cap { display:flex; justify-content:space-between; font:600 10px/1.2 ui-monospace,"SF Mono",Menlo,monospace; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); }
#scale { display:flex; flex-direction:column; align-items:flex-end; gap:6px; flex:0 0 auto; }
#bar { height:6px; border:2px solid #fff; border-top:none; box-shadow:0 1px 3px rgba(0,0,0,.8); }
#hint { position:absolute; left:50%; top:50%; transform:translate(-50%,-50%); padding:10px 16px; border-radius:999px; background:var(--glass);
        border:1px solid var(--line); pointer-events:none; transition:opacity .6s; color:var(--muted); white-space:nowrap; }
.max { color:#ff4d6a; } .min { color:var(--acc); }
@media (prefers-reduced-motion: reduce) { * { animation:none !important; transition:none !important; } }
@media (max-width:560px) { #spark { height:56px; } #spark svg { height:30px; } .chip { padding:5px 9px; } .chip .val { font-size:12.5px; } .opt { display:none; } .btn { width:30px; height:30px; }
  #panel { left:12px; right:50px; width:auto; } #legend { bottom:78px; min-width:0; } }
</style></head><body>
<div id="v" tabindex="0" aria-label="The crack photo: drag to move, scroll or pinch to zoom, click a crack to see only it">
  <div id="stage">
    <div id="layer"><img id="photo" alt="Crack photo"><img id="mask" alt=""><img id="heat" alt=""><svg id="lines"></svg></div>
    <div id="dtags"></div><div id="btags" class="off"></div><div id="marks"></div><div id="hov"></div>
    <div id="lens"><div class="glass"></div><svg></svg><div class="read"></div><div class="x" title="Close (Esc)">✕</div></div>
  </div>
  <div id="alert"></div>
  <div class="hud" id="tl"></div>
  <div class="hud" id="tr">
    <button class="chip btn" id="zin" title="Zoom in (+)">+</button>
    <button class="chip btn" id="zout" title="Zoom out (−)">−</button>
    <button class="chip btn" id="fit" title="Whole photo (0)">⤢</button>
    <button class="chip btn" id="tlay" title="Layers: what is drawn on the photo">☰</button>
    <button class="chip btn" id="tedit" title="Correct the app: remove a false crack, draw a missed crack or spalling">✎</button>
  </div>
  <div id="panel"></div>
  <div id="legend"></div>
  <div id="editbar"></div>
  <div class="hud" id="bot"><div id="spark"></div><div id="scale"><div class="chip mono" id="zoomv">1.0×</div><div class="chip" id="scalev"><span><div id="bar"></div></span><span class="val mono" id="scalet"></span></div></div></div>
  <div id="hint">Scroll or pinch to zoom · drag to move · click a crack</div>
</div>
<script>
(function () {
const D = __DATA__, PHOTO = "__PHOTO__", MASK = "__MASK__";
const $ = function (id) { return document.getElementById(id); };
const v = $("v"), stage = $("stage"), layer = $("layer"), photo = $("photo"), mask = $("mask"), lines = $("lines");
const lens = $("lens"), glass = lens.querySelector(".glass"), lsvg = lens.querySelector("svg"), read = lens.querySelector(".read");
const C = D || {}, U = C.unit || "none", APPROX = !!C.approx, HAS = !!(D && D.line && D.line.length);
let W = 1, H = 1, s = 1, s0 = 1, tx = 0, ty = 0, lensAt = null, lensZ = 4;
const R = function () { return Math.min(110, Math.max(70, Math.min(stage.clientWidth, stage.clientHeight) * 0.22)); };
// D116: only millimetres are shown; without a scale no width number at all (pixels stay inside the app)
function num(w) { return w < 1 ? w.toFixed(2) : w.toFixed(1); }
function fmt(w) { if (w == null) return "–"; if (U !== "mm") return "no scale"; return (APPROX ? "≈ " : "") + num(w) + " mm"; }
function lenFmt(l) { if (l == null || U !== "mm") return "–"; return (APPROX ? "≈ " : "") + (l >= 1000 ? (l / 1000).toFixed(2) + " m" : Math.round(l) + " mm"); }
function rgb(a) { return "rgb(" + a.map(Math.round).join(",") + ")"; }
function grad(stops, t) { t = Math.max(0, Math.min(1, t)) * (stops.length - 1); const i = Math.min(stops.length - 2, Math.floor(t)), f = t - i, a = stops[i], b = stops[i + 1];
  return rgb([0, 1, 2].map(function (k) { return a[k] + (b[k] - a[k]) * f; })); }
// width colours: violet (thin) to pink, orange and yellow (wide); never green, which reads as "safe"
const WSTOPS = [[124,58,237],[236,72,153],[251,146,60],[253,224,71]];
// D117: angle colours, 0° cyan, 45° amber, 90° red
const ASTOPS = [[34,211,238],[251,191,36],[239,68,68]];
function acol(a) { return grad(ASTOPS, a / 90); }
const PAL = ["#f472b6", "#38bdf8", "#fbbf24", "#a78bfa", "#fb923c", "#e879f9", "#60a5fa", "#f87171"];
const CR = C.cracks || [], MULTI = CR.length > 1;
const DF = C.defects || [], DCOL = {steel: "#ef4444", spalling: "#ff5c7a", rust: "#f97316", dark: "#94a3b8", other: "#e2e8f0", water: "#38bdf8", roots: "#a3e635", pipe: "#cbd5e1"},
  DSHORT = {steel: "Steel?", spalling: "Spalling?", rust: "Rust?", dark: "Dark patch", other: "Unusual", water: "Water?", roots: "Roots?", pipe: "Pipe?"};
const KIN = C.kin || [], HASKIN = KIN.some(function (k) { return k && k.st && k.st.length; });
const REF = C.ref || {}, FROMH = !!REF.from_horizontal, AREF = FROMH ? "from horizontal" : "from vertical";
// D118: layers, all switchable; less on at first
const L = {lines: HAS, mode: "width", nums: false, kin: false, def: DF.length > 0, heat: false, mask: false, grow: 0};
const hidden = new Set(), removed = new Set(), added = []; let sel = -1, edit = false, tool = null, drawing = null, sent = false;

photo.onload = function () { W = photo.naturalWidth; H = photo.naturalHeight; layer.style.width = W + "px"; layer.style.height = H + "px";
  lines.setAttribute("viewBox", "0 0 " + W + " " + H); redraw(); fit(false); };
const AREA = C.area || null;  // the detected area alone, see-through; the old annotated picture only as a fallback
photo.src = PHOTO; if (AREA) mask.src = AREA; else if (MASK) mask.src = MASK;
if (C.heat) $("heat").src = C.heat;
const ENG = C.engineer !== false;  // the engineer's wall: corrections and the unusual-surface layer
if (!HAS || !ENG) $("tedit").style.display = "none";
if (!HAS) $("spark").style.visibility = "hidden";

// ---- which cracks count, and their widths, for the colours and the numbers ----
function live(ci) { return !hidden.has(ci) && !removed.has(ci) && (sel < 0 || sel === ci); }
function widthsOf(list) { const w = []; list.forEach(function (ci) { CR[ci].paths.forEach(function (P) { P.forEach(function (q) { if (q[2] != null) w.push(q[2]); }); }); }); return w.sort(function (a, b) { return a - b; }); }
let WS = [];
function shown() { const a = []; CR.forEach(function (_, ci) { if (live(ci)) a.push(ci); }); return a; }
// D118: the colour curve stretched over this crack's own widths: each colour covers about the same share of its points
function rank(w) { const n = WS.length; if (n < 2) return 0.5; let lo = 0, hi = n - 1; while (lo < hi) { const m = (lo + hi) >> 1; if (WS[m] < w) lo = m + 1; else hi = m; } return lo / (n - 1); }
function wcol(w) { return grad(WSTOPS, rank(w)); }
function q(p) { if (!WS.length) return null; return WS[Math.min(WS.length - 1, Math.round(p * (WS.length - 1)))]; }

// the photo can be moved at every zoom, but never leaves the frame: smaller than the frame, it moves anywhere inside
// it; larger, its edges stop at the frame's
function clampPan() { const sw = stage.clientWidth, sh = stage.clientHeight, w = W * s, h = H * s;
  tx = w <= sw ? Math.min(sw - w, Math.max(0, tx)) : Math.min(0, Math.max(sw - w, tx));
  ty = h <= sh ? Math.min(sh - h, Math.max(0, ty)) : Math.min(0, Math.max(sh - h, ty)); }
function apply() { clampPan(); layer.style.transform = "translate(" + tx + "px," + ty + "px) scale(" + s + ")";
  $("zoomv").textContent = (s / s0).toFixed(s / s0 < 10 ? 1 : 0) + "×"; place(); placeTags(); scalebar(); if (lensAt) drawLens(); }
function fit(anim) { const sw = stage.clientWidth, sh = stage.clientHeight; s0 = Math.min(sw / W, sh / H);
  go(s0, (sw - W * s0) / 2, (sh - H * s0) / 2, anim); }
let raf = null;
function go(ns, ntx, nty, anim) { cancelAnimationFrame(raf);
  if (!anim || matchMedia("(prefers-reduced-motion: reduce)").matches) { s = ns; tx = ntx; ty = nty; apply(); return; }
  const sw = stage.clientWidth / 2, sh = stage.clientHeight / 2, s1 = s, t0 = performance.now();
  const ca = [(sw - tx) / (W * s), (sh - ty) / (H * s)], cb = [(sw - ntx) / (W * ns), (sh - nty) / (H * ns)];
  (function step(t) { const k = Math.min(1, (t - t0) / 420), e = 1 - Math.pow(1 - k, 3);
    s = s1 * Math.pow(ns / s1, e); const cx = ca[0] + (cb[0] - ca[0]) * e, cy = ca[1] + (cb[1] - ca[1]) * e;
    tx = sw - cx * W * s; ty = sh - cy * H * s; apply(); if (k < 1) raf = requestAnimationFrame(step); })(t0); }
function clampS(x) { return Math.max(s0 * 0.5, Math.min(s0 * 60, x)); }  // at the smallest, half the frame
function zoomAt(f, cx, cy, anim) { const ns = clampS(s * f), k = ns / s; go(ns, cx - (cx - tx) * k, cy - (cy - ty) * k, anim); }
function scr(x, y) { return [tx + x * W * s, ty + y * H * s]; }
function img(X, Y) { return [(X - tx) / (W * s), (Y - ty) / (H * s)]; }
function flyTo(p, z) { const ns = clampS(Math.max(s, s0 * (z || 6))), sw = stage.clientWidth, sh = stage.clientHeight;
  go(ns, sw / 2 - p.x * W * ns, sh / 2.2 - p.y * H * ns, true); }

function redraw() { WS = widthsOf(shown()); drawLines(); legend(); hud(); spark(); alertBar(); panel(); editbar(); apply(); }

function ln(x1, y1, x2, y2, c, w, o, dash) { return '<line x1="' + x1.toFixed(1) + '" y1="' + y1.toFixed(1) + '" x2="' + x2.toFixed(1) + '" y2="' + y2.toFixed(1) +
  '" stroke="' + c + '" stroke-width="' + w + '" vector-effect="non-scaling-stroke" stroke-linecap="round" opacity="' + o + '"' + (dash ? ' stroke-dasharray="' + dash + '"' : "") + "/>"; }
function grow(w) { if (!L.grow || !C.growth || w == null) return 1; const now = C.growth.now_mm || w; return Math.min(4, (now + C.growth.per_day_mm * L.grow) / Math.max(now, 1e-6)); }
function drawLines() { let out = "";
  if (HAS && L.lines) CR.forEach(function (cr, ci) {
    const off = hidden.has(ci), gone = removed.has(ci), focus = sel < 0 || sel === ci;
    if (off) return;
    cr.paths.forEach(function (P, bi) {
      const segs = (cr.branches && cr.branches[bi] && cr.branches[bi].segs) || [];
      for (let i = 1; i < P.length; i++) { const a = P[i - 1], b = P[i], w = (a[2] != null ? a[2] : b[2]);
        let c, o = .95, sw = 3.2;
        if (gone) { c = "#9ca3af"; o = .5; }
        else if (!focus) { c = "#cbd5e1"; o = .28; sw = 2; }
        else if (L.mode === "crack") c = PAL[ci % PAL.length];
        else if (L.mode === "angle") { let ang = null; for (let k = 0; k < segs.length; k++) if (i - 1 >= segs[k][0] && i - 1 < segs[k][1]) { ang = segs[k][2]; break; }
          c = ang == null ? "#ffffff" : acol(ang); }
        else { c = w == null ? "#ffffff" : wcol(w); if (w == null) o = .35; }
        if (focus && !gone) sw *= grow(w);
        out += ln(a[0] * W, a[1] * H, b[0] * W, b[1] * H, c, sw, o, gone ? "4 5" : null); } }); });
  if (L.def) DF.forEach(function (d) { const pts = d.o.map(function (p) { return (p[0] * W).toFixed(1) + "," + (p[1] * H).toFixed(1); }).join(" ");
    out += '<polygon points="' + pts + '" fill="' + DCOL[d.kind] + '" fill-opacity=".12" stroke="' + DCOL[d.kind] +
      '" stroke-width="2" stroke-dasharray="6 4" vector-effect="non-scaling-stroke"/>'; });
  added.concat(drawing ? [drawing] : []).forEach(function (a) { const pts = a.pts.map(function (p) { return (p[0] * W).toFixed(1) + "," + (p[1] * H).toFixed(1); }).join(" ");
    out += a.kind === "spalling" ? '<polygon points="' + pts + '" fill="#ff5c7a" fill-opacity=".25" stroke="#ff5c7a" stroke-width="2.5" vector-effect="non-scaling-stroke"/>'
      : '<polyline points="' + pts + '" fill="none" stroke="#f472b6" stroke-width="3.5" stroke-dasharray="7 4" vector-effect="non-scaling-stroke" stroke-linecap="round"/>'; });
  if (L.kin) out += kinArrows();
  lines.innerHTML = '<g style="filter:drop-shadow(0 0 2px rgba(0,0,0,.9))">' + out + "</g>"; }
function kinArrows() {  // at each point: a double arrow across the crack (opening, cyan), one along it (sliding, amber)
  let g = ""; const l2 = function (x1, y1, x2, y2, c) { return ln(x1, y1, x2, y2, c, 2, 1); };
  const head = function (x, y, ux, uy, h, c) { return l2(x, y, x - ux * h + uy * h * 0.6, y - uy * h - ux * h * 0.6, c) + l2(x, y, x - ux * h - uy * h * 0.6, y - uy * h + ux * h * 0.6, c); };
  KIN.forEach(function (k, ci) { if (!k || !k.st || (sel >= 0 && sel !== ci) || removed.has(ci) || hidden.has(ci)) return; k.st.forEach(function (p) {
    const x = p[0] * W, y = p[1] * H, ox = p[2], oy = p[3], m = Math.hypot(ox, oy) || 1, ux = ox / m, uy = oy / m, Ln = Math.max(8, 1.4 * m), h = Ln * 0.35;
    g += l2(x - ux * Ln, y - uy * Ln, x + ux * Ln, y + uy * Ln, "#22d3ee") + head(x + ux * Ln, y + uy * Ln, ux, uy, h, "#22d3ee") + head(x - ux * Ln, y - uy * Ln, -ux, -uy, h, "#22d3ee");
    if (p[4] != null) { const sm = Math.hypot(p[4], p[5]), S = Ln * Math.min(2, sm / m); if (S > 1.5) { const vx = p[4] / sm, vy = p[5] / sm;
      g += l2(x - vx * S, y - vy * S, x + vx * S, y + vy * S, "#fbbf24") + head(x + vx * S, y + vy * S, vx, vy, Math.min(h, S * 0.6), "#fbbf24"); } } }); });
  return g; }

// the widest and narrowest point: of the whole photo, or of the crack chosen
function extreme(ci) { let mx = null, mn = null; const len = []; CR[ci].paths.forEach(function (P) { P.forEach(function (p) { if (p[2] == null) return;
  if (!mx || p[2] > mx.w) mx = {x: p[0], y: p[1], w: p[2]}; }); }); return mx; }
function marksNow() { if (sel >= 0) { const m = extreme(sel); return m ? [["max", "Widest", m]] : []; }
  return [["max", "Widest", C.widest], ["min", "Narrowest", C.narrowest]]; }
function place() { const m = $("marks"); m.innerHTML = ""; if (!HAS || C.steel || U !== "mm" || !L.lines || edit) return;
  marksNow().forEach(function (d) { const p = d[2]; if (!p) return; const el = document.createElement("div"); el.className = "mk " + d[0];
    el.innerHTML = '<div class="ring"></div><div class="tag">' + d[1] + ' <b>' + fmt(p.w) + "</b></div>";
    el.addEventListener("pointerdown", function (e) { e.stopPropagation(); });
    el.addEventListener("click", function (e) { e.stopPropagation(); flyTo(p, 8); openLens(p.x, p.y, d[0] === "max" ? "widest" : "narrowest", p); });
    const qq = scr(p.x, p.y); el.style.left = qq[0] + "px"; el.style.top = qq[1] + "px"; el.classList.toggle("left", qq[0] > stage.clientWidth - 170);
    m.appendChild(el); });
  m.classList.toggle("lensopen", !!lensAt); }

function placeTags() { const box = $("dtags"); box.classList.toggle("off", !L.def);
  if (box.childElementCount !== DF.length) { box.innerHTML = ""; DF.forEach(function (d) { const el = document.createElement("div"); el.className = "df";
    el.style.color = DCOL[d.kind] || "#fff"; el.textContent = DSHORT[d.kind] || d.kind; el.title = d.label + " (an engineer decides)"; box.appendChild(el); }); }
  Array.prototype.forEach.call(box.children, function (el, i) { const p = scr(DF[i].c[0], DF[i].c[1]); el.style.left = p[0] + "px"; el.style.top = p[1] + "px"; });
  // D117: the branches' numbers, counted from the crack's centre
  const bt = $("btags"), on = L.nums && L.lines; bt.classList.toggle("off", !on); bt.innerHTML = "";
  if (on) CR.forEach(function (cr, ci) { if (!live(ci)) return; (cr.branches || []).forEach(function (b) { if (!b.mid) return; const el = document.createElement("div"); el.className = "bn";
    el.textContent = (MULTI ? (ci + 1) + "." : "") + b.n + (b.angle != null ? " · " + Math.round(b.angle) + "°" : ""); const p = scr(b.mid[0], b.mid[1]);
    el.style.left = p[0] + "px"; el.style.top = p[1] + "px"; bt.appendChild(el); }); }); }

function nearest(x, y, px) { if (!HAS) return null; let best = null, bd = 1e18;
  C.net.forEach(function (n) { const dx = (n[0] - x) * W * s, dy = (n[1] - y) * H * s, d = dx * dx + dy * dy; if (d < bd) { bd = d; best = n; } });
  return Math.sqrt(bd) <= px ? best : null; }
function crackAt(x, y, px) { let best = -1, bd = px * px; CR.forEach(function (cr, ci) { if (hidden.has(ci)) return; cr.paths.forEach(function (P) { P.forEach(function (p) {
  const dx = (p[0] - x) * W * s, dy = (p[1] - y) * H * s, d = dx * dx + dy * dy; if (d < bd) { bd = d; best = ci; } }); }); }); return best; }

function openLens(x, y, kind, at) { lensAt = {x: x, y: y, kind: kind || null, at: at || null}; $("marks").classList.add("lensopen"); lens.style.display = "block"; $("hint").style.opacity = 0; drawLens(); }
function closeLens() { lensAt = null; $("marks").classList.remove("lensopen"); lens.style.display = "none"; }
function drawLens() { const r = R(), p = scr(lensAt.x, lensAt.y), touch = matchMedia("(pointer: coarse)").matches;
  const cx = p[0], cy = touch ? p[1] - r - 30 : p[1];   // on a phone the loupe sits above the finger
  lens.style.left = (cx - r) + "px"; lens.style.top = (cy - r) + "px"; lens.style.width = 2 * r + "px"; lens.style.height = 2 * r + "px";
  const k = s * lensZ;
  glass.style.width = glass.style.height = 2 * r + "px";
  const lay = L.mask && AREA ? ["url(" + AREA + ")", "url(" + PHOTO + ")"] : [L.mask && MASK ? "url(" + MASK + ")" : "url(" + PHOTO + ")"];
  const sz = (W * k) + "px " + (H * k) + "px", ps = (r - lensAt.x * W * k) + "px " + (r - lensAt.y * H * k) + "px";
  glass.style.backgroundImage = lay.join(",");
  glass.style.backgroundSize = lay.map(function () { return sz; }).join(",");
  glass.style.backgroundPosition = lay.map(function () { return ps; }).join(",");
  let n = lensAt.at || (lensAt.kind ? C[lensAt.kind] : null);
  if (!n) { const qq = nearest(lensAt.x, lensAt.y, 22); if (qq) n = {x: qq[0], y: qq[1], w: qq[2], e: qq.length > 3 ? [[qq[3], qq[4]], [qq[5], qq[6]]] : null}; }
  const Lp = function (u, w) { return [r + (u - lensAt.x) * W * k, r + (w - lensAt.y) * H * k]; };
  let g = '<circle cx="' + r + '" cy="' + r + '" r="' + (r - 1) + '" fill="none" stroke="rgba(255,255,255,.15)"/>' +
          '<path d="M' + (r - 9) + " " + r + "h6M" + (r + 3) + " " + r + "h6M" + r + " " + (r - 9) + "v6M" + r + " " + (r + 3) + 'v6" stroke="rgba(255,255,255,.7)" stroke-width="1"/>';
  if (n && n.e) { const a = Lp(n.e[0][0], n.e[0][1]), b = Lp(n.e[1][0], n.e[1][1]); let dx = b[0] - a[0], dy = b[1] - a[1]; const dl = Math.hypot(dx, dy) || 1; dx /= dl; dy /= dl;
    const ex = function (pp, sg) { return [pp[0] - dx * sg * 14, pp[1] - dy * sg * 14]; }, a2 = ex(a, 1), b2 = ex(b, -1), t = 8;
    g += '<g stroke="#fff" stroke-width="2" stroke-linecap="round" style="filter:drop-shadow(0 0 2px #000)">' +
      '<line x1="' + a2[0] + '" y1="' + a2[1] + '" x2="' + a[0] + '" y2="' + a[1] + '"/><line x1="' + b2[0] + '" y1="' + b2[1] + '" x2="' + b[0] + '" y2="' + b[1] + '"/>' +
      '<line x1="' + (a[0] - dy * t) + '" y1="' + (a[1] + dx * t) + '" x2="' + (a[0] + dy * t) + '" y2="' + (a[1] - dx * t) + '"/>' +
      '<line x1="' + (b[0] - dy * t) + '" y1="' + (b[1] + dx * t) + '" x2="' + (b[0] + dy * t) + '" y2="' + (b[1] - dx * t) + '"/></g>';
    g += '<line x1="' + a[0] + '" y1="' + a[1] + '" x2="' + b[0] + '" y2="' + b[1] + '" stroke="' + wcol(n.w) + '" stroke-width="3"/>'; }
  lsvg.setAttribute("width", 2 * r); lsvg.setAttribute("height", 2 * r); lsvg.innerHTML = g;
  const name = lensAt.kind === "widest" ? "Widest point · used by the level" : lensAt.kind === "narrowest" ? "Narrowest point · shown only" : (n ? "Width here" : "No crack under the loupe");
  read.style.top = (2 * r + 10) + "px";
  const u = lensAt.kind === "widest" && !lensAt.at && C.widest.u ? ' <span class="small">± ' + num(C.widest.u) + "</span>" : "";
  read.innerHTML = '<div class="big ' + (lensAt.kind === "widest" ? "max" : lensAt.kind === "narrowest" ? "min" : "") + '">' + (n ? fmt(n.w) + u : "–") +
    '</div><div class="small">' + name + " · loupe " + (s / s0 * lensZ).toFixed(0) + "×" + (U !== "mm" ? " · add a bank card for mm" : "") + "</div>";
  const x = lens.querySelector(".x"); x.style.left = (2 * r - 18) + "px"; x.style.top = "-4px"; }

// ---- D112, D115: exposed steel first, then what may come before the crack ----
function alertBar() { const a = $("alert"), CN = {water: "water or damp", roots: "roots", pipe: "a pipe"};
  const causes = (C.causes || []).map(function (k) { return CN[k] || k; });
  if (C.steel) { a.className = ""; a.innerHTML = '⚠ Exposed steel seen: critical <span class="sub">An engineer must look, whatever the crack looks like.' +
    (causes.length ? " Also possible: " + causes.join(", ") + "." : "") + "</span>"; }
  else if (causes.length) { a.className = "cause"; a.innerHTML = "Possible " + causes.join(", ") + ' near the crack <span class="sub">It may come before the crack: tell the engineer.</span>'; }
  else { a.style.display = "none"; $("tl").style.top = "12px"; return; }
  a.style.display = "flex"; $("tl").style.top = (a.offsetHeight + 20) + "px"; }

function chip(k, val, cls, tip, id) { return (id ? '<button class="chip ' + (cls || "") + '" id="' + id + '"' : '<span class="chip ' + (cls || "") + '"') +
  (tip ? ' title="' + tip + '"' : "") + '><span class="k">' + k + '</span><span class="val">' + val + (id ? "</span></button>" : "</span></span>"); }
function scaleChip() { const sc = C.scale || {};
  if (U !== "mm") return chip("Scale", "none · add a bank card", "", "No bank card and no door in photo 2: the width is not shown. The colours still show where the crack is wider or thinner.");
  if (sc.measured) return chip("Scale", "bank card" + (sc.rel != null ? " · ± " + Math.round(sc.rel * 100) + " %" : ""), "opt", "Measured with the bank card next to the crack. ± is the widest width's error (one pixel and the card).");
  return chip("Scale", "≈ " + (sc.name || "estimate") + (sc.rel != null ? " · ± " + Math.round(sc.rel * 100) + " %" : ""), "", (sc.basis || "An estimate") + ". A bank card next to the crack measures it."); }
function hud() { const tl = $("tl"); if (!HAS) { tl.innerHTML = chip("Crack", "no line to measure"); return; }
  if (C.steel) { tl.innerHTML = chip("Widths", "not used: the lines may follow the steel", "", "The traced lines may follow the steel and the broken concrete, so their widths are not used and not shown.") + defectChips(); return; }
  let h = "";
  if (sel >= 0) { const cr = CR[sel], m = extreme(sel), br = (cr.branches || []).slice().sort(function (a, b) { return a.n - b.n; });
    const main = br.slice().sort(function (a, b) { return (b.length || 0) - (a.length || 0); })[0];
    h += chip("Crack", (sel + 1) + " of " + CR.length, "", "", "csel") + chip("Widest", fmt(m ? m.w : cr.max), "max") + chip("Length", lenFmt(cr.length)) +
      (main && main.angle != null ? chip("Angle", Math.round(main.angle) + "° " + AREF, "", "Of the longest branch, read against " + (REF.source || "the photo") + ".") : "") +
      chip("Branches", String(br.length)) + '<button class="chip sm" id="call"><span class="k">Show all</span><span class="val">✕</span></button>';
    if (br.length > 1) h += '<span class="row"></span>' + br.map(function (b) { return '<span class="chip sm" title="Branch ' + b.n + ', counted from the crack\'s centre"><span class="k">' + b.n +
      '</span><span class="val">' + (b.angle != null ? Math.round(b.angle) + "°" : "–") + (U === "mm" && b.length != null ? " · " + lenFmt(b.length) : "") + (U === "mm" && b.max != null ? " · " + fmt(b.max) : "") + "</span></span>"; }).join("");
  } else {
    // D111 (Jerome): spalling on the close-up: the crack is not traced or measured inside it
    if (C.spall) h += '<span class="chip" style="border-color:#ff5c7a" title="SAM 3.1 sees spalling: a piece of plaster or concrete has broken away. The crack is not traced or measured inside it. An engineer should look."><span class="k" style="color:#ff5c7a">Spalling</span><span class="val">crack measured outside it</span></span>';
    h += (U === "mm" ? chip("Widest", fmt(C.widest.w), "max", "", "cmax") + chip("Length", lenFmt(C.length)) : "") + scaleChip() +
      (MULTI ? chip("Cracks", String(CR.length - removed.size), "", "Click a crack to see only it: its colours, its widths and its branches.") : "") + defectChips();
    if (!MULTI && C.shape) h += shapeRow(C.shape);
  }
  if (L.kin) h += kinRow();
  if (L.grow && C.growth) h += '<span class="row"></span>' + chip("In " + L.grow + " days", "≈ " + num((C.growth.now_mm || 0) + C.growth.per_day_mm * L.grow) + " mm", "max",
    "If it kept growing at the rate measured between the two photos (" + num(C.growth.per_day_mm * 30) + " mm a month). An illustration, not a forecast.");
  tl.innerHTML = h;
  if ($("cmax")) $("cmax").onclick = function () { flyTo(C.widest, 8); openLens(C.widest.x, C.widest.y, "widest"); };
  if ($("call")) $("call").onclick = function () { sel = -1; closeLens(); redraw(); };
  if ($("csel")) $("csel").onclick = function () { const m = extreme(sel); if (m) flyTo(m, 4); }; }
function shapeRow(sh) {
  const sm = function (k, val, tip) { return '<span class="chip sm" title="' + tip + '"><span class="k">' + k + '</span><span class="val">' + val + "</span></span>"; };
  const f = function (x, d) { return x === null || x === undefined ? "–" : x.toFixed(d); };
  return '<span class="row"></span>' + sm("Angle", Math.round(sh.angle) + "° " + AREF, "Angle of the main crack, read against " + (sh.ref || "the photo") + ". 0° " + (FROMH ? "horizontal" : "vertical") + ", 90° " + (FROMH ? "vertical" : "horizontal") + ". Colour by angle in the layers for every segment.") +
    sm("Straight", f(sh.straightness, 2), "Straightness: distance between the two ends divided by the length of the main crack. 1 = a straight line.") +
    sm("Cells", String(sh.loops), "Closed cells: areas fully enclosed by crack lines, like the cells of a net. From 2 the rules read a network of surface cracks (map cracking).") +
    sm("Branches", f(sh.branches, 1) + "×", "Length of all branches divided by the main crack's length. 1.0× = no side branches."); }
function kinRow() {  // how the crack moved (D99): the opening, sliding and the reading
  const k = KIN.filter(function (x, ci) { return x && (sel < 0 || ci === sel); })[0]; if (!k) return "";
  const f = function (x) { if (x == null || k.unit !== "mm") return null; return (k.approx ? "≈ " : "") + num(x) + " mm"; };
  const tip = "Read from the crack's two faces: they were one surface, so the bumps of one face match the hollows of the other. " +
    "Moving one face back onto the other gives how the crack moved: across it (opening, cyan) and along it (sliding, amber). One photo: the movement since the crack formed. Shown only; an engineer reads it.";
  const sm = function (k2, val, c) { return val == null ? "" : '<span class="chip sm" title="' + tip + '"><span class="k"' + (c ? ' style="color:' + c + '"' : "") + ">" + k2 + '</span><span class="val">' + val + "</span></span>"; };
  return '<span class="row"></span>' + sm("Opening", f(k.opening_median), "#22d3ee") + sm("Sliding", k.sliding_median == null ? (k.unit === "mm" ? "not readable" : null) : f(k.sliding_median), "#fbbf24") + sm("Movement", k.reading, null); }
function defectChips() { const n = {}; DF.forEach(function (d) { n[d.kind] = (n[d.kind] || 0) + 1; });
  let h = ""; Object.keys(n).forEach(function (k) { if (k === "steel") return;
    h += '<span class="chip opt" style="border-style:dashed;border-color:' + (DCOL[k] || "#fff") + '" title="Shown only, separate from the cracks; an engineer decides."><span class="k" style="color:' + (DCOL[k] || "#fff") + '">' + (DSHORT[k] || k) + '</span><span class="val">' + n[k] + "</span></span>"; });
  return h; }

// ---- D118: the legend, always relative to the crack shown ----
function legend() { const el = $("legend");
  if (!HAS || !L.lines || C.steel || edit) { el.style.display = "none"; return; } el.style.display = "block";
  if (L.mode === "angle") { let st = ""; for (let i = 0; i <= 6; i++) st += acol(i * 15) + (i < 6 ? "," : "");
    el.innerHTML = '<div class="cap">Angle of each segment, ' + AREF + '</div><div class="bar" style="background:linear-gradient(90deg,' + st + ')"></div>' +
      '<div class="ticks"><span>0°</span><span>45°</span><span>90°</span></div><div class="sub">Read against ' + (REF.source || "the photo") + ".</div>"; return; }
  if (L.mode === "crack") { el.innerHTML = '<div class="cap">One colour per crack</div><div class="ticks" style="justify-content:flex-start;gap:10px;flex-wrap:wrap">' +
      CR.map(function (_, i) { return removed.has(i) ? "" : '<span><span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:' + PAL[i % PAL.length] + '"></span> ' + (i + 1) + "</span>"; }).join("") + "</div>"; return; }
  let st = ""; for (let i = 0; i <= 4; i++) st += grad(WSTOPS, i / 4) + (i < 4 ? "," : "");
  const t = U === "mm" ? [0, .5, 1].map(function (p) { return fmt(q(p)); }) : ["thinner", "", "wider"];
  el.innerHTML = '<div class="cap">Width' + (sel >= 0 ? " of crack " + (sel + 1) : MULTI ? " of the cracks shown" : "") + '</div><div class="bar" style="background:linear-gradient(90deg,' + st + ')"></div>' +
    '<div class="ticks"><span>' + t[0] + "</span><span>" + t[1] + "</span><span>" + t[2] + '</span></div><div class="sub">Colours stretched over this crack\'s own widths: each colour covers about the same share of it' + (U === "mm" ? "" : "; no scale, so no mm") + ".</div>"; }

// ---- D118: the layers panel ----
function panel() { const p = $("panel"); if (p.style.display !== "block") return;
  // a switch with its name and, under it, what it shows
  const box = function (id, on, label, sub) { return '<label><span class="t">' + label + (sub ? "<small>" + sub + "</small>" : "") +
    '</span><input type="checkbox" role="switch" id="' + id + '"' + (on ? " checked" : "") + "></label>"; };
  const clock = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>';
  const mb = function (m, t) { return '<button class="chip sm' + (L.mode === m ? " on" : "") + '" data-mode="' + m + '"><span class="val">' + t + "</span></button>"; };
  let h = "<h4>Colour the crack by</h4><div class=\"seg\">" + mb("width", "Width") + mb("angle", "Angle") + (MULTI ? mb("crack", "Crack") : "") + "</div>";
  if (HAS) { h += "<h4>Cracks</h4>" + box("l-lines", L.lines, "Crack lines", "The traced cracks, in the colours above");
    if (MULTI) CR.forEach(function (cr, i) { if (removed.has(i)) return; h += box("l-c" + i, !hidden.has(i), '<span class="dot" style="background:' + PAL[i % PAL.length] + '"></span> Crack ' + (i + 1) + (U === "mm" && cr.max != null ? " · " + fmt(cr.max) : "")); });
    h += box("l-nums", L.nums, "Branch numbers", "Each branch numbered from the centre, with its angle"); }
  h += "<h4>More</h4>";
  if (HASKIN) h += box("l-kin", L.kin, "Movement", "How the two sides moved: apart (blue), along (amber)");
  if (DF.length) h += box("l-def", L.def, "Other damage", "Possible steel, spalling or stains, dashed");
  if (C.heat && ENG) h += box("l-heat", L.heat, "Unusual surface", "Where the wall looks unlike sound concrete");
  if (AREA || MASK) h += box("l-mask", L.mask, "Detected crack area", "Everything the AI marked as crack, in soft red");
  if (C.growth) h += "<h4>If it keeps growing</h4><div class=\"seg\">" + [0, 5, 10, 20].map(function (d) { return '<button class="chip sm' + (L.grow === d ? " on" : "") + '" data-grow="' + d + '"><span class="val">' + (d ? d + " d" : "now") + "</span></button>"; }).join("") +
    '</div><div class="info">' + clock + '<span><b>An illustration, not a forecast</b>At the rate measured between your two photos.</span></div>';
  else h += '<div class="info">' + clock + '<span><b>See if it grows</b>Photograph the same crack again with the card in a day or more, and enter this result\'s number.</span></div>';
  p.innerHTML = h;
  p.querySelectorAll("[data-mode]").forEach(function (b) { b.onclick = function () { L.mode = b.dataset.mode; redraw(); }; });
  p.querySelectorAll("[data-grow]").forEach(function (b) { b.onclick = function () { L.grow = +b.dataset.grow; redraw(); }; });
  const on = function (id, f) { const el = $(id); if (el) el.onchange = function () { f(el.checked); redraw(); }; };
  on("l-lines", function (x) { L.lines = x; }); on("l-nums", function (x) { L.nums = x; }); on("l-kin", function (x) { L.kin = x; }); on("l-def", function (x) { L.def = x; });
  on("l-heat", function (x) { L.heat = x; $("heat").style.opacity = x ? .85 : 0; }); on("l-mask", function (x) { L.mask = x; mask.style.opacity = x ? 1 : 0; });
  CR.forEach(function (_, i) { on("l-c" + i, function (x) { if (x) hidden.delete(i); else { hidden.add(i); if (sel === i) sel = -1; } }); }); }

// ---- D119: corrections by the sender ----
function editbar() { const e = $("editbar"); if (!edit) { e.style.display = "none"; stage.classList.remove("draw"); return; } e.style.display = "flex";
  const b = function (id, t, on, dis) { return '<button class="chip sm' + (on ? " on" : "") + '" id="' + id + '"' + (dis ? " disabled" : "") + '><span class="val">' + t + "</span></button>"; };
  e.innerHTML = b("e-rm", sel >= 0 && removed.has(sel) ? "Bring this crack back" : "Remove this crack", false, sel < 0) + b("e-crack", "Draw a missed crack", tool === "crack") +
    b("e-spall", "Draw spalling", tool === "spalling") + b("e-undo", "Undo", false, !added.length && !removed.size) +
    b("e-send", sent ? "Sent ✓" : "Send corrections", false, !added.length && !removed.size) + b("e-done", "Done");
  stage.classList.toggle("draw", !!tool);
  $("e-rm").onclick = function () { if (sel < 0) return; if (removed.has(sel)) removed.delete(sel); else removed.add(sel); sent = false; redraw(); };
  $("e-crack").onclick = function () { tool = tool === "crack" ? null : "crack"; closeLens(); redraw(); };
  $("e-spall").onclick = function () { tool = tool === "spalling" ? null : "spalling"; closeLens(); redraw(); };
  $("e-undo").onclick = function () { if (added.length) added.pop(); else if (removed.size) removed.delete(Array.from(removed).pop()); sent = false; redraw(); };
  $("e-send").onclick = function () { const msg = {type: "sl-corrections", removed: Array.from(removed), added: added};
    try { window.parent.postMessage(msg, "*"); } catch (err) {} sent = true; redraw(); };
  $("e-done").onclick = function () { edit = false; tool = null; $("tedit").classList.remove("on"); redraw(); };
  ["e-rm", "e-crack", "e-spall", "e-undo", "e-send", "e-done"].forEach(function (id) { $(id).addEventListener("pointerdown", function (ev) { ev.stopPropagation(); }); }); }

function spark() { const el = $("spark"); if (!HAS || C.steel) { el.style.visibility = "hidden"; return; } el.style.visibility = "visible";
  const ANG = L.mode === "angle";
  // the line shown: the chosen crack's longest branch, else the main line; in angle mode a branch with its segments
  let Lr = C.line, segs = null;
  if (sel >= 0 || ANG) { let best = null;
    CR.forEach(function (cr, ci) { if ((sel >= 0 && ci !== sel) || removed.has(ci) || hidden.has(ci)) return;
      cr.paths.forEach(function (P, bi) { if (!best || P.length > best.P.length) best = {P: P, segs: (cr.branches && cr.branches[bi] && cr.branches[bi].segs) || []}; }); });
    if (best) { Lr = best.P; segs = best.segs; } }
  const n = Lr.length; if (n < 2) { el.style.visibility = "hidden"; return; }
  // D118: in angle mode the chart shows each segment's angle along the crack (0 to 90°), else the width
  const angAt = function (i) { if (!segs) return null; for (let k = 0; k < segs.length; k++) if (i >= segs[k][0] && i <= segs[k][1]) return segs[k][2]; return null; };
  const ws = ANG ? Lr.map(function (_, i) { return angAt(i); }) : Lr.map(function (p) { return p[2]; });
  const wm = ANG ? 90 : Math.max.apply(null, ws.filter(function (x) { return x != null; }).concat([1e-6]));
  let st = 0; const cum = [0]; for (let i = 1; i < n; i++) { st += Math.hypot((Lr[i][0] - Lr[i - 1][0]) * W, (Lr[i][1] - Lr[i - 1][1]) * H); cum.push(st); }
  const X = function (i) { return 1000 * cum[i] / (st || 1); }, Y = function (w) { return 38 - 34 * (w / wm); };
  let path = "", area = "", open = false;
  ws.forEach(function (w, i) { if (w == null) { open = false; return; } path += (open ? "L" : "M") + X(i).toFixed(1) + " " + Y(w).toFixed(1); open = true; });
  const UN = !ANG && U === "mm" ? (C.unc || 0) : 0, Yc = function (w) { return Math.max(0, Math.min(40, Y(w))); }; let runs = [], cur = [];
  ws.forEach(function (w, i) { if (w == null) { if (cur.length) runs.push(cur); cur = []; return; } cur.push(i); }); if (cur.length) runs.push(cur);
  runs.forEach(function (r) { if (r.length < 2 || !UN) return; area += "M" + r.map(function (i) { return X(i).toFixed(1) + " " + Yc(ws[i] + UN).toFixed(1); }).join("L") +
    "L" + r.slice().reverse().map(function (i) { return X(i).toFixed(1) + " " + Yc(Math.max(0, ws[i] - UN)).toFixed(1); }).join("L") + "Z"; });
  let gr = ""; for (let i = 0; i <= 4; i++) gr += '<stop offset="' + (i * 25) + '%" stop-color="' + (ANG ? acol(90 * (1 - i / 4)) : grad(WSTOPS, 1 - i / 4)) + '"/>';
  const guides = ANG ? [0, 45, 90].map(function (a) { return '<line x1="0" x2="1000" y1="' + Y(a) + '" y2="' + Y(a) + '" stroke="rgba(255,255,255,.12)" stroke-width="1" vector-effect="non-scaling-stroke" stroke-dasharray="3 4"/>'; }).join("") : "";
  const what = ANG ? "Angle along " + (sel >= 0 ? "crack " + (sel + 1) : "the longest branch") + ", " + AREF + " · 0° to 90°"
    : "Width along " + (sel >= 0 ? "crack " + (sel + 1) : "the crack") + (U !== "mm" ? " · relative, no scale" : UN ? ' <span title="Grey band: the error of every width, one pixel either way (' + fmt(UN) +
    '). Dust and worn crack edges make it larger: wipe the crack before the photo." style="text-transform:none;letter-spacing:0">· ± ' + num(UN) + " mm</span>" : "");
  el.innerHTML = '<div class="cap"><span>' + what + '</span><span id="spv"></span></div><svg viewBox="0 0 1000 40" preserveAspectRatio="none">' +
    '<defs><linearGradient id="g" gradientUnits="userSpaceOnUse" x1="0" y1="4" x2="0" y2="38">' + gr + '</linearGradient></defs>' + guides +
    (area ? '<path d="' + area + '" fill="#8b95a3" fill-opacity=".28" stroke="none"/>' : "") +
    '<path d="' + path + '" fill="none" stroke="url(#g)" stroke-width="2.2" vector-effect="non-scaling-stroke"/>' +
    '<line id="spl" x1="0" x2="0" y1="0" y2="40" stroke="#fff" stroke-width="1" vector-effect="non-scaling-stroke" opacity="0"/></svg>';
  const pick = function (e) { const b = el.querySelector("svg").getBoundingClientRect(), f = Math.max(0, Math.min(1, (e.clientX - b.left) / b.width));
    let i = 0; while (i < n - 1 && X(i) / 1000 < f) i++; return i; };
  el.onpointermove = function (e) { const i = pick(e), p = Lr[i], sp = scr(p[0], p[1]), h = $("hov");
    h.style.display = "block"; h.style.left = sp[0] + "px"; h.style.top = sp[1] + "px";
    const l = $("spl"); l.setAttribute("x1", X(i)); l.setAttribute("x2", X(i)); l.setAttribute("opacity", 1);
    $("spv").textContent = ANG ? (ws[i] == null ? "–" : Math.round(ws[i]) + "° " + AREF) : U === "mm" ? fmt(p[2]) + (UN && p[2] != null ? " ± " + num(UN) : "") : ""; };
  el.onpointerleave = function () { $("hov").style.display = "none"; $("spl").setAttribute("opacity", 0); $("spv").textContent = ""; };
  el.onclick = function (e) { const p = Lr[pick(e)]; flyTo({x: p[0], y: p[1]}, 8); openLens(p[0], p[1], null); }; }

function scalebar() { const px = W * s;   // screen px across the photo
  if (!C.mm_per_x) { $("scalev").style.display = "none"; return; } $("scalev").style.display = "";
  const per = px / C.mm_per_x;  // screen px per mm
  const nice = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000];
  let Ln = nice[0]; for (let i = 0; i < nice.length; i++) { if (nice[i] * per <= 130) Ln = nice[i]; }
  $("bar").style.width = Math.max(8, Ln * per) + "px"; $("scalet").textContent = "≈ " + (Ln >= 1000 ? Ln / 1000 + " m" : Ln + " mm"); }

// pointer: drag to move, pinch to zoom; a click picks a crack and opens the loupe; in edit mode, drawing
const ptr = new Map(); let moved = 0, pinch = null, lensDrag = false;
stage.addEventListener("pointerdown", function (e) { v.focus({preventScroll: true}); stage.setPointerCapture(e.pointerId); ptr.set(e.pointerId, [e.clientX, e.clientY]); moved = 0;
  if (tool && ptr.size === 1) { const b = stage.getBoundingClientRect(); drawing = {kind: tool, pts: [img(e.clientX - b.left, e.clientY - b.top)]}; return; }
  lensDrag = lens.contains(e.target) && !e.target.classList.contains("x");
  if (ptr.size === 2) { drawing = null; const a = Array.from(ptr.values()); pinch = {d: Math.hypot(a[0][0] - a[1][0], a[0][1] - a[1][1]), s: s}; }
  if (!lensDrag) stage.classList.add("drag"); $("hint").style.opacity = 0; });
stage.addEventListener("pointermove", function (e) { if (!ptr.has(e.pointerId)) return; const last = ptr.get(e.pointerId), dx = e.clientX - last[0], dy = e.clientY - last[1];
  ptr.set(e.pointerId, [e.clientX, e.clientY]); moved += Math.abs(dx) + Math.abs(dy);
  if (drawing) { const b = stage.getBoundingClientRect(); drawing.pts.push(img(e.clientX - b.left, e.clientY - b.top)); drawLines(); return; }
  if (ptr.size === 2 && pinch) { const a = Array.from(ptr.values()), d = Math.hypot(a[0][0] - a[1][0], a[0][1] - a[1][1]), b = stage.getBoundingClientRect();
    zoomAt(pinch.s * d / pinch.d / s, (a[0][0] + a[1][0]) / 2 - b.left, (a[0][1] + a[1][1]) / 2 - b.top, false); return; }
  if (lensDrag && lensAt) { lensAt.x += dx / (W * s); lensAt.y += dy / (H * s); lensAt.kind = null; lensAt.at = null; drawLens(); return; }
  tx += dx; ty += dy; apply(); });
function up(e) { if (!ptr.has(e.pointerId)) return; ptr.delete(e.pointerId); if (ptr.size < 2) pinch = null; stage.classList.remove("drag");
  if (drawing) { if (drawing.pts.length >= 3) { drawing.pts = drawing.pts.filter(function (_, i) { return i % 2 === 0 || i === drawing.pts.length - 1; })
      .map(function (p) { return [Math.round(p[0] * 1e5) / 1e5, Math.round(p[1] * 1e5) / 1e5]; }); added.push(drawing); sent = false; }
    drawing = null; redraw(); return; }
  if (moved < 6 && ptr.size === 0 && e.type === "pointerup") {
    if (e.target.classList && e.target.classList.contains("x")) { closeLens(); return; }
    if (lensDrag) return;
    const b = stage.getBoundingClientRect(), qq = img(e.clientX - b.left, e.clientY - b.top);
    if (qq[0] < 0 || qq[1] < 0 || qq[0] > 1 || qq[1] > 1) { closeLens(); return; }
    const ci = HAS ? crackAt(qq[0], qq[1], 18) : -1;
    if (ci < 0) { if (sel >= 0) { sel = -1; closeLens(); redraw(); return; } }
    else if (ci !== sel && (MULTI || edit)) { sel = ci; redraw(); }
    const n = nearest(qq[0], qq[1], 26); if (U === "mm" || n) openLens(n ? n[0] : qq[0], n ? n[1] : qq[1], null); }
  lensDrag = false; }
stage.addEventListener("pointerup", up); stage.addEventListener("pointercancel", up);
stage.addEventListener("dblclick", function (e) { const b = stage.getBoundingClientRect(); zoomAt(2.5, e.clientX - b.left, e.clientY - b.top, true); });
v.addEventListener("wheel", function (e) { if ($("panel").contains(e.target)) return; e.preventDefault(); const b = stage.getBoundingClientRect();
  if (lensAt && lens.contains(e.target)) { lensZ = Math.max(2, Math.min(12, lensZ * Math.exp(-e.deltaY * 0.002))); drawLens(); return; }
  zoomAt(Math.exp(-e.deltaY * 0.0018), e.clientX - b.left, e.clientY - b.top, false); }, {passive: false});
v.addEventListener("keydown", function (e) { const st = 70, k = e.key, c = [stage.clientWidth / 2, stage.clientHeight / 2];
  if (k === "ArrowLeft") { tx += st; apply(); } else if (k === "ArrowRight") { tx -= st; apply(); } else if (k === "ArrowUp") { ty += st; apply(); }
  else if (k === "ArrowDown") { ty -= st; apply(); } else if (k === "+" || k === "=") zoomAt(1.5, c[0], c[1], true); else if (k === "-") zoomAt(1 / 1.5, c[0], c[1], true);
  else if (k === "0") fit(true); else if (k === "Escape") { closeLens(); if (sel >= 0) { sel = -1; redraw(); } } else return; e.preventDefault(); });
$("zin").onclick = function () { zoomAt(1.6, stage.clientWidth / 2, stage.clientHeight / 2, true); };
$("zout").onclick = function () { zoomAt(1 / 1.6, stage.clientWidth / 2, stage.clientHeight / 2, true); };
$("fit").onclick = function () { closeLens(); fit(true); };
$("tlay").onclick = function () { const p = $("panel"), open = p.style.display !== "block"; p.style.display = open ? "block" : "none"; this.classList.toggle("on", open); panel(); };
$("tedit").onclick = function () { edit = !edit; tool = null; this.classList.toggle("on", edit); closeLens(); redraw(); };
["zin", "zout", "fit", "tlay", "tedit"].forEach(function (id) { $(id).addEventListener("pointerdown", function (e) { e.stopPropagation(); }); });
["tl", "spark", "panel", "alert", "editbar"].forEach(function (id) { $(id).addEventListener("pointerdown", function (e) { e.stopPropagation(); }); });
window.addEventListener("resize", function () { fit(false); alertBar(); });
setTimeout(function () { $("hint").style.opacity = 0; }, 4500);
})();
</script></body></html>"""


def viewer_html(photo: bytes, data: dict | None, mask: bytes | None = None, engineer: bool = True) -> str:
    """The viewer for one crack: the photo (JPEG), the crack's numbers (view_data) and, optionally, the model's red mask.
    engineer=False is the resident's phone (review 26 Nov 2026): no corrections and no unusual-surface layer; those
    are the engineer's, on the wall. Only numbers go into the script; '</' is escaped so no text in the data can end
    it early."""
    b64 = lambda b: "data:image/jpeg;base64," + base64.b64encode(b).decode()  # noqa: E731
    if data is not None:
        data = {**data, "engineer": engineer}
    return (PAGE.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
            .replace("__PHOTO__", b64(photo)).replace("__MASK__", b64(mask) if mask else ""))
