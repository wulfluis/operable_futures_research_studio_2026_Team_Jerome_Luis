"""The crack, to explore (D83): zoom in and out, drag the photo, click the narrowest or widest point and a loupe opens
with the width measured there; drag the loupe along the crack and the width follows. The numbers live in the picture.

The width at every point comes from the measurement's profile (D81): millimetres with a bank card, else the photo's
pixels (never turned into mm without a card). The narrowest point leaves out the crack's two ends, where every crack
thins to nothing; it is shown only, the level uses the widest point (src/triage/matrix.py).
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


def _pick(n: int, most: int) -> np.ndarray:
    return np.arange(n) if n <= most else np.linspace(0, n - 1, most).round().astype(int)


def _source(result):
    """(crack measurement, unit, factor from its width numbers to the unit). mm only when the width was measured with
    a bank card; otherwise pixels, never mm (rule 7)."""
    c = result.crack
    if result.measured:
        return c, "mm", 1.0
    if c is not None and c.found and c.profile_px is not None and c.mm_per_px:
        return c, "px", 1.0 / c.mm_per_px  # a card, but the width is not trusted: shown in pixels
    m = getattr(result, "pixel_crack", None)
    if m is not None and m.found and m.profile_px is not None:
        return m, "px", 1.0  # no card: measured in the photo's pixels (core.pixel_line)
    return None, None, 1.0


def view_data(result, size: tuple[int, int] | None = None) -> dict | None:
    """The crack for the viewer, in the photo's own coordinates divided by its size (0 to 1). None: nothing to draw."""
    c, unit, k = _source(result)
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

    parts = [p for p in (getattr(c, "cracks", None) or [c]) if p is not None]
    cracks = crack_paths(parts, tree, wid, near_px, n, k)

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
    if unit == "mm" and c.width_uncertainty_mm is not None:
        widest["u"] = round(float(c.width_uncertainty_mm), 3)

    card = getattr(result, "card", None)
    mm_per_px = float(c.mm_per_px) if unit == "mm" and c.mm_per_px else None
    an = getattr(result, "anomaly", None) or None  # the autoencoder's layers (D89), shown only
    return {"unit": unit, "line": line, "net": net, "cracks": cracks, "widest": widest, "narrowest": point(j_min),
            "median": round(float(np.median(wid)), 4),
            "length": round(float(c.length_main_mm) * k, 1) if c.length_main_mm is not None else round(length_px, 1),
            "mm_per_x": None if mm_per_px is None else round(mm_per_px * W, 4),  # mm across the photo's width
            "camera": None if card is None else camera_angle(card),
            "shape": shape_data(c, path, unit),
            "defects": [{"kind": f["kind"], "label": f["label"], "c": f["centre"], "o": f["outline"]}
                        for f in an["defects"]] if an else [],
            "kin": kin_data(getattr(result, "kinematics", None), W, H),
            # D107: possible exposed steel: the traced lines may follow it, so no width is shown as the crack's
            "steel": bool((getattr(result, "exposed_steel", None) or {}).get("seen")),
            # the error of every width: one pixel, as found against a crack microscope on an old crack ([46] ch. 5.2)
            "unc": round(WIDTH_UNCERTAINTY_PX * (float(c.mm_per_px) if unit == "mm" and c.mm_per_px else 1.0), 4),
            "surface_score": None,  # not shown: not calibrated for photos at the app's scale (D89)
            "heat": ("data:image/png;base64," + base64.b64encode(an["heat_png"]).decode()) if an and an.get("heat_png") else None,
            "aspect": round(H / W, 5)}


def crack_paths(parts, tree, wid, near_px: float, n, k: float) -> list[dict]:
    """Every independent crack (D88) with every branch as a line of [x, y, width or None], its length and its widest
    point, in the viewer's unit. Points are thinned evenly so all cracks together stay under MAX_PATHS."""
    lines = []
    for ci, part in enumerate(parts):
        for b in (getattr(part, "branch_paths_px", None) or [getattr(part, "main_path_px", None)]):
            if b is not None and len(b) >= 2:
                lines.append((ci, np.asarray(b, float).reshape(-1, 2)))
    total = sum(len(b) for _, b in lines) or 1
    out = [{"paths": [], "length": None, "max": None} for _ in parts]
    for ci, b in lines:
        idx = _pick(len(b), max(2, int(round(MAX_PATHS * len(b) / total))))
        d, j = tree.query(b[idx])
        out[ci]["paths"].append([n(b[i]) + [None if dd > near_px else round(float(wid[jj]), 4)]
                                 for i, dd, jj in zip(idx, d, j)])
    for ci, part in enumerate(parts):
        if getattr(part, "length_main_mm", None) is not None:
            out[ci]["length"] = round(float(part.length_main_mm) * k, 1)
        if getattr(part, "width_max_mm", None) is not None:
            out[ci]["max"] = round(float(part.width_max_mm) * k, 4)
    return [c for c in out if c["paths"]]


MAX_STATIONS = 60  # arrows of all cracks together (more hide the crack)


def kin_data(kins, W: int, H: int) -> list | None:
    """Opening and sliding per crack (D99) for the viewer: its summary, and its stations as
    [x, y, opening dx, opening dy, sliding dx, sliding dy] (photo shares; the arrows in photo pixels), in crack order."""
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
            n, t, d = np.asarray(x.normal), np.asarray(x.tangent), np.asarray(x.vector_px, float)
            op, sl = float(d @ n), (None if x.sliding is None else float(d @ t))
            rows.append([round(x.point_px[0] / W, 5), round(x.point_px[1] / H, 5), round(op * n[0], 2), round(op * n[1], 2),
                         None if sl is None else round(sl * t[0], 2), None if sl is None else round(sl * t[1], 2)])
        out.append({**k.to_dict(), "st": rows})
    return out if any(o is not None for o in out) else None


def shape_data(c, path: np.ndarray, unit: str) -> dict | None:
    """The crack's shape as the type rules read it (src/triage/pattern_rules.py), for the viewer's second row.
    The angle is to the photo's up-down direction (the phone held upright); without a card it is the photo's, not the
    wall's (D65). `lean` only tilts the little icon: +1 the top leans right, -1 left, from the main line's two ends."""
    a = getattr(c, "angle_to_vertical_deg", None)
    if a is None:
        return None
    top, bottom = (path[0], path[-1]) if path[0][1] <= path[-1][1] else (path[-1], path[0])
    r = lambda v, k=2: None if v is None else round(float(v), k)  # noqa: E731
    return {"angle": r(a, 1), "lean": 1 if top[0] >= bottom[0] else -1, "in_photo": unit == "px",
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
:root { --fg:#eef1f5; --muted:#8b95a3; --acc:#22d3ee; --glass:rgba(12,14,18,.84); --line:rgba(255,255,255,.12); }
* { box-sizing:border-box; }
html,body { margin:0; height:100%; background:#050607; font:13px/1.35 -apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI",Helvetica,Arial,sans-serif; color:var(--fg); -webkit-font-smoothing:antialiased; }
.mono { font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace; }
#v { position:relative; height:100%; border-radius:18px; overflow:hidden; background:#050607; border:1px solid var(--line);
     user-select:none; -webkit-user-select:none; touch-action:none; outline:none; }
#v:focus-visible { border-color:var(--acc); }
#stage { position:absolute; inset:0; cursor:grab; overflow:hidden; }
#stage.drag { cursor:grabbing; }
#layer { position:absolute; left:0; top:0; transform-origin:0 0; will-change:transform; }
#layer img { position:absolute; left:0; top:0; width:100%; height:100%; display:block; pointer-events:none; }
#mask, #heat { opacity:0; transition:opacity .2s; }
.df { position:absolute; transform:translate(-50%,-50%); pointer-events:none; white-space:nowrap; padding:3px 8px; border-radius:999px;
      background:var(--glass); border:1px dashed currentColor; font-size:11.5px; font-weight:600; }
#dtags.off { display:none; }
#lines { position:absolute; left:0; top:0; width:100%; height:100%; overflow:visible; pointer-events:none; transition:opacity .2s; }
.mk { position:absolute; transform:translate(-50%,-50%); pointer-events:auto; cursor:pointer; }
.mk .ring { width:30px; height:30px; border-radius:50%; border:2px solid currentColor; box-shadow:0 0 0 3px rgba(0,0,0,.35), 0 0 18px currentColor;
            animation:pulse 2.2s ease-in-out infinite; }
.mk .tag { position:absolute; left:22px; top:-12px; white-space:nowrap; padding:3px 8px; border-radius:999px; background:var(--glass);
           border:1px solid currentColor; font-weight:600; font-size:12px; }
.mk .tag b { color:#fff; font-weight:700; }
.mk:hover .ring { transform:scale(1.15); }
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
        white-space:nowrap; }
.chip .k { font:600 10px/1 ui-monospace,"SF Mono",Menlo,monospace; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); }
.chip .val { font-weight:700; font-size:14px; font-variant-numeric:tabular-nums; }
button.chip { cursor:pointer; color:var(--fg); font:inherit; transition:border-color .15s, background .15s, transform .1s; }
button.chip:hover { border-color:currentColor; }
button.chip:active { transform:scale(.96); }
button.chip.on { border-color:var(--acc); color:var(--acc); }
.hud .row { flex-basis:100%; height:0; }
.chip.sm { padding:4px 9px; } .chip.sm .val { font-size:12.5px; }
.btn { width:34px; height:34px; border-radius:50%; padding:0; justify-content:center; align-items:center; font-size:16px; font-weight:600; }
#tl { left:12px; top:12px; right:120px; }
#tr { right:12px; top:12px; flex-direction:column; align-items:flex-end; }
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
@media (max-width:560px) { #spark { height:56px; } #spark svg { height:30px; } .chip { padding:5px 9px; } .chip .val { font-size:12.5px; } #tl { right:56px; } .opt { display:none; } .btn { width:30px; height:30px; } }
</style></head><body>
<div id="v" tabindex="0" aria-label="The crack photo: drag to move, scroll or pinch to zoom, click to open the loupe">
  <div id="stage">
    <div id="layer"><img id="photo" alt="Crack photo"><img id="mask" alt=""><img id="heat" alt=""><svg id="lines"></svg></div>
    <div id="dtags"></div><div id="marks"></div><div id="hov"></div>
    <div id="lens"><div class="glass"></div><svg></svg><div class="read"></div><div class="x" title="Close (Esc)">✕</div></div>
  </div>
  <div class="hud" id="tl"></div>
  <div class="hud" id="tr">
    <button class="chip btn" id="zin" title="Zoom in (+)">+</button>
    <button class="chip btn" id="zout" title="Zoom out (−)">−</button>
    <button class="chip btn" id="fit" title="Whole photo (0)">⤢</button>
    <button class="chip btn" id="tline" title="Width line on or off">≋</button>
    <button class="chip btn" id="tcol" title="Colours: each crack its own, or by width">◈</button>
    <button class="chip btn" id="tkin" title="How the crack moved: opening (across, cyan) and sliding (along, amber), read from its two faces (Pantoja-Rosero et al. 2022, as in Corres Sojo 2024)">⇄</button>
    <button class="chip btn" id="tdef" title="Other defects (possible rust, dark patches, other unusual areas) on or off">◇</button>
    <button class="chip btn" id="theat" title="Heatmap: what looks different from sound concrete (autoencoder)">▦</button>
    <button class="chip btn" id="tmask" title="The model's red mask on or off">◐</button>
  </div>
  <div class="hud" id="bot"><div id="spark"></div><div id="scale"><div class="chip mono" id="zoomv">1.0×</div><div class="chip" id="scalev"><span><div id="bar"></div></span><span class="val mono" id="scalet"></span></div></div></div>
  <div id="hint">Scroll or pinch to zoom · drag to move · click the crack</div>
</div>
<script>
(function () {
const D = __DATA__, PHOTO = "__PHOTO__", MASK = "__MASK__";
const $ = function (id) { return document.getElementById(id); };
const v = $("v"), stage = $("stage"), layer = $("layer"), photo = $("photo"), mask = $("mask"), lines = $("lines");
const lens = $("lens"), glass = lens.querySelector(".glass"), lsvg = lens.querySelector("svg"), read = lens.querySelector(".read");
const C = D || {}, U = C.unit || "", HAS = !!(D && D.line && D.line.length);
let W = 1, H = 1, s = 1, s0 = 1, tx = 0, ty = 0, lensAt = null, lensZ = 4, showLine = true, showMask = false;
const R = function () { return Math.min(110, Math.max(70, Math.min(stage.clientWidth, stage.clientHeight) * 0.22)); };
function fmt(w) { if (w == null) return "–"; return U === "mm" ? (w < 1 ? w.toFixed(2) : w.toFixed(1)) + " mm" : w.toFixed(1) + " px"; }
function lenFmt(l) { return U === "mm" ? (l >= 1000 ? (l / 1000).toFixed(2) + " m" : Math.round(l) + " mm") : Math.round(l) + " px"; }
// width colours: violet (thin) to pink, orange and yellow (wide); never green, which reads as "safe"
const STOPS = [[124,58,237],[236,72,153],[251,146,60],[253,224,71]];
function col(t) { t = Math.max(0, Math.min(1, t)) * (STOPS.length - 1); const i = Math.min(STOPS.length - 2, Math.floor(t)), f = t - i, a = STOPS[i], b = STOPS[i + 1];
  return "rgb(" + [0,1,2].map(function (k) { return Math.round(a[k] + (b[k] - a[k]) * f); }).join(",") + ")"; }
const WMAX = HAS ? Math.max(C.widest.w, 1e-6) : 1;
// every independent crack its own colour (D88); never green, which reads as "safe"
const PAL = ["#f472b6", "#38bdf8", "#fbbf24", "#a78bfa", "#fb923c", "#e879f9", "#60a5fa", "#f87171"];
const CR = C.cracks || [], MULTI = CR.length > 1;
let byCrack = MULTI;
// other defects, separate from the cracks (D89): outlined, dashed, each with its label; shown only
const DF = C.defects || [], DCOL = {steel: "#ef4444", spalling: "#ff5c7a", rust: "#f97316", dark: "#94a3b8", other: "#e2e8f0"},
  DSHORT = {steel: "Steel?", spalling: "Spalling?", rust: "Rust?", dark: "Dark patch", other: "Unusual"};
let showDef = DF.length > 0, showHeat = false;
// how the crack moved (D99): opening across it, sliding along it, from its two faces; shown only
const KIN = C.kin || [], HASKIN = KIN.some(function (k) { return k && k.st && k.st.length; });
let showKin = HASKIN;

photo.onload = function () { W = photo.naturalWidth; H = photo.naturalHeight; layer.style.width = W + "px"; layer.style.height = H + "px";
  lines.setAttribute("viewBox", "0 0 " + W + " " + H); drawLines(); fit(false); hud(); spark(); };
photo.src = PHOTO; if (MASK) mask.src = MASK; else $("tmask").style.display = "none";
if (!HAS) { $("tline").style.display = "none"; $("spark").style.visibility = "hidden"; }
if (!MULTI) $("tcol").style.display = "none";
if (!DF.length) $("tdef").style.display = "none";
if (!HASKIN) $("tkin").style.display = "none";
if (C.heat) $("heat").src = C.heat; else $("theat").style.display = "none";

function apply() { layer.style.transform = "translate(" + tx + "px," + ty + "px) scale(" + s + ")";
  $("zoomv").textContent = (s / s0).toFixed(s / s0 < 10 ? 1 : 0) + "×"; place(); placeDefects(); scalebar(); if (lensAt) drawLens(); }
function fit(anim) { const sw = stage.clientWidth, sh = stage.clientHeight; s0 = Math.min(sw / W, sh / H);
  go(s0, (sw - W * s0) / 2, (sh - H * s0) / 2, anim); }
let raf = null;
function go(ns, ntx, nty, anim) { cancelAnimationFrame(raf);
  if (!anim || matchMedia("(prefers-reduced-motion: reduce)").matches) { s = ns; tx = ntx; ty = nty; apply(); return; }
  // the flight follows the photo point at the centre of the view, so the view never leaves the photo on the way
  const sw = stage.clientWidth / 2, sh = stage.clientHeight / 2, s1 = s, t0 = performance.now();
  const ca = [(sw - tx) / (W * s), (sh - ty) / (H * s)], cb = [(sw - ntx) / (W * ns), (sh - nty) / (H * ns)];
  (function step(t) { const k = Math.min(1, (t - t0) / 420), e = 1 - Math.pow(1 - k, 3);
    s = s1 * Math.pow(ns / s1, e); const cx = ca[0] + (cb[0] - ca[0]) * e, cy = ca[1] + (cb[1] - ca[1]) * e;
    tx = sw - cx * W * s; ty = sh - cy * H * s; apply(); if (k < 1) raf = requestAnimationFrame(step); })(t0); }
function clampS(x) { return Math.max(s0 * 0.6, Math.min(s0 * 60, x)); }
function zoomAt(f, cx, cy, anim) { const ns = clampS(s * f), k = ns / s; go(ns, cx - (cx - tx) * k, cy - (cy - ty) * k, anim); }
function scr(x, y) { return [tx + x * W * s, ty + y * H * s]; }
function img(X, Y) { return [(X - tx) / (W * s), (Y - ty) / (H * s)]; }
function flyTo(p, z) { const ns = clampS(Math.max(s, s0 * (z || 6))), sw = stage.clientWidth, sh = stage.clientHeight;
  go(ns, sw / 2 - p.x * W * ns, sh / 2.2 - p.y * H * ns, true); }

function drawLines() { if (!HAS) return; let out = "";
  const seg = function (L, crackColour) {  // one line: by crack (its colour) or by width (gaps where no width was read)
    for (let i = 1; i < L.length; i++) { const a = L[i - 1], b = L[i], w = (a[2] != null ? a[2] : b[2]);
      const c = crackColour || (w == null ? "#ffffff" : col(w / WMAX)), o = crackColour ? .95 : (w == null ? .35 : .95);
      out += '<line x1="' + a[0] * W + '" y1="' + a[1] * H + '" x2="' + b[0] * W + '" y2="' + b[1] * H + '" stroke="' + c +
        '" stroke-width="3" vector-effect="non-scaling-stroke" stroke-linecap="round" opacity="' + o + '"/>'; } };
  if (CR.length) CR.forEach(function (cr, ci) { cr.paths.forEach(function (P) { seg(P, byCrack ? PAL[ci % PAL.length] : null); }); });
  else seg(C.line, null);
  if (showDef) DF.forEach(function (d) { const pts = d.o.map(function (q) { return (q[0] * W).toFixed(1) + "," + (q[1] * H).toFixed(1); }).join(" ");
    out += '<polygon points="' + pts + '" fill="' + DCOL[d.kind] + '" fill-opacity=".12" stroke="' + DCOL[d.kind] +
      '" stroke-width="2" stroke-dasharray="6 4" vector-effect="non-scaling-stroke"/>'; });
  if (showKin) out += kinArrows();
  lines.innerHTML =
    '<g style="filter:drop-shadow(0 0 2px rgba(0,0,0,.9))">' + out + "</g>"; }
function kinArrows() {  // at each point: a double arrow across the crack (opening, cyan), one along it (sliding, amber)
  // the heads are drawn as lines in the photo's own units, so an arrow keeps its shape at every zoom, and a small
  // sliding gives a small arrow
  let g = ""; const ln = function (x1, y1, x2, y2, c) { return '<line x1="' + x1.toFixed(1) + '" y1="' + y1.toFixed(1) + '" x2="' + x2.toFixed(1) + '" y2="' + y2.toFixed(1) +
    '" stroke="' + c + '" stroke-width="2" vector-effect="non-scaling-stroke" stroke-linecap="round"/>'; };
  const head = function (x, y, ux, uy, h, c) { return ln(x, y, x - ux * h + uy * h * 0.6, y - uy * h - ux * h * 0.6, c) + ln(x, y, x - ux * h - uy * h * 0.6, y - uy * h + ux * h * 0.6, c); };
  KIN.forEach(function (k) { if (!k || !k.st) return; k.st.forEach(function (q) {
    const x = q[0] * W, y = q[1] * H, ox = q[2], oy = q[3], m = Math.hypot(ox, oy) || 1, ux = ox / m, uy = oy / m, L = Math.max(8, 1.4 * m), h = L * 0.35;
    g += ln(x - ux * L, y - uy * L, x + ux * L, y + uy * L, "#22d3ee") + head(x + ux * L, y + uy * L, ux, uy, h, "#22d3ee") + head(x - ux * L, y - uy * L, -ux, -uy, h, "#22d3ee");
    if (q[4] != null) { const sm = Math.hypot(q[4], q[5]), S = L * Math.min(2, sm / m); if (S > 1.5) { const vx = q[4] / sm, vy = q[5] / sm;
      g += ln(x - vx * S, y - vy * S, x + vx * S, y + vy * S, "#fbbf24") + head(x + vx * S, y + vy * S, vx, vy, Math.min(h, S * 0.6), "#fbbf24"); } } }); });
  return g; }

function place() { const m = $("marks"); if (!HAS || C.steel) { m.innerHTML = ""; return; }
  if (!m.childElementCount) {
    [["widest", "max", "Widest"], ["narrowest", "min", "Narrowest"]].forEach(function (d) {
      const p = C[d[0]], el = document.createElement("div"); el.className = "mk " + d[1]; el.dataset.k = d[0];
      el.innerHTML = '<div class="ring"></div><div class="tag">' + d[2] + ' <b>' + fmt(p.w) + "</b></div>";
      el.addEventListener("pointerdown", function (e) { e.stopPropagation(); });
      el.addEventListener("click", function (e) { e.stopPropagation(); flyTo(p, 8); openLens(p.x, p.y, d[0]); });
      m.appendChild(el); }); }
  Array.prototype.forEach.call(m.children, function (el) { const p = C[el.dataset.k], q = scr(p.x, p.y); el.style.left = q[0] + "px"; el.style.top = q[1] + "px"; el.classList.toggle("left", q[0] > stage.clientWidth - 170); }); }

function placeDefects() { const box = $("dtags"); box.classList.toggle("off", !showDef);
  if (!box.childElementCount) DF.forEach(function (d) { const el = document.createElement("div"); el.className = "df";
    el.style.color = DCOL[d.kind]; el.textContent = DSHORT[d.kind]; el.title = d.label + " (autoencoder and colour rule: an engineer decides)"; box.appendChild(el); });
  Array.prototype.forEach.call(box.children, function (el, i) { const q = scr(DF[i].c[0], DF[i].c[1]); el.style.left = q[0] + "px"; el.style.top = q[1] + "px"; }); }

function nearest(x, y, px) { if (!HAS) return null; let best = null, bd = 1e18;
  C.net.forEach(function (n) { const dx = (n[0] - x) * W * s, dy = (n[1] - y) * H * s, d = dx * dx + dy * dy; if (d < bd) { bd = d; best = n; } });
  return Math.sqrt(bd) <= px ? best : null; }

function openLens(x, y, kind) { lensAt = {x: x, y: y, kind: kind || null}; $("marks").classList.add("lensopen"); lens.style.display = "block"; $("hint").style.opacity = 0; drawLens(); }
function closeLens() { lensAt = null; $("marks").classList.remove("lensopen"); lens.style.display = "none"; }
function drawLens() { const r = R(), p = scr(lensAt.x, lensAt.y), touch = matchMedia("(pointer: coarse)").matches;
  const cx = p[0], cy = touch ? p[1] - r - 30 : p[1];   // on a phone the loupe sits above the finger
  lens.style.left = (cx - r) + "px"; lens.style.top = (cy - r) + "px"; lens.style.width = 2 * r + "px"; lens.style.height = 2 * r + "px";
  const k = s * lensZ;
  glass.style.width = glass.style.height = 2 * r + "px";
  glass.style.backgroundImage = "url(" + (showMask && MASK ? MASK : PHOTO) + ")";
  glass.style.backgroundSize = (W * k) + "px " + (H * k) + "px";
  glass.style.backgroundPosition = (r - lensAt.x * W * k) + "px " + (r - lensAt.y * H * k) + "px";
  let n = lensAt.kind ? C[lensAt.kind] : null;
  if (!n) { const q = nearest(lensAt.x, lensAt.y, 22); if (q) n = {x: q[0], y: q[1], w: q[2], e: q.length > 3 ? [[q[3], q[4]], [q[5], q[6]]] : null}; }
  const L = function (u, w) { return [r + (u - lensAt.x) * W * k, r + (w - lensAt.y) * H * k]; };
  let g = '<circle cx="' + r + '" cy="' + r + '" r="' + (r - 1) + '" fill="none" stroke="rgba(255,255,255,.15)"/>' +
          '<path d="M' + (r - 9) + " " + r + "h6M" + (r + 3) + " " + r + "h6M" + r + " " + (r - 9) + "v6M" + r + " " + (r + 3) + 'v6" stroke="rgba(255,255,255,.7)" stroke-width="1"/>';
  if (n && n.e) { const a = L(n.e[0][0], n.e[0][1]), b = L(n.e[1][0], n.e[1][1]); let dx = b[0] - a[0], dy = b[1] - a[1]; const dl = Math.hypot(dx, dy) || 1; dx /= dl; dy /= dl;
    const ex = function (q, sgn) { return [q[0] - dx * sgn * 14, q[1] - dy * sgn * 14]; }, a2 = ex(a, 1), b2 = ex(b, -1), t = 8;
    g += '<g stroke="#fff" stroke-width="2" stroke-linecap="round" style="filter:drop-shadow(0 0 2px #000)">' +
      '<line x1="' + a2[0] + '" y1="' + a2[1] + '" x2="' + a[0] + '" y2="' + a[1] + '"/><line x1="' + b2[0] + '" y1="' + b2[1] + '" x2="' + b[0] + '" y2="' + b[1] + '"/>' +
      '<line x1="' + (a[0] - dy * t) + '" y1="' + (a[1] + dx * t) + '" x2="' + (a[0] + dy * t) + '" y2="' + (a[1] - dx * t) + '"/>' +
      '<line x1="' + (b[0] - dy * t) + '" y1="' + (b[1] + dx * t) + '" x2="' + (b[0] + dy * t) + '" y2="' + (b[1] - dx * t) + '"/></g>';
    const c = col(n.w / WMAX); g += '<line x1="' + a[0] + '" y1="' + a[1] + '" x2="' + b[0] + '" y2="' + b[1] + '" stroke="' + c + '" stroke-width="3"/>'; }
  lsvg.setAttribute("width", 2 * r); lsvg.setAttribute("height", 2 * r); lsvg.innerHTML = g;
  const name = lensAt.kind === "widest" ? "Widest point · used by the level" : lensAt.kind === "narrowest" ? "Narrowest point · shown only" : (n ? "Width here" : "No crack under the loupe");
  read.style.top = (2 * r + 10) + "px";
  read.innerHTML = '<div class="big ' + (lensAt.kind === "widest" ? "max" : lensAt.kind === "narrowest" ? "min" : "") + '">' + (n ? fmt(n.w) + (lensAt.kind === "widest" && C.widest.u ? ' <span class="small">± ' + C.widest.u.toFixed(2) + "</span>" : "") : "–") +
    '</div><div class="small">' + name + " · loupe " + (s / s0 * lensZ).toFixed(0) + "×" + (U === "px" ? " · no card, no mm" : "") + "</div>";
  const x = lens.querySelector(".x"); x.style.left = (2 * r - 18) + "px"; x.style.top = "-4px"; }

function cam(c) {  // the camera at its angle to the wall: a wall line, the camera and its line of sight
  const a = (90 - c.to_wall) * Math.PI / 180, x = 9 + 9 * Math.sin(a), y = 10 + 9 * Math.cos(a), bad = c.status === "retake";
  const icon = '<svg width="20" height="20" viewBox="0 0 20 20" style="align-self:center"><line x1="2" y1="1" x2="18" y2="1" stroke="#8b95a3" stroke-width="2"/>' +
    '<line x1="' + x + '" y1="' + y + '" x2="10" y2="1" stroke="' + (bad ? "#ff4d6a" : "#22d3ee") + '" stroke-width="1.5" stroke-dasharray="2 2"/>' +
    '<circle cx="' + x + '" cy="' + y + '" r="3" fill="' + (bad ? "#ff4d6a" : "#22d3ee") + '"/></svg>';
  return '<span class="chip opt" title="Camera angle to the wall, from the bank card: 90° is straight on. Left-right ' + c.left_right +
    '°, up-down ' + c.up_down + '° off straight.">' + icon + '<span class="k">Camera</span><span class="val' + (bad ? ' max' : '') + '">' +
    Math.round(c.to_wall) + '° <span class="k" style="letter-spacing:0">to wall</span></span><span class="k" style="letter-spacing:0">↔' +
    Math.round(c.left_right) + '° ↕' + Math.round(c.up_down) + '°</span></span>'; }
function shapeRow(sh) {  // the crack's shape, as the type rules read it; each chip explains itself on hover
  const a = sh.angle * Math.PI / 180, x = 10 + sh.lean * 8 * Math.sin(a), y = 10 - 8 * Math.cos(a);
  const icon = '<svg width="20" height="20" viewBox="0 0 20 20" style="align-self:center"><line x1="10" y1="1" x2="10" y2="19" stroke="#8b95a3" stroke-width="1.5" stroke-dasharray="2 2"/>' +
    '<line x1="' + (20 - x) + '" y1="' + (20 - y) + '" x2="' + x + '" y2="' + y + '" stroke="#22d3ee" stroke-width="2" stroke-linecap="round"/></svg>';
  const sm = function (k, val, tip) { return '<span class="chip sm" title="' + tip + '"><span class="k">' + k + '</span><span class="val">' + val + "</span></span>"; };
  const f = function (v, d) { return v === null || v === undefined ? "–" : v.toFixed(d); };
  return '<span class="row"></span>' +
    '<span class="chip sm" title="Angle of the main crack to vertical: 0° vertical, 90° horizontal. Vertical is the photo\'s up-down, so it assumes the phone was held upright' +
    (sh.in_photo ? '; no card: read in the photo, not on the wall' : '') + '.">' + icon + '<span class="k">Angle</span><span class="val">' + Math.round(sh.angle) + '°</span>' +
    '<span class="k" style="letter-spacing:0">' + (sh.in_photo ? "from vertical · in photo" : "from vertical") + '</span></span>' +
    sm("Straight", f(sh.straightness, 2), "Straightness: distance between the two ends divided by the length of the main crack. 1 = a straight line; below 0.6 the direction is not read.") +
    sm("Coherence", f(sh.coherence, 2), "How much the whole crack runs one way: 1 = all in one direction, 0 = no main direction (a network).") +
    sm("Cells", String(sh.loops), "Closed cells: areas fully enclosed by crack lines, like the cells of a net. A single crack has 0; from 2 the rules read a network of surface cracks (map cracking).") +
    sm("Branches", f(sh.branches, 1) + "×", "Length of all branches divided by the main crack's length. 1.0× = no side branches."); }
function kinRow() {  // how the crack moved (D99): the main crack's opening, sliding and the reading
  const k = KIN.filter(function (x) { return x && x.opening_median != null; })[0]; if (!k) return "";
  const f = function (v) { if (v == null) return "–"; return k.unit === "mm" ? (v < 1 ? v.toFixed(2) : v.toFixed(1)) + " mm" : v.toFixed(1) + " px"; };
  const tip = "Read from the crack's two faces: they were one surface, so the bumps of one face match the hollows of the other. " +
    "Moving one face back onto the other gives how the crack moved: across it (opening) and along it (sliding). Method of Pantoja-Rosero et al. 2022, " +
    "used in E. Corres Sojo's thesis (EPFL 2024). One photo: the movement since the crack formed. Shown only; an engineer reads it, the level does not use it.";
  const sm = function (k2, val, c) { return '<span class="chip sm" title="' + tip + '"><span class="k"' + (c ? ' style="color:' + c + '"' : "") + ">" + k2 + '</span><span class="val">' + val + "</span></span>"; };
  return '<span class="row"></span>' + sm("Opening", f(k.opening_median), "#22d3ee") + sm("Sliding", k.sliding_median == null ? "not readable" : f(k.sliding_median), "#fbbf24") +
    sm("Movement", k.reading, null); }
function defectChips() {  // counts of the other defects and the photo's surface score (D89)
  const n = {steel: 0, spalling: 0, rust: 0, dark: 0, other: 0}; DF.forEach(function (d) { n[d.kind]++; });
  const TIP = {steel: "Possible exposed steel: SAM 3.1 asked for bars. The concrete cover is gone here; the crack widths on this photo are not used, since the lines may follow the steel.",
    spalling: "Possible spalling or missing plaster: SAM 3.1 and written rules (an area, not a crack line, clearly different in colour from the wall around it). Unverified, tested on a few photos only.",
    rust: "Possible rust stains: what the autoencoder found unusual, labelled by its colour.", dark: "Possible dark patches (damp or dirt?): what the autoencoder found unusual, labelled by its colour.",
    other: "Other unusual areas the autoencoder found."};
  let h = ""; [["steel", "Steel?"], ["spalling", "Spalling?"], ["rust", "Rust?"], ["dark", "Dark"], ["other", "Unusual"]].forEach(function (k) { if (n[k[0]])
    h += '<span class="chip" style="border-style:dashed;border-color:' + DCOL[k[0]] + '" title="' + TIP[k[0]] +
      ' Separate from the cracks; shown only, an engineer decides."><span class="k" style="color:' + DCOL[k[0]] + '">' + k[1] + '</span><span class="val">' + n[k[0]] + "</span></span>"; });
  if (C.surface_score != null) h += '<span class="chip opt" title="How unusual the whole surface is compared with the sound concrete the autoencoder learned: above 1, more unusual than 95 % of it. Shown only."><span class="k">Surface</span><span class="val' +
    (C.surface_score > 1 ? " max" : "") + '">' + C.surface_score.toFixed(2) + "</span></span>";
  return h; }
function hud() { const tl = $("tl"); if (!HAS) { tl.innerHTML = '<span class="chip"><span class="k">Crack</span><span class="val">no line to measure</span></span>'; return; }
  const chip = function (id, k, val, cls) { return '<button class="chip ' + (cls || "") + '" id="' + id + '"><span class="k">' + k + '</span><span class="val">' + val + "</span></button>"; };
  if (C.steel) {  // D107: the lines may follow the steel and the broken concrete: their widths are not the crack's
    tl.innerHTML = '<span class="chip" style="border-color:#ef4444" title="SAM 3.1 sees possible exposed steel. The traced lines may follow the steel and the broken concrete, so their widths are not used and not shown. An expert should look."><span class="k" style="color:#ef4444">Exposed steel?</span><span class="val">widths not used</span></span>' +
      defectChips();
    return; }
  tl.innerHTML = chip("cmax", "Max", fmt(C.widest.w), "max") + chip("cmin", "Min", fmt(C.narrowest.w), "min") +
    '<span class="chip opt"><span class="k">Median</span><span class="val">' + fmt(C.median) + "</span></span>" +
    '<span class="chip"><span class="k">Length</span><span class="val">' + lenFmt(C.length) + "</span></span>" +
    (U === "px" ? '<span class="chip opt"><span class="k">Scale</span><span class="val">no card · pixels</span></span>' : "") +
    (MULTI ? '<span class="chip" title="' + CR.map(function (cr, i) { return "Crack " + (i + 1) + ": " + (cr.length != null ? lenFmt(cr.length) : "–") +
      ", widest " + fmt(cr.max); }).join("&#10;") + '"><span class="k">Cracks</span><span class="val">' + CR.length + '</span>' +
      CR.map(function (cr, i) { return '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:' + PAL[i % PAL.length] + '"></span>'; }).join("") + "</span>" : "") +
    defectChips() + (C.shape ? shapeRow(C.shape) : "") + kinRow();
  $("cmax").onclick = function () { flyTo(C.widest, 8); openLens(C.widest.x, C.widest.y, "widest"); };
  $("cmin").onclick = function () { flyTo(C.narrowest, 8); openLens(C.narrowest.x, C.narrowest.y, "narrowest"); }; }

function spark() { if (!HAS) return; const L = C.line, n = L.length, el = $("spark");
  if (C.steel) { el.style.visibility = "hidden"; return; }  // D107: no width chart for lines that may follow steel
  const pts = [], ws = L.map(function (q) { return q[2]; });
  let st = 0; const cum = [0]; for (let i = 1; i < n; i++) { st += Math.hypot((L[i][0] - L[i - 1][0]) * W, (L[i][1] - L[i - 1][1]) * H); cum.push(st); }
  const X = function (i) { return 1000 * cum[i] / (st || 1); }, Y = function (w) { return 38 - 34 * (w / WMAX); };
  let path = "", area = "", open = false;
  ws.forEach(function (w, i) { if (w == null) { open = false; return; } path += (open ? "L" : "M") + X(i).toFixed(1) + " " + Y(w).toFixed(1); open = true; });
  // the error band: every width ± one pixel ([46] ch. 5.2: photo widths matched a crack microscope within one pixel)
  const UN = C.unc || 0, Yc = function (w) { return Math.max(0, Math.min(40, Y(w))); }; let runs = [], cur = [];
  ws.forEach(function (w, i) { if (w == null) { if (cur.length) runs.push(cur); cur = []; return; } cur.push(i); }); if (cur.length) runs.push(cur);
  runs.forEach(function (r) { if (r.length < 2 || !UN) return; area += "M" + r.map(function (i) { return X(i).toFixed(1) + " " + Yc(ws[i] + UN).toFixed(1); }).join("L") +
    "L" + r.slice().reverse().map(function (i) { return X(i).toFixed(1) + " " + Yc(Math.max(0, ws[i] - UN)).toFixed(1); }).join("L") + "Z"; });
  let grad = ""; for (let i = 0; i <= 4; i++) grad += '<stop offset="' + (i * 25) + '%" stop-color="' + col(1 - i / 4) + '"/>';
  el.innerHTML = '<div class="cap"><span>Width along the crack' + (UN ? ' <span title="Grey band: the error of every width, one pixel either way (' + fmt(UN) +
    '). On an old bridge crack, photo widths matched a crack microscope within one pixel at most points (E. Corres Sojo, EPFL 2024, ch. 5.2); dust and worn crack edges make it larger. Wipe the crack before the photo." style="text-transform:none;letter-spacing:0">· ± ' + fmt(UN) + '</span>' : "") +
    '</span><span id="spv"></span></div><svg viewBox="0 0 1000 40" preserveAspectRatio="none">' +
    '<defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1">' + grad + '</linearGradient></defs>' +
    (area ? '<path d="' + area + '" fill="#8b95a3" fill-opacity=".28" stroke="none"/>' : "") +
    '<path d="' + path + '" fill="none" stroke="url(#g)" stroke-width="2.2" vector-effect="non-scaling-stroke"/>' +
    '<line id="spl" x1="0" x2="0" y1="0" y2="40" stroke="#fff" stroke-width="1" vector-effect="non-scaling-stroke" opacity="0"/></svg>';
  const pick = function (e) { const b = el.querySelector("svg").getBoundingClientRect(), f = Math.max(0, Math.min(1, (e.clientX - b.left) / b.width));
    let i = 0; while (i < n - 1 && X(i) / 1000 < f) i++; return i; };
  el.onpointermove = function (e) { const i = pick(e), q = L[i], p = scr(q[0], q[1]), h = $("hov");
    h.style.display = "block"; h.style.left = p[0] + "px"; h.style.top = p[1] + "px";
    const l = $("spl"); l.setAttribute("x1", X(i)); l.setAttribute("x2", X(i)); l.setAttribute("opacity", 1);
    $("spv").textContent = fmt(q[2]) + (UN && q[2] != null ? " ± " + fmt(UN) : "") + " · at " + lenFmt(cum[i] * (C.length / (st || 1))); };
  el.onpointerleave = function () { $("hov").style.display = "none"; $("spl").setAttribute("opacity", 0); $("spv").textContent = ""; };
  el.onclick = function (e) { const q = L[pick(e)]; flyTo({x: q[0], y: q[1]}, 8); openLens(q[0], q[1], null); }; }

function scalebar() { const px = W * s;   // screen px across the photo
  let per = C.mm_per_x ? px / C.mm_per_x : px / W, unit = C.mm_per_x ? "mm" : "px";  // screen px per mm (or per photo px)
  const nice = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000];
  let L = nice[0]; for (let i = 0; i < nice.length; i++) { if (nice[i] * per <= 130) L = nice[i]; }
  $("bar").style.width = Math.max(8, L * per) + "px"; $("scalet").textContent = (C.mm_per_x ? "≈ " : "") + (L >= 1000 && unit === "mm" ? L / 1000 + " m" : L + " " + unit); }

// pointer: drag to move, pinch to zoom, a click opens the loupe; dragging the loupe moves it along the photo
const ptr = new Map(); let moved = 0, pinch = null, lensDrag = false;
stage.addEventListener("pointerdown", function (e) { v.focus({preventScroll: true}); stage.setPointerCapture(e.pointerId); ptr.set(e.pointerId, [e.clientX, e.clientY]); moved = 0;
  lensDrag = lens.contains(e.target) && !e.target.classList.contains("x");
  if (ptr.size === 2) { const a = Array.from(ptr.values()); pinch = {d: Math.hypot(a[0][0] - a[1][0], a[0][1] - a[1][1]), s: s}; }
  if (!lensDrag) stage.classList.add("drag"); $("hint").style.opacity = 0; });
stage.addEventListener("pointermove", function (e) { if (!ptr.has(e.pointerId)) return; const last = ptr.get(e.pointerId), dx = e.clientX - last[0], dy = e.clientY - last[1];
  ptr.set(e.pointerId, [e.clientX, e.clientY]); moved += Math.abs(dx) + Math.abs(dy);
  if (ptr.size === 2 && pinch) { const a = Array.from(ptr.values()), d = Math.hypot(a[0][0] - a[1][0], a[0][1] - a[1][1]), b = stage.getBoundingClientRect();
    zoomAt(pinch.s * d / pinch.d / s, (a[0][0] + a[1][0]) / 2 - b.left, (a[0][1] + a[1][1]) / 2 - b.top, false); return; }
  if (lensDrag && lensAt) { lensAt.x += dx / (W * s); lensAt.y += dy / (H * s); lensAt.kind = null; drawLens(); return; }
  tx += dx; ty += dy; apply(); });
function up(e) { if (!ptr.has(e.pointerId)) return; ptr.delete(e.pointerId); if (ptr.size < 2) pinch = null; stage.classList.remove("drag");
  if (moved < 6 && ptr.size === 0 && e.type === "pointerup") {
    if (e.target.classList && e.target.classList.contains("x")) { closeLens(); return; }
    if (lensDrag) return;
    const b = stage.getBoundingClientRect(), q = img(e.clientX - b.left, e.clientY - b.top);
    if (q[0] < 0 || q[1] < 0 || q[0] > 1 || q[1] > 1) { closeLens(); return; }
    const n = nearest(q[0], q[1], 26); openLens(n ? n[0] : q[0], n ? n[1] : q[1], null); }
  lensDrag = false; }
stage.addEventListener("pointerup", up); stage.addEventListener("pointercancel", up);
stage.addEventListener("dblclick", function (e) { const b = stage.getBoundingClientRect(); zoomAt(2.5, e.clientX - b.left, e.clientY - b.top, true); });
v.addEventListener("wheel", function (e) { e.preventDefault(); const b = stage.getBoundingClientRect();
  if (lensAt && lens.contains(e.target)) { lensZ = Math.max(2, Math.min(12, lensZ * Math.exp(-e.deltaY * 0.002))); drawLens(); return; }
  zoomAt(Math.exp(-e.deltaY * 0.0018), e.clientX - b.left, e.clientY - b.top, false); }, {passive: false});
v.addEventListener("keydown", function (e) { const st = 70, k = e.key, c = [stage.clientWidth / 2, stage.clientHeight / 2];
  if (k === "ArrowLeft") { tx += st; apply(); } else if (k === "ArrowRight") { tx -= st; apply(); } else if (k === "ArrowUp") { ty += st; apply(); }
  else if (k === "ArrowDown") { ty -= st; apply(); } else if (k === "+" || k === "=") zoomAt(1.5, c[0], c[1], true); else if (k === "-") zoomAt(1 / 1.5, c[0], c[1], true);
  else if (k === "0") fit(true); else if (k === "Escape") closeLens(); else return; e.preventDefault(); });
$("zin").onclick = function () { zoomAt(1.6, stage.clientWidth / 2, stage.clientHeight / 2, true); };
$("zout").onclick = function () { zoomAt(1 / 1.6, stage.clientWidth / 2, stage.clientHeight / 2, true); };
$("fit").onclick = function () { closeLens(); fit(true); };
$("tline").classList.add("on"); $("tline").onclick = function () { showLine = !showLine; lines.style.opacity = showLine ? 1 : 0; $("marks").style.display = showLine ? "" : "none"; this.classList.toggle("on", showLine); };
$("tmask").onclick = function () { showMask = !showMask; mask.style.opacity = showMask ? 1 : 0; this.classList.toggle("on", showMask); if (lensAt) drawLens(); };
$("tcol").classList.toggle("on", byCrack);
$("tdef").classList.toggle("on", showDef);
$("tdef").onclick = function () { showDef = !showDef; this.classList.toggle("on", showDef); drawLines(); placeDefects(); };
$("theat").onclick = function () { showHeat = !showHeat; $("heat").style.opacity = showHeat ? .85 : 0; this.classList.toggle("on", showHeat); };
$("tkin").classList.toggle("on", showKin);
$("tkin").onclick = function () { showKin = !showKin; this.classList.toggle("on", showKin); drawLines(); };
$("tcol").onclick = function () { byCrack = !byCrack; this.classList.toggle("on", byCrack); drawLines(); };
["zin", "zout", "fit", "tline", "tcol", "tkin", "tdef", "theat", "tmask"].forEach(function (id) { $(id).addEventListener("pointerdown", function (e) { e.stopPropagation(); }); });
$("tl").addEventListener("pointerdown", function (e) { e.stopPropagation(); }); $("spark").addEventListener("pointerdown", function (e) { e.stopPropagation(); });
window.addEventListener("resize", function () { fit(false); });
setTimeout(function () { $("hint").style.opacity = 0; }, 4500);
})();
</script></body></html>"""


def viewer_html(photo: bytes, data: dict | None, mask: bytes | None = None) -> str:
    """The viewer for one crack: the photo (JPEG), the crack's numbers (view_data) and, optionally, the model's red mask.
    Only numbers go into the script; '</' is escaped so no text in the data can end it early."""
    b64 = lambda b: "data:image/jpeg;base64," + base64.b64encode(b).decode()  # noqa: E731
    return (PAGE.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
            .replace("__PHOTO__", b64(photo)).replace("__MASK__", b64(mask) if mask else ""))
