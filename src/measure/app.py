"""Second Look: crack measurement prototype. Upload a crack photo with a bank card; get width, length and tilt.

Run, from the repository root:
    streamlit run src/measure/app.py
Runs on this laptop only. Photos stay in memory and are not stored; "Save result" writes only the overlay and the
JSON to data/measurements/ (ignored by git). Spec: spec/features/measurement.md (D30, D31).
"""

import io
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from measure.analyse import analyse  # noqa: E402
from measure.card import TILT_OK_DEG, TILT_RETAKE_DEG  # noqa: E402
from measure.crack import BACKGROUND_PX  # noqa: E402
from measure.overlay import draw_overlay  # noqa: E402
from measure.sample_photo import ensure_sample  # noqa: E402
from measure.synthetic import make_demo  # noqa: E402
from triage.matrix import ELEMENTS  # noqa: E402

ELEMENT_KEYS = {"unknown": None, **{name: key for key, name in ELEMENTS.items()}}

ROOT = Path(__file__).resolve().parents[2]

st.set_page_config(page_title="Second Look · crack measurement", layout="wide")
st.title("Second Look · crack measurement")
st.caption("Prototype. A measurement, never a safety assessment: the level needs the crack pattern and the "
           "element, and an engineer confirms every Category 3. Thresholds are illustrative until calibrated.")

with st.expander("How to take the photo", expanded=False):
    st.markdown(
        "- Put a **bank-card-sized card** (85.60 x 53.98 mm: bank, ID, loyalty or an expired card) **flat on the "
        "wall right next to the crack**. All four corners visible; no fingers over its edges.\n"
        "- **Cover the card number and name** (paper or tape on top is fine, the outline is what counts), or use the "
        "back of the card.\n"
        "- Hold the phone **parallel to the wall**, 10 to 30 cm away; tap to focus on the crack.\n"
        "- At least **3 pixels across the crack**, 6 to 8 better [46]: go closer if the app says so.\n"
        "- Brush off dust first. No faces, house numbers or street signs in the frame.\n"
        "- iPhone: save as JPEG (Settings > Camera > Formats > Most Compatible); HEIC is not read.")

# ---------- input ----------
up = st.file_uploader("Crack photo with the bank card (on a phone: Take Photo or Photo Library)",
                      type=["jpg", "jpeg", "png"])  # on the main page: the sidebar is folded away on phones
SOURCES = ["Uploaded photo", "Sample photo: crack and bank card", "Simple demo (sliders)"]
with st.sidebar:
    st.header("Photo")
    wanted = {"sample": 1, "demo": 2}.get(st.query_params.get("example", ""), None)
    source = st.radio("Source", SOURCES, index=0 if up is not None else (wanted if wanted is not None else 1))
    use_demo = source == SOURCES[2]
    if use_demo:
        demo_w = st.slider("Demo: crack width (mm)", 0.2, 3.0, 1.0, 0.1)
        demo_t = st.slider("Demo: camera tilt (degrees)", 0, 35, 12)
    st.header("Search box")
    st.caption("Narrow it to the crack if stains, joints or shadows are picked up.")
    xr = st.slider("Left to right (%)", 0, 100, (0, 100))
    yr = st.slider("Top to bottom (%)", 0, 100, (0, 100))
    longest = st.checkbox("Keep only the longest crack piece", value=True)
    st.header("Field worker")
    st.caption("For the pattern suggestion. Answered by a field worker, never asked of the resident (D25). "
               "Hold the phone upright: the rules read the crack's direction against up and down.")
    element = ELEMENT_KEYS[st.selectbox("Element", list(ELEMENT_KEYS), index=0)]
    at_joint = st.checkbox("At a beam-column joint")
    spalling = st.checkbox("Spalling or exposed bar")
    rust = st.checkbox("Rust stain along the crack")
    with st.expander("Advanced"):
        bg = st.number_input("Background size (px, wider than the crack)", 11, 301, BACKGROUND_PX, 2)
        thr_on = st.checkbox("Fixed threshold instead of automatic")
        thr = st.number_input("Threshold", 1.0, 200.0, 25.0) if thr_on else None
        manual = st.text_input("Card corners by hand (x,y;x,y;x,y;x,y)", "",
                               help="Only if the card is not found automatically.")


@st.cache_data(show_spinner=False)
def load_upload(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as img:
        return np.asarray(ImageOps.exif_transpose(img).convert("RGB"))


@st.cache_data(show_spinner=False)
def load_demo(width_mm: float, tilt: int):
    grey, truth = make_demo(width_mm, tilt_lr_deg=float(tilt), tilt_ud_deg=-tilt / 3)
    return np.repeat(grey.astype(np.uint8)[..., None], 3, axis=2), truth


@st.cache_data(show_spinner=False)
def load_sample():
    path, t = ensure_sample()
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB")), t


truth = None
if use_demo:
    rgb, truth = load_demo(demo_w, demo_t)
    name = f"demo_{demo_w:g}mm_{demo_t}deg"
elif source == SOURCES[1]:
    with st.spinner("Making the sample photo (first time only)..."):
        rgb, truth = load_sample()
    name = "sample_crack_card"
    st.image(rgb, caption="Sample photo (synthetic, made to look like a phone photo; the card is a sample with no "
             "real number). Upload your own photo above to measure it instead.", width="stretch")
elif up is not None:
    try:
        rgb = load_upload(up.getvalue())
    except Exception:  # noqa: BLE001 - an unreadable upload is a retake, not a crash
        st.error("Retake: this file could not be read as a photo (HEIC is not supported; save as JPEG).")
        st.stop()
    name = Path(up.name).stem
else:
    st.info("Upload a photo above, or choose the sample photo in the sidebar.")
    st.stop()

h, w = rgb.shape[:2]
roi = (xr[0] / 100 * w, yr[0] / 100 * h, xr[1] / 100 * w, yr[1] / 100 * h)
corners = None
if manual.strip():
    try:
        corners = [tuple(float(v) for v in p.split(",")) for p in manual.replace(" ", "").split(";") if p]
        assert len(corners) == 4
    except (ValueError, AssertionError):
        st.sidebar.error("Corners: four x,y pairs separated by ';'.")
        corners = None

with st.spinner("Finding the card and measuring the crack..."):
    a = analyse(rgb, roi, corners, longest_only=longest, background_px=int(bg), threshold=thr, element=element,
                at_joint=at_joint, spalling=spalling, rust_stain=rust)

# ---------- output ----------
outcome = a.outcome
if outcome == "Retake":
    st.error("**Retake the photo.** " + " ".join(a.retake))
elif outcome == "Width not measured":
    st.warning("**Width not measured.** The case goes to an engineer (Expert needed), or retake with the crack "
               "clearly in the search box.")
else:
    st.success("**Measured.** Check the overlay: the orange pixels must be the crack, and the red line its widest "
               "point.")

c = a.crack
m = st.columns(5)
if c is not None and c.width_max_mm is not None:
    m[0].metric("Width, widest point", f"{c.width_max_mm:.2f} mm", f"± {c.width_uncertainty_mm:.2f} mm",
                delta_color="off")
    m[1].metric("Width, median", f"{c.width_median_mm:.2f} mm")
    m[3].metric("Pixels across", f"{c.px_across:.1f} px", "need ≥ 3, best 6–8", delta_color="off")
else:
    m[0].metric("Width, widest point", "–")
if c is not None and c.length_main_mm is not None:
    m[2].metric("Length", f"{c.length_main_mm:.0f} mm", f"all branches {c.length_total_mm:.0f} mm",
                delta_color="off")
if a.card is not None:
    m[4].metric("Camera tilt", f"{a.card.tilt_deg:.0f}°", a.card.status, delta_color="off")
else:
    m[4].metric("Camera tilt", "card not found")

if a.card is not None:
    t = a.card.tilt_deg
    (st.success if t <= TILT_OK_DEG else st.warning if t <= TILT_RETAKE_DEG else st.error)(
        f"{a.card.message}  (left-right {abs(a.card.tilt_left_right_deg):.0f}°, up-down "
        f"{abs(a.card.tilt_up_down_deg):.0f}°; straight up to {TILT_OK_DEG:g}°, retake above {TILT_RETAKE_DEG:g}°, "
        "illustrative)")
for x in a.warnings:
    if a.card is None or x != a.card.message:
        st.warning(x)
if a.band():
    st.info(f"Width band of the risk matrix for the upper end of the width: **{a.band()}**. This is not a level: the "
            "level also needs the crack pattern and the element.")
if c is not None and c.angle_to_vertical_deg is not None:
    st.caption(f"Crack shape: {c.angle_to_vertical_deg:.0f}° from vertical · straightness {c.straightness:.2f} · "
               f"coherence {c.coherence:.2f} · {c.n_loops} closed cells · branches {c.branch_ratio:.1f}x main length")
if a.pattern is not None:
    p = a.pattern.to_dict()
    text = (f"**Pattern suggestion (rule {p['rule']}): {p['pattern_name'] or 'none'}**"
            f"{'' if p['clear'] else ' · unclear'}. {p['reason']} A field worker confirms or corrects it; only a "
            "confirmed pattern goes into the risk matrix (draft rules, illustrative).")
    (st.info if p["clear"] else st.warning)(text)
if truth:
    w_true = truth.get("width_max_mm", truth.get("width_mm"))
    l_true = truth.get("length_main_mm", truth.get("length_mm"))
    st.info(f"**Known answer of this synthetic photo:** widest point {w_true:g} mm, main length {l_true:.1f} mm, "
            f"camera tilt {truth['tilt_deg']:.1f}°. Compare it with the measurement above.")

buf = io.BytesIO()
lines = [f"Outcome: {outcome}"] + [f"RETAKE: {r}" for r in a.retake]
if a.card is not None:
    lines.append(f"Card tilt: {a.card.tilt_deg:.1f} degrees ({a.card.status})")
draw_overlay(rgb, None if a.card is None else a.card.corners_px, a.roi, c, f"{name}  ·  reference: bank card", buf,
             extra_lines=lines)
st.image(buf.getvalue(), width="stretch")

report = {"photo": name, "measured_at": datetime.now().isoformat(timespec="seconds"), **a.to_dict()}
if truth:
    report["demo_truth"] = {k: round(v, 2) if isinstance(v, float) else v for k, v in truth.items()
                            if k in ("width_mm", "length_mm", "width_max_mm", "length_main_mm", "tilt_deg")}
d1, d2, d3 = st.columns(3)
d1.download_button("Download result (JSON)", json.dumps(report, indent=2), f"{name}_measurement.json",
                   "application/json")
d2.download_button("Download overlay (PNG)", buf.getvalue(), f"{name}_overlay.png", "image/png")
if d3.button("Save result to data/measurements/"):
    out = ROOT / "data" / "measurements"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}_measurement.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / f"{name}_overlay.png").write_bytes(buf.getvalue())
    d3.success(f"Saved to {out} (the photo itself is not stored).")
with st.expander("Full result (JSON)"):
    st.json(report)
