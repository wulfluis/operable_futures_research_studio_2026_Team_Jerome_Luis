"""Copy of the class demo page with bank-card measurement: phone page at /, live wall at /?wall=<key>, QR at /?qr.
Spec: spec/features/app_measure.md. The original (src/app/second_look.py) is not changed; its look is reused.

Start it with src/app_measure/launch_measure.py (https on this network and a QR); on its own, for a local check:
    python3 -m streamlit run src/app_measure/second_look_measure.py --server.port 8503 --theme.base dark
"""

import base64
import io
import json
import os
import sys
import threading
from pathlib import Path

import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from app.background import background_css, crack_underline_css, model_line  # the original's look, unchanged
from app.logo import LOGO_CSS, logo_title
from app_measure.core_measure import (NOT_A_SAFETY_CHECK, STATEMENTS, ConsentMissing, PhotoRejected, Wall, analyse,
                                      make_qr, wall_key_ok)
from measure.card import TILT_OK_DEG, TILT_RETAKE_DEG
from segmentation.train import load_checkpoint
from triage.matrix import ELEMENTS

MODEL = ROOT / "models" / "unet_crack.pt"
PUBLIC_URL = os.environ.get("SECOND_LOOK_URL", "")
WALL_KEY = os.environ.get("SECOND_LOOK_WALL_KEY", "")
APP_VERSION = "v1.0 + mm"
ELEMENT_KEYS = {"unknown": None, **{name: key for key, name in ELEMENTS.items()}}
WALL_GALLERY_COLUMNS = 6  # photos per row in an opened collection
WALL_QUOTE = ("A repair that conceals a crack without addressing its root cause is not a solution, It is a "
              "temporary disguise for a potentially continuing defect.")
MODEL_LINE = ("Model: U-Net crack segmentation, ResNet-34 encoder (crack model v3), trained on 9,441 public crack photos "
              "and brick walls without cracks, plus written rules for the statement, plus bank-card measurement in mm")

STYLE = """<style>
""" + background_css() + LOGO_CSS + """
[data-testid="stHeader"] { background: transparent; }
.stApp, .stApp p, .stApp label, .stApp li { color: #f2f2f2; }
h1, h2, h3 { color: #ffffff !important; font-weight: 800 !important; text-transform: uppercase; letter-spacing: 0.01em; }
.statement { font-size: 2.1rem; font-weight: 800; text-transform: uppercase; margin: 0.4rem 0 0.2rem; color: #ffffff; }
.stApp .reason { font-weight: 600; color: #d9d9d9; }
.stApp .note { font-size: 0.85rem; color: #a6a6a6; }
.count { font-size: 1.25rem; font-weight: 800; text-transform: uppercase; }
.version { font-size: 1.4rem; font-weight: 800; color: #a6a6a6; margin-left: 0.6rem; }
.count-n { font-size: 2rem; margin-left: 0.4rem; }
/* result buttons: no box, white text; red when opened or under the mouse; one cracked line runs under the row */
.stApp .stButton button { min-height: 4.2rem; background: transparent !important; border: none !important;
                          box-shadow: none !important; color: #ffffff !important; }
.stApp .stButton button p { font-size: 1.35rem; font-weight: 800; text-transform: uppercase; }
/* the label is a <p>, which ".stApp p" keeps white: colour it directly */
.stApp .stButton button[kind="primary"], .stApp .stButton button[kind="primary"] p,
.stApp .stButton [data-testid="stBaseButton-primary"] p,
.stApp .stButton button:hover, .stApp .stButton button:hover p { color: #d62728 !important; }
""" + crack_underline_css(".crack-rule", len(STATEMENTS),
                          '.stApp [data-testid="stVerticalBlock"]:has([data-testid="stColumn"]:nth-child({n}) '
                          '.stButton :is(button:hover, button[kind="primary"]))') + """
/* the QR page: the code alone, centred, as large as the screen allows */
.qr-alone { position: fixed; inset: 0; display: flex; align-items: center; justify-content: center; z-index: 50; }
.qr-alone img { height: 80vh; max-width: 90vw; object-fit: contain; }
.stApp .quote { font-weight: 800; text-transform: uppercase; color: #ffffff; letter-spacing: 0.01em; }
/* wall: title and quote in one block as wide as the quote, centred on the screen; both start at its left edge */
.stApp .wall-head { width: max-content; position: relative; left: 50%; transform: translateX(-50%); }
/* "Reading cracks..." right of the title, under the version: typed letter by letter, held, gone, again */
.stApp .wall-version { position: relative; }
.stApp .reading { position: absolute; left: 0; top: 100%; white-space: nowrap; color: #a6a6a6;
                  font: 700 1.1rem/1.4 Consolas, "Courier New", monospace; letter-spacing: 0; }
.reading > span { display: inline-block; overflow: hidden; white-space: nowrap; vertical-align: bottom; width: 0;
                  animation: reading-type 4s steps(17, end) infinite; }
@keyframes reading-type { 0% { width: 0; } 55% { width: 17ch; } 85% { width: 17ch; } 86%, 100% { width: 0; } }
.stApp .wall-quote { margin: -0.2rem 0 0.8rem; text-align: left; white-space: nowrap; font-size: 0.9vw; }
.stApp h1.wall-title { font-size: 6rem !important; line-height: 1.05; text-align: left; padding: 0.2rem 0 0.4rem; }
.stApp .wall-version { font-size: 3rem; font-weight: 800; text-transform: none; color: #a6a6a6; vertical-align: super; margin-left: 0.4rem; }
/* the wall refreshes every few seconds: keep it from dimming or showing "Running" while it does */
[data-stale="true"] { opacity: 1 !important; transition: none !important; }
[data-testid="stStatusWidget"] { visibility: hidden; }
/* added in the copy: the measurement block, in the same bold capitals and greys */
.stApp .m-head { font-size: 1.25rem; font-weight: 800; text-transform: uppercase; color: #ffffff; margin: 1.2rem 0 0.3rem; }
.stApp .m-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.9rem 1.2rem; margin: 0.4rem 0 0.6rem; }
.stApp .m-lab { font-size: 0.78rem; font-weight: 700; text-transform: uppercase; color: #a6a6a6; letter-spacing: 0.02em; }
.stApp .m-val { font-size: 1.7rem; font-weight: 800; color: #ffffff; line-height: 1.15; }
.stApp .m-sub { font-size: 0.85rem; font-weight: 600; color: #a6a6a6; }
.stApp .m-red { color: #d62728; }
.stApp .m-line { font-weight: 600; color: #d9d9d9; margin: 0.25rem 0; }
/* added in the copy: wall photos are links to their result page */
.stApp .wall-gallery { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 0.8rem; margin-top: 0.6rem; }
.stApp .wall-photo { position: relative; display: block; border: 2px solid transparent; transition: border-color .15s; }
.stApp .wall-photo:hover { border-color: #d62728; }
.stApp .wall-photo img { width: 100%; display: block; }
.stApp .wall-photo .badge { position: absolute; left: 0.4rem; top: 0.4rem; background: #0b0b0b; color: #ffffff;
                            font: 800 0.95rem/1.2 inherit; padding: 0.15rem 0.45rem; text-transform: uppercase; }
</style>"""

DETAIL_STYLE = """<style>
/* a result's own page: plain black, no crack behind, the numbers large */
.stApp { background: #000000 !important; background-image: none !important; }
.block-container { max-width: 1500px; padding-top: 2.2rem !important; }
.stApp a.d-back { color: #a6a6a6 !important; text-decoration: none; font-weight: 800; text-transform: uppercase;
                  letter-spacing: 0.04em; font-size: 0.95rem; }
.stApp a.d-back:hover { color: #d62728 !important; }
.stApp .d-title { font-size: 4.2rem; font-weight: 800; text-transform: uppercase; color: #ffffff; line-height: 1.02;
                  margin: 0.8rem 0 0.4rem; letter-spacing: 0.01em; }
.stApp .d-num { font-size: 2rem; color: #6f6f6f; margin-left: 0.8rem; vertical-align: super; }
.stApp .d-photo { width: 100%; display: block; border: 1px solid #262626; }
.stApp .d-hero .d-val { font-size: 6.5rem; line-height: 1; }
.stApp .d-hero { border-bottom: 3px solid #d62728; padding-bottom: 1rem; margin-bottom: 1.4rem; }
.stApp .d-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1.6rem 2rem; margin-bottom: 1.4rem; }
.stApp .d-lab { font-size: 0.85rem; font-weight: 800; text-transform: uppercase; color: #a6a6a6; letter-spacing: 0.05em; }
.stApp .d-val { font-size: 3.4rem; font-weight: 800; color: #ffffff; line-height: 1.05; }
.stApp .d-unit { font-size: 0.42em; color: #a6a6a6; margin-left: 0.25rem; }
.stApp .d-sub { font-size: 0.95rem; font-weight: 600; color: #a6a6a6; }
.stApp .d-red { color: #d62728; }
.stApp .d-none { font-size: 2.6rem; font-weight: 800; text-transform: uppercase; color: #d62728; margin: 0 0 0.3rem; }
.stApp .d-line { font-size: 1.15rem; font-weight: 600; color: #d9d9d9; }
.stApp .d-note { font-size: 0.9rem; color: #8c8c8c; margin: 0.35rem 0; }
@media (max-width: 800px) { .stApp .d-title { font-size: 2.4rem; } .stApp .d-hero .d-val { font-size: 4rem; }
                            .stApp .d-val { font-size: 2.4rem; } }
</style>"""


@st.cache_resource
def shared():
    """One model, one GPU lock and one wall for every visitor of this run."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return load_checkpoint(MODEL, device), device, threading.Lock(), Wall()


def phone_page() -> None:
    st.markdown(f"<h1>{logo_title()}</h1>", unsafe_allow_html=True)
    st.markdown(f'<p class="quote">{WALL_QUOTE}</p>', unsafe_allow_html=True)  # wraps: a phone is too narrow
    st.markdown("Photograph a crack in concrete or plaster. Second Look marks the crack in red and says what it sees.")
    st.markdown("**Get close:** hold the phone 20 to 40 cm from the wall so the wall fills the whole photo. "
                "No windows, edges, sky or people: the model only knows close-ups and reads other lines as cracks.")
    st.markdown("**Measure it in mm (optional):** put a bank-card-sized card flat on the wall next to the crack, all "
                "four corners visible. Cover the number and name, or use the back. Hold the phone parallel to the "
                "wall and upright. Without a card you get the same answer as before, without millimetres.")
    share = st.checkbox("Share results")
    # one tick only (D34): sending the photo is the consent, so its terms stand right above the upload
    st.markdown('<p class="note">By sending a photo you confirm you took it, it shows no faces, house numbers or '
                'addresses, and you agree it is analysed.</p>', unsafe_allow_html=True)
    consent = True
    upload = st.file_uploader("Take or choose a photo", type=["jpg", "jpeg", "png"], max_upload_size=10)
    st.markdown('<p class="note">The photo is analysed in memory and not saved. Its location data is not read.</p>',
                unsafe_allow_html=True)
    if upload is None:
        return
    done = st.session_state.setdefault("done", {})
    if upload.file_id not in done:  # analyse each upload once, even when the page reruns
        model, device, gpu, wall = shared()
        try:
            with st.spinner("Looking at the photo..."):
                done[upload.file_id] = analyse(upload.getvalue(), consent, share, wall, model, device, gpu)
        except (ConsentMissing, PhotoRejected) as e:
            st.error(str(e))
            return
    result = done[upload.file_id]
    st.markdown(f'<p class="statement">{result.statement}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="reason">{result.reason}</p>', unsafe_allow_html=True)
    st.image(result.overlay, width="stretch")
    if result.card is not None:
        measurement_block(result, upload.name)
    st.markdown(f'<p class="note">{NOT_A_SAFETY_CHECK}</p>', unsafe_allow_html=True)


def _cell(label: str, value: str, sub: str = "", red: bool = False) -> str:
    return (f'<div><div class="m-lab">{label}</div><div class="m-val{" m-red" if red else ""}">{value}</div>'
            f'<div class="m-sub">{sub}</div></div>')


def measurement_block(result, name: str) -> None:
    """Added in the copy: shown only when a bank card is in the photo (spec/features/app_measure.md)."""
    card, c = result.card, result.crack
    st.markdown('<p class="m-head">Measurement · bank card</p>', unsafe_allow_html=True)
    tilt_sub = (f"{card.status} · left-right {abs(card.tilt_left_right_deg):.0f}°, up-down "
                f"{abs(card.tilt_up_down_deg):.0f}°")
    cells = []
    if result.measured:
        cells += [_cell("Width, widest point", f"{c.width_max_mm:.2f} mm", f"± {c.width_uncertainty_mm:.2f} mm"),
                  _cell("Width, median", f"{c.width_median_mm:.2f} mm"),
                  _cell("Length", f"{c.length_main_mm:.0f} mm", f"all branches {c.length_total_mm:.0f} mm"),
                  _cell("Pixels across", f"{c.px_across:.1f} px", "need ≥ 3, best 6–8")]
    cells.append(_cell("Camera tilt", f"{card.tilt_deg:.0f}°", tilt_sub, red=card.status == "retake"))
    st.markdown('<div class="m-grid">' + "".join(cells) + "</div>", unsafe_allow_html=True)
    st.markdown(f'<p class="note">Straight up to {TILT_OK_DEG:g}°, retake above {TILT_RETAKE_DEG:g}° '
                '(illustrative). The white ring marks the widest point, the white outline the card.</p>',
                unsafe_allow_html=True)
    if result.measure_note:
        st.markdown(f'<p class="m-line"><b>Width not measured.</b> {result.measure_note}</p>', unsafe_allow_html=True)
    for w in result.warnings:
        st.markdown(f'<p class="m-line">{w}</p>', unsafe_allow_html=True)
    if result.band():
        st.markdown(f'<p class="m-line">Width band of the risk matrix for the upper end of the width: '
                    f'<b>{result.band()}</b>. This is not a level: the level also needs the crack pattern and the '
                    'element.</p>', unsafe_allow_html=True)
    if c is not None and c.found and c.angle_to_vertical_deg is not None:
        st.markdown(f'<p class="note">Crack shape: {c.angle_to_vertical_deg:.0f}° from vertical · straightness '
                    f'{c.straightness:.2f} · coherence {c.coherence:.2f} · {c.n_loops} closed cells · branches '
                    f'{c.branch_ratio:.1f}x main length</p>', unsafe_allow_html=True)
    pattern = None
    if c is not None and c.found:
        with st.expander("Field worker: pattern suggestion"):
            st.markdown('<p class="note">Answered by a field worker, never asked of the resident (D25). The rules '
                        'read the direction against up and down, so the phone must be held upright.</p>',
                        unsafe_allow_html=True)
            element = ELEMENT_KEYS[st.selectbox("Element", list(ELEMENT_KEYS), index=0)]
            at_joint = st.checkbox("At a beam-column joint")
            spalling = st.checkbox("Spalling or exposed bar")
            rust = st.checkbox("Rust stain along the crack")
            pattern = result.pattern(element, at_joint, spalling, rust)
            if pattern is not None:
                p = pattern.to_dict()
                st.markdown(f'<p class="m-line"><b>Rule {p["rule"]}: {p["pattern_name"] or "none"}</b>'
                            f'{"" if p["clear"] else " · unclear"}. {p["reason"]}</p>'
                            '<p class="note">A field worker confirms or corrects it; only a confirmed pattern goes '
                            'into the risk matrix (draft rules, illustrative).</p>', unsafe_allow_html=True)
    report = {"photo": Path(name).stem, **result.to_dict(),
              "pattern_suggestion": None if pattern is None else pattern.to_dict()}
    buf = io.BytesIO()
    result.overlay.save(buf, "PNG")
    d1, d2 = st.columns(2)
    d1.download_button("Result (JSON)", json.dumps(report, indent=2), f"{Path(name).stem}_measurement.json",
                       "application/json", width="stretch")
    d2.download_button("Overlay (PNG)", buf.getvalue(), f"{Path(name).stem}_overlay.png", "image/png",
                       width="stretch")


def _wall_href(**extra) -> str:
    """A link inside the wall that keeps its key (needed when the wall is opened through the public link)."""
    from urllib.parse import urlencode
    return "?" + urlencode({"wall": st.query_params.get("wall") or "", **extra})


@st.fragment(run_every=4)
def wall_grid() -> None:
    """One button per statement with its count; no photo is shown until a button opens that collection.
    Added in the copy: each photo is a link to its own page with the stats (/?wall=...&item=<n>)."""
    wall = shared()[3]
    counts, entries = wall.counts(), wall.entries()
    if "opened" not in st.session_state:  # coming back from a result's page: reopen its collection
        back = st.query_params.get("open")
        st.session_state["opened"] = back if back in STATEMENTS else None
    opened = st.session_state.get("opened")
    for col, s in zip(st.columns(len(STATEMENTS)), STATEMENTS):
        if col.button(f"{s}   {counts[s]}", key=f"open_{s}", width="stretch",
                      type="primary" if opened == s else "secondary"):
            opened = None if opened == s else s  # a second click closes it
            st.session_state["opened"] = opened
    st.markdown('<div class="crack-rule">' + '<div></div>' * len(STATEMENTS) + '</div>',  # one line under all four,
                unsafe_allow_html=True)  # in pieces: the piece under a hovered or opened button turns red
    if opened is None:
        return
    mine = [e for e in entries if e.statement == opened]
    if not mine:
        st.markdown(f'<p class="note">No photo in "{opened}" yet.</p>', unsafe_allow_html=True)
        return
    cells = "".join(
        f'<a class="wall-photo" href="{_wall_href(item=e.number)}" target="_self" title="Open result {e.number}">'
        f'<img src="data:image/jpeg;base64,{base64.b64encode(e.thumb).decode()}" alt="Result {e.number}">'
        f'{_photo_badge(e.stats)}</a>' for e in mine)
    st.markdown(f'<div class="wall-gallery">{cells}</div>', unsafe_allow_html=True)


def _photo_badge(stats: dict) -> str:
    """The widest width on the photo in the gallery, when it was measured."""
    c = stats.get("crack") or {}
    if stats.get("measured") and c.get("width_max_mm") is not None:
        return f'<span class="badge">{c["width_max_mm"]:.2f} mm</span>'
    return ""


def _big(label: str, value: str, unit: str = "", sub: str = "", red: bool = False) -> str:
    return (f'<div class="d-cell"><div class="d-lab">{label}</div><div class="d-val{" d-red" if red else ""}">'
            f'{value}<span class="d-unit">{unit}</span></div><div class="d-sub">{sub}</div></div>')


def result_page(number: int) -> None:
    """Added in the copy: one shared result on its own page, the stats large, on black."""
    entry = shared()[3].get(number)
    st.markdown(f'<a class="d-back" href="{_wall_href(**({"open": entry.statement} if entry else {}))}" '
                'target="_self">&larr; Back to the wall</a>', unsafe_allow_html=True)
    if entry is None:
        st.markdown('<p class="d-title">Result gone</p><p class="d-line">The wall keeps only the newest 40 shared '
                    'results, and none after the app stops.</p>', unsafe_allow_html=True)
        return
    d = entry.stats
    card, c = d.get("card"), d.get("crack") or {}
    conf = d.get("confidence")
    st.markdown(f'<p class="d-title">{entry.statement}<span class="d-num">#{entry.number}</span></p>'
                f'<p class="d-line">{d.get("reason", "")}</p>', unsafe_allow_html=True)
    left, right = st.columns([1.15, 1], gap="large")
    left.markdown(f'<img class="d-photo" src="data:image/jpeg;base64,{base64.b64encode(entry.large).decode()}" '
                  f'alt="Result {entry.number}">', unsafe_allow_html=True)
    html = []
    if d.get("measured"):
        html.append('<div class="d-hero">' + _big("Width, widest point", f'{c["width_max_mm"]:.2f}', " mm",
                                                  f'± {c["width_uncertainty_mm"]:.2f} mm') + "</div>")
        cells = [_big("Width, median", f'{c["width_median_mm"]:.2f}', " mm"),
                 _big("Length", f'{c["length_main_mm"]:.0f}', " mm", f'all branches {c["length_total_mm"]:.0f} mm'),
                 _big("Pixels across", f'{c["px_across_at_widest"]:.1f}', " px", "need ≥ 3, best 6–8")]
    else:
        why = d.get("measure_note") or "No bank card in the photo: no measurement in mm."
        html.append(f'<p class="d-none">Width not measured</p><p class="d-line">{why}</p>')
        cells = []
    if card:
        cells.append(_big("Camera tilt", f'{card["tilt_deg"]:.0f}', "°",
                          f'{card["status"]} · left-right {abs(card["tilt_left_right_deg"]):.0f}°, up-down '
                          f'{abs(card["tilt_up_down_deg"]):.0f}°', red=card["status"] == "retake"))
    if conf is not None:
        cells.append(_big("Model confidence", f"{conf:.0%}"))
    if cells:
        html.append('<div class="d-grid">' + "".join(cells) + "</div>")
    if d.get("width_band_of_upper_end"):
        html.append(f'<p class="d-line">Width band of the risk matrix: <b>{d["width_band_of_upper_end"]}</b>. Not a '
                    'level: the level also needs the crack pattern and the element.</p>')
    sh = c.get("shape") or {}
    if c.get("found") and sh.get("angle_to_vertical_deg") is not None:
        html.append(f'<p class="d-note">Crack shape: {sh["angle_to_vertical_deg"]:.0f}° from vertical · straightness '
                    f'{sh["straightness"]:.2f} · coherence {sh["coherence"]:.2f} · {sh["n_loops"]} closed cells · '
                    f'branches {sh["branch_ratio"]:.1f}x main length</p>')
    for w in d.get("warnings") or []:
        html.append(f'<p class="d-note">{w}</p>')
    html.append(f'<p class="d-note">{NOT_A_SAFETY_CHECK} Thresholds illustrative until calibrated.</p>')
    right.markdown("".join(html), unsafe_allow_html=True)


def wall_page() -> None:
    """Title and quote on top, the result buttons and an opened collection below; the model line stands along the
    bottom. No QR here (1 Nov): it has its own page, /?qr."""
    st.markdown("<style>.block-container { padding-left: calc(11vw + 4.5vw) !important; padding-right: "
                "calc(11vw + 4.5vw) !important; padding-bottom: calc(3vw + 1rem) !important; }"
                # the title and the quote are wider than this column: let them overflow it, centred on the screen
                ".stApp [data-testid='stElementContainer'], .stApp [data-testid='stMarkdown'], "
                ".stApp [data-testid='stMarkdownContainer'] { overflow: visible; }</style>",
                unsafe_allow_html=True)  # equal sides: buttons centred on the screen, photos clear of the QR corner
    st.markdown(f'<div class="wall-head"><h1 class="wall-title">{logo_title()} <span class="wall-version">{APP_VERSION}'
                '<span class="reading"><span>Reading cracks...</span></span></span></h1>'
                f'<p class="quote wall-quote">{WALL_QUOTE}</p></div>', unsafe_allow_html=True)
    st.markdown(model_line(MODEL_LINE), unsafe_allow_html=True)
    wall_grid()


def qr_page() -> None:
    """The QR alone, to show on the projector when students should scan it."""
    if not PUBLIC_URL:
        st.error("No phone link: start the app with src/app_measure/launch_measure.py.")
        return
    buf = io.BytesIO()
    make_qr(PUBLIC_URL).save(buf, "PNG")
    st.markdown(f'<div class="qr-alone"><img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}" '
                'alt="QR code"></div>', unsafe_allow_html=True)


def on_this_laptop() -> bool:
    """Opened in the laptop's own browser: host is localhost and the request did not come through the
    Cloudflare link (which adds Cf-* headers and keeps the public host name)."""
    headers = {k.lower(): v for k, v in st.context.headers.items()}
    host = headers.get("host", "").split(":")[0]
    return host in ("localhost", "127.0.0.1") and not any(k.startswith("cf-") for k in headers)


st.set_page_config(page_title="Second Look", layout="wide" if st.query_params.get("wall") is not None else "centered")
st.markdown(STYLE, unsafe_allow_html=True)
if st.query_params.get("wall") is not None:
    if on_this_laptop() or wall_key_ok(st.query_params.get("wall"), WALL_KEY):
        item = st.query_params.get("item", "")
        if item.isdigit():
            st.markdown(DETAIL_STYLE, unsafe_allow_html=True)
            result_page(int(item))
        else:
            wall_page()
    else:
        st.error("This page is only shown on the presenter's screen.")
elif st.query_params.get("qr") is not None:
    qr_page()
else:
    phone_page()
