"""Class demo web page: phone page at /, live wall at /?wall=<key>. Spec: spec/features/class_demo.md.

Start it with src/app/launch.py (adds the tunnel and the QR); on its own, for a local check:
    py -m streamlit run src/app/second_look.py
"""

import base64
import io
import os
import sys
import threading
from pathlib import Path

import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from app.background import background_css, crack_underline_css, model_line
from app.logo import LOGO_CSS, logo_title
from app.core import (NOT_A_SAFETY_CHECK, STATEMENTS, ConsentMissing, PhotoRejected, Wall, analyse, make_qr,
                      wall_key_ok)
from segmentation.train import load_checkpoint

MODEL = ROOT / "models" / "unet_crack.pt"
PUBLIC_URL = os.environ.get("SECOND_LOOK_URL", "")
WALL_KEY = os.environ.get("SECOND_LOOK_WALL_KEY", "")
APP_VERSION = "v1.0"
WALL_GALLERY_COLUMNS = 6  # photos per row in an opened collection
WALL_QUOTE = ("A repair that conceals a crack without addressing its root cause is not a solution, It is a "
              "temporary disguise for a potentially continuing defect.")
MODEL_LINE = ("Model: U-Net crack segmentation, ResNet-34 encoder (crack model v3), trained on 9,441 public crack photos "
              "and brick walls without cracks, plus written rules for the statement")  # update with models/unet_crack.pt

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
    st.markdown(f'<p class="note">{NOT_A_SAFETY_CHECK}</p>', unsafe_allow_html=True)


@st.fragment(run_every=4)
def wall_grid() -> None:
    """One button per statement with its count; no photo is shown until a button opens that collection."""
    wall = shared()[3]
    counts, items = wall.counts(), wall.items()
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
    mine = [img for statement, img in items if statement == opened]
    if not mine:
        st.markdown(f'<p class="note">No photo in "{opened}" yet.</p>', unsafe_allow_html=True)
    for row in range(0, len(mine), WALL_GALLERY_COLUMNS):
        for col, img in zip(st.columns(WALL_GALLERY_COLUMNS), mine[row:row + WALL_GALLERY_COLUMNS]):
            col.image(img, width="stretch")


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
        st.error("No public link: start the app with src/app/launch.py.")
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
        wall_page()
    else:
        st.error("This page is only shown on the presenter's screen.")
elif st.query_params.get("qr") is not None:
    qr_page()
else:
    phone_page()
