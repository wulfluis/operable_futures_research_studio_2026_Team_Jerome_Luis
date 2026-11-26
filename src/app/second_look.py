"""Class demo web page: phone page at /, live wall at /?wall=<key>, a result's own page at /?wall=<key>&item=<n>,
QR at /?qr. With a bank card in the photo the crack is measured in mm (Luis's measurement, D41 to D46).
Specs: spec/features/class_demo.md, spec/features/app_measure.md.

Start it with src/app/launch.py (the team's fixed ngrok link and QR); on its own, for a local check:
    py -m streamlit run src/app/second_look.py
"""

import base64
import io
import json
import os
import sys
import threading
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from app.theme import FONT, MODE_JS, MONO, THEME_CSS, label, phone_css, scan_frame, state, wall_css  # dark concrete look (D82)
from app.logo import LOGO_CSS, logo_title
from app.viewer3d import HEIGHT as VIEW_HEIGHT, viewer_html
from app import crack_view  # the crack to explore: zoom, drag, loupe (D83)
from app import result_blocks as blocks  # the result in reading order (D115 to D122)
from app.inbox import INBOX, start as start_inbox
from app.core import (EXPORT_DIR, NOT_A_SAFETY_CHECK, RISK_TABS, WAITING, WAITING_SHORT, risk_group, ConsentMissing, PhotoRejected, Wall,
                      analyse, analyse_element, make_qr, wall_key_ok)
from element.predict import load as load_element_model
from quality.scene import load as load_scene_model
from element.spalling import load as load_spalling_model
from measure.card import TILT_OK_DEG, TILT_RETAKE_DEG
from segmentation.decide import CRACK, UNSURE
from segmentation.train import load_checkpoint
from triage.element_rule import UNSURE as UNSURE_ELEMENT
from triage.matrix import ELEMENTS, PATTERN_NAMES

MODEL = ROOT / "models" / "unet_crack.pt"
PUBLIC_URL = os.environ.get("SECOND_LOOK_URL", "")
WALL_KEY = os.environ.get("SECOND_LOOK_WALL_KEY", "")
ELEMENT_KEYS = {"unknown": None, **{name: key for key, name in ELEMENTS.items()}}
ELEMENT_WORDS = {"column": "Column", "beam": "Beam", "wall": "Wall", "others": "Not an element", "unsure": "Unsure"}
WALL_GALLERY_COLUMNS = 6  # photos per row in an opened collection
MODEL_LINE = ("Model: U-Net crack segmentation, ResNet-34 encoder (crack model v5), trained on 19,357 photos: public crack "
              "close-ups, building concrete with crack masks, brick walls without cracks, and the EPFL photos of "
              "buildings cracked by earthquakes, plus written rules for the statement, plus bank-card measurement in mm"
              )  # update with models/unet_crack.pt (D48, D100: v4's 12,477 + 6,511 EPFL building + 369 EPFL lab tiles)

STYLE = "<style>" + THEME_CSS + LOGO_CSS + """
/* no "Made with Streamlit" line */
footer, .stApp footer, [data-testid="stFooter"], [class*="viewerBadge"], a[href*="streamlit.io"] { display: none !important; }
[data-testid="stMainMenuPopover"] div:has(> button[title="Copy version to clipboard"]) { display: none !important; }
/* the QR page: the code alone, centred, as large as the screen allows */
.qr-alone { position: fixed; inset: 0; display: flex; align-items: center; justify-content: center; z-index: 50;
            background: #07080a; }
.qr-alone img { height: 80vh; max-width: 90vw; object-fit: contain; background: #ffffff; padding: 2vh;
                border-radius: 24px; box-shadow: 0 0 80px rgba(34, 211, 238, 0.25); }  /* a QR scans on white */
/* wall: one tab per risk level with its counter, a glowing cyan line under the opened one */
.stApp .st-key-gh-tabs { gap: 8px; border-bottom: 1px solid var(--gh-border); margin-bottom: 16px; }
.stApp .st-key-gh-tabs .stButton button, .stApp .st-key-gh-tabs .stButton button[kind="primary"],
.stApp .st-key-gh-tabs [data-testid="stBaseButton-primary"] {
    background: transparent !important; border: none !important; border-bottom: 2px solid transparent !important;
    border-radius: 0 !important; box-shadow: none !important; min-height: 52px; padding: 0 12px; margin-bottom: -1px;
    color: var(--gh-muted) !important; backdrop-filter: none !important; }
.stApp .st-key-gh-tabs .stButton button p { font-size: 15px; color: var(--gh-muted) !important; transition: color .18s ease; }
.stApp .st-key-gh-tabs .stButton button:hover p { color: var(--gh-fg) !important; }
.stApp .st-key-gh-tabs .stButton button:hover { border-bottom-color: var(--gh-border) !important; }
.stApp .st-key-gh-tabs .stButton button[kind="primary"],
.stApp .st-key-gh-tabs [data-testid="stBaseButton-primary"] { border-bottom-color: var(--gh-tab) !important;
    box-shadow: 0 10px 18px -12px rgba(34, 211, 238, 0.9) !important; }
.stApp .st-key-gh-tabs .stButton button[kind="primary"] p,
.stApp .st-key-gh-tabs [data-testid="stBaseButton-primary"] p { font-weight: 600; color: var(--gh-fg) !important; }
.stApp .st-key-gh-tabs button code { display: inline-block; min-width: 20px; margin-left: 6px; padding: 0 6px;
    font-family: """ + MONO + """ !important; font-size: 12px; font-weight: 600; line-height: 20px; text-align: center; color: var(--gh-fg) !important;
    background: rgba(255, 255, 255, 0.08) !important; border-radius: 2em; }
.stApp .st-key-gh-tab-Category-3 .stButton button p::before { content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; background: #f85149; box-shadow: 0 0 8px #f85149; vertical-align: 1px; }
.stApp .st-key-gh-tab-Expert-needed .stButton button p::before { content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; background: #b180d7; box-shadow: 0 0 8px #b180d7; vertical-align: 1px; }
.stApp .st-key-gh-tab-Category-2 .stButton button p::before { content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; background: #e5a50a; box-shadow: 0 0 8px #e5a50a; vertical-align: 1px; }
.stApp .st-key-gh-tab-Category-1 .stButton button p::before { content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; background: #86c5ff; box-shadow: 0 0 8px #86c5ff; vertical-align: 1px; }
.stApp .map-legend { display: flex; flex-wrap: wrap; gap: 8px 18px; margin: -4px 0 10px; font-size: 13px; color: var(--gh-muted); }
.stApp .map-legend span { display: inline-flex; align-items: center; gap: 7px; }
.stApp .map-legend i { width: 11px; height: 11px; border-radius: 50%; border: 2px solid #fff; display: inline-block; }
.stApp .map-legend b { color: var(--gh-fg); font-weight: 600; }
.stApp p.tab-desc { margin: -4px 12px 18px !important; font-size: 12.5px !important; line-height: 1.4 !important;
                    color: var(--gh-muted) !important; white-space: nowrap; }
.stApp [class*="st-key-gh-tab-"] { gap: 0 !important; min-width: 250px; flex: 0 1 auto !important; }
/* wall: each shared result is a card; a see-through button over it opens the result over the wall (D77) */
.stApp .st-key-gh-cards { display: grid !important; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
                          gap: 16px !important; }
.stApp [class*="st-key-card-"] { position: relative; gap: 0 !important; }
.stApp [class*="st-key-open-item-"] { position: absolute !important; inset: 0; z-index: 2; margin: 0 !important; }
.stApp [class*="st-key-open-item-"] .stButton, .stApp [class*="st-key-open-item-"] button {
    width: 100% !important; height: 100% !important; opacity: 0; cursor: pointer; }
.stApp .wall-photo { position: relative; display: block; border: 1px solid var(--gh-border); border-radius: 16px;
                     overflow: hidden; background: var(--gh-canvas); color: var(--gh-fg);
                     transition: transform .25s cubic-bezier(.2, .8, .2, 1), border-color .25s ease, box-shadow .25s ease; }
/* on hover the card lifts and the logo's scan frame locks onto the photo */
.stApp .wall-photo .pic::after { content: ""; position: absolute; inset: 10px; pointer-events: none; opacity: 0;
    transform: scale(1.08); transition: opacity .25s ease, transform .25s cubic-bezier(.2, .8, .2, 1);
    background: """ + scan_frame(size="18px") + """; background-repeat: no-repeat; }
.stApp [class*="st-key-card-"]:hover .wall-photo { transform: translateY(-4px); border-color: rgba(34, 211, 238, 0.55);
    box-shadow: 0 18px 40px rgba(0, 0, 0, 0.5), 0 0 30px rgba(34, 211, 238, 0.15); }
.stApp [class*="st-key-card-"]:hover .wall-photo .pic::after { opacity: 1; transform: scale(1); }
/* a result opened over the wall: the wall blurred and light, the crack sharp, a window beside it (D77) */
.stApp .st-key-sl-overlay { position: fixed !important; inset: 0; z-index: 999990; display: flex !important;
    align-items: center; justify-content: center; padding: 6vh 6vw; background: rgba(5, 7, 10, 0.6);
    backdrop-filter: blur(14px) brightness(0.8); -webkit-backdrop-filter: blur(14px) brightness(0.8);
    animation: sl-fade .25s ease; }
@keyframes sl-fade { from { opacity: 0; } to { opacity: 1; } }
.stApp .st-key-sl-backdrop { position: absolute !important; inset: 0; margin: 0 !important; }
.stApp .st-key-sl-backdrop .stButton, .stApp .st-key-sl-backdrop button { width: 100% !important;
    height: 100% !important; opacity: 0; cursor: zoom-out; }
.stApp .st-key-sl-stage { position: relative; z-index: 1; width: min(1180px, 100%); flex-wrap: nowrap !important;
    align-items: center !important; gap: 24px !important; }
.stApp .st-key-sl-stage > [data-testid="stElementContainer"] { flex: 0 1 58%; min-width: 0; }
.stApp iframe { border: none; color-scheme: dark; border-radius: 18px; background: #050607; }
.stApp .sl-crack { background: var(--gh-solid); border: 1px solid rgba(34, 211, 238, 0.35); border-radius: 18px;
    overflow: hidden; box-shadow: 0 30px 80px rgba(0, 0, 0, 0.6), 0 0 40px rgba(34, 211, 238, 0.12); }
.stApp .sl-crack img { width: 100%; max-height: 70vh; object-fit: contain; display: block; }
.stApp .sl-crack .meta { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; padding: 8px 12px;
    border-top: 1px solid var(--gh-border); font-size: 12px; color: var(--gh-muted); }
.stApp .sl-crack .num { font-size: 14px; font-weight: 600; color: var(--gh-fg); margin-right: 4px; }
.stApp .st-key-sl-window { flex: 1 1 42%; min-width: 0; max-height: 80vh; overflow-y: auto; overscroll-behavior: contain;
    background: rgba(16, 19, 24, 0.96); border: 1px solid var(--gh-border-strong); border-radius: 22px; padding: 8px 28px 28px;
    box-shadow: 0 30px 80px rgba(0, 0, 0, 0.6); }
.stApp .st-key-sl-close { position: sticky; top: 0; z-index: 3; align-self: flex-end; margin-bottom: -40px; }
.stApp .st-key-sl-close button { min-height: 32px; padding: 0 10px; }
.stApp .st-key-sl-window .d-title { margin-top: 0; padding-right: 48px; }
@media (max-width: 800px) {
  .stApp .st-key-sl-overlay { align-items: flex-start; overflow-y: auto; padding: 16px; }
  .stApp .st-key-sl-stage { flex-direction: column !important; }
  .stApp .st-key-sl-window { max-height: none; } }
.stApp .wall-photo .pic { position: relative; background: #000000; }
.stApp .wall-photo .pic img { width: 100%; display: block; }
.stApp .wall-photo .pic img.inset { position: absolute; right: 8px; bottom: 8px; width: 36%; border: 1px solid rgba(255, 255, 255, 0.6);
                                    border-radius: 8px; box-shadow: 0 6px 16px rgba(0, 0, 0, 0.6); }
.stApp .wall-photo .meta { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; padding: 8px 12px;
                           border-top: 1px solid var(--gh-border); font-size: 12px; color: var(--gh-muted); }
.stApp .wall-photo .num { font-family: """ + MONO + """; font-size: 13px; font-weight: 600; color: var(--gh-fg); margin-right: 4px; }
.stApp .gh-blank { padding: 56px 32px; text-align: center; color: var(--gh-muted); font-size: 15px;
                   border: 1px dashed var(--gh-border-strong); border-radius: 18px; background: var(--gh-subtle); }
.stApp .gh-footer { margin-top: 48px; padding-top: 16px; border-top: 1px solid var(--gh-border); font-family: """ + MONO + """;
                    font-size: 11.5px; line-height: 1.6; color: var(--gh-muted); }
/* phone page: one line, three tips, then the photo (D83: less text) */
.stApp .lead { font-size: 22px; font-weight: 600; letter-spacing: -0.015em; margin: 0 0 14px; }
.stApp .lead-s { font-size: 15px; color: var(--gh-muted); margin: 0 0 8px; }
.stApp .tips { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; margin: 0 0 18px; }
.stApp .tips > span { padding: 12px; border-radius: 14px; background: var(--gh-subtle); border: 1px solid var(--gh-border);
                      font-size: 12.5px; line-height: 1.35; color: var(--gh-muted); }
.stApp .tips svg { display: block; width: 100%; height: 64px; margin: 0 0 8px; color: var(--gh-accent); }
.stApp .tips b { display: block; font-size: 15px; color: var(--gh-accent); margin-bottom: 2px; }
@media (max-width: 480px) { .stApp .tips { grid-template-columns: 1fr; } .stApp .tips > span { display: flex; gap: 12px;
                            align-items: center; } .stApp .tips svg { flex: 0 0 76px; width: 76px; height: 46px; margin: 0; }
                            .stApp .tips b { display: inline; } }
/* D115: exposed steel first; D121: what to do now */
.stApp .sl-alert { margin: 0 0 14px; padding: 14px 16px; border-radius: 16px; background: rgba(127, 29, 29, 0.55);
                   border: 1px solid #f87171; font-size: 15px; line-height: 1.4; }
.stApp .sl-alert b { color: #fecaca; }
.stApp .sl-alert.cause { background: rgba(67, 20, 7, 0.55); border-color: #fb923c; }
.stApp .sl-alert.cause b { color: #fed7aa; }
.stApp ul.todo { margin: 4px 0 6px; padding-left: 20px; font-size: 15px; line-height: 1.45; }
.stApp ul.todo li { margin: 0 0 6px; }
/* everything that is not the answer, folded away */
.stApp details.more { margin: 8px 0 0; border: 1px solid var(--gh-border); border-radius: 16px; background: var(--gh-subtle); }
.stApp details.more > summary { cursor: pointer; padding: 12px 16px; font-size: 14px; font-weight: 600; color: var(--gh-fg);
                                list-style: none; }
.stApp details.more > summary::-webkit-details-marker { display: none; }
.stApp details.more > summary::before { content: "＋"; color: var(--gh-accent); margin-right: 8px; }
.stApp details.more[open] > summary::before { content: "－"; }
.stApp details.more > :not(summary) { margin: 0 16px 16px; }
/* phone page: the measurement and the element, in one bordered grid */
.stApp .m-head { font-family: """ + MONO + """; font-size: 12px; font-weight: 600; letter-spacing: 0.14em;
                 text-transform: uppercase; color: var(--gh-accent); margin: 32px 0 10px; padding-bottom: 8px;
                 border-bottom: 1px solid var(--gh-border); }
.stApp .m-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1px; margin: 8px 0 12px;
                 background: var(--gh-border); border: 1px solid var(--gh-border); border-radius: 16px; overflow: hidden; }
.stApp .m-grid > div { background: var(--gh-solid); padding: 14px 16px; }
.stApp .m-grid > div:last-child:nth-child(odd) { grid-column: span 2; }
.stApp .m-lab { font-family: """ + MONO + """; font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase;
                color: var(--gh-muted); }
.stApp .m-val { font-size: 22px; font-weight: 600; color: var(--gh-fg); line-height: 1.3; font-variant-numeric: tabular-nums; }
.stApp .m-sub { font-size: 12px; color: var(--gh-muted); }
.stApp .m-red { color: var(--gh-danger); }
.stApp .m-line { font-size: 14px; color: var(--gh-fg); margin: 4px 0; }
</style>"""

DETAIL_STYLE = """<style>
/* a result's own page: title with its number, a state pill, glass boxes */
.block-container { max-width: 1280px; padding-top: 2rem !important; }
.stApp a.d-back { font-size: 14px; }
.stApp .d-title { font-size: 34px; font-weight: 600; letter-spacing: -0.02em; line-height: 1.2; color: var(--gh-fg); margin: 8px 0; }
.stApp .d-num { color: var(--gh-muted); font-weight: 400; margin-left: 10px; font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.7em; }
.stApp .d-meta { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; padding-bottom: 16px; margin-bottom: 24px;
                 border-bottom: 1px solid var(--gh-border); }
.stApp .d-meta p { margin: 0; color: var(--gh-muted); }
.stApp .d-hero { border-bottom: 1px solid var(--gh-border); padding-bottom: 16px; margin-bottom: 16px; }
.stApp .d-hero .d-val { font-size: 48px; letter-spacing: -0.03em; color: #ffffff; }
.stApp .d-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px 24px; margin-bottom: 16px; }
.stApp .d-lab { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--gh-muted); }
.stApp .d-val { font-size: 26px; font-weight: 600; color: var(--gh-fg); line-height: 1.25; font-variant-numeric: tabular-nums; }
.stApp .d-unit { font-size: 0.6em; font-weight: 400; color: var(--gh-muted); margin-left: 2px; }
.stApp .d-sub { font-size: 12px; color: var(--gh-muted); }
.stApp .d-red { color: var(--gh-danger); }
.stApp .d-none { font-size: 16px; font-weight: 600; color: var(--gh-danger); margin: 0 0 4px; }
.stApp .d-line { font-size: 14px; color: var(--gh-fg); }
.stApp .d-words .d-val { font-size: 16px; }  /* words (element, type, level), not numbers */
.stApp .d-note { font-size: 12px; color: var(--gh-muted); margin: 6px 0; }
.stApp .d-views { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }
.stApp .gh-h { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 12px; font-weight: 600; letter-spacing: 0.14em;
               text-transform: uppercase; color: var(--gh-accent); margin: 32px 0 10px; padding-bottom: 8px;
               border-bottom: 1px solid var(--gh-border); }
@media (max-width: 800px) { .stApp .d-title { font-size: 24px; } .stApp .d-hero .d-val { font-size: 32px; } }
</style>"""


def _box(title: str, body: str) -> str:
    """GitHub's bordered panel with a grey header row."""
    return f'<div class="Box"><div class="Box-header">{title}</div><div class="Box-body">{body}</div></div>'


@st.cache_resource
def shared():
    """The crack model, one GPU lock, one wall, the element model (D53) and the scene model (D54) for every visitor
    of this run; the last two are None when their file is missing."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    export = Path(os.environ.get("SECOND_LOOK_EXPORT_DIR") or EXPORT_DIR)  # read now: tests point it elsewhere
    out = (load_checkpoint(MODEL, device), device, threading.Lock(), Wall(export_dir=export),
           load_element_model(device=device), load_scene_model(device=device), load_spalling_model(device=device))
    model, _, gpu, wall, element_model, scene_model, spalling_model = out
    start_inbox(wall, model, device, gpu, element_model, scene_model, spalling_model,
                inbox=Path(os.environ.get("SECOND_LOOK_INBOX") or INBOX),  # when the inbox folder exists
                parts_loader=parts_model)  # SAM 3.1, the same copy as the phone page, loaded on first use
    return out


@st.cache_resource(show_spinner="Loading SAM 3.1 (about 10 s, once)...")
def parts_model():
    """SAM 3.1 for the part the crack lies on (D70); None when its weights are not on this laptop."""
    from element.parts import load_parts_model
    return load_parts_model()


# Line sketches for the three tips on the phone page (D84): drawn here in SVG, no image files, in the scan colour.
_SVG = ('<svg viewBox="0 0 120 72" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
        'stroke-linejoin="round" aria-hidden="true">')
_CRACK = '<path d="M{x} 8 l-4 12 l6 8 l-5 12 l4 10 l-3 14"/>'
TIP_SKETCHES = {
    # the phone held close to the wall, a short distance between them
    "close": _SVG + '<path d="M84 4 V68 M84 14 l8 -8 M84 30 l8 -8 M84 46 l8 -8 M84 62 l8 -8"/>'
             + '<rect x="22" y="18" width="20" height="36" rx="4"/><path d="M29 49 h6"/>'
             + '<path d="M32 64 H82 M32 60 v8 M82 60 v8" stroke-width="1.5"/>'
             + '<path d="M44 36 h4 m4 0 h4 m4 0 h4 m4 0 h4 m4 0 h4" stroke-width="1.5" opacity="0.6"/></svg>',
    # a card flat beside the crack, its number covered
    "card": _SVG + _CRACK.format(x=40) + '<rect x="56" y="22" width="44" height="28" rx="4"/>'
            + '<rect x="62" y="38" width="26" height="6" rx="1" fill="currentColor" stroke="none" opacity="0.6"/>'
            + '<rect x="62" y="28" width="9" height="7" rx="1.5" stroke-width="1.5"/></svg>',
    # the frame filled with wall and crack; a window outside it, crossed out
    "wall": _SVG + '<path d="M14 16 V6 H24 M70 6 H80 V16 M80 56 V66 H70 M24 66 H14 V56"/>'
            + _CRACK.format(x=50) + '<g opacity="0.6"><rect x="90" y="20" width="22" height="26" rx="2"/>'
            + '<path d="M101 20 V46 M90 33 H112"/></g><path d="M88 52 L114 14" stroke-width="2.5"/></svg>',
}


@st.cache_resource
def _mode_watcher():
    import streamlit.components.v2 as components_v2
    return components_v2.component("sl_mode_watch", js=MODE_JS, isolate_styles=False)


def phone_page() -> None:
    """The photo taker (review 26 Nov 2026): instructions, then the photo button, then the options, on a light
    concrete wall (theme.phone_css)."""
    st.markdown(phone_css(), unsafe_allow_html=True)
    _mode_watcher()(key="sl_mode")  # light or dark, as chosen in the menu (top right) or by the phone
    st.markdown(f'<div class="gh-header">{logo_title()}</div>', unsafe_allow_html=True)
    with st.container(key="ph-intro"):
        st.markdown('<p class="lead">Photograph a crack.<br>See it measured.</p>'
                    '<p class="lead-sub">Free, about a minute. An engineer looks at anything serious.</p>'
                    '<div class="tips">'
                    f'<span>{TIP_SKETCHES["close"]}<span><b><span class="step-n">1</span>Close</b>within 1 m, the wall '
                    'fills the photo</span></span>'
                    f'<span>{TIP_SKETCHES["card"]}<span><b><span class="step-n">2</span>Card</b>a bank or credit card '
                    'flat beside the crack gives mm; cover its number</span></span>'
                    f'<span>{TIP_SKETCHES["wall"]}<span><b><span class="step-n">3</span>Wall only</b>no windows, sky '
                    'or people</span></span>'
                    '</div>'
                    # D99: the photo advice of a crack engineer's method ([46] ch. 5.2)
                    '<p class="note">Tip: hold the phone flat to the wall, wipe the dust off the crack, and on a shiny '
                    'wall try without flash.</p>', unsafe_allow_html=True)
    with st.container(key="ph-options"):
        st.markdown('<p class="m-head">Before the photo</p>', unsafe_allow_html=True)
        card_placed = st.checkbox("I put a bank or credit card next to the crack")  # no tick: no card search (D56)
        share = st.checkbox("Share with the engineers' wall")
        share_location = st.checkbox("Share where the photo was taken (only engineers see it, on their map)",
                                     disabled=not share) and share
        # D99: a second visit of the same crack, compared only beyond the measurement's error ([46])
        before = st.text_input("Photographed this crack before? Its result number", placeholder="optional, e.g. 12",
                               max_chars=6)
        # one confirmation (D34): it is the consent, so its terms stand right above the button
        st.markdown('<p class="note">By confirming you agree: you took the photo, it shows no faces, house numbers '
                    'or addresses, and it is analysed. It is analysed in memory, not saved. Its location is read only '
                    'if you tick "Share where the photo was taken", and then only the engineers see it.</p>',
                    unsafe_allow_html=True)
        if not st.session_state.get("ph_ready"):
            if st.button("Confirm", key="ph-confirm", type="primary", width="stretch"):
                st.session_state["ph_ready"] = True
                st.rerun()
    if not st.session_state.get("ph_ready"):
        return
    with st.container(key="ph-photo"):
        upload = st.file_uploader("📷  Take or choose a photo", type=["jpg", "jpeg", "png", "heic", "heif"],
                                  max_upload_size=10)
    previous = int(before.strip().lstrip("#")) if before.strip().lstrip("#").isdigit() else None
    consent = True
    if upload is None:
        return
    done = st.session_state.setdefault("done", {})
    key = f"{upload.file_id}-{int(card_placed)}-{int(share)}-{previous}-{int(share_location)}"  # an option changed: read again
    if key not in done:  # analyse each upload once, even when the page reruns
        model, device, gpu, wall, _, scene_model, _ = shared()
        try:
            with st.spinner("Looking at the photo..."):
                done[key] = analyse(upload.getvalue(), consent, share, wall, model, device, gpu,
                                    scene_model, card_placed, previous=previous, share_location=share_location)
        except (ConsentMissing, PhotoRejected) as e:
            st.error(str(e))
            return
    upload_key = key
    result = done[upload_key]
    wall = shared()[3]
    if share_location:
        blocks.where(result, upload_key, wall)  # the photo's place, or the phone's, for the engineer's map
    blocks.steel_alert(result)  # D115: exposed steel first
    with st.container(key="ph-result"):
        st.markdown(f'<p class="statement">{state(result.statement)}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="reason">{result.reason}</p>', unsafe_allow_html=True)
        if result.triage:
            level_block(result)
        if result.growth:  # D99: since the last visit, beyond the error only
            st.markdown(f'<p class="reason"><b>Since the last visit:</b> {result.growth["reason"]}</p>',
                        unsafe_allow_html=True)
        if result.statement in (CRACK, UNSURE):
            blocks.todo(result)  # D121: what to do now
    if result.photo is None or result.statement not in (CRACK, UNSURE):
        st.image(result.overlay, width="stretch")
    else:
        blocks.explore(result, upload.file_id, wall=wall)  # D118, D119
    blocks.before_after(result, wall)  # D122
    if result.statement in (CRACK, UNSURE):
        blocks.ask(result, upload.file_id, wall)  # D120
        blocks.knock(result, upload.file_id, wall)  # D120
        classification_block(result, upload.file_id)
    if result.card is not None:
        measurement_block(result, upload.name)
    blocks.how_sure()  # D120
    st.markdown(f'<p class="note">{NOT_A_SAFETY_CHECK}</p>', unsafe_allow_html=True)


def classification_block(result, key: str) -> None:
    """Where the crack is and what type it is (spec/features/app_classification.md, D53). The resident only sends a
    second photo; the element model and the written rules read element and type; the written risk matrix gives the
    level by itself (D55). Any doubt gives Expert needed."""
    _, device, gpu, wall, element_model, scene_model, _ = shared()
    with st.container(key=f"ph-step2-{key}"):  # its own card on the phone page
        st.markdown('<p class="m-head">Step 2 · the whole element</p>', unsafe_allow_html=True)
        st.markdown('<p class="lead-s">Step back: photograph the whole column, beam or wall, phone upright.</p>',
                    unsafe_allow_html=True)
        second = st.file_uploader("Photo of the whole element", type=["jpg", "jpeg", "png", "heic", "heif"], max_upload_size=10,
                                  key=f"element_{key}")
        seen = st.session_state.setdefault("element_seen", {})
        if second is not None and seen.get(key) != second.file_id:
            try:
                sam = parts_model()
                with st.spinner("Looking at the element..." + (" SAM 3.1 adds about 10 s." if sam else "")):
                    analyse_element(second.getvalue(), result, wall, element_model, device, gpu, scene_model,
                                    crack_model=shared()[0], spalling_model=shared()[6], parts_model=sam)
                seen[key] = second.file_id
                st.rerun()  # the level, the advice and the viewer at the top follow photo 2
            except PhotoRejected as e:
                st.error(str(e))
        a, s = result.element, result.suggested_pattern()
        cells = [_cell("Element", "—" if a is None else (ELEMENT_WORDS.get(a.element, a.element)),
                       "send the photo above" if a is None else
                       (f"model, {a.probability:.0%}" if a.element != UNSURE_ELEMENT else "model unsure"),
                       red=a is not None and a.could_be_column),
                 _cell("Position", *_position_cell(result))]
        if s is not None and s.pattern is not None:  # the crack's kind, only when the rules could read it
            cells.insert(1, _cell("Type of crack", PATTERN_NAMES[s.pattern].capitalize(),
                                  "read from the crack's shape and direction" + ("" if s.clear else ", unclear")))
        if result.parts is not None:
            cells.append(_cell("Lies on", *parts_words(result.parts)))
        if result.on_element is not None and result.on_element.overlay is not None:
            st.image(result.on_element.overlay, caption="Photo 2: the element the crack lies on in light blue, the crack in red", width="stretch")
        st.markdown('<div class="m-grid">' + "".join(cells) + "</div>", unsafe_allow_html=True)
        if result.triage is None:
            level_block(result)  # the level itself stands at the top of the page once given (D121)
        if a is None and result.location is None:
            return
        with st.expander("How it was read"):  # the notes behind element, type and position, folded away (D83)
            _classification_notes(result, a)


def _classification_notes(result, a) -> None:
    if a is not None:
        st.markdown(f'<p class="note">{a.reason}</p>', unsafe_allow_html=True)
    if result.parts is not None:
        pr = result.parts
        st.markdown(f'<p class="note"><b>SAM 3.1:</b> {pr["reason"]}'
                    + (f' Wall: {pr["note"]}.' if pr.get("note") else "")
                    + " Shown only: the level does not use it.</p>", unsafe_allow_html=True)
    if result.location is not None:
        loc = result.location
        if loc.get("direction"):  # against the room's true vertical (D69); no type or level by itself
            st.markdown(f'<p class="m-line"><b>Direction: {loc["direction"]}</b>, '
                        f'{loc["crack_to_vertical_deg"]:.0f}° from vertical, read against the room\'s vertical lines '
                        '(door and window frames, corners, the ceiling) in the second photo.</p>',
                        unsafe_allow_html=True)
        st.markdown(f'<p class="note">{loc["reason"]}</p>', unsafe_allow_html=True)
    if result.location_preview is not None:  # the sender's own element photo; never on the wall (rule 9)
        st.image(result.location_preview, width=360,
                 caption="Red ring: where the crack is. White: your close-up. Cyan: the bank card. Yellow: the "
                         "element's edges. Orange (not linked): where the card's look matched best.")
    if result.parts_preview is not None:  # the sender's own element photo; never on the wall (rule 9)
        from element.parts import COLOURS, PART_NAMES
        legend = " · ".join(f'<span style="color:rgb{COLOURS[p]}">■</span> {PART_NAMES[p]}'
                            for p in (result.parts or {}).get("found", {}))
        st.image(result.parts_preview, width=360, caption="SAM 3.1: the parts it found. Red ring: the crack.")
        if legend:
            st.markdown(f'<p class="note">{legend}</p>', unsafe_allow_html=True)
    if result.location is not None:
        st.markdown('<p class="m-lab">What the app found in the two photos (for testing)</p>'
                    + diagnosis_html(result.location.get("diag") or {}), unsafe_allow_html=True)


def parts_words(parts: dict) -> tuple[str, str]:
    """The 'Lies on' cell: the part SAM 3.1 holds most of the crack (D70), or not clear."""
    if not parts.get("on"):
        return "not clear", "SAM 3.1"
    sub = f'SAM 3.1, {parts["confidence"]:.0%}' if parts.get("confidence") is not None else "SAM 3.1"
    if parts.get("border_name"):
        sub = f'border with {parts["border_name"].lower()} · ' + sub
    return parts["name"], sub


POSITION_WORDS = {"end_zone": "End zone", "middle": "Middle", "unknown": "not known"}


def diagnosis_html(d: dict) -> str:
    """Each step of linking the two photos, in plain words, for field tests (D67). No photo, only numbers."""
    yes = lambda v: "yes" if v else "no"  # noqa: E731
    rows = [("Bank card in the close-up", yes(d.get("card_in_close_up")) + ("" if d.get("card_in_close_up") else
             " (tick \"I put a bank or credit card next to the crack\", or the card was not found there)")),
            ("Crack line in the close-up", yes(d.get("crack_line_in_close_up"))),
            ]
    if "card_plainness" in d:
        rows.append(("Card print (needs 4 or more)", str(d["card_plainness"])))
    if "card_upright_in_photo" in d:
        rows.append(("Card lies upright in the photo", yes(d["card_upright_in_photo"])))
    if "look_scores" in d:
        rows.append(("Best places by the card's look (needs 0.5)", ", ".join(map(str, d["look_scores"])) or "none"))
        rows.append(("Card outline found there", ", ".join(yes(v) for v in d.get("outline_found", [])) or "none"))
        if d.get("outline_look"):
            rows.append(("Outline looks like the card (needs 0.5)", ", ".join(map(str, d["outline_look"]))))
    if "card_link" in d:
        rows.append(("Link through the card", d["card_link"]))
    if "key_points" in d:
        rows.append(("Matching key points (needs 20)", str(d["key_points"])))
    if "true_vertical_found" in d:
        rows.append(("Room lines for the true vertical (needs 2)", f'{d.get("room_lines", 0)}, '
                     + ("found" if d["true_vertical_found"] else "not enough")))
    if "element_edges" in d:
        rows.append(("Two long edges of the element found", yes(d["element_edges"])))
    return "".join(f'<p class="note"><b>{k}:</b> {v}</p>' for k, v in rows)


def _position_cell(result) -> tuple[str, str]:
    """Position on the element (core.where_on_element): from the two photos linked (D66), else from the crack on
    photo 2 (D62), else not known, counted as an end zone. Said in plain words for the resident."""
    t = result.triage or {}
    pos = t.get("position", "unknown")
    if result.element is None:
        return "—", "where on the column, beam or wall the crack sits: send photo 2"
    if pos == "end_zone":
        return "Near an end", ("close to where it meets a floor, beam or column. The steel bars are anchored there, "
                               "so a crack there counts more.")
    if pos == "middle":
        return "In the middle", ("away from the ends of the " + (t.get("element_name") or "element")
                                 + ", where a crack counts less.")
    return "Not known", ("the app could not place the crack on the element, so it counts it as near an end, the "
                         "more serious case.")


def position_source(t: dict) -> str:
    return "from photo 2" if "photo of the whole element" in (t.get("position_reason") or "") else "from the two photos"


def _type_sub(result, s) -> str:
    if (result.crack is None or not result.crack.found) and result.pixel_shape is None:
        return "no crack line to read its direction and shape"
    if s is None or s.pattern is None:
        return "not clear from the photo: an engineer decides"
    no_card = result.crack is None or not result.crack.found
    return ("suggested by the written rules" + (", read in the photo (no card)" if no_card else "")
            + ("" if s.clear else ", unclear"))


def level_block(result) -> None:
    t = result.triage
    if t is None:
        st.markdown(f'<p class="m-line"><b>{WAITING}</b></p>', unsafe_allow_html=True)
        return
    st.markdown(f'<p class="statement">{state(t["level"])}</p><p class="reason">{t["reason"]}</p>'
                + (f'<p class="m-line">{t["why"]}</p>' if t.get("why") else "")
                + f'<p class="m-line">{t["next_step"]}</p>'
                + ('<p class="m-line m-red"><b>An engineer must confirm this result.</b></p>'
                   if t["engineer_review"] else ""), unsafe_allow_html=True)


def _cell(label: str, value: str, sub: str = "", red: bool = False) -> str:
    return (f'<div><div class="m-lab">{label}</div><div class="m-val{" m-red" if red else ""}">{value}</div>'
            f'<div class="m-sub">{sub}</div></div>')


def measurement_block(result, name: str) -> None:
    """Shown only when a bank card is in the photo (spec/features/app_measure.md)."""
    card, c = result.card, result.crack
    tilt_sub = (f"90° = straight on · {card.status} · left-right {abs(card.tilt_left_right_deg):.0f}°, up-down "
                f"{abs(card.tilt_up_down_deg):.0f}° off")
    cells = []
    if result.measured:
        cells += [_cell("Width, widest point", f"{c.width_max_mm:.2f} mm", f"± {c.width_uncertainty_mm:.2f} mm"),
                  _cell("Width, median", f"{c.width_median_mm:.2f} mm"),
                  _cell("Length", f"{c.length_main_mm:.0f} mm", f"all branches {c.length_total_mm:.0f} mm")]
    cells.append(_cell("Camera angle to the wall", f"{90 - card.tilt_deg:.0f}°", tilt_sub, red=card.status == "retake"))
    if result.measure_note:
        st.markdown(f'<p class="m-line"><b>Width not measured.</b> {result.measure_note}</p>', unsafe_allow_html=True)
    for w in result.warnings:
        st.markdown(f'<p class="note">{w}</p>', unsafe_allow_html=True)
    with st.expander("Measurement details"):  # the main numbers are in the picture above (D83)
        st.markdown('<div class="m-grid">' + "".join(cells) + "</div>", unsafe_allow_html=True)
        st.markdown(f'<p class="note">Camera: straight from {90 - TILT_OK_DEG:g}° to the wall, retake below '
                    f'{90 - TILT_RETAKE_DEG:g}° (illustrative).</p>', unsafe_allow_html=True)
        if result.band():
            st.markdown(f'<p class="note">Width band of the risk matrix (upper end of the width): '
                        f'<b>{result.band()}</b>. Not a level: the level also needs the pattern and the element.</p>',
                        unsafe_allow_html=True)
        if c is not None and c.found and c.angle_to_vertical_deg is not None:
            st.markdown(f'<p class="note">Crack shape: {c.angle_to_vertical_deg:.0f}° from vertical · straightness '
                        f'{c.straightness:.2f} · coherence {c.coherence:.2f} · {c.n_loops} closed cells · branches '
                        f'{c.branch_ratio:.1f}x main length</p>', unsafe_allow_html=True)
        _downloads(result, name)


def _downloads(result, name: str) -> None:
    report = {"photo": Path(name).stem, **result.to_dict()}
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


# what each risk tab means, in a line under it (the five outputs of AGENTS.md)
TAB_WORDS = {"Category 3": "An engineer must confirm, sent now",
             "Expert needed": "The app is unsure: an engineer decides",
             "Category 2": "Low-cost advice, recheck in a month",
             "Category 1": "Routine monitoring, recheck in 6 months"}


@st.fragment(run_every=4)
def wall_grid() -> None:
    """One tab per risk level with its count, highest risk first (D78); no photo is shown until a tab opens that collection. Each card opens
    its result over the wall, without leaving the page (D77)."""
    wall = shared()[3]
    groups = {e.number: risk_group(e.stats) for e in wall.entries()}  # by risk, not by what the model saw (D78)
    entries = [e for e in wall.entries() if groups[e.number] is not None]
    counts = {s: sum(groups[e.number] == s for e in entries) for s in RISK_TABS}
    if "opened" not in st.session_state:  # coming back from a result's page: reopen its collection
        back = st.query_params.get("open")
        st.session_state["opened"] = back if back in RISK_TABS else None
    opened = st.session_state.get("opened")
    with st.container(key="gh-tabs", horizontal=True, gap="small"):  # GitHub's tabs, each with its counter (D76)
        for s in RISK_TABS:
            with st.container(key=f"gh-tab-{s.replace(' ', '-')}", gap=None):
                if st.button(f"{s} `{counts[s]}`", key=f"open_{s}", width="content",
                             type="primary" if opened == s else "secondary"):
                    opened = None if opened == s else s  # a second click closes it
                    st.session_state["opened"] = opened
                st.markdown(f'<p class="tab-desc">{TAB_WORDS[s]}</p>', unsafe_allow_html=True)
    if opened is None:
        st.markdown('<div class="gh-blank">Choose a tab to see its results.</div>', unsafe_allow_html=True)
        return
    mine = [e for e in entries if groups[e.number] == opened]
    if not mine:
        st.markdown(f'<div class="gh-blank">No result in "{opened}" yet.</div>', unsafe_allow_html=True)
        return
    with st.container(key="gh-cards"):  # each card opens its result over the wall, on the same page (D77)
        for e in mine:
            with st.container(key=f"card-{e.number}"):
                st.markdown(
                    f'<div class="wall-photo" title="Open result {e.number}"><div class="pic">'
                    f'<img src="data:image/jpeg;base64,{base64.b64encode(e.thumb).decode()}" alt="Result {e.number}">'
                    + (f'<img class="inset" src="data:image/jpeg;base64,{base64.b64encode(e.photo2).decode()}" '
                       f'alt="Photo 2 of result {e.number}">' if e.photo2 else "")
                    + f'</div><div class="meta"><span class="num">#{e.number}</span>{_photo_badge(e.stats)}</div></div>',
                    unsafe_allow_html=True)
                if st.button(f"Open result {e.number}", key=f"open-item-{e.number}"):
                    st.session_state["item"] = e.number
                    st.rerun(scope="app")  # the overlay belongs to the page, not to this refreshing grid


def _photo_badge(stats: dict) -> str:
    """Under the photo in the gallery, as labels: TEST, the widest width when measured, the level once given, and
    the element (D53)."""
    c = stats.get("crack") or {}
    out = label("TEST", "severe") if stats.get("test") else ""
    if stats.get("measured") and c.get("width_max_mm") is not None:
        out += label(f'{c["width_max_mm"]:.2f} mm', "accent")
    elif ((stats.get("scale_estimate") or {}).get("crack") or {}).get("width_max_mm") is not None:  # D116
        out += label(f'≈ {stats["scale_estimate"]["crack"]["width_max_mm"]:.2f} mm', "accent")
    t, e = stats.get("triage"), stats.get("element")
    if t:
        out += label(t["level"]) + f'<span>{(t["element_name"] or "element unknown").capitalize()}</span>'
    else:
        out += label(WAITING_SHORT, "done")
        if e:
            out += f'<span>{(e["name"] or ELEMENT_WORDS.get(e["answer"], e["answer"])).capitalize()}</span>'
    return out


def where_what_html(d: dict) -> str:
    """The result page's block: element, position, type and level, each with where it comes from (D53)."""
    t, e, s = d.get("triage"), d.get("element"), d.get("pattern_suggestion")
    if not (t or e):
        return ""
    if t:
        cells = [_big("Element", (t["element_name"] or "not clear").capitalize(), "",
                      f'element model, {t["element_probability"]:.0%}'),
                 _big("Position", t["position_name"].split(" (")[0].capitalize(), "",
                      position_source(t) if t.get("position") in ("end_zone", "middle") else "counts as an end zone"),
                 _big("Type", (t["pattern_name"] or "not clear").capitalize(), "", "written rules"),
                 *_parts_big(d),
                 _big("Level", state(t["level"]), "", "risk matrix, written rules")]
        tail = (f'<p class="d-line">{t["reason"]} {t.get("why") or ""} {t["next_step"]}</p>'
                + ('<p class="d-line d-red"><b>An engineer must confirm this result.</b></p>'
                   if t["engineer_review"] else ""))
    else:
        cells = [_big("Element", (e["name"] or ELEMENT_WORDS.get(e["answer"], e["answer"])).capitalize(), "",
                      f'model, {e["probability"]:.0%}', red=e["could_be_column"]),
                 _big("Type", (s["pattern_name"] or "not clear").capitalize() if s else "—", "",
                      "suggested by the written rules" if s and s["pattern_name"] else "not clear"),
                 *_parts_big(d)]
        tail = f'<p class="d-line">{WAITING}</p>'
    return '<div class="d-grid d-words">' + "".join(cells) + "</div>" + tail


def _parts_big(d: dict) -> list[str]:
    """The result page's 'Lies on' cell from SAM 3.1 (D70), when it ran; shown only, not used by the level."""
    if not d.get("parts"):
        return []
    value, sub = parts_words(d["parts"])
    return [_big("Lies on", value, "", sub)]


def _big(label: str, value: str, unit: str = "", sub: str = "", red: bool = False) -> str:
    return (f'<div class="d-cell"><div class="d-lab">{label}</div><div class="d-val{" d-red" if red else ""}">'
            f'{value}<span class="d-unit">{unit}</span></div><div class="d-sub">{sub}</div></div>')


def _img(data: bytes, alt: str) -> str:
    return f'<img class="d-photo" src="data:image/jpeg;base64,{base64.b64encode(data).decode()}" alt="{alt}">'


def _head_html(entry) -> str:
    """Title (its risk tab) with its number, the level pill, and the reason, as a GitHub issue (D78)."""
    d = entry.stats
    level = (d.get("triage") or {}).get("level")
    group = risk_group(d) or (level or entry.statement)
    return (f'<p class="d-title">{group}<span class="d-num">#{entry.number}</span></p>'
            f'<div class="d-meta">{state(level) if level else label(WAITING_SHORT, "done")}'
            f'{label("TEST", "severe") if d.get("test") else ""}<p>{d.get("reason", "")}</p></div>')


def _views_html(entry) -> str:
    """Photo 1 seen three more ways (D81): width heatmap, confidence heatmap, enhanced view (not an X-ray)."""
    views = getattr(entry, "views", None) or {}
    if not views:
        return ""
    cells = "".join(f'<div><p class="d-lab">{name}</p>{_img(data, f"Result {entry.number}, {name}")}</div>'
                    for name, data in views.items())
    return _box("Photo 1 · heatmaps and enhanced view",
                f'<div class="d-views">{cells}</div><p class="d-note">Shown only: the level does not use them. No '
                'stress map: that needs a structural model with the loads and materials, not a photo.</p>')


def _steps_html(entry) -> str:
    """The photo in steps, for the engineer (review 26 Nov 2026): photo, crack over it, crack alone, unusual surface."""
    from app.sequence import steps_from_entry
    s = steps_from_entry(entry.zoom, entry.marks)
    if len(s) < 2:
        return ""
    cells = "".join(f'<div><p class="d-lab">{cap}</p>{_img(jpg, f"Result {entry.number}, {cap}")}</div>' for cap, jpg in s)
    return _box("Photo 1 · step by step", f'<div class="d-views">{cells}</div>')


def _photo2_html(entry) -> str:
    d = entry.stats
    return ((_box("Photo 2 · the whole element", _img(entry.photo2, f"Result {entry.number}, photo 2"))
             if entry.photo2 else _box("Photo 2 · the whole element",
                                       '<p class="d-note">No photo of the whole element yet.</p>'))
            + (_box("Photo 2 read by SAM 3.1 · the parts it found",
                    _img(entry.parts_img, f"Result {entry.number}, parts found by SAM 3.1")
                    + f'<p class="d-note">{(d.get("parts") or {}).get("reason", "")} Shown only: the level '
                    'does not use it.</p>') if entry.parts_img else ""))


def _facts_html(d: dict) -> str:
    """The measurement, element, position, type and level, and the notes."""
    card, c = d.get("card"), d.get("crack") or {}
    conf = d.get("confidence")
    html = []
    if d.get("measured"):
        html.append('<div class="d-hero">' + _big("Width, widest point", f'{c["width_max_mm"]:.2f}', " mm",
                                                  f'± {c["width_uncertainty_mm"]:.2f} mm') + "</div>")
        cells = [_big("Width, median", f'{c["width_median_mm"]:.2f}', " mm"),
                 _big("Length", f'{c["length_main_mm"]:.0f}', " mm", f'all branches {c["length_total_mm"]:.0f} mm')]
    elif (d.get("scale_estimate") or {}).get("crack"):  # D116: from a door in photo 2, an estimate
        e = d["scale_estimate"]
        html.append('<div class="d-hero">' + _big("Width, widest point, ESTIMATED", f'≈ {e["crack"]["width_max_mm"]:.2f}',
                                                  " mm", f'± {e["crack"]["uncertainty_mm"]:.2f} mm · {e["basis"]}')
                    + "</div>")
        cells = [_big("Length, estimated", f'≈ {e["crack"]["length_mm"]:.0f}', " mm", e["name"])] \
            if e["crack"].get("length_mm") is not None else []
    elif d.get("estimate"):  # TEST cases only (D62): an estimate, never a measurement
        e = d["estimate"]
        html.append('<div class="d-hero">' + _big("Width of the marked damage, ESTIMATED (not measured)",
                                                  f'≈ {e["width_max_mm"]:g}', " mm",
                                                  f'± {e["uncertainty_mm"]:g} mm · {e["basis"]}') + "</div>")
        cells = [_big("Length, estimated", f'≈ {e["length_mm"]}', " mm", "from photo 2")]
    else:
        why = d.get("measure_note") or "No bank card in the photo: no measurement in mm."
        html.append(f'<p class="d-none">Width not measured</p><p class="d-line">{why}</p>')
        cells = []
    if card:
        cells.append(_big("Camera angle to the wall", f'{90 - card["tilt_deg"]:.0f}', "°",
                          f'90° = straight on · {card["status"]} · left-right {abs(card["tilt_left_right_deg"]):.0f}°, '
                          f'up-down {abs(card["tilt_up_down_deg"]):.0f}° off', red=card["status"] == "retake"))
    if conf is not None:
        cells.append(_big("Model confidence", f"{conf:.0%}"))
    if cells:
        html.append('<div class="d-grid">' + "".join(cells) + "</div>")
    html = [_box("Measurement", "".join(html))]
    if where_what_html(d):
        html.append(_box("Element, position, type and level", where_what_html(d)))
    if d.get("width_band_of_upper_end"):
        html.append(f'<p class="d-line">Width band of the risk matrix: <b>{d["width_band_of_upper_end"]}</b>. Not a '
                    'level: the level also needs the crack pattern and the element.</p>')
    sh = (c.get("shape") if c.get("found") else None) or d.get("shape_in_pixels") or {}
    if sh.get("angle_to_vertical_deg") is not None:
        html.append(f'<p class="d-note">Crack shape{" (in the photo, no card)" if sh.get("source") else ""}: '
                    f'{sh["angle_to_vertical_deg"]:.0f}° from vertical · straightness '
                    f'{sh["straightness"]:.2f} · coherence {sh["coherence"]:.2f} · {sh["n_loops"]} closed cells · '
                    f'branches {sh["branch_ratio"]:.1f}x main length</p>')
    for w in d.get("warnings") or []:
        html.append(f'<p class="d-note">{w}</p>')
    html.append(f'<p class="d-note">{NOT_A_SAFETY_CHECK} Thresholds illustrative until calibrated.</p>')
    return "".join(html)


def result_page(number: int) -> None:
    """One shared result on its own page (a direct link, /?wall=...&item=<n>), laid out as a GitHub issue (D76).
    From the wall, a card opens the same result over the wall instead (result_overlay, D77)."""
    entry = shared()[3].get(number)
    st.markdown(wall_css(), unsafe_allow_html=True)  # the same design as the wall
    _mode_watcher()(key="sl_mode")
    st.markdown(f'<a class="d-back" href="{_wall_href(**({"open": risk_group(entry.stats)} if entry and risk_group(entry.stats) else {}))}" '
                'target="_self">&larr; Back to the wall</a>', unsafe_allow_html=True)
    if entry is None:
        st.markdown('<p class="d-title">Result gone</p><p class="d-line">The wall keeps only the newest 200 shared '
                    'results, and none after the app stops.</p>', unsafe_allow_html=True)
        return
    st.markdown(_head_html(entry), unsafe_allow_html=True)
    left, right = st.columns([1.4, 1], gap="large")
    with left:
        entry_viewer(entry, 640)
    right.markdown(_window_html(entry), unsafe_allow_html=True)
    model_3d(number, entry.stats)


def close_result() -> None:
    st.session_state["item"] = None


def result_overlay(number: int) -> None:
    """A result opened from the wall, on the same page (D77): the wall behind goes blurred and light, the clicked
    crack stays sharp, and a window next to it holds everything else, scrolled inside. A click beside them or the
    close button brings the wall back."""
    entry = shared()[3].get(number)
    if entry is None:
        close_result()
        return
    with st.container(key="sl-overlay"):
        st.button("Close", key="sl-backdrop", on_click=close_result)  # the blurred area itself closes it
        with st.container(key="sl-stage", horizontal=True):
            entry_viewer(entry, 680)
            with st.container(key="sl-window"):
                st.button("✕", key="sl-close", on_click=close_result, help="Back to the wall")
                st.markdown(_head_html(entry) + _window_html(entry), unsafe_allow_html=True)
                model_3d(number, entry.stats)


def entry_viewer(entry, height: int) -> None:
    """Photo 1 to explore (D83): zoom, drag, the loupe on the narrowest and widest point. Older results without the
    viewer's photo keep the picture."""
    if entry.zoom is None:
        st.markdown(f'<div class="sl-crack">{_img(entry.large, f"Result {entry.number}, photo 1")}'
                    f'<div class="meta"><span class="num">#{entry.number}</span>{_photo_badge(entry.stats)}</div>'
                    '</div>', unsafe_allow_html=True)
        return
    # the engineer's viewer: corrections (add, remove) come back and are kept with the result (D119)
    corr = blocks.viewer(crack_view.viewer_html(entry.zoom, entry.marks, entry.large, engineer=True),
                         f"wall_viewer_{entry.number}", height)
    seen = st.session_state.setdefault("wall_corr_seen", {})
    if corr and seen.get(entry.number) != json.dumps(corr, sort_keys=True):
        seen[entry.number] = json.dumps(corr, sort_keys=True)
        e = blocks.save_entry_corrections(entry, corr, shared()[3])
        st.success(f"Saved: {len(e['removed'])} crack(s) removed, {len(e['added'])} drawn. Kept for retraining.")


def _window_html(entry) -> str:
    """Beside the crack: level first, everything else folded away (D83). The numbers are in the picture."""
    d = entry.stats
    out = ""
    if (d.get("exposed_steel") or {}).get("seen"):  # D115: first
        out += ('<div class="sl-alert"><b>⚠ Exposed steel seen: critical.</b> An engineer must look, whatever the '
                'crack looks like.</div>')
    causes = [c["kind"] for c in (d.get("exposed_steel") or {}).get("causes") or []]
    if causes:
        out += f'<div class="sl-alert cause"><b>Possible {", ".join(blocks.CAUSE_WORDS.get(c, c) for c in causes)}' \
               '</b> near the crack (SAM 3.1).</div>'
    out += _box("Element, position, type and level", where_what_html(d)) if where_what_html(d) else ""
    out += _when_where_html(d)
    out += _resident_html(d)
    if d.get("statement") in ("Crack found", "Expert needed"):  # D99: the crack read as a crack engineer reads it
        from app import engineer
        out += _box("Read like an engineer", engineer.html(d))
    more = _facts_html(d) + _steps_html(entry) + _photo2_html(entry) + _views_html(entry)
    return out + f'<details class="more"><summary>Measurement, photo 2 and heatmaps</summary>{more}</details>'


def _when_where_html(d: dict) -> str:
    """When the photo was taken (its own record) and where, if the sender shared it (review 26 Nov 2026)."""
    p = d.get("place") or {}
    rows = [f'<p class="d-line">Taken: <b>{d.get("taken_at") or "not in the photo"}</b></p>']
    if p.get("lat") is not None:
        src = "the photo's record" if p.get("source") == "photo" else "the sender's phone" + (
            f", ± {p['accuracy_m']} m" if p.get("accuracy_m") else "")
        rows.append(f'<p class="d-line">Place: <b>{p["lat"]:.5f}, {p["lon"]:.5f}</b> (from {src}) · '
                    f'<a href="https://www.openstreetmap.org/?mlat={p["lat"]}&mlon={p["lon"]}#map=19/{p["lat"]}/{p["lon"]}" '
                    'target="_blank">open map</a></p>')
    else:
        rows.append('<p class="d-line">Place: not shared</p>')
    return _box("When and where", "".join(rows))


def _resident_html(d: dict) -> str:
    """D119, D120: what the resident answered, the knock's numbers and their corrections, for the engineer."""
    from app.questions import words
    rows = [f'<p class="d-line">{q} <b>{a}</b></p>' for q, a in words(d.get("answers") or {})]
    if d.get("tap"):
        t = d["tap"]
        rows.append(f'<p class="d-line">Knock: {t["knocks"]} knock(s), pitch {t["pitch_hz"]} Hz, rings '
                    f'{t["ring_ms"]:.0f} ms. Hollow or solid not read yet.</p>')
    for c in d.get("corrections") or []:
        rows.append(f'<p class="d-line d-red">Resident\'s correction: {len(c["removed"])} crack(s) marked as wrong, '
                    f'{len(c["added"])} drawn ({", ".join(a["kind"] for a in c["added"]) or "none"}). Check them.</p>')
    return _box("What the resident told us", "".join(rows)) if rows else ""


def model_3d(number: int, stats: dict) -> None:
    """The crack in 3D, built by Grasshopper (D57): shown when its file is there, else waited for."""
    st.markdown('<p class="gh-h">3D model · Grasshopper</p>', unsafe_allow_html=True)
    from gh3d.layout import build
    if build({"number": number, **stats}) is None:  # the same test Grasshopper uses: no model to wait for
        st.markdown('<p class="d-note">No 3D model: the crack is drawn in 3D when it was measured with a bank card, '
                    'or, for test cases, placed from the photo of the whole element.</p>', unsafe_allow_html=True)
        return
    path = shared()[3].mesh_path(number)
    if path is None:
        waiting_for_3d(number)
        return
    try:
        model = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        waiting_for_3d(number)  # being written: next check
        return
    components.html(viewer_html(model), height=VIEW_HEIGHT + 4)


@st.fragment(run_every=3)
def waiting_for_3d(number: int) -> None:
    """Until Grasshopper has written the model: a note, checked again every 3 s; then the whole page shows it."""
    if shared()[3].mesh_path(number) is not None:
        st.rerun(scope="app")
    st.markdown('<p class="d-note">3D model being drawn (Grasshopper). It appears here by itself; Rhino must be open '
                'with second_look.gh.</p>', unsafe_allow_html=True)


def wall_page() -> None:
    """GitHub's layout (D76): header with the logo, one tab per risk level with its counter (D78),
    the opened tab's cards, the model in the footer. No QR here (1 Nov): it has its own page, /?qr."""
    st.markdown("<style>.block-container { max-width: 1280px; padding-top: 2rem !important; }</style>",
                unsafe_allow_html=True)
    st.markdown(wall_css(), unsafe_allow_html=True)  # the phone page's design, in VS Code's colours
    _mode_watcher()(key="sl_mode")  # light or dark, as chosen in the menu
    st.markdown(f'<div class="gh-header gh-large">{logo_title()}'
                '<span class="gh-right"><span class="gh-dot"></span>Live · shared results appear by themselves</span>'
                '</div>', unsafe_allow_html=True)
    wall_map()
    wall_grid()
    st.markdown(f'<div class="gh-footer">{MODEL_LINE}.</div>', unsafe_allow_html=True)


# one colour per category, the same on the tabs, the legend and the map (never green: it reads as "safe")
CATEGORY_HEX = {"Category 3": "#f85149", "Expert needed": "#b180d7", "Category 2": "#e5a50a", "Category 1": "#86c5ff"}
MAP_COLOURS = {k: [int(v[i:i + 2], 16) for i in (1, 3, 5)] for k, v in CATEGORY_HEX.items()}


@st.fragment(run_every=8)
def wall_map() -> None:
    """Where the shared results were taken, on one map (review 26 Nov 2026). Only results whose sender shared the
    place; a click on a point opens that result. Locations stay on this laptop's wall."""
    import pydeck as pdk
    from app.place import valid
    pts = []
    for e in shared()[3].entries():
        p, g = e.stats.get("place") or {}, risk_group(e.stats)
        if g and valid(p.get("lat"), p.get("lon")):
            pts.append({"number": e.number, "lat": p["lat"], "lon": p["lon"], "level": g,
                        "when": e.stats.get("taken_at") or "time not in the photo",
                        "source": "photo" if p.get("source") == "photo" else "phone", "colour": MAP_COLOURS[g]})
    st.markdown(f'<p class="gh-h">Map · {len(pts)} shared place{"s" if len(pts) != 1 else ""}</p>'
                '<div class="map-legend">' + "".join(
                    f'<span><i style="background:{c}"></i>{k} <b>{sum(p["level"] == k for p in pts)}</b></span>'
                    for k, c in CATEGORY_HEX.items()) + "</div>", unsafe_allow_html=True)
    if not pts:
        st.markdown('<div class="gh-blank">No shared places yet. A result appears here when the sender ticks "Share '
                    'where the photo was taken".</div>', unsafe_allow_html=True)
        return
    lat = sum(p["lat"] for p in pts) / len(pts)
    lon = sum(p["lon"] for p in pts) / len(pts)
    spread = max(max(abs(p["lat"] - lat) for p in pts), max(abs(p["lon"] - lon) for p in pts))
    zoom = 15 if spread < 0.002 else 13 if spread < 0.02 else 10 if spread < 0.3 else 6
    layer = pdk.Layer("ScatterplotLayer", data=pts, id="results", get_position="[lon, lat]", get_fill_color="colour",
                      get_radius=16, radius_min_pixels=8, radius_max_pixels=20, stroked=True,
                      get_line_color=[255, 255, 255], line_width_min_pixels=2, pickable=True)
    deck = pdk.Deck(layers=[layer], initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom),
                    map_style="light" if st.context.theme.type == "light" else "dark", tooltip={"html": "<b>#{number}</b> · {level}<br>{when}<br>place from the {source}"})
    ev = st.pydeck_chart(deck, on_select="rerun", selection_mode="single-object", key="wall-map", height=380)
    picked = ((getattr(ev, "selection", None) or {}).get("objects") or {}).get("results") or []
    if picked and st.session_state.get("item") != picked[0]["number"]:
        st.session_state["item"] = picked[0]["number"]
        st.rerun(scope="app")


def qr_page() -> None:
    """The QR alone, to show on the projector when students should scan it."""
    if not PUBLIC_URL:
        st.error("No phone link: start the app with src/app/launch.py.")
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
            if st.session_state.get("item"):  # a card was clicked: its result over the wall (D77)
                st.markdown(DETAIL_STYLE.replace(".block-container {", ".x-unused {"), unsafe_allow_html=True)
                result_overlay(st.session_state["item"])
    else:
        st.error("This page is only shown on the presenter's screen.")
elif st.query_params.get("qr") is not None:
    qr_page()
else:
    phone_page()
