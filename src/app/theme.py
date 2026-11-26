"""The app's look since 18 Nov (D82): dark and technical, a computer-vision instrument. A dark concrete wall behind
everything (drawn by the browser from noise, no photo), a faint measuring grid, frosted-glass panels, the system font
(SF on Apple devices), one cyan "scan" accent and the scan frame of the logo on what can be clicked.
The class names and `--gh-*` variables are kept from GitHub's look (D76) so the pages did not change shape.
Spec: spec/features/class_demo.md (Look)."""

import html

FONT = '-apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif'
DISPLAY = '-apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif'
MONO = 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace'

# Colours by role. "accent" is the interface's scan colour (links, focus, the opened tab); Category 1 has its own
# blue ("info") so a level never borrows the interface's colour.
TOKENS = {"fg": "#eef1f5", "muted": "#8b95a3", "canvas": "rgba(18, 21, 26, 0.88)", "solid": "#101318",
          "subtle": "rgba(255, 255, 255, 0.045)", "border": "rgba(255, 255, 255, 0.10)",
          "border-strong": "rgba(255, 255, 255, 0.18)", "accent": "#22d3ee", "info": "#6c8cff",
          "attention": "#ffb224", "severe": "#ff8a3d", "danger": "#ff4d6a", "done": "#b38cff", "tab": "#22d3ee",
          "page": "#0a0c0f"}

# One colour per answer. Never green: green reads as "safe", which the tool never says (rule 1).
TONES = {"Retake": "muted", "Expert needed": "done", "Category 1": "info", "Category 2": "attention",
         "Category 3": "danger", "Crack found": "danger", "No crack found": "muted"}
LEVEL_KINDS = ("info", "accent", "attention", "severe", "danger", "done")


def tone(text: str) -> str:
    return TONES.get(text, "muted")


def label(text: str, kind: str | None = None) -> str:
    """A small rounded label, tinted with its colour."""
    return f'<span class="Label Label--{kind or tone(text)}">{html.escape(text)}</span>'


def state(text: str, kind: str | None = None) -> str:
    """A filled pill with a glowing dot: the main answer of a page."""
    return f'<span class="State State--{kind or tone(text)}">{html.escape(text)}</span>'


def _vars() -> str:
    return ":root { " + " ".join(f"--gh-{k}: {v};" for k, v in TOKENS.items()) + " }"


def _hex_rgb(colour: str) -> str:
    c = colour.lstrip("#")
    return ", ".join(str(int(c[i:i + 2], 16)) for i in (0, 2, 4))


# The concrete: fractal noise in dark grey, coarse blotches and fine grain, made once here as a small tileable JPEG
# (no photo). A raster, not an SVG filter: Safari redraws SVG filters behind blurred panels on every frame and hung (D82).
def _concrete(side: int = 512, seed: int = 7) -> str:
    import base64
    import io

    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(seed)
    acc = np.zeros((side, side))
    for cells, weight in ((4, 1.0), (8, 0.6), (16, 0.35), (32, 0.2), (64, 0.12)):
        small = rng.random((cells, cells))
        big = np.asarray(Image.fromarray(np.tile(small, (3, 3)).astype(np.float32), "F")
                         .resize((3 * side, 3 * side), Image.BICUBIC))[side:2 * side, side:2 * side]  # wraps: tiles
        acc += weight * big
    acc = (acc - acc.min()) / (acc.max() - acc.min())
    grain = rng.normal(0, 1, (side, side))
    v = 16 + 16 * acc ** 1.4 + 2.6 * grain  # dark grey 16 to 32, with grain
    rgb = np.clip(np.stack([v, v + 1.5, v + 3.5], axis=2), 0, 255).astype(np.uint8)  # a touch cool
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=72)
    return "url(data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode() + ")"


CONCRETE = _concrete()


def scan_frame(colour: str = "var(--gh-accent)", size: str = "14px", width: str = "2px") -> str:
    """Four corner brackets, as the O's of the logo, drawn with gradients (no extra markup)."""
    c, s, w = colour, size, width
    return (f"linear-gradient({c},{c}) top left / {s} {w}, linear-gradient({c},{c}) top left / {w} {s}, "
            f"linear-gradient({c},{c}) top right / {s} {w}, linear-gradient({c},{c}) top right / {w} {s}, "
            f"linear-gradient({c},{c}) bottom left / {s} {w}, linear-gradient({c},{c}) bottom left / {w} {s}, "
            f"linear-gradient({c},{c}) bottom right / {s} {w}, linear-gradient({c},{c}) bottom right / {w} {s}")


GLASS = ""  # no backdrop blur on small panels: Safari hung with many of them (D82); the overlay keeps its own blur

THEME_CSS = _vars() + f"""
/* the page: dark concrete, lit from the top, a faint 40 px measuring grid, dark edges */
html, body {{ background: var(--gh-page); }}
.stApp {{ color: var(--gh-fg); background-color: var(--gh-page) !important;
  background-image:
    radial-gradient(120% 70% at 50% -10%, rgba(34, 211, 238, 0.10), transparent 55%),
    radial-gradient(140% 110% at 50% 20%, transparent 40%, rgba(0, 0, 0, 0.55) 100%),
    linear-gradient(rgba(255, 255, 255, 0.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255, 255, 255, 0.025) 1px, transparent 1px),
    {CONCRETE} !important;
  background-size: 100% 100vh, 100% 100vh, 40px 40px, 40px 40px, 512px 512px !important;
  background-repeat: no-repeat, no-repeat, repeat, repeat, repeat !important; }}
.stApp [data-testid="stMarkdownContainer"], .stApp button, .stApp label, .stApp input, .stApp textarea,
.stApp [data-testid="stCaptionContainer"] {{ font-family: {FONT} !important; -webkit-font-smoothing: antialiased; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stAppDeployButton"], [data-testid="stDecoration"] {{ display: none; }}
.stApp p, .stApp li, .stApp label {{ color: var(--gh-fg); font-size: 15px; line-height: 1.55; }}
.stApp strong, .stApp b {{ color: #ffffff; font-weight: 600; }}
.stApp a {{ color: var(--gh-accent); text-decoration: none; }}
.stApp a:hover {{ text-decoration: underline; text-underline-offset: 3px; }}
.stApp h1, .stApp h2, .stApp h3 {{ color: var(--gh-fg) !important; font-family: {DISPLAY} !important;
  font-weight: 600 !important; letter-spacing: -0.02em; text-transform: none; }}
.stApp code {{ font-family: {MONO}; }}
.stApp ::selection {{ background: rgba(34, 211, 238, 0.35); color: #ffffff; }}
/* the page header: the word mark, a live note on the right */
.gh-header {{ display: flex; align-items: center; column-gap: 16px; row-gap: 8px; flex-wrap: wrap; padding: 4px 0 20px;
              border-bottom: 1px solid var(--gh-border); margin: 0 0 24px; }}
.gh-header .logo {{ font-family: {DISPLAY}; font-size: 34px; font-weight: 700; letter-spacing: -0.01em;
                    color: var(--gh-fg); white-space: nowrap; line-height: 1.1; }}
.gh-header.gh-large .logo {{ font-size: 60px; letter-spacing: -0.025em; }}
.gh-header .gh-right {{ margin-left: auto; font-family: {MONO}; font-size: 12px; letter-spacing: 0.04em;
                        color: var(--gh-muted); }}
.gh-dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--gh-accent);
           margin-right: 8px; vertical-align: 0; box-shadow: 0 0 0 0 rgba(34, 211, 238, 0.6);
           animation: sl-pulse 2.4s ease-out infinite; }}
@keyframes sl-pulse {{ 0% {{ box-shadow: 0 0 0 0 rgba(34, 211, 238, 0.55); }}
                       70% {{ box-shadow: 0 0 0 10px rgba(34, 211, 238, 0); }}
                       100% {{ box-shadow: 0 0 0 0 rgba(34, 211, 238, 0); }} }}
.stApp .gh-quote {{ margin: 0 0 16px; padding: 0 1em; color: var(--gh-muted); border-left: 2px solid var(--gh-accent);
                    font-size: 15px; }}
.stApp .note {{ font-size: 12.5px; color: var(--gh-muted); }}
.stApp .reason {{ color: var(--gh-fg); font-size: 16px; }}
.stApp .statement {{ margin: 20px 0 6px; }}
/* labels and state pills: tinted glass, the colour as text; the main answer filled, with a glowing dot */
.Label {{ display: inline-block; padding: 1px 9px; font-size: 11.5px; font-weight: 600; line-height: 18px;
          letter-spacing: 0.01em; white-space: nowrap; border: 1px solid; border-radius: 2em; }}
.State {{ display: inline-flex; align-items: center; gap: 8px; padding: 6px 14px 6px 12px; font-size: 15px;
          font-weight: 600; line-height: 20px; border-radius: 2em; white-space: nowrap; border: 1px solid; }}
.State::before {{ content: ""; width: 8px; height: 8px; border-radius: 50%; background: currentColor;
                  box-shadow: 0 0 10px currentColor; }}
""" + "".join(
    f".Label--{k} {{ color: var(--gh-{k}); border-color: rgba({_hex_rgb(TOKENS[k])}, .45); "
    f"background: rgba({_hex_rgb(TOKENS[k])}, .10); }} "
    f".State--{k} {{ color: var(--gh-{k}); border-color: rgba({_hex_rgb(TOKENS[k])}, .55); "
    f"background: rgba({_hex_rgb(TOKENS[k])}, .14); box-shadow: 0 0 24px rgba({_hex_rgb(TOKENS[k])}, .18); }} "
    for k in LEVEL_KINDS) + f"""
.Label--muted {{ color: var(--gh-muted); border-color: var(--gh-border-strong); background: var(--gh-subtle); }}
.State--muted {{ color: var(--gh-fg); border-color: var(--gh-border-strong); background: var(--gh-subtle); }}
/* boxes: frosted-glass panels with a quiet header row */
.Box {{ border: 1px solid var(--gh-border); border-radius: 16px; background: var(--gh-canvas); margin: 0 0 16px;
        overflow: hidden; {GLASS} }}
.Box-header {{ padding: 10px 16px; background: var(--gh-subtle); border-bottom: 1px solid var(--gh-border);
               font-family: {MONO}; font-size: 11.5px; font-weight: 600; letter-spacing: 0.12em;
               text-transform: uppercase; color: var(--gh-muted); }}
.Box-body {{ padding: 16px; }}
.Box-body > p:last-child {{ margin-bottom: 0; }}
.Box img.d-photo {{ width: 100%; display: block; border-radius: 10px; }}
/* Streamlit's own widgets */
.stApp .stButton button, .stApp .stDownloadButton button {{ border-radius: 999px; border: 1px solid var(--gh-border-strong);
    background: rgba(255, 255, 255, 0.06); color: var(--gh-fg); font-weight: 500; padding: 6px 18px;
    transition: background .18s ease, border-color .18s ease, transform .18s ease, box-shadow .18s ease; {GLASS} }}
.stApp .stButton button:hover, .stApp .stDownloadButton button:hover {{ background: rgba(34, 211, 238, 0.12);
    border-color: rgba(34, 211, 238, 0.6); color: #ffffff; box-shadow: 0 0 24px rgba(34, 211, 238, 0.18); }}
.stApp .stButton button:active, .stApp .stDownloadButton button:active {{ transform: scale(0.97); }}
.stApp button:focus-visible, .stApp input:focus-visible {{ outline: 2px solid var(--gh-accent) !important;
    outline-offset: 2px; }}
/* the upload: a viewfinder with the logo's corner brackets and a scan line that sweeps across */
.stApp [data-testid="stFileUploaderDropzone"] {{ position: relative; overflow: hidden; border-radius: 16px;
    border: 1px solid var(--gh-border); padding: 28px 20px;
    background: {scan_frame(size="22px")}, rgba(255, 255, 255, 0.035) !important;
    background-repeat: no-repeat !important; {GLASS} transition: border-color .2s ease, box-shadow .2s ease; }}
.stApp [data-testid="stFileUploaderDropzone"]::after {{ content: ""; position: absolute; left: 0; right: 0; top: 0;
    height: 40%; pointer-events: none;
    background: linear-gradient(180deg, transparent, rgba(34, 211, 238, 0.10) 85%, rgba(34, 211, 238, 0.7) 100%);
    animation: sl-scan 3.2s cubic-bezier(.45, 0, .55, 1) infinite; }}
@keyframes sl-scan {{ 0% {{ transform: translateY(-100%); }} 100% {{ transform: translateY(250%); }} }}
.stApp [data-testid="stFileUploaderDropzone"]:hover {{ border-color: rgba(34, 211, 238, 0.5);
    box-shadow: 0 0 40px rgba(34, 211, 238, 0.12), inset 0 0 40px rgba(34, 211, 238, 0.05); }}
.stApp [data-testid="stFileUploaderDropzone"] button {{ background: var(--gh-accent) !important; color: #031216 !important;
    border: none !important; font-weight: 600; }}
.stApp [data-testid="stFileUploaderDropzone"] span, .stApp [data-testid="stFileUploaderDropzone"] small,
.stApp [data-testid="stFileUploaderDropzone"] svg {{ color: var(--gh-muted); }}
.stApp [data-testid="stFileUploaderFile"] {{ color: var(--gh-fg); }}
.stApp [data-testid="stCheckbox"] label span:first-child {{ border-radius: 6px; }}
.stApp [data-testid="stExpander"] details {{ border-radius: 16px; border-color: var(--gh-border);
    background: var(--gh-canvas); {GLASS} }}
.stApp [data-testid="stExpander"] summary:hover {{ color: var(--gh-accent); }}
.stApp [data-testid="stImage"] img {{ border-radius: 14px; border: 1px solid var(--gh-border);
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.45); }}
/* the wall refreshes every few seconds: keep it from dimming or showing "Running" while it does */
[data-stale="true"] {{ opacity: 1 !important; transition: none !important; }}
[data-testid="stStatusWidget"] {{ visibility: hidden; }}
@media (prefers-reduced-motion: reduce) {{
  .stApp *, .stApp *::before, .stApp *::after {{ animation: none !important; transition: none !important; }} }}
"""


# ---- The phone page's own look (review 26 Nov 2026): a light, warm concrete wall, friendly and easy ----------------
# Plain light concrete, drawn here (no photo): no joints or tie holes, they read as marks on the wall. White frosted
# cards, dark ink, one indigo accent. Never green (rule 1). The wall and the engineer pages keep the dark look.
# The phone page's own look (review 26 Nov 2026): a light, warm concrete wall, white frosted cards, dark ink. The
# details (buttons, ticks, labels, focus) in georoad.ai's light blue. Never green (rule 1).
ACCENT_LIGHT = "#0c9aab"  # light blue for details on light (georoad.ai's teal-blue)
ACCENT_DARK = "#22d3ee"
PHONE_TOKENS = {"fg": "#1f2430", "muted": "#5f6672", "canvas": "rgba(255, 255, 255, 0.86)", "solid": "#ffffff",
                "subtle": "rgba(255, 255, 255, 0.72)", "border": "rgba(31, 36, 48, 0.10)",
                "border-strong": "rgba(31, 36, 48, 0.18)", "accent": ACCENT_LIGHT, "info": "#2563eb",
                "attention": "#b45309", "severe": "#c2410c", "danger": "#d6334f", "done": "#7c3aed", "tab": ACCENT_LIGHT,
                "page": "#cbc6be"}


def _light_concrete(side: int = 480, seed: int = 11) -> str:
    """Light warm concrete: noise, grain and small air holes; no joints or tie holes (they read as marks)."""
    import base64
    import io

    import numpy as np
    from PIL import Image, ImageFilter
    rng = np.random.default_rng(seed)
    acc = np.zeros((side, side))
    for cells, weight in ((3, 1.0), (6, 0.6), (12, 0.35), (24, 0.2), (60, 0.12)):
        small = rng.random((cells, cells))
        big = np.asarray(Image.fromarray(np.tile(small, (3, 3)).astype(np.float32), "F")
                         .resize((3 * side, 3 * side), Image.BICUBIC))[side:2 * side, side:2 * side]
        acc += weight * big
    acc = (acc - acc.min()) / (acc.max() - acc.min())
    v = 178 + 44 * acc ** 1.2 + 4.0 * rng.normal(0, 1, (side, side))
    pores = rng.random((side, side)) < 0.0018  # small air holes of cast concrete
    v[pores] -= 38
    rgb = np.clip(np.stack([v + 3, v + 1, v - 4], axis=2), 0, 255).astype(np.uint8)  # a touch warm
    im = Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(0.4))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=74)
    return "url(data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode() + ")"


def _phone_css(t: dict, texture: str, surf: str, surf_hover: str, card_edge: str, shadow: str) -> str:
    a = t["accent"]
    on = "#04222a" if a == ACCENT_DARK else "#ffffff"  # text on the light blue: dark on the bright one
    tints = "".join(
        f".stApp .Label--{k} {{ color: {t[k]}; border-color: rgba({_hex_rgb(t[k])}, .45); background: rgba({_hex_rgb(t[k])}, .10); }} "
        f".stApp .State--{k} {{ color: {t[k]}; border-color: rgba({_hex_rgb(t[k])}, .5); background: rgba({_hex_rgb(t[k])}, .12); "
        f"box-shadow: none; }} " for k in LEVEL_KINDS + ("muted",) if k in t)
    return ("<style>:root { " + " ".join(f"--gh-{k}: {v};" for k, v in t.items()) + " }" + f"""
html, body {{ background: {t['page']}; }}
.stApp {{ color: {t['fg']} !important; background-color: {t['page']} !important;
          background-image: linear-gradient(180deg, rgba(255,255,255,.18), rgba(255,255,255,0) 40%), {texture} !important;
          background-size: 100% 100%, 480px 480px !important; background-repeat: no-repeat, repeat !important;
          background-attachment: fixed, fixed !important; }}
.stApp::before {{ display: none !important; }}
.stApp [data-testid="stHeader"] {{ background: transparent !important; }}
.stApp .block-container {{ padding-top: 28px; padding-bottom: 48px; max-width: 640px; }}
.stApp p, .stApp li, .stApp label, .stApp span, .stApp small, .stApp b, .stApp strong {{ color: inherit; }}
.stApp .gh-header {{ padding: 4px 0 6px; }}
.stApp .gh-header .logo, .stApp .gh-header .logo * {{ color: {t['fg']} !important; }}
.stApp .logo-oo .oo-frame {{ stroke: {a} !important; }}
.stApp .st-key-ph-intro, .stApp .st-key-ph-photo, .stApp .st-key-ph-options, .stApp .st-key-ph-result,
.stApp [class*="st-key-ask_"], .stApp [class*="st-key-ph-step2"] {{ background: {t['canvas']}; border: 1px solid {card_edge};
    border-radius: 22px; padding: 18px 18px 10px; box-shadow: 0 10px 30px {shadow}; backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px); margin: 0 0 4px; }}
.stApp .st-key-ph-photo {{ border: 2px solid rgba({_hex_rgb(a)}, .25); }}
.stApp .lead {{ font-size: 25px; line-height: 1.2; color: {t['fg']}; margin: 0 0 4px; }}
.stApp .lead-sub {{ font-size: 15px; color: {t['muted']}; margin: 0 0 14px; }}
.stApp .tips {{ gap: 10px; margin: 0 0 12px; }}
.stApp .tips > span {{ background: {surf}; border: 1px solid {t['border']}; color: {t['muted']}; border-radius: 16px;
                       box-shadow: 0 2px 8px rgba(40,34,26,.06); }}
.stApp .tips svg {{ color: {a}; }}
.stApp .tips b {{ color: {t['fg']}; display: block !important; margin-bottom: 2px; }}
.stApp .step-n {{ display: inline-flex; width: 22px; height: 22px; border-radius: 50%; background: {a}; color: {on} !important;
                  font-size: 12px; font-weight: 700; align-items: center; justify-content: center; margin-right: 6px; }}
.stApp .note {{ color: {t['muted']}; }}
.stApp .reason, .stApp .m-line, .stApp .lead-s {{ color: {t['fg']}; }}
.stApp .m-line.m-red, .stApp .m-red {{ color: {t['danger']}; }}
.stApp .m-head {{ color: {a}; }}
.stApp .m-grid {{ background: {t['border']}; border-color: {t['border']}; }}
.stApp .m-grid > div {{ background: {surf}; }}
/* the big photo button */
.stApp [data-testid="stFileUploader"] > label {{ font-size: 17px !important; font-weight: 700; color: {t['fg']} !important; }}
.stApp [data-testid="stFileUploaderDropzone"] {{ background: {surf} !important; border: 2px dashed rgba({_hex_rgb(a)}, .55) !important;
    border-radius: 20px !important; padding: 22px 16px !important; box-shadow: 0 8px 24px rgba({_hex_rgb(a)}, .12); }}
.stApp [data-testid="stFileUploaderDropzone"]::after {{ display: none !important; }}
.stApp [data-testid="stFileUploaderDropzone"]:hover {{ border-color: {a} !important; background: {surf_hover} !important; }}
.stApp [data-testid="stFileUploaderDropzone"] button {{ background: {a} !important; color: {on} !important; border: none !important;
    border-radius: 999px !important; font-weight: 700 !important; padding: 10px 22px !important; font-size: 16px !important; }}
.stApp [data-testid="stFileUploaderDropzone"] span, .stApp [data-testid="stFileUploaderDropzone"] small,
.stApp [data-testid="stFileUploaderDropzone"] svg {{ color: {t['muted']} !important; }}
.stApp [data-testid="stFileUploaderDropzone"] button, .stApp [data-testid="stFileUploaderDropzone"] button * {{ color: {on} !important; }}
.stApp [data-testid="stFileUploaderFile"], .stApp [data-testid="stFileUploaderFile"] * {{ color: {t['fg']} !important; }}
/* options: checkboxes and the earlier number */
.stApp [data-testid="stCheckbox"] label, .stApp [data-testid="stCheckbox"] p {{ color: {t['fg']} !important; font-size: 15px; }}
.stApp [data-testid="stCheckbox"] label > div:first-of-type {{ width: 22px !important; height: 22px !important;
    border: 2px solid {t['muted']} !important; background: {surf} !important; border-radius: 7px !important; }}
.stApp [data-testid="stCheckbox"] label[data-selected] > div:first-of-type {{ background: {a} !important; border-color: {a} !important; }}
.stApp [data-testid="stCheckbox"] label[data-selected] > div:first-of-type svg, .stApp [data-testid="stCheckbox"] label[data-selected] > div:first-of-type * {{ color: {on} !important; fill: {on} !important; }}
.stApp [data-testid="stTextInput"] label, .stApp [data-testid="stTextInput"] p {{ color: {t['fg']} !important; }}
.stApp [data-testid="stTextInputRootElement"], .stApp [data-testid="stTextInputField"],
.stApp [data-baseweb="input"], .stApp [data-baseweb="base-input"], .stApp [data-baseweb="input"] input {{ background: {surf} !important;
    color: {t['fg']} !important; border-color: {t['border-strong']} !important; border-radius: 12px !important; }}
.stApp [data-baseweb="input"] input::placeholder, .stApp [data-testid="stTextInputField"]::placeholder {{ color: #9aa1ab !important; }}
.stApp [data-testid="stTextInputRootElement"] {{ border: 1px solid {t['border-strong']} !important; min-height: 46px; }}
.stApp [data-testid="stTextInputRootElement"]:focus-within {{ border-color: {a} !important; box-shadow: 0 0 0 3px rgba({_hex_rgb(a)}, .2); }}
.stApp [data-testid="stTextInputField"] {{ font-size: 16px !important; }}
/* buttons, expanders, messages */
.stApp .stButton button {{ background: {surf} !important; color: {t['fg']} !important; border: 1px solid {t['border-strong']} !important;
    border-radius: 999px !important; font-weight: 600; min-height: 44px; box-shadow: 0 2px 6px rgba(40,34,26,.06); }}
.stApp .stButton button p {{ color: {t['fg']} !important; }}
.stApp .stButton button:hover {{ border-color: {a} !important; color: {a} !important; }}
.stApp .stButton button[kind="primary"], .stApp [data-testid="stBaseButton-primary"] {{ background: {a} !important;
    border-color: {a} !important; min-height: 52px; font-size: 17px; box-shadow: 0 8px 22px rgba({_hex_rgb(a)}, .35); }}
.stApp .stButton button[kind="primary"] p, .stApp [data-testid="stBaseButton-primary"] p {{ color: {on} !important; font-weight: 700; font-size: 17px; }}
.stApp .stButton button[kind="primary"]:hover {{ filter: brightness(1.08); color: {on} !important; }}
.stApp [data-testid="stExpander"] details {{ background: {t['canvas']} !important; border: 1px solid {card_edge} !important;
    border-radius: 18px !important; box-shadow: 0 6px 18px {shadow}; }}
.stApp [data-testid="stExpander"] summary, .stApp [data-testid="stExpander"] summary * {{ color: {t['fg']} !important; }}
.stApp [data-testid="stAlert"] * {{ color: {t['fg']} !important; }}
.stApp [data-testid="stImageCaption"], .stApp [data-testid="caption"] {{ color: {t['muted']} !important; }}
.stApp [data-testid="stSpinner"] * {{ color: {t['fg']} !important; }}
.stApp .sl-alert {{ color: #ffffff; }} .stApp .sl-alert * {{ color: inherit; }}
.stApp .sl-alert {{ background: #b91c1c; border-color: #7f1d1d; box-shadow: 0 8px 24px rgba(185,28,28,.3); }}
.stApp .sl-alert b {{ color: #ffffff; }}
.stApp .sl-alert.cause {{ background: #c2410c; border-color: #7c2d12; }}
.stApp .sl-alert.cause b {{ color: #ffffff; }}
.stApp iframe {{ box-shadow: 0 12px 30px rgba(40,34,26,.25); }}
/* a tapped step, large in the middle of the screen (the card's blur would hold it inside the card: off meanwhile) */
.stApp .st-key-ph-intro:has(.big) {{ backdrop-filter: none !important; -webkit-backdrop-filter: none !important; }}
.stApp .tips > span {{ cursor: zoom-in; transition: transform .2s ease, box-shadow .2s ease; }}
.stApp .tips > span:active {{ transform: scale(.98); }}
.stApp .tips > span.big {{ position: fixed !important; z-index: 1000001; left: 50%; top: 50%; width: min(440px, calc(100vw - 32px));
    transform: translate(-50%, -50%); display: block !important; padding: 28px 26px !important; cursor: zoom-out;
    font-size: 17px !important; line-height: 1.5 !important; background: {surf} !important; color: {t['fg']} !important;
    border: 2px solid {a} !important; box-shadow: 0 30px 80px rgba(0,0,0,.45) !important; animation: tip-pop .22s ease; }}
.stApp .tips > span.big svg {{ width: 100% !important; height: 170px !important; margin: 0 0 18px !important; flex: none !important; }}
.stApp .tips > span.big b {{ font-size: 24px !important; margin-bottom: 8px; display: block !important; }}
.stApp .tips > span.big .step-n {{ width: 30px; height: 30px; font-size: 15px; }}
@keyframes tip-pop {{ from {{ opacity: 0; transform: translate(-50%, -50%) scale(.85); }} to {{ opacity: 1; transform: translate(-50%, -50%) scale(1); }} }}
{tints}
</style>""")


PHONE_DARK_TOKENS = {"fg": "#eef1f5", "muted": "#9aa3af", "canvas": "rgba(24, 28, 34, 0.84)", "solid": "#181c22",
                     "subtle": "rgba(255, 255, 255, 0.05)", "border": "rgba(255, 255, 255, 0.10)",
                     "border-strong": "rgba(255, 255, 255, 0.20)", "accent": ACCENT_DARK, "info": "#6c8cff",
                     "attention": "#ffb224", "severe": "#ff8a3d", "danger": "#ff4d6a", "done": "#b38cff",
                     "tab": ACCENT_DARK, "page": "#16181b"}


def _scoped(css: str, scope: str) -> str:
    """The same rules, applied only under `scope` (light or dark mode on the <html> element)."""
    css = css.replace(":root {", scope + " {", 1).replace("html, body {", f"{scope}, {scope} body {{", 1)
    return css.replace(".stApp", scope + " .stApp")


def phone_css() -> str:
    """The phone page in light and in dark mode, as chosen in Streamlit's menu (Settings, top right) or, with
    "System", by the phone itself. mode_watcher sets data-sl-mode on <html>; light is the default."""
    light = _phone_css(PHONE_TOKENS, LIGHT_CONCRETE, "#ffffff", "#f3fbfc", "rgba(255,255,255,.75)", "rgba(40, 34, 26, .14)")
    dark = _phone_css(PHONE_DARK_TOKENS, CONCRETE, "#1f242c", "#22303a", "rgba(255,255,255,.08)", "rgba(0, 0, 0, .45)")
    strip = lambda c: c.replace("<style>", "").replace("</style>", "")  # noqa: E731
    return ("<style>" + _scoped(strip(light), 'html:not([data-sl-mode="dark"])')
            + _scoped(strip(dark), 'html[data-sl-mode="dark"]') + ".stApp .st-key-sl_mode { display: none; }"
            ".tip-back { position: fixed; inset: 0; z-index: 1000000; background: rgba(10, 12, 15, .45);"
            " backdrop-filter: blur(4px); -webkit-backdrop-filter: blur(4px); }</style>")


# Reads the mode chosen in Streamlit's own menu (kept by Streamlit in the browser as "Light", "Dark" or "System") and
# marks <html> with it, checked twice a second, so the page follows a switch at once.
MODE_JS = """
export default function () {
  const root = document.documentElement;
  // dark is the default: on the first visit Streamlit's own choice is set to Dark once; the menu changes it after
  try {
    if (!localStorage.getItem('sl-mode-default')) {
      localStorage.setItem('sl-mode-default', '1');
      const key = Object.keys(localStorage).find(function (k) { return k.indexOf('stActiveTheme') === 0; }) || 'stActiveTheme-/-v2';
      if (JSON.parse(localStorage.getItem(key) || 'null') !== 'Dark') {
        localStorage.setItem(key, JSON.stringify('Dark'));
        location.reload();
        return;
      }
    }
  } catch (e) {}
  function mode() {
    let v = null;
    try { for (const k of Object.keys(localStorage)) if (k.indexOf('stActiveTheme') === 0) v = JSON.parse(localStorage.getItem(k)); }
    catch (e) {}
    if (v === 'Dark') return 'dark';
    if (v === 'Light') return 'light';
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  function apply() { const m = mode(); if (root.dataset.slMode !== m) root.dataset.slMode = m; }
  apply();
  if (!window.__slModeTimer) window.__slModeTimer = setInterval(apply, 500);
  // a tapped step opens large; a second tap, a tap elsewhere or a scroll puts it back
  if (!window.__slTips) {
    window.__slTips = true;
    let back = null;
    const close = function () {
      document.querySelectorAll('.tips > span.big').forEach(function (e) { e.classList.remove('big'); });
      if (back) { back.remove(); back = null; }
    };
    document.addEventListener('click', function (e) {
      const tip = e.target.closest && e.target.closest('.tips > span');
      const open = document.querySelector('.tips > span.big');
      close();
      if (tip && tip !== open) {
        back = document.createElement('div'); back.className = 'tip-back'; document.body.appendChild(back);
        tip.classList.add('big');
      }
    }, true);
    document.addEventListener('scroll', function () { if (document.querySelector('.tips > span.big')) close(); }, true);
    window.addEventListener('wheel', function () { if (document.querySelector('.tips > span.big')) close(); }, {passive: true});
    window.addEventListener('touchmove', function () { if (document.querySelector('.tips > span.big')) close(); }, {passive: true});
  }
}
"""


LIGHT_CONCRETE = _light_concrete()


# ---- The engineer wall in the phone page's design (review 26 Nov 2026), in VS Code's own colours --------------------
# Dark Modern and Light Modern, VS Code's default themes: their backgrounds, text, borders and blue accent. The levels
# keep their own colours (Category 3 red, Expert needed violet, Category 2 amber, Category 1 blue), never green.
VSCODE_DARK = {"fg": "#cccccc", "muted": "#9d9d9d", "canvas": "rgba(31, 31, 31, 0.92)", "solid": "#1f1f1f",
               "subtle": "rgba(255, 255, 255, 0.04)", "border": "#2b2b2b", "border-strong": "#3c3c3c",
               "accent": "#8fd0ff", "info": "#86c5ff", "attention": "#e5a50a", "severe": "#d18616",
               "danger": "#f85149", "done": "#b180d7", "tab": "#8fd0ff", "page": "#181818"}
VSCODE_LIGHT = {"fg": "#3b3b3b", "muted": "#616161", "canvas": "rgba(255, 255, 255, 0.92)", "solid": "#ffffff",
                "subtle": "rgba(0, 0, 0, 0.03)", "border": "#e5e5e5", "border-strong": "#cecece",
                "accent": "#2f86c9", "info": "#4aa3f0", "attention": "#bf8803", "severe": "#c2410c",
                "danger": "#e51400", "done": "#652d90", "tab": "#2f86c9", "page": "#f8f8f8"}


def _wall_css(t: dict, texture: str, veil: str, button: str, shadow: str) -> str:
    a = t["accent"]
    tints = "".join(
        f".stApp .Label--{k} {{ color: {t[k]}; border-color: rgba({_hex_rgb(t[k])}, .45); background: rgba({_hex_rgb(t[k])}, .10); }} "
        f".stApp .State--{k} {{ color: {t[k]}; border-color: rgba({_hex_rgb(t[k])}, .5); background: rgba({_hex_rgb(t[k])}, .12); "
        f"box-shadow: none; }} " for k in LEVEL_KINDS + ("muted",) if k in t)
    card = (f"background: {t['canvas']} !important; border: 1px solid {t['border']} !important; border-radius: 14px !important; "
            f"box-shadow: {shadow} !important;")
    return (":root { " + " ".join(f"--gh-{k}: {v};" for k, v in t.items()) + " }" + f"""
html, body {{ background: {t['page']}; }}
.stApp {{ color: {t['fg']} !important; background-color: {t['page']} !important;
          background-image: linear-gradient({veil}, {veil}), {texture} !important;
          background-size: 100% 100%, 480px 480px !important; background-repeat: no-repeat, repeat !important;
          background-attachment: fixed, fixed !important; }}
.stApp::before {{ display: none !important; }}
.stApp [data-testid="stHeader"] {{ background: transparent !important; }}
.stApp p, .stApp li, .stApp span, .stApp b, .stApp strong {{ color: inherit; }}
.stApp .gh-header .logo, .stApp .gh-header .logo * {{ color: {t['fg']} !important; }}
.stApp .logo-oo .oo-frame {{ stroke: {a} !important; }}
.stApp .gh-right {{ color: {t['muted']} !important; }}
.stApp .gh-dot {{ background: {a} !important; box-shadow: 0 0 10px {a}; }}
.stApp .gh-h {{ color: {a} !important; }}
.stApp .gh-blank {{ {card} color: {t['muted']} !important; border-style: dashed !important; }}
.stApp .tab-desc {{ color: {t['muted']} !important; }}
.stApp .st-key-gh-tabs {{ border-bottom-color: {t['border']} !important; }}
.stApp .st-key-gh-tabs .stButton button p {{ color: {t['muted']} !important; }}
.stApp .st-key-gh-tabs .stButton button:hover p {{ color: {t['fg']} !important; }}
.stApp .st-key-gh-tabs .stButton button[kind="primary"], .stApp .st-key-gh-tabs [data-testid="stBaseButton-primary"] {{
    border-bottom-color: {a} !important; box-shadow: 0 10px 18px -12px {a} !important; }}
.stApp .st-key-gh-tabs .stButton button[kind="primary"] p {{ color: {t['fg']} !important; }}
.stApp .st-key-gh-tabs button code {{ color: {t['fg']} !important; background: {t['subtle']} !important; border: 1px solid {t['border']}; }}
.stApp .wall-photo {{ {card} color: {t['fg']} !important; }}
.stApp .wall-photo .meta, .stApp .sl-crack .meta {{ border-top-color: {t['border']} !important; color: {t['muted']} !important; }}
.stApp .wall-photo .num, .stApp .sl-crack .num {{ color: {t['fg']} !important; }}
.stApp [class*="st-key-card-"]:hover .wall-photo {{ border-color: {a} !important; box-shadow: 0 18px 40px rgba(0,0,0,.25), 0 0 0 1px {a} !important; }}
.stApp .st-key-sl-window {{ background: {t['solid']} !important; border: 1px solid {t['border-strong']} !important; border-radius: 16px !important; }}
.stApp .st-key-sl-overlay {{ background: {veil} !important; }}
.stApp .Box, .stApp .d-box, .stApp details.more {{ background: {t['canvas']} !important; border-color: {t['border']} !important; border-radius: 12px !important; }}
.stApp .Box-header, .stApp .d-box > p:first-child {{ border-bottom-color: {t['border']} !important; }}
.stApp details.more > summary {{ color: {t['fg']} !important; }}
.stApp details.more > summary::before {{ color: {a} !important; }}
.stApp .d-title, .stApp .d-val, .stApp .d-line {{ color: {t['fg']} !important; }}
.stApp .d-lab, .stApp .d-sub, .stApp .d-note, .stApp .d-num {{ color: {t['muted']} !important; }}
.stApp .d-red {{ color: {t['danger']} !important; }}
.stApp .d-grid, .stApp .d-grid > div, .stApp .d-cell {{ border-color: {t['border']} !important; }}
.stApp a, .stApp .d-back {{ color: {a} !important; }}
.stApp .stButton button {{ background: {t['solid']} !important; color: {t['fg']} !important; border: 1px solid {t['border-strong']} !important;
    border-radius: 999px !important; }}
.stApp .stButton button p {{ color: {t['fg']} !important; }}
.stApp .stButton button:hover {{ border-color: {a} !important; }}
.stApp .st-key-gh-tabs .stButton button {{ background: transparent !important; border: none !important; border-radius: 0 !important; }}
.stApp .stButton button[kind="primary"]:not(.st-key-gh-tabs *) {{ background: {button} !important; border-color: {button} !important; }}
.stApp .stButton button[kind="primary"]:not(.st-key-gh-tabs *) p {{ color: {"#0b2233" if button == "#8fd0ff" else "#ffffff"} !important; }}
.stApp .gh-footer {{ color: {t['muted']} !important; border-top-color: {t['border']} !important; }}
.stApp .sl-alert {{ color: #fff; background: {t['danger']}; border: 1px solid {t['danger']}; border-radius: 12px; }}
.stApp .sl-alert.cause {{ background: {t['severe']}; border-color: {t['severe']}; }}
.stApp iframe {{ border-radius: 14px !important; }}
.stApp [data-testid="stDeckGlJsonChart"], .stApp [data-testid="stDeckGlJsonChart"] > div {{ border-radius: 14px; overflow: hidden; }}
{tints}""")


def wall_css() -> str:
    """The engineer wall in light and dark mode (menu, top right), as the phone page; VS Code's colours."""
    light = _wall_css(VSCODE_LIGHT, LIGHT_CONCRETE, "rgba(248, 248, 248, .35)", "#2f86c9", "0 8px 24px rgba(0, 0, 0, .10)")
    dark = _wall_css(VSCODE_DARK, CONCRETE, "rgba(24, 24, 24, .25)", "#8fd0ff", "0 12px 32px rgba(0, 0, 0, .45)")
    return ("<style>" + _scoped(light, 'html:not([data-sl-mode="dark"])') + _scoped(dark, 'html[data-sl-mode="dark"]')
            + ".stApp .st-key-sl_mode { display: none; }</style>")
