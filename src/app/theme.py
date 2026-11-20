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
