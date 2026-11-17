"""The app's look since 12 Nov (D76): GitHub's design (Primer, light). White page, system font, thin grey borders,
6 px corners, tabs with counters, rounded labels. No background picture and nothing moves.
Spec: spec/features/class_demo.md (Look)."""

import html

FONT = '-apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif'
MONO = 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace'

# Primer's light colours (GitHub's own design tokens, named here by role)
TOKENS = {"fg": "#1f2328", "muted": "#59636e", "canvas": "#ffffff", "subtle": "#f6f8fa", "border": "#d1d9e0",
          "accent": "#0969da", "attention": "#9a6700", "severe": "#bc4c00", "danger": "#d1242f", "done": "#8250df",
          "tab": "#fd8c73"}

# One colour per answer. Never green: green reads as "safe", which the tool never says (rule 1).
TONES = {"Retake": "muted", "Expert needed": "done", "Category 1": "accent", "Category 2": "attention",
         "Category 3": "danger", "Crack found": "danger", "No crack found": "muted"}


def tone(text: str) -> str:
    return TONES.get(text, "muted")


def label(text: str, kind: str | None = None) -> str:
    """A small outlined label, as GitHub's issue labels and the "Public" tag of a repository."""
    return f'<span class="Label Label--{kind or tone(text)}">{html.escape(text)}</span>'


def state(text: str, kind: str | None = None) -> str:
    """A filled pill, as the Open / Closed state of an issue: the main answer of a page."""
    return f'<span class="State State--{kind or tone(text)}">{html.escape(text)}</span>'


def _vars() -> str:
    return ":root { " + " ".join(f"--gh-{k}: {v};" for k, v in TOKENS.items()) + " }"


THEME_CSS = _vars() + f"""
.stApp {{ background: var(--gh-canvas) !important; background-image: none !important; color: var(--gh-fg); }}
.stApp [data-testid="stMarkdownContainer"], .stApp button, .stApp label, .stApp input, .stApp textarea,
.stApp [data-testid="stCaptionContainer"] {{ font-family: {FONT} !important; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stAppDeployButton"] {{ display: none; }}
.stApp p, .stApp li, .stApp label {{ color: var(--gh-fg); font-size: 14px; line-height: 1.5; }}
.stApp a {{ color: var(--gh-accent); text-decoration: none; }}
.stApp a:hover {{ text-decoration: underline; }}
.stApp h1, .stApp h2, .stApp h3 {{ color: var(--gh-fg) !important; font-weight: 600 !important; text-transform: none;
                                  letter-spacing: 0; }}
.stApp code {{ font-family: {MONO}; }}
/* the page header: logo, version label, a note on the right; one grey line under it */
.gh-header {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; padding: 0 0 16px;
              border-bottom: 1px solid var(--gh-border); margin: 0 0 16px; }}
.gh-header {{ column-gap: 14px; }}
.gh-header .logo {{ font-size: 30px; font-weight: 800; letter-spacing: 0.06em; color: var(--gh-fg); white-space: nowrap;
                    line-height: 1.1; }}
.gh-header.gh-large .logo {{ font-size: 56px; }}
/* "HUMAN IN THE LOOP" beside the name: small, bold capitals (12 Nov) */
.gh-header .gh-tag {{ font-size: 11px; font-weight: 800; letter-spacing: 0.14em; text-transform: uppercase;
                      color: var(--gh-fg); white-space: nowrap; padding-left: 14px; line-height: 1.3;
                      align-self: stretch; display: flex; align-items: center;  /* the grey rule: as tall as the name */
                      border-left: 2px solid var(--gh-border); }}
.gh-header.gh-large .gh-tag {{ font-size: 13px; }}
.gh-header .gh-right {{ margin-left: auto; font-size: 12px; color: var(--gh-muted); }}
.gh-dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--gh-accent);
           margin-right: 6px; vertical-align: 0; }}
.stApp .gh-quote {{ margin: 0 0 16px; padding: 0 1em; color: var(--gh-muted); border-left: .25em solid var(--gh-border);
                    font-size: 14px; }}
.stApp .note {{ font-size: 12px; color: var(--gh-muted); }}
.stApp .reason {{ color: var(--gh-fg); }}
.stApp .statement {{ margin: 16px 0 4px; }}
/* labels and state pills */
.Label {{ display: inline-block; padding: 0 7px; font-size: 12px; font-weight: 500; line-height: 18px;
          white-space: nowrap; border: 1px solid; border-radius: 2em; background: var(--gh-canvas); }}
.State {{ display: inline-flex; align-items: center; padding: 5px 12px; font-size: 14px; font-weight: 500;
          line-height: 20px; border-radius: 2em; color: #ffffff; white-space: nowrap; }}
""" + "".join(f".Label--{k} {{ color: var(--gh-{k}); border-color: var(--gh-{k}); }} "
              f".State--{k} {{ background: var(--gh-{k}); }} "
              for k in ("accent", "attention", "severe", "danger", "done")) + f"""
.Label--muted {{ color: var(--gh-muted); border-color: var(--gh-border); }}
.State--muted {{ background: var(--gh-muted); }}
/* boxes: GitHub's bordered panels with a grey header row */
.Box {{ border: 1px solid var(--gh-border); border-radius: 6px; background: var(--gh-canvas); margin: 0 0 16px;
        overflow: hidden; }}
.Box-header {{ padding: 8px 16px; background: var(--gh-subtle); border-bottom: 1px solid var(--gh-border);
               font-size: 14px; font-weight: 600; color: var(--gh-fg); }}
.Box-body {{ padding: 16px; }}
.Box-body > p:last-child {{ margin-bottom: 0; }}
.Box img.d-photo {{ width: 100%; display: block; }}
/* Streamlit's own widgets, in GitHub's shape */
.stApp .stButton button, .stApp .stDownloadButton button {{ border-radius: 6px; border: 1px solid var(--gh-border);
    background: var(--gh-subtle); color: var(--gh-fg); font-weight: 500; box-shadow: 0 1px 0 rgba(31,35,40,.04); }}
.stApp .stButton button:hover, .stApp .stDownloadButton button:hover {{ background: #eff2f5; border-color: var(--gh-border);
    color: var(--gh-fg); }}
.stApp [data-testid="stFileUploaderDropzone"] {{ border-radius: 6px; border: 1px dashed var(--gh-border);
    background: var(--gh-subtle); }}
.stApp [data-testid="stExpander"] details {{ border-radius: 6px; border-color: var(--gh-border); }}
.stApp [data-testid="stImage"] img {{ border-radius: 6px; border: 1px solid var(--gh-border); }}
/* the wall refreshes every few seconds: keep it from dimming or showing "Running" while it does */
[data-stale="true"] {{ opacity: 1 !important; transition: none !important; }}
[data-testid="stStatusWidget"] {{ visibility: hidden; }}
"""
