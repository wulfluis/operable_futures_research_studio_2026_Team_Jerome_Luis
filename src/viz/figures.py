"""Figures for the 29 Oct deliverables. Spec: spec/features/figures.md.

Run from the repository root:  py src/viz/figures.py   (writes PNGs to figures/)
Every figure is drawn from the repo's own rules and data map. None shows a measured result.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from triage.matrix import (_ESCALATOR_EFFECT, _LOCATION_RULE, BAND_NAMES, BANDS, BASIS, ELEMENT_RAISE, ELEMENTS,
                           ESCALATORS, MATRIX, PATTERN_NAMES, PATTERNS, VERSION)

# Palette: black, white and greys only (the user's house style); differences also carried by labels, line styles
# or hatching, never by grey alone.
SURFACE, INK, INK_2, MUTED, GRID = "#ffffff", "#111111", "#4d4d4d", "#8a8a8a", "#e0e0e0"
BLUE, AQUA, VIOLET = "#333333", "#8c8c8c", "#5c5c5c"
SEQ = {"covered": "#262626", "partial": "#7a7a7a", "planned": "#cfcfcf", "none": SURFACE}
# Status colours are reserved for triage levels, always with a text label.
LEVEL = {"Category 1": "#e6e6e6", "Category 2": "#a6a6a6", "Category 3": "#111111", "Expert needed": "#595959", "Retake": MUTED}
ON_FILL = {"Category 1": INK, "Category 2": INK}
SLIDE_PX = 4800  # every figure is 16:9 and saved 4800 px wide (4800 x 2700), to drop straight into a slide  # dark text on the light fills, white elsewhere

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "text.color": INK,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def _house_title(fig, title, subtitle=None):
    """The title every figure carries: bold capitals, 38 pt on a 24 in wide figure (no number: added in the deck)."""
    H, W = fig.get_figheight(), fig.get_figwidth()
    fig.text(0.02, 1 - 0.5 / H, title, fontsize=38 * W / 24, weight="bold", color=INK, va="center")
    if subtitle:
        fig.text(0.02, 1 - 1.0 / H, subtitle, fontsize=18 * W / 24, weight="bold", color=INK_2, va="center")


def _house_legend(fig, handles, labels):
    """The legend every figure carries: one line, top right, level with the title, same size on every slide."""
    H, W = fig.get_figheight(), fig.get_figwidth()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(0.985, 1 - 0.5 / H), ncol=len(handles),
               frameon=False, prop={"size": 13 * W / 24, "weight": "bold"}, handlelength=1.8, handleheight=1.0,
               handletextpad=0.6, columnspacing=1.8, borderaxespad=0)


def _flow_key(fig):
    """Colour key of the flow figures: white = people and photos, grey = written rules, black = learned models."""
    _house_legend(fig, [Patch(facecolor="white", edgecolor=FLOW_INK, lw=1.6),
                        Patch(facecolor=FLOW_RULE, edgecolor=FLOW_RULE),
                        Patch(facecolor=FLOW_MODEL, edgecolor=FLOW_MODEL)],
                  ["PEOPLE AND PHOTOS", "WRITTEN RULES", "LEARNED MODELS"])


def _box(ax, x, y, w, h, text, fill, edge, size=9.5, color=INK, weight="normal", lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                facecolor=fill, edgecolor=edge, linewidth=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=color,
            weight=weight, wrap=True)


def _arrow(ax, start, end, label=None, label_offset=(0, 0.12)):
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=1.2,
                                                          shrinkA=0, shrinkB=0))
    if label:
        mx, my = (start[0] + end[0]) / 2 + label_offset[0], (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=9, color=INK_2)


def _canvas(w, h, title, subtitle=None):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_axis_off()
    fig.suptitle(title, x=0.02, ha="left", fontsize=14, weight="bold")
    if subtitle:
        fig.text(0.02, 0.905, subtitle, ha="left", fontsize=10, color=INK_2)
    return fig, ax


# Pipeline v3 (spec/features/pipeline.md). Boxes: (row, column) -> (kind, title, body, status).
# Rows follow PHI-Net task 1 scene levels: 0 = S structural, 1 = O object, 2 = P pixel [55].
PIPE_LANES = [("S", "STRUCTURAL", "from far:\nthe building"), ("O", "OBJECT", "mid-range:\none element"),
              ("P", "PIXEL", "close-up:\nthe crack")]
PIPE_PHASES = [("1  Capture", "photos and answers"), ("2  Check", "written rules"),
               ("3  Perceive", "learned models, each with a confidence"), ("4  Decide", "written rules, auditable"),
               ("5  Act", "people")]
PIPE_BOXES = {
    (0, 0): ("human", "Building case (once)", "questionnaire: age, storeys, added floors,\ntypology + changes, open ground floor\n+ facade photo + map data (MCP)", "spec"),
    (0, 1): ("rule", "Completeness check", "every answer known or 'unknown'\nunknown counts as a risk flag\nand stays visible as unknown", "spec"),
    (0, 2): ("learned", "Building reading", "now: the questionnaire answers\nlater: model on the facade photo\n(storeys, typology; PHI-Net task 5)", "spec"),
    (0, 3): ("rule", "Building risk score", "worst crack level → P1 · P2 · P3\n+1 tier if ≥ 2 risk flags\n+1 tier if Category 2+ in ≥ 2 elements", "spec"),
    (0, 4): ("human", "Engineer", "ranked list of buildings + dossier\nconfirms or corrects every Category 3\nand every referral; rates on site", "spec"),
    (1, 0): ("human", "Element photo", "the whole element\nwith the scale card\nno question to the resident", "spec"),
    (1, 1): ("rule", "Scale and angle", "scale card visible?\nphone parallel to the wall?\nno → retake", "spec"),
    (1, 2): ("learned", "Material + element", "material model: concrete or not\nelement model: column, beam, slab, wall\n(PHI-Net task 6) · each + confidence", "spec"),
    (1, 3): ("rule", "Element rules", "not concrete or unsure → Expert\npattern does not fit element → Expert\ncolumn → +1 level", "code"),
    (2, 0): ("human", "Crack close-up", "scale card + 2 marker stickers\nburst of photos, flash on and off\nresident or field worker", "spec"),
    (2, 1): ("rule", "Photo checks", "sharp and lit? ≥ 3 px across the crack?\nno → retake closer; still no →\nwidth band read on the card", "partly"),
    (2, 2): ("learned", "Crack reading", "crack pixels → width mm ± 1 px\npattern model (PHI-Net tasks 3, 8)\nfield worker until confident", "spec"),
    (2, 3): ("rule", "Crack level + growth", "matrix v3 → level, reason, next step\ngrowth counts only if larger\nthan the measurement uncertainty", "partly"),
    (2, 4): ("human", "Resident", "level, reason, next step,\nrecheck date\nnever 'safe', never a clearance", "partly"),
}
PIPE_STATUS = {"code": ("in code", "#262626", "#262626", "white"),
               "partly": ("partly in code", "#e6e6e6", "#262626", "#262626"),
               "spec": ("specified", "#FFFFFF", MUTED, INK_2)}


def pipeline():
    """Pipeline v3: three scales (S, O, P) x five phases, scene check, monitoring and learning loops, status."""
    kinds = {"learned": (VIOLET, "#ececec"), "rule": (AQUA, "#f2f2f2"), "human": (MUTED, "#efefef")}
    x0, w, y0, h = [0, 4.1, 9.35, 13.45, 17.55], 3.5, [7.0, 4.25, 1.5], 2.05
    rx, rw = 8.0, 0.95  # scene check
    fig, ax = plt.subplots(figsize=(21, 12.6))
    ax.set_axis_off()
    fig.suptitle("Second Look pipeline v3: three scales, a building risk score and monitoring", x=0.02, ha="left",
                 fontsize=17, weight="bold")
    fig.text(0.02, 0.905, "Scales as in PHI-Net task 1: S structural (the building), O object (the element), P pixel "
             "(the crack) [53][55]. Learned models report a confidence; written rules decide; any doubt goes to an "
             "engineer.\nResidents are never asked the material, the element or the crack pattern: models identify "
             "them. Status on 27 Oct 2026. Source: spec/features/pipeline.md", fontsize=11, color=INK_2,
             linespacing=1.5)

    def arrow(p0, p1, colour=INK_2, ls="-", lw=1.5):
        ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>", color=colour, lw=lw, ls=ls,
                                                          shrinkA=0, shrinkB=0, mutation_scale=15))

    for c, (head, sub) in enumerate(PIPE_PHASES):
        ax.text(x0[c] + w / 2, 9.85, head, ha="center", fontsize=13, weight="bold", color=INK)
        ax.text(x0[c] + w / 2, 9.5, sub, ha="center", fontsize=9.5, color=INK_2)
    for r, (code, name, what) in enumerate(PIPE_LANES):
        y = y0[r]
        ax.add_patch(FancyBboxPatch((-2.25, y), 1.75, h, boxstyle="round,pad=0,rounding_size=0.1",
                                    facecolor=TREE_NAVY, edgecolor="none"))
        ax.text(-1.375, y + h - 0.5, code, ha="center", va="center", color=TREE_YELLOW, fontsize=26, weight="bold")
        ax.text(-1.375, y + h / 2 - 0.15, name, ha="center", va="center", color="white", fontsize=10, weight="bold")
        ax.text(-1.375, y + 0.42, what, ha="center", va="center", color="#e6e6e6", fontsize=9, linespacing=1.2)
    for (r, c), (kind, title, body, status) in PIPE_BOXES.items():
        edge, fill = kinds[kind]
        x, y = x0[c], y0[r]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.1", facecolor=fill,
                                    edgecolor=edge, lw=1.6))
        ax.text(x + w / 2, y + h - 0.42, title, ha="center", va="center", fontsize=11.3, weight="bold", color=INK)
        ax.text(x + w / 2, y + 0.8, body, ha="center", va="center", fontsize=9.3, color=INK_2, linespacing=1.25)
        label, chip_fill, chip_edge, chip_ink = PIPE_STATUS[status]
        cw = 0.2 + 0.085 * len(label)
        ax.add_patch(FancyBboxPatch((x + w - 0.15 - cw, y + h - 0.15), cw, 0.3,
                                    boxstyle="round,pad=0,rounding_size=0.15", facecolor=chip_fill,
                                    edgecolor=chip_edge, lw=1.1, zorder=4))
        ax.text(x + w - 0.15 - cw / 2, y + h, label, ha="center", va="center", fontsize=8, color=chip_ink,
                weight="bold", zorder=5)
        if c == 1:  # Check -> scene check -> Perceive
            arrow((x + w, y + h / 2), (rx, y + h / 2))
            arrow((rx + rw, y + h / 2), (x0[2], y + h / 2))
        elif c < 3 and (r, c + 1) in PIPE_BOXES:
            arrow((x + w, y + h / 2), (x0[c + 1], y + h / 2))

    # scene check: PHI-Net task 1
    ax.add_patch(FancyBboxPatch((rx, y0[2]), rw, y0[0] + h - y0[2], boxstyle="round,pad=0,rounding_size=0.1",
                                facecolor=kinds["learned"][1], edgecolor=kinds["learned"][0], lw=1.6))
    ax.text(rx + rw / 2, (y0[2] + y0[0] + h) / 2, "SCENE CHECK (PHI-Net task 1):  is this photo S, O or P?\n"
            "now: the app's guided capture sets it  ·  later: a scene model", rotation=90, ha="center",
            va="center", fontsize=9.3, weight="bold", color=INK, linespacing=1.6)

    # between the rows: material and element down to the crack level; crack results up to the building score
    arrow((x0[3] + w / 2, y0[1]), (x0[3] + w / 2, y0[2] + h))
    ax.text(x0[3] + w / 2 + 0.12, y0[2] + h + 0.35, "material, element", fontsize=9.3, color=INK_2, va="center")
    gx = x0[3] + w + 0.3
    ax.plot([x0[3] + w, gx, gx], [y0[2] + 0.55, y0[2] + 0.55, y0[0] + 0.55], color=INK_2, lw=1.5)
    arrow((gx, y0[0] + 0.55), (x0[3] + w, y0[0] + 0.55))
    ax.text(gx + 0.08, y0[1] + 0.3, "crack levels,\ngrowth, damage\nspread across\nelements", fontsize=9.3,
            color=INK_2, va="center", linespacing=1.2)
    arrow((x0[3] + w, y0[0] + 1.45), (x0[4], y0[0] + 1.45))
    arrow((x0[3] + w, y0[2] + 1.45), (x0[4], y0[2] + 1.45))

    # learning loop: engineer corrections retrain the learned models of phase 3
    ly, lx = 10.6, x0[4] + 0.45
    ax.plot([lx, lx], [y0[0] + h + 0.15, ly], color=VIOLET, lw=1.8, ls="--")
    ax.plot([lx, x0[2] + w / 2], [ly, ly], color=VIOLET, lw=1.8, ls="--")
    arrow((x0[2] + w / 2, ly), (x0[2] + w / 2, 10.2), colour=VIOLET, ls="--", lw=1.8)
    ax.text((x0[2] + w / 2 + lx) / 2, ly + 0.42, "LEARNING LOOP: engineer corrections and on-site ratings become "
            "labels and retrain the learned models (phase 3)", ha="center", fontsize=9.8, color=VIOLET, weight="bold")
    ax.text((x0[2] + w / 2 + lx) / 2, ly + 0.14, "each new version is checked on held-out photos before use; "
            "written rules change only by a reviewed decision", ha="center", fontsize=9.3, color=VIOLET)

    # monitoring loop: the same crack again on its recheck date
    my = 0.85
    ax.plot([x0[4] + w / 2, x0[4] + w / 2], [y0[2], my], color=BLUE, lw=1.8, ls="--")
    ax.plot([x0[4] + w / 2, x0[0] + w / 2], [my, my], color=BLUE, lw=1.8, ls="--")
    arrow((x0[0] + w / 2, my), (x0[0] + w / 2, y0[2]), colour=BLUE, ls="--", lw=1.8)
    ax.text((x0[0] + x0[4]) / 2 + w / 2, my - 0.3, "MONITORING LOOP: on the recheck date (Category 1: 6 months, Category 2: "
            "1 month) the same crack ID and stickers are photographed again, with the last photo as an overlay",
            ha="center", fontsize=9.8, color=BLUE, weight="bold")
    ax.text((x0[0] + x0[4]) / 2 + w / 2, my - 0.58, "growth larger than the measurement uncertainty raises the "
            "crack level, and through it the building risk score", ha="center", fontsize=9.3, color=BLUE)

    handles = []
    for (klabel, kkey), skey in zip([("Learned model (reports a confidence)", "learned"),
                                     ("Written rule, auditable", "rule"), ("People and their inputs", "human")],
                                    ["code", "partly", "spec"]):
        e, f = kinds[kkey]
        handles.append(Patch(facecolor=f, edgecolor=e, label=klabel))
        slabel, sf, se, _ = PIPE_STATUS[skey]
        handles.append(Patch(facecolor=sf, edgecolor=se, label=f"Status: {slabel}" + (" (designed, not coded yet)" if skey == "spec" else "")))
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False,
              fontsize=10.5)
    ax.set_xlim(-2.4, x0[4] + w + 0.2)
    ax.set_ylim(-0.1, 11.3)
    fig.subplots_adjust(left=0.02, right=0.99, top=0.85, bottom=0.08)
    return fig


# Decisions and players: five players as lifelines, time going down, numbered hand-offs between them.
# Decisions are diamonds on the lifeline of the player who decides (written rules, engineer); models only measure.
# Sources: src/triage/dispatch.py, src/triage/matrix.py, spec/features/building_score.md, spec/features/pipeline.md.
PLAYERS = {"R": 15.0, "M": 43.0, "W": 71.0, "F": 99.0, "E": 127.0}  # lifeline x
PLAYER_ICON = {"RESIDENT": "person", "APP · MODELS": "network", "APP · WRITTEN RULES": "rulebook",
               "FIELD WORKER": "worker", "ENGINEER": "hardhat"}
PLAYER_HEADS = [("R", "photo", "RESIDENT", "any phone"), ("M", "model", "APP · MODELS", "measure + confidence"),
                ("W", "rule", "APP · WRITTEN RULES", "decide, auditable"),
                ("F", "photo", "FIELD WORKER", "reliable phone + scale card"), ("E", "term", "ENGINEER", "final word")]
# (y, sender, receiver, symbol, word, meaning, dashed); time goes down. The figure shows the symbol and the word;
# the meaning is the key (spec/features/figures.md).
HANDOFFS = [
    (73.4, "R", "M", "camera", "Report", "photo of the crack + answers", False),
    (69.9, "M", "W", "ruler", "Measure", "crack found? rough width · confidence", False),
    (60.2, "W", "R", "clock", "Wait", "Retake, or V3: not assessed yet, recheck in 1 month", False),
    (56.8, "W", "F", "house", "Visit", "V1: visit within 1 week · V2: within 1 month", False),
    (53.4, "W", "E", "bell", "Alert", "V1 only: the engineer is told the same day", False),
    (48.8, "F", "R", "clipboard", "Survey", "asks the building questions, places stickers + crack ID", False),
    (45.4, "F", "M", "camera", "Photos", "reliable photos: the building, every cracked element, every crack", False),
    (40.8, "M", "W", "bars", "Readings", "material, element, width, pattern + confidence", False),
    (37.4, "F", "W", "pencil", "Pattern", "pattern picked from drawings while the model is unsure", False),
    (28.6, "W", "R", "document", "Result", "Category 1 or 2: level, reason, recheck date", False),
    (25.2, "W", "E", "warning", "Escalate", "Category 3, Expert needed (any doubt), buildings P1 · P2", False),
    (16.4, "E", "R", "check", "Confirmed", "confirmed level, reason, next step · never 'safe'", False),
    (13.0, "E", "M", "layers", "Retrain", "corrections become labels → retraining", True),
    (8.4, "R", "M", "calendar", "Recheck", "recheck photo of the same crack ID → back to step 8", True),
]


def _handoff_icon(ax, key, cx, cy, s=2.8, ink="#111111", paper="white"):
    """Small black-and-white symbol for a player or a hand-off, in an s x s box centred on (cx, cy).
    ink and paper swap for a symbol drawn on a black box (the models)."""
    K, W, u = ink, paper, s / 10.0

    def P(x, y):
        return cx - s / 2 + x * u, cy - s / 2 + y * u

    def rect(x, y, w, h, fc=None, lw=1.3):
        ax.add_patch(matplotlib.patches.Rectangle(P(x, y), w * u, h * u, facecolor=fc or W, edgecolor=K, lw=lw,
                                                  zorder=7))

    def poly(pts, fc=None, lw=1.3):
        ax.add_patch(matplotlib.patches.Polygon([P(*q) for q in pts], closed=True, facecolor=fc or W, edgecolor=K,
                                                lw=lw, zorder=7))

    def ln(pts, lw=1.4):
        xs, ys = zip(*[P(*q) for q in pts])
        ax.plot(xs, ys, color=K, lw=lw, zorder=8, solid_capstyle="round")

    def circ(x, y, r, fc=None, lw=1.3):
        ax.add_patch(plt.Circle(P(x, y), r * u, facecolor=fc or W, edgecolor=K, lw=lw, zorder=8))

    def shoulders(y_top, half):
        theta = [i / 20 * 3.14159 for i in range(21)]
        import math
        poly([(5 + half * math.cos(t), 0.4 + (y_top - 0.4) * math.sin(t)) for t in theta], fc=W)

    if key == "camera":
        rect(0.5, 1.5, 9, 6.5)
        rect(3.3, 8.0, 3.4, 1.4, fc=K)
        circ(5, 4.7, 2.0)
    elif key == "ruler":
        rect(0, 3, 10, 4)
        for i, x in enumerate((1.5, 3, 4.5, 6, 7.5, 9)):
            ln([(x, 7), (x, 5.6 if i % 2 else 4.6)], lw=1.1)
    elif key == "clock":
        circ(5, 5, 4.6)
        ln([(5, 5), (5, 8.2)])
        ln([(5, 5), (7.4, 5)])
    elif key == "house":
        poly([(1.2, 0.5), (8.8, 0.5), (8.8, 5.6), (5, 9.4), (1.2, 5.6)])
        rect(4, 0.5, 2, 3.2, fc=K)
    elif key == "bell":
        poly([(2, 2.6), (8, 2.6), (7.2, 4), (7.2, 6.8), (5, 8.8), (2.8, 6.8), (2.8, 4)])
        circ(5, 1.3, 1.0, fc=K)
    elif key == "clipboard":
        rect(1.5, 0.3, 7, 9)
        rect(3.5, 8.4, 3, 1.4, fc=K)
        for y in (6.4, 4.4, 2.4):
            ln([(3, y), (7, y)], lw=1.1)
    elif key == "bars":
        for x, h in ((1, 4), (4, 8), (7, 6)):
            rect(x, 0.5, 2.2, h, fc=K)
    elif key == "pencil":
        poly([(1.2, 1.2), (2.6, 4.0), (7.6, 9.0), (9.0, 7.6), (4.0, 2.6)], fc="white")
        poly([(1.2, 1.2), (2.6, 4.0), (4.0, 2.6)], fc=K)
    elif key == "document":
        poly([(1.8, 0.3), (8.2, 0.3), (8.2, 7.2), (6.0, 9.6), (1.8, 9.6)])
        for y in (6.6, 4.6, 2.6):
            ln([(3.2, y), (6.8, y)], lw=1.1)
    elif key == "warning":
        poly([(5, 9.6), (9.8, 0.6), (0.2, 0.6)])
        ln([(5, 6.6), (5, 3.6)], lw=1.8)
        circ(5, 2.2, 0.5, fc=K, lw=0.5)
    elif key == "check":
        circ(5, 5, 4.6)
        ln([(2.8, 5.0), (4.4, 3.2), (7.4, 7.0)], lw=1.8)
    elif key == "layers":
        for k in (2, 1, 0):
            rect(1 + k * 0.8, 1 + k * 2.2, 7.2, 3.4, fc=K if k == 0 else "#7a7a7a", lw=1.0)
    elif key == "network":  # the learned models: a small neural network
        nodes = [[(1.2, 2.6), (1.2, 7.4)], [(5, 1.2), (5, 5), (5, 8.8)], [(8.8, 5)]]
        for left, right in zip(nodes, nodes[1:]):
            for a in left:
                for b in right:
                    ln([a, b], lw=1.0)
        for layer in nodes:
            for x, y in layer:
                circ(x, y, 1.1, fc=K)
    elif key == "column":  # a structural element
        rect(2.2, 8.6, 5.6, 1.2, fc=K)
        rect(3.5, 1.2, 3.0, 7.4)
        rect(2.2, 0.2, 5.6, 1.0, fc=K)
    elif key == "crack":
        ln([(5.2, 9.8), (4.0, 7.6), (5.9, 5.2), (4.1, 2.8), (5.3, 0.2)], lw=2.2)
    elif key == "person":  # the resident
        circ(5, 7.4, 2.0)
        shoulders(5.0, 4.2)
    elif key == "worker":  # the field worker: a cap and a clipboard
        circ(4.2, 7.2, 1.8)
        rect(2.0, 8.4, 4.4, 0.9, fc=K)
        shoulders(4.8, 3.6)
        rect(6.6, 0.6, 3.0, 4.2)
        ln([(7.2, 3.6), (9.0, 3.6)], lw=1.0)
        ln([(7.2, 2.2), (9.0, 2.2)], lw=1.0)
    elif key == "hardhat":  # the engineer
        import math
        poly([(5 + 4.0 * math.cos(i / 20 * math.pi), 3.6 + 4.0 * math.sin(i / 20 * math.pi)) for i in range(21)], fc=K)
        rect(0.4, 2.6, 9.2, 1.2, fc=K)
        rect(4.4, 5.8, 1.2, 1.9, fc=W, lw=0.6)
    elif key == "rulebook":  # the written rules: a rule sheet with a tick
        poly([(1.8, 0.3), (8.2, 0.3), (8.2, 9.6), (1.8, 9.6)])
        for y in (7.6, 5.8):
            ln([(3.2, y), (6.8, y)], lw=1.1)
        ln([(3.2, 3.0), (4.4, 1.8), (6.8, 4.4)], lw=1.8)
    elif key == "calendar":
        rect(0.8, 0.5, 8.4, 8.0)
        rect(0.8, 6.6, 8.4, 1.9, fc=K)
        for x, y in ((2.8, 4.6), (5, 4.6), (7.2, 4.6), (2.8, 2.4), (5, 2.4)):
            circ(x, y, 0.55, fc=K, lw=0.5)


PHASES = [(76.4, 67.9, "REPORT"), (67.9, 51.4, "DISPATCH"), (51.4, 43.4, "FIELD VISIT"), (43.4, 23.2, "TRIAGE"),
          (23.2, 11.0, "REVIEW"), (11.0, 6.4, "FOLLOW-UP")]
WHO_DECIDES = [
    ("photo", "RESIDENT", "Decides to report a crack and answers\nwhat they know (age, floors added,\nnew? growing? doors stick?).",
     "Never asked the material, the element\nor the pattern. Never told 'safe'."),
    ("model", "APP · MODELS", "Decides nothing. Measures the crack,\nwidth, material, element, pattern,\nand says how sure it is.",
     "Unsure → the case goes up to\na person, never down to Category 1."),
    ("rule", "APP · WRITTEN RULES", "Decides: retake, visit priority\n(V1 to V3), the level, building\npriority (P1 to P3).",
     "Auditable and versioned; changed\nonly by a reviewed decision."),
    ("photo", "FIELD WORKER", "Decides nothing on risk. Takes the\nreliable photos, places stickers,\npicks the pattern when unsure.",
     "Visits only when the dispatch\nrule sends them (V1, V2, audit)."),
    ("term", "ENGINEER", "Has the final word: confirms or\ncorrects every Category 3 and every\ndoubt; visits P1 and P2 buildings.",
     "Every correction becomes a label\nthat retrains the models."),
]


def decisions_and_players():
    """Who decides what, and when: players as lifelines, hand-offs as numbered arrows, decisions as diamonds."""
    W = 19.2 * 16 / 9  # 16:9 like every figure, drawn across the full width (171 units of 0.2 in)
    fig = plt.figure(figsize=(W, 19.2), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W / 0.2)
    ax.set_ylim(0, 96)
    ax.set_axis_off()
    t = _flow_tools(ax)
    label, photo_box, rule_box, model_box, diamond, terminal, line = (
        t.label, t.photo_box, t.rule_box, t.model_box, t.diamond, t.terminal, t.line)
    X = PLAYERS

    def white_diamond(cx, cy, hw, hh, text):  # a decision taken by a person
        ax.add_patch(matplotlib.patches.Polygon([(cx - hw, cy), (cx, cy + hh), (cx + hw, cy), (cx, cy - hh)],
                                                closed=True, facecolor="white", edgecolor=FLOW_INK, lw=2.0, zorder=4))
        label(cx, cy, text, size=13, weight="bold")

    def head(kind, cx, cy, w, h, name, sub="", size=15):
        if kind == "photo":
            photo_box(cx, cy, w, h, "")
        elif kind == "rule":
            rule_box(cx, cy, w, h, "")
        elif kind == "model":
            model_box(cx, cy, w, h, "")
        elif kind == "model_flat":
            ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.5",
                                        facecolor=FLOW_MODEL, edgecolor="none", zorder=4))
        else:
            terminal(cx, cy, w, h, "")
        ink = "white" if kind.startswith("model") else FLOW_INK
        paper = FLOW_MODEL if kind.startswith("model") else ("#d2d2d2" if kind == "rule" else "white")
        icon_s = min(h * 0.72, 4.6)
        _handoff_icon(ax, PLAYER_ICON[name], cx - w / 2 + 1.0 + icon_s / 2, cy, s=icon_s, ink=ink, paper=paper)
        tx = cx + (icon_s + 1.0) / 2
        if sub:
            label(tx, cy + 1.0, name, size=size, colour=ink, weight="bold")
            label(tx, cy - 1.3, sub, size=13, colour=ink)
        else:
            label(tx, cy, name, size=size, colour=ink, weight="bold")

    # title, heading, colour key
    label(2.0, 93.5, "DECISIONS AND PLAYERS", size=38 * W / 24, weight="bold", ha="left")
    label(2.0, 87.0, "Who decides what, and what passes between players", size=18 * W / 24, colour=FLOW_TEXT, weight="bold",
          ha="left")
    _flow_key(fig)

    # players and lifelines
    for key, kind, name, sub in PLAYER_HEADS:
        head(kind, X[key], 80.6, 24.0, 6.2, name, sub)
        if key == "M":  # the models: a solid black lifeline, so they stand out
            ax.plot([X[key], X[key]], [6.4, 77.8], color=FLOW_MODEL, lw=3.2, zorder=1)
        else:
            ax.plot([X[key], X[key]], [6.4, 77.8], color="#9a9a9a", lw=1.6, ls=(0, (6, 4)), zorder=1)
    # phases (time goes down)
    for top, bottom, name in PHASES:
        ax.plot([1.0, 141.0], [bottom, bottom], color="#e0e0e0", lw=1.2, zorder=0)
        label(3.0, (top + bottom) / 2, name, size=13, weight="bold", colour=FLOW_SOFT, rotation=90)
    # models at work: black bars on the models lifeline
    for y0, y1 in ((73.4, 69.9), (45.4, 40.8)):
        ax.add_patch(FancyBboxPatch((X["M"] - 1.5, y1 - 0.4), 3.0, y0 - y1 + 0.8, boxstyle="round,pad=0,rounding_size=0.5",
                                    facecolor=FLOW_MODEL, edgecolor="white", lw=1.5, zorder=3))
        _handoff_icon(ax, "network", X["M"], (y0 + y1) / 2, s=2.6, ink="white", paper=FLOW_MODEL)
    # hand-offs
    for n, (y, a, b, icon, word, _meaning, dashed) in enumerate(HANDOFFS, start=1):
        x0, x1 = X[a], X[b]
        sign = 1 if x1 > x0 else -1
        gap0, gap1 = (1.8 if a == "M" else 0.7), (1.8 if b == "M" else 0.7)  # clear of the models' work boxes
        line([(x0 + sign * gap0, y), (x1 - sign * gap1, y)], ls="--" if dashed else "-", lw=2.2)
        # step number, symbol and one word, on a white tab above the arrow
        width = 2.4 + 0.8 + 3.3 + 0.8 + 0.8 * len(word)
        mid = (X["F"] + X["E"]) / 2 if word == "Alert" else (x0 + x1) / 2  # clear of arrow 4's head
        left, ym = mid - width / 2, y + 1.75
        ax.add_patch(FancyBboxPatch((left - 0.5, ym - 1.6), width + 1.0, 3.2, boxstyle="round,pad=0,rounding_size=0.4",
                                    facecolor="white", edgecolor="none", zorder=5))
        ax.add_patch(plt.Circle((left + 1.2, ym), 1.2, color=FLOW_INK, zorder=6))
        label(left + 1.2, ym, str(n), size=12, colour="white", weight="bold", zorder=7)
        _handoff_icon(ax, icon, left + 2.4 + 0.8 + 1.65, ym, s=3.3)
        label(left + 2.4 + 0.8 + 3.3 + 0.8, ym, word, size=16, weight="bold", ha="left")
    # decisions: only on the written rules and the engineer
    diamond(X["W"], 64.4, 10.0, 3.2, "DISPATCH RULE\nV1 · V2 · V3 · RETAKE")
    diamond(X["W"], 32.8, 15.0, 3.8, "TRIAGE RULES + BUILDING SCORE\nRetake · Expert needed · Category 1 · Category 2 · Category 3\n"
            "building priority P1 · P2 · P3")
    white_diamond(X["E"], 20.0, 9.5, 3.2, "CONFIRMS OR\nCORRECTS")
    label(X["E"] + 10.3, 20.0, "visits P1, P2\nbuildings", size=12, colour=FLOW_TEXT, ha="left")

    # who decides what
    label(157.0, 80.6, "WHO DECIDES WHAT", size=17, weight="bold")
    ax.plot([143.5, 143.5], [8.0, 83.5], color="#cfcfcf", lw=1.2, zorder=0)
    for k, (kind, name, decides, never) in enumerate(WHO_DECIDES):
        top = 76.4 - k * 13.6
        head("model_flat" if kind == "model" else kind, 157.0, top - 1.2, 25.0, 3.0, name, size=13.5)
        label(145.0, top - 3.4, decides, size=12.5, colour=FLOW_INK, ha="left", va="top")
        label(145.0, top - 9.0, never, size=12.5, colour=FLOW_TEXT, ha="left", va="top", style="italic")
    # legend
    ly0 = 2.8
    ax.add_patch(matplotlib.patches.Polygon([(2.2, ly0), (3.5, ly0 + 0.95), (4.8, ly0), (3.5, ly0 - 0.95)],
                                            closed=True, facecolor=FLOW_RULE, edgecolor="none", zorder=4))
    label(5.6, ly0, "DECISION BY A WRITTEN RULE", size=13, ha="left")
    ax.add_patch(matplotlib.patches.Polygon([(34.2, ly0), (35.5, ly0 + 0.95), (36.8, ly0), (35.5, ly0 - 0.95)],
                                            closed=True, facecolor="white", edgecolor=FLOW_INK, lw=1.6, zorder=4))
    label(37.6, ly0, "DECISION BY A PERSON", size=13, ha="left")
    line([(59.5, ly0), (64.5, ly0)], lw=2.0)
    label(65.4, ly0, "WHAT PASSES BETWEEN PLAYERS", size=13, ha="left")
    line([(93.5, ly0), (98.5, ly0)], ls="--", lw=2.0)
    label(99.4, ly0, "LOOP (RECHECK, RETRAINING)", size=13, ha="left")
    ax.add_patch(FancyBboxPatch((124.4, ly0 - 1.3), 2.6, 2.6, boxstyle="round,pad=0,rounding_size=0.4",
                                facecolor=FLOW_MODEL, edgecolor="none", zorder=3))
    _handoff_icon(ax, "network", 125.7, ly0, s=2.0, ink="white", paper=FLOW_MODEL)
    label(128.6, ly0, "A MODEL AT WORK (MEASURES ONLY)", size=13, ha="left")
    label(169.0, 0.9, "Sources: src/triage/dispatch.py, src/triage/matrix.py, spec/features/building_score.md · "
          "27 Oct 2026 · thresholds illustrative until calibrated", size=9, colour=FLOW_SOFT, ha="right")
    return fig


NEW_IN_V3 = {"thin_cover", "crossing_bar_crack", "damp"}
PATTERN_WORD = {"surface_map": "Map", "flexural": "Bending", "diagonal_shear": "Shear", "along_bar": "Bar",
                "column_horizontal": "Column", "spalling_exposed_bar": "Spalling", "joint": "Joint"}
CAT_FILL = {"Category 1": ("white", "#111111", "#111111"),
            "Category 2": ("#bdbdbd", "#bdbdbd", "#111111"), "Category 3": ("#111111", "#111111", "white")}


def _icon(ax, key, cx, cy, s=9.0):
    """Black, white and grey pictogram of a crack pattern, in an s x s box centred on (cx, cy).
    Symbol key in spec/features/figures.md."""
    K, G, u = "#111111", "#9a9a9a", s / 10.0
    x0, y0 = cx - s / 2, cy - s / 2

    def P(x, y):
        return x0 + x * u, y0 + y * u

    def rect(x, y, w, h, fc="white", ec=K, lw=1.6, ls="-", z=3):
        ax.add_patch(matplotlib.patches.Rectangle(P(x, y), w * u, h * u, facecolor=fc, edgecolor=ec, lw=lw, ls=ls,
                                                  zorder=z))

    def poly(pts, fc="white", ec=K, lw=1.6, z=3, closed=True):
        ax.add_patch(matplotlib.patches.Polygon([P(*p) for p in pts], closed=closed, facecolor=fc, edgecolor=ec,
                                                lw=lw, zorder=z))

    def ln(pts, lw=1.8, c=K, ls="-", z=4):
        xs, ys = zip(*[P(*p) for p in pts])
        ax.plot(xs, ys, color=c, lw=lw, ls=ls, zorder=z, solid_capstyle="round", solid_joinstyle="round")

    def circ(x, y, r, fc="white", ec=K, lw=1.6, z=4):
        ax.add_patch(plt.Circle(P(x, y), r * u, facecolor=fc, edgecolor=ec, lw=lw, zorder=z))

    def arrow(p0, p1, lw=1.8):
        ax.annotate("", xy=P(*p1), xytext=P(*p0), zorder=5,
                    arrowprops=dict(arrowstyle="-|>", color=K, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=11))

    if key == "surface_map":
        rect(0.5, 0.5, 9, 9, fc="#f2f2f2", ec=G, lw=1.2)
        for seg in ([(1, 3), (3, 4), (5, 2), (7, 4), (9, 3)], [(3, 4), (4, 7), (2, 9)], [(5, 2), (6, 6), (4, 7)],
                    [(6, 6), (8, 8), (9, 6)], [(7, 4), (8, 8)], [(4, 7), (6, 9)]):
            ln(seg, lw=1.3)
    elif key in ("flexural", "diagonal_shear", "along_bar"):
        rect(0, 3, 10, 4)
        if key == "flexural":
            for x in (3, 5, 7):
                ln([(x, 3), (x + 0.3, 4), (x - 0.2, 4.8), (x + 0.1, 5.5)], lw=1.7)
        elif key == "diagonal_shear":
            poly([(0.5, 3), (1.5, 1.4), (-0.5, 1.4)], fc=G, ec="none")
            ln([(1.6, 3.1), (2.3, 4.0), (2.6, 4.4), (3.4, 5.4), (3.7, 5.9), (4.4, 6.9)], lw=1.9)
        else:
            ln([(0.4, 4.3), (9.6, 4.3)], lw=2.6, c=G, ls=(0, (2.5, 1.5)), z=3)
            ln([(1, 4.9), (2.5, 4.7), (4, 5.0), (5.5, 4.75), (7, 5.0), (8.8, 4.8)], lw=1.7)
    elif key == "column_horizontal":
        rect(3.3, 0, 3.4, 10)
        for y in (4, 6.3):
            ln([(3.3, y), (4.2, y + 0.3), (5.0, y - 0.1), (5.8, y + 0.3), (6.7, y)], lw=1.7)
    elif key == "spalling_exposed_bar":
        rect(0, 2, 10, 6)
        poly([(2.5, 3.2), (6.5, 2.8), (7.8, 5.0), (6.4, 7.0), (3.2, 6.6), (2.2, 5.0)], fc="#d9d9d9", ec=K, lw=1.3)
        ln([(1.5, 5), (8.5, 5)], lw=3.2, c=K, z=5)
    elif key == "joint":
        rect(4, 0, 2, 10, z=3)
        rect(0, 4.5, 10, 2.5, z=2)
        ln([(4.1, 4.6), (4.8, 5.4), (5.2, 6.1), (5.9, 6.9)], lw=1.8, z=5)
        ln([(5.9, 4.6), (5.2, 5.3), (4.8, 6.2), (4.1, 6.9)], lw=1.8, z=5)


def risk_matrix():
    """Matrix v3 in black, white and grey: base category per pattern (drawn symbol) x width band, and what raises it.
    Values read from src/triage/risk_matrix_v3.json; pattern symbol key in spec/features/figures.md."""
    K, GREY_TXT = "#111111", "#555555"
    W = 10.4 * 16 / 9  # 16:9 like every figure
    pad = (W / 0.1 - 180) / 2
    fig = plt.figure(figsize=(W, 10.4), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-pad, 180 + pad)
    ax.set_ylim(0, 104)
    ax.set_axis_off()
    ax.text(3 - pad, 99.0, "RISK MATRIX", fontsize=38 * W / 24, weight="bold", color=K, va="center")
    ax.text(3, 92.6, f"Reinforced concrete, v{VERSION}. Base category from the crack pattern × width; the element, the "
            "location on the element and warning signs can only raise it. Values illustrative until calibrated with "
            "the engineering partners.", fontsize=11, color=GREY_TXT, va="center")

    def cat_cell(x, y, w, h, level, size=13):
        fc, ec, ink = CAT_FILL[level]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8", facecolor=fc,
                                    edgecolor=ec, lw=1.6, zorder=3))
        ax.text(x + w / 2, y + h / 2, level, ha="center", va="center", fontsize=size, weight="bold", color=ink,
                zorder=4)

    # column headers: crack width, thin to wide
    cw, x_first, top = 19.0, 21.0, 84.0
    for c, band in enumerate(BANDS):
        cx = x_first + c * (cw + 1.2) + cw / 2
        ax.plot([cx - 6, cx + 6], [top + 2.6, top + 2.6], color=K, lw=[1.0, 2.6, 5.0, 8.5][c], solid_capstyle="butt")
        ax.text(cx, top - 0.4, BAND_NAMES[band], ha="center", va="center", fontsize=12, weight="bold", color=K)
    ax.text(x_first + 2 * (cw + 1.2), top + 5.2, "Crack width (upper end of the measurement)", ha="center",
            fontsize=11, color=GREY_TXT)
    # rows: one drawn symbol per pattern (key in spec/features/figures.md), then the category cells
    rh = 10.2
    for r, pattern in enumerate(PATTERNS):
        y = top - 4.2 - (r + 1) * rh
        ax.add_patch(FancyBboxPatch((4.0, y + 0.3), 13.5, rh - 0.6, boxstyle="round,pad=0,rounding_size=0.8",
                                    facecolor="white", edgecolor="#cfcfcf", lw=1.2, zorder=2))
        _icon(ax, pattern, 10.75, y + rh / 2 + 1.1, s=6.4)
        ax.text(10.75, y + 1.7, PATTERN_WORD[pattern], ha="center", va="center", fontsize=10.5, weight="bold",
                color=K)
        for c, band in enumerate(BANDS):
            cat_cell(x_first + c * (cw + 1.2), y + 0.3, cw, rh - 0.6, MATRIX[band][r])

    # what raises the category, in words
    ax.plot([104.5, 104.5], [10, 88], color="#cfcfcf", lw=1.2)
    ax.text(108, 86.5, "What raises the category (never lowers it)", fontsize=14, weight="bold", color=K,
            va="center")
    end = _LOCATION_RULE["end_zone"]
    plural = {"along_bar": "cracks along bars", "diagonal_shear": "diagonal shear cracks"}
    lines = [(f"It is a {ELEMENTS[e]}", f"+{n}", "") for e, n in ELEMENT_RAISE.items() if n]
    lines.append(("In an end zone (or location unknown):\n" + ", ".join(plural.get(p, PATTERN_NAMES[p])
                                                                       for p in end["patterns"]),
                  f"+{end['raise']}", "assumption" if "assumption" in BASIS["end_zone"] else ""))
    lines += [(ESCALATORS[k][0].upper() + ESCALATORS[k][1:], "+1" if _ESCALATOR_EFFECT[k] == "+1" else "to 3",
               "assumption" if BASIS[k].startswith("assumption") else ("new in v3" if k in NEW_IN_V3 else ""))
              for k in ESCALATORS]
    for i, (text_, effect, tag) in enumerate(lines):
        y = 80.2 - i * 5.75
        fill, ink = ("#bdbdbd", K) if effect != "to 3" else (K, "white")
        ax.add_patch(FancyBboxPatch((108, y - 2.0), 8.0, 4.0, boxstyle="round,pad=0,rounding_size=0.6",
                                    facecolor=fill, edgecolor="none", zorder=3))
        ax.text(112, y, effect, ha="center", va="center", fontsize=11.5, weight="bold", color=ink, zorder=4)
        ax.text(118, y, text_, va="center", fontsize=11.5, color=K)
        if tag:
            ax.text(177, y, tag, va="center", ha="right", fontsize=10, color=GREY_TXT, style="italic")
    ax.text(108, 8.6, "Categories stop at 3. Any doubt (photo, material, element, pattern, width) goes to Expert "
            "needed.", fontsize=10.5, color=GREY_TXT, va="center")

    # legend: what each category means
    ly, x = 3.2, 4.0
    for level, meaning in (("Category 1", "routine monitoring, recheck in 6 months"),
                           ("Category 2", "low-cost advice, recheck in 1 month"),
                           ("Category 3", "an engineer must confirm now")):
        cat_cell(x, ly - 2.0, 15.0, 4.0, level, size=11)
        ax.text(x + 16.5, ly, meaning, fontsize=11, color=K, va="center")
        x += 59.0
    return fig


# Segmentation tree v3 (docs/segmentation_tree.md; decisions D16, D21, D25). Only what the tool works on.
# Leaf: (code, label, path, scale, evidence). Paths: "crack" feeds the atomic problem, "building" feeds the building
# risk score, "context" explains the need. Scale as in PHI-Net task 1 [55]. Evidence: source numbers in
# docs/sources.md, expert input (studio professor, 27 Oct), measured (our data map) or assumption (to test).
TREE = [
    ("A", "Buildings change unwatched", "building", [
        ("A1", "Ageing, no maintenance", "building", "S", "[3] [9]"),
        ("A2", "Floors added on top", "building", "S", "expert input"),
        ("A3", "Structure or use changed", "building", "S", "expert input"),
    ]),
    ("B", "Residents cannot read the signs", "crack", [
        ("B1", "Harmless vs structural", "crack", "P", "assumption"),
        ("B2", "No record of growth", "crack", "P", "expert input, [46]"),
        ("B3", "Cheap phones, no scale", "crack", "P", "expert input, [46]"),
        ("B4", "Is it concrete? Can't tell", "crack", "O", "assumption"),
        ("B5", "Which element? Can't tell", "crack", "O", "assumption"),
        ("B6", "One crack hides the pattern", "building", "S", "expert input"),
    ]),
    ("C", "Too few engineers", "building", [
        ("C1", "No ranked list of buildings", "building", "S", "assumption"),
        ("C2", "Reports not acted on", "context", "", "[3] [7] [10]"),
        ("C3", "Backlog: 114 buildings", "context", "", "[3] [5]"),
    ]),
    ("D", "Data is scarce", "context", [
        ("D1", "Public data: bridges, roads", "context", "", "measured: data map"),
        ("D2", "No crack labels by engineers", "context", "", "measured: data map"),
    ]),
]
TREE_YELLOW, TREE_NAVY, TREE_GREY = "#c4c4c4", "#262626", "#8c8c8c"
TREE_PATH = {"crack": TREE_YELLOW, "building": TREE_NAVY, "context": TREE_GREY}


def _pill(ax, x, y, w, h, text, fill, ink, size):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={h / 2}",
                                facecolor=fill, edgecolor="none", zorder=3))
    ax.text(x, y, text, ha="center", va="center", color=ink, fontsize=size, weight="bold", zorder=4)


def _elbow(ax, x0, y0, x1, y1, ymid, colour, r=0.28):
    """Vertical down from (x0, y0), across at ymid with rounded corners, down to (x1, y1)."""
    from matplotlib.path import Path as MPath
    from matplotlib.patches import PathPatch

    dx = x1 - x0
    if abs(dx) < 1e-6:
        verts, codes = [(x0, y0), (x1, y1)], [MPath.MOVETO, MPath.LINETO]
    else:
        s, r = (1 if dx > 0 else -1), min(r, abs(dx) / 2)
        verts = [(x0, y0), (x0, ymid + r), (x0, ymid), (x0 + s * r, ymid), (x1 - s * r, ymid), (x1, ymid),
                 (x1, ymid - r), (x1, y1)]
        codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CURVE3, MPath.CURVE3,
                 MPath.LINETO]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor="none", edgecolor=colour, lw=2.2, zorder=1,
                           capstyle="round"))


def _tag(ax, x, y, text, colour, r=0.33, size=11):
    ax.add_patch(plt.Circle((x, y), r, facecolor="white", edgecolor=colour, lw=2.2, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, weight="bold", color=TREE_NAVY, zorder=4)


def segmentation_tree():
    """Flowchart tree: macro pill, four branch pills, leaf circles with scale badge and evidence, then the crack
    path converges into the atomic problem and the building path into the building risk score."""
    import textwrap

    fig, ax = plt.subplots(figsize=(22, 22 * 9 / 16))
    ax.set_axis_off()
    _house_title(fig, "PROBLEM SEGMENTATION TREE")
    fig.text(0.02, 1 - 1.45 / fig.get_figheight(), "Macro problem → four branches → causes and needs → the atomic problem (one crack) and the "
             "building risk score (one building). Under each node: its evidence, as a source number (docs/sources.md),"
             "\nexpert input (studio professor, 27 Oct), measured (our data map) or an assumption still to test. "
             "S, O, P: the scale of the photo, as in PHI-Net task 1 (structural, object, pixel).", fontsize=11,
             color=INK_2, linespacing=1.5)
    leaf_gap, group_gap, r_leaf = 1.85, 0.9, 0.58
    y_top, y_meso, y_leaf = 12.2, 10.0, 7.3
    xs, x = {}, 0.0
    for letter, _, _, kids in TREE:
        xs[letter] = []
        for _ in kids:
            xs[letter].append(x)
            x += leaf_gap
        x += group_gap
    width = x - group_gap - leaf_gap
    centre = width / 2
    y_label = y_leaf - r_leaf - 0.18
    y_start = y_label - 1.05  # below label and evidence

    _pill(ax, centre, y_top, 14.6, 0.95, "PEOPLE CANNOT JUDGE WHETHER THE CRACKS IN THEIR AGEING BUILDINGS ARE "
          "DANGEROUS", TREE_YELLOW, TREE_NAVY, 12.5)
    n = len(TREE)
    crack_x, building_x = [], []
    for i, (letter, title, path, kids) in enumerate(TREE):
        colour = TREE_PATH[path]
        mx = sum(xs[letter]) / len(xs[letter])
        offset = (i - (n - 1) / 2) * 1.25
        ymid = y_top - 0.95 if abs(offset) > 1 else y_top - 1.55
        _elbow(ax, centre + offset, y_top - 0.48, mx, y_meso + 0.36, ymid, colour)
        _tag(ax, (centre + offset + mx) / 2, ymid, letter, colour)
        pill_ink = TREE_NAVY if path == "crack" else ("white" if path == "building" else INK)
        _pill(ax, mx, y_meso, max(3.0, 0.165 * len(title)), 0.72, title.upper(), colour, pill_ink, 10.5)
        for lx, (code, label, leaf_path, scale, evidence) in zip(xs[letter], kids):
            lc = TREE_PATH[leaf_path]
            _elbow(ax, mx, y_meso - 0.36, lx, y_leaf + r_leaf, y_meso - 1.15, lc)
            ax.add_patch(plt.Circle((lx, y_leaf), r_leaf, facecolor="white", edgecolor=lc, lw=3, zorder=3))
            ax.text(lx, y_leaf, code, ha="center", va="center", fontsize=14, weight="bold", color=TREE_NAVY,
                    zorder=4)
            if scale:
                _tag(ax, lx + 0.5, y_leaf + 0.5, scale, lc, r=0.24, size=9)
            ax.text(lx, y_label, textwrap.fill(label.upper(), 15, break_on_hyphens=False), ha="center",
                    va="top", fontsize=8.8, color=INK, linespacing=1.15)
            ax.text(lx, y_label - 0.62, evidence, ha="center", va="top", fontsize=8.3, color=INK_2, style="italic")
            if leaf_path == "crack":
                crack_x.append(lx)
            elif leaf_path == "building":
                building_x.append(lx)

    # crack path: converge into the atomic problem
    ax_c = sum(crack_x) / len(crack_x)
    y_atomic, atomic_w = 4.35, max(crack_x) - min(crack_x) + 1.3
    for lx in crack_x:
        _elbow(ax, lx, y_start, ax_c, y_atomic + 0.55, y_atomic + 0.95, TREE_YELLOW)
    ax.add_patch(FancyBboxPatch((ax_c - atomic_w / 2, y_atomic - 0.55), atomic_w, 1.1,
                                boxstyle="round,pad=0,rounding_size=0.55", facecolor=TREE_YELLOW, edgecolor="none",
                                zorder=3))
    ax.text(ax_c, y_atomic + 0.2, "ATOMIC PROBLEM: IS THIS CRACK DANGEROUS, AND IS IT GROWING?", ha="center",
            va="center", fontsize=10.5, weight="bold", color=TREE_NAVY, zorder=4)
    ax.text(ax_c, y_atomic - 0.22, "one crack in one concrete element, with a scale, seen again → level, confidence, "
            "reason, growth", ha="center", va="center", fontsize=9.3, color=TREE_NAVY, zorder=4)

    # building path: straight down into the building risk score, joined by the atomic problem
    y_build = 2.45
    for lx in building_x:
        ax.plot([lx, lx], [y_start, y_build + 0.55], color=TREE_NAVY, lw=2.2, zorder=1)
    ax.annotate("", xy=(ax_c, y_build + 0.55), xytext=(ax_c, y_atomic - 0.55),
                arrowprops=dict(arrowstyle="-|>", color=TREE_YELLOW, lw=2.4, shrinkA=0, shrinkB=0,
                                mutation_scale=18), zorder=2)
    ax.add_patch(FancyBboxPatch((-0.6, y_build - 0.55), width + 1.2, 1.1, boxstyle="round,pad=0,rounding_size=0.55",
                                facecolor=TREE_NAVY, edgecolor="none", zorder=3))
    ax.text(centre, y_build + 0.2, "BUILDING RISK SCORE: WHICH BUILDING DOES AN ENGINEER VISIT FIRST?   "
            "P1 · P2 · P3", ha="center", va="center", fontsize=11, weight="bold", color="white", zorder=4)
    ax.text(centre, y_build - 0.22, "crack levels and growth from the atomic problem + building data (age, added floors, "
            "structure or use changed) + damage spread across elements", ha="center", va="center", fontsize=9.5,
            color="#e6e6e6", zorder=4)

    _house_legend(fig, [Line2D([], [], color=c, lw=5) for c in (TREE_YELLOW, TREE_NAVY, TREE_GREY)],
                  ["CRACK PATH", "BUILDING PATH", "CONTEXT"])
    ax.set_xlim(-1.0, width + 1.0)
    ax.set_ylim(0.9, 12.9)
    ax.set_aspect("equal")
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - 2.1 / fig.get_figheight(), bottom=0.03)
    return fig


STAGES = ["Quality gate", "Material model (O)", "Element model (O)", "Crack segmentation (P)", "Measurement in mm (P)",
          "Pattern model (P)", "Building risk score (S)", "Agent: building data", "Triage rules sign-off",
          "Engineer review + retraining", "Evaluation (ground truth)"]
SOURCES = ["SDNET2018", "CrackSeg9k", "crack-seg\n(Ultralytics)", "METU", "PHI-Net", "CODEBRIM", "Haiti + Nepal\nsurveys",
           "Brick masonry\n(Darbhanga)", "Brick facades\n(Strathclyde)", "Masonry masks\n(Dais et al.)", "Pilot photos",
           "Engineer-\nlabelled photos", "Engineering\npartner"]
# From docs/data_map.md, section 1. Rows follow STAGES, columns follow SOURCES.
COVERAGE = [
    ["none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "planned", "planned", "none"],
    ["partial", "none", "none", "partial", "none", "partial", "none", "partial", "partial", "partial", "planned", "planned", "none"],
    ["none", "none", "none", "none", "partial", "none", "none", "none", "none", "none", "planned", "planned", "none"],
    ["none", "covered", "covered", "none", "none", "none", "none", "none", "none", "partial", "none", "planned", "none"],
    ["none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "planned", "planned", "none"],
    ["none", "none", "none", "none", "partial", "partial", "none", "none", "none", "none", "none", "planned", "none"],
    ["none", "none", "none", "none", "none", "none", "partial", "none", "none", "none", "none", "none", "planned"],
    ["none", "none", "none", "none", "none", "none", "partial", "none", "none", "none", "none", "none", "none"],
    ["none", "none", "none", "none", "none", "none", "partial", "none", "none", "none", "none", "none", "planned"],
    ["none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "planned", "planned"],
    ["none", "none", "none", "none", "none", "none", "partial", "none", "none", "none", "none", "planned", "planned"],
]
COVERAGE_LABEL = {"covered": "covers", "partial": "partly", "planned": "planned", "none": ""}


def data_coverage():
    fig, ax = plt.subplots(figsize=(21, 21 * 9 / 16))
    for r, row in enumerate(COVERAGE):
        for c, state in enumerate(row):
            ax.add_patch(FancyBboxPatch((c + 0.04, r + 0.08), 0.92, 0.84, boxstyle="round,pad=0,rounding_size=0.05",
                                        facecolor=SEQ[state], edgecolor=GRID if state == "none" else "none", lw=0.8))
            ax.text(c + 0.5, r + 0.5, COVERAGE_LABEL[state], ha="center", va="center", fontsize=9,
                    color="white" if state in ("covered", "partial") else INK)
    ax.set_xlim(0, len(SOURCES))
    ax.set_ylim(len(STAGES), 0)
    ax.set_xticks([c + 0.5 for c in range(len(SOURCES))], SOURCES)
    ax.set_yticks([r + 0.5 for r in range(len(STAGES))], STAGES)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    _house_legend(fig, [Patch(facecolor=SEQ[k], edgecolor="#9a9a9a") for k in ("covered", "partial", "planned", "none")],
                  ["COVERS", "PARTLY", "PLANNED", "NOTHING"])
    _house_title(fig, "DATA MAP", "Which source feeds which pipeline stage")
    fig.text(0.02, 1 - 1.4 / fig.get_figheight(), "Public data covers only crack segmentation (P); every other stage needs our own photos or an "
             "engineer. From docs/data_map.md, section 1 (27 Oct 2026).", fontsize=10, color=INK_2)
    fig.subplots_adjust(left=0.13, right=0.99, top=1 - 2.3 / fig.get_figheight(), bottom=0.1)
    return fig


# From docs/data_map.md, sections 0b, 1, 2 and 6: (name, images, count and licence, train, test, measure, what for).
# Train: teach a model. Test: check a model or the quality gate on data it has not seen. Measure: the photos carry a
# scale, so crack widths in mm can be checked.
DATASETS = [
    ("SDNET2018", 56_000, "56,000+  ·  CC BY 4.0", True, True, False, "crack / no crack; quality gate"),
    ("METU", 40_000, "40,000  ·  CC BY 4.0", True, True, False, "crack / no crack; material: concrete"),
    ("PHI-Net", 36_413, "36,413  ·  CC BY-NC-SA 4.0", True, True, False, "element, pattern, scene"),
    ("CrackSeg9k", 9_255, "9,255  ·  CC0", True, True, False, "crack pixels (U-Net)"),
    ("crack-seg (Ultralytics)", 4_029, "4,029  ·  Public Domain", True, True, False, "crack pixels (U-Net)"),
    ("Haiti 2010 survey", 6_526, "6,526 photos  ·  licence not stated", False, True, False,
     "engineer ratings: check of levels"),
    ("Brick masonry (Darbhanga)", 4_259, "4,259  ·  CC BY 4.0", True, True, False, "crack boxes; material: not concrete"),
    ("Brick facades (Strathclyde)", 700, "700  ·  CC BY 4.0", True, True, False, "crack / no crack; not concrete"),
    ("Pilot phone photos", 0, "0, about 30 cracks this week", False, True, True, "width in mm; tune the gate"),
    ("Engineer-labelled phone photos", 0, "0, to collect in Phase 2", True, True, True,
     "ground truth, baseline, calibration"),
]


def dataset_sizes():
    """What data exists and what each set can be used for (train, test, measure), in black and white."""
    K = "#111111"
    fig, ax = plt.subplots(figsize=(17, 17 * 9 / 16))
    names = [d[0] for d in DATASETS]
    ys = range(len(DATASETS))
    ax.barh(ys, [d[1] for d in DATASETS], height=0.5, color=K)
    ax.set_yticks(list(ys), names)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(15)
        tick.set_fontweight("bold")
        tick.set_color(K)
    ax.invert_yaxis()
    ax.set_xlim(0, 168_000)
    ax.set_xticks([0, 20_000, 40_000, 60_000])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}K" if v else "0"))
    ax.tick_params(axis="x", colors=K, labelsize=11.5, length=0)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.grid(axis="x", color="#d9d9d9", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_bounds(0, 60_000)
    ax.spines["bottom"].set_color(K)
    ax.text(30_000, len(DATASETS) + 0.15, "Images", ha="center", va="top", fontsize=12, color=K)
    fig.subplots_adjust(left=0.22, right=0.985, top=0.8, bottom=0.08)
    # two aligned columns after the bars: count and licence, then what the set is for (bold)
    for y, (_, n, count, *_flags, what) in zip(ys, DATASETS):
        ax.text(62_000, y, count, va="center", fontsize=11.5, color=K)
        ax.text(96_000, y, what, va="center", fontsize=12, color=K, weight="bold")
    ax.text(62_000, -0.95, "IMAGES · LICENCE", va="center", fontsize=11, color="#444444")
    ax.text(96_000, -0.95, "WHAT IT IS FOR", va="center", fontsize=11, color="#444444")

    # what each set can be used for: a filled dot means yes
    trans = ax.get_yaxis_transform()
    for k, head in enumerate(("TRAIN", "TEST", "MEASURE")):
        x = 0.835 + k * 0.073
        ax.text(x, -0.95, head, transform=trans, ha="center", va="center", fontsize=13, weight="bold", color=K)
        for y, row in zip(ys, DATASETS):
            used = row[3 + k]
            ax.plot(x, y, transform=trans, marker="o" if used else "_", ms=15 if used else 12,
                    color=K if used else "#9a9a9a", mew=2, clip_on=False)
    fig.suptitle("WHAT WE FOUND AND HOW IT COULD HELP", x=0.02, ha="left", fontsize=38 * 17 / 24, weight="bold",
                 color=K)
    fig.text(0.02, 0.905, "Train: teach a model.  Test: check a model or the quality gate on data it has not seen.  "
             "Measure: the photos carry a scale, so crack widths in mm can be checked.\nCounts and licences from each "
             "dataset's page; uses from docs/data_map.md. Only our own photos have a scale.", fontsize=11.5,
             color="#444444", va="top", linespacing=1.5)
    return fig


def confusion_template():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 14 * 9 / 16), gridspec_kw={"width_ratios": [1, 1.6]})
    cells = [  # (row, col, title, meaning, colour)
        (0, 0, "TP", "danger caught,\nengineer visits", LEVEL["Category 1"]),
        (0, 1, "FN", "missed danger:\nwrong all-clear. Stop condition", LEVEL["Category 3"]),
        (1, 0, "FP", "unneeded visit,\ncosts engineer time", LEVEL["Category 2"]),
        (1, 1, "TN", "correctly kept off\nthe engineer's list", LEVEL["Category 1"]),
    ]
    for r, c, title, meaning, colour in cells:
        a1.add_patch(FancyBboxPatch((c + 0.04, r + 0.04), 0.92, 0.92, boxstyle="round,pad=0,rounding_size=0.05",
                                    facecolor=colour, edgecolor="none"))
        ink = "white" if colour == LEVEL["Category 3"] else INK
        a1.text(c + 0.5, r + 0.36, title, ha="center", va="center", fontsize=18, weight="bold", color=ink)
        a1.text(c + 0.5, r + 0.66, meaning, ha="center", va="center", fontsize=9, color=ink)
    a1.set_xlim(0, 2)
    a1.set_ylim(2, 0)
    a1.set_xticks([0.5, 1.5], ["Tool flags\n(Category 3 or Expert needed)", "Tool does not flag\n(Category 1 or Category 2)"])
    a1.set_yticks([0.5, 1.5], ["Engineer:\nCategory 3", "Engineer:\nCategory 1 or 2"])
    a1.xaxis.tick_top()
    a1.set_title("Table 1: the safety question", loc="left", weight="bold", pad=44)

    kinds = {"correct": ("#f2f2f2", INK), "over-cautious": ("#d9d9d9", INK), "false alarm": ("#d9d9d9", INK),
             "referral": ("#bfbfbf", INK), "retake": ("#efefef", INK_2),
             "false all-clear": (LEVEL["Category 3"], "white"), "missed danger": (LEVEL["Category 3"], "white"),
             "missed danger\n(worst)": ("#111111", "white")}
    grid = [["correct", "over-cautious", "false alarm", "referral", "retake"],
            ["false all-clear", "correct", "over-cautious", "referral", "retake"],
            ["missed danger\n(worst)", "missed danger", "correct", "referral", "retake"]]
    for r, row in enumerate(grid):
        for c, kind in enumerate(row):
            fill, ink = kinds[kind]
            a2.add_patch(FancyBboxPatch((c + 0.04, r + 0.06), 0.92, 0.88, boxstyle="round,pad=0,rounding_size=0.05",
                                        facecolor=fill, edgecolor="none"))
            a2.text(c + 0.5, r + 0.5, kind, ha="center", va="center", fontsize=9, color=ink,
                    weight="bold" if "missed" in kind or "all-clear" in kind else "normal")
    a2.set_xlim(0, 5)
    a2.set_ylim(3, 0)
    a2.set_xticks([c + 0.5 for c in range(5)], ["Category 1", "Category 2", "Category 3", "Expert\nneeded", "Retake"])
    a2.set_yticks([0.5, 1.5, 2.5], ["Category 1", "Category 2", "Category 3"])
    a2.xaxis.tick_top()
    a2.set_xlabel("Tool output")
    a2.xaxis.set_label_position("top")
    a2.set_ylabel("Engineer on site")
    a2.set_title("Table 2: full confusion matrix", loc="left", weight="bold", pad=44)
    for ax in (a1, a2):
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    _house_title(fig, "CONFUSION MATRIX", "How the tool will be scored")
    fig.text(0.02, 0.02, "Template: counts come from a held-out set of engineer-labelled photos, which does not "
             "exist yet. The non-AI baseline (engineer looks at the photo only) is scored the same way. "
             "spec/features/evaluation.md", fontsize=9, color=INK_2)
    fig.subplots_adjust(left=0.1, right=0.98, top=1 - 2.2 / fig.get_figheight(), bottom=0.09, wspace=0.25)
    return fig


# From spec/constitution/roadmap.md. (workstream, task, first week, last week, at risk)
PLAN = [  # (group, task, first week, last week, at risk); [n] = box n of the pipeline story (figure 01d); scope D30
    ("Photos", "[2] Quality gate (done)", 1, 1, False),
    ("Photos", "[5] Pilot photos + scale card", 2, 2, False),
    ("Photos", "[2] Tune the gate on pilot photos", 2, 3, False),
    ("Models", "[6] U-Net crack segmentation", 3, 4, False),
    ("Models", "[6] Width in mm (1st feature)", 3, 4, False),
    ("Models", "[6] Pattern model (stretch)", 4, 5, False),
    ("Written rules", "[4] [7] Dispatch + risk matrix (done)", 1, 1, False),
    ("Written rules", "[8] Building score", 2, 3, False),
    ("Written rules", "[7] Calibrate with the engineer", 3, 5, True),
    ("Engineer", "[9] Partner scoping", 1, 2, False),
    ("Engineer", "[9] Engineer labels", 4, 6, True),
    ("Engineer", "[9] Baseline comparison", 6, 6, True),
    ("Prototype", "First MCP: building data", 3, 3, False),
    ("Prototype", "Beta, end to end", 4, 4 - 2 / 7, False),  # hackathon, Monday 16 to Friday 20 Nov
    ("Prototype", "Harness + MCP to all sources", 5, 5, False),
    ("Prototype", "Recheck + learn loops", 5, 6, True),
    ("Prototype", "Proof of concept + ethics", 7, 7, False),
    ("Prototype", "UI + realistic tests", 7, 8, False),
    ("Syllabus", "Midterm deck", 1, 2, False),
    ("Syllabus", "Lean Canvas", 6, 7, False),
    ("Syllabus", "Final pitch", 7, 8, False),
    ("Syllabus", "Blog post", 8, 9, False),
]
# One date per week: the Thursday at the centre of each Monday-to-Sunday column, from 29 Oct.
# One milestone per week, from the syllabus (the 2 Nov expert package is a task bar, the midterm is Fri 6 Nov).
WEEK_DATES = ["29 Oct", "5 Nov", "12 Nov", "19 Nov", "26 Nov", "3 Dec", "10 Dec", "17 Dec", "24 Dec"]
WEEK_MILESTONES = [("data map", False), ("midterm", False), ("1st feature", False), ("beta", False),
                   ("harness + MCP", False), ("baseline", True), ("PoC + ethics", False), ("final", False),
                   ("blog", False)]
# One look per group, following the pipeline's colour logic (white = photos, black = models, grey = written rules)
# plus red, green and blue (the user's pick for the roadmap). (fill, hatch); every edge is black.
WORKSTREAM_STYLE = {"Photos": ("white", None), "Models": ("#111111", None), "Written rules": ("#9a9a9a", None),
                    "Engineer": ("#d03b3b", None), "Prototype": ("#2e9e44", None), "Syllabus": ("#1f5fbf", None)}
# At risk (needs the engineering partners): dark-red hatch, unlike any workstream; the row still shows the stream.
RISK_STYLE = dict(facecolor="white", edgecolor="#9e1b1b", hatch="\\\\\\\\", lw=1.6)


def timeline():
    weeks = len(WEEK_DATES)
    fig, ax = plt.subplots(figsize=(10.9 * 16 / 9, 10.9))
    n = len(PLAN)
    ax.axvspan(3, 3 + 5 / 7, color="#e6e6e6", zorder=0)  # hackathon: Monday 16 to Friday 20 Nov (columns run Mon-Sun)
    ax.text(3 + 2.5 / 7, n - 0.9, "HACKATHON\nMon 16 - Fri 20 Nov", ha="center", va="center", fontsize=10.5,
            weight="bold", color=INK_2)
    previous = None
    plt.rcParams["hatch.linewidth"] = 1.3
    for i, (stream, task, start, end, risk) in enumerate(PLAN):
        fill, hatch = WORKSTREAM_STYLE[stream]
        if stream != previous and previous is not None:
            ax.axhline(i - 0.5, color=GRID, lw=0.8, zorder=1)
        previous = stream
        ax.add_patch(FancyBboxPatch((start - 1 + 0.06, i - 0.36), end - start + 1 - 0.12, 0.72,
                                    boxstyle="round,pad=0,rounding_size=0.12", zorder=2,
                                    **(RISK_STYLE if risk else dict(facecolor=fill, edgecolor=INK, hatch=hatch, lw=1.4))))
    for w, (label, risk) in enumerate(WEEK_MILESTONES):  # one milestone per week, on one row
        ax.text(w + 0.5, -0.95, label + (" (at risk)" if risk else ""), ha="center", va="center", fontsize=10.5,
                color=INK_2 if risk else INK, style="italic" if risk else "normal")
    ax.set_xlim(0, weeks)
    ax.set_ylim(n - 0.4, -1.6)
    ax.set_xticks([w + 0.5 for w in range(weeks)], WEEK_DATES, fontsize=13, weight="bold", color=INK)
    ax.set_yticks(range(n), [t[1] for t in PLAN], fontsize=12.5)
    ax.xaxis.tick_top()
    for w in range(weeks + 1):
        ax.axvline(w, color=GRID, lw=0.6, zorder=0)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    handles = [Patch(facecolor=fill, edgecolor=INK, hatch=hatch, lw=1.4, label=s)
               for s, (fill, hatch) in WORKSTREAM_STYLE.items()]
    handles.append(Patch(label="At risk", **RISK_STYLE))
    _house_legend(fig, handles, [h.get_label().upper() for h in handles])
    H, W = fig.get_figheight(), fig.get_figwidth()  # title as in the other figures: 38 pt on a 24 in figure
    fig.text(0.02, 1 - 0.45 / H, "PROJECT ROADMAP", fontsize=38 * W / 24, weight="bold", color=INK, va="center")
    fig.text(0.02, 1 - 0.95 / H, "Plan to the final presentation, 29 Oct to 24 Dec 2026", fontsize=18 * W / 24,
             weight="bold", color=INK_2, va="center")
    fig.text(0.02, 1 - 1.25 / H, "We build the crack line of the pipeline: [n] = box n of the pipeline story (01d). Designed, "
             "not built: the field worker's app, the facade and element models (the field worker picks the element, D30).\n"
             "At risk = needs Engineers Without Borders or the structural engineer. Fallback if no engineer ratings by 26 Nov: "
             "compare on pilot photos only. Source: spec/constitution/roadmap.md", fontsize=11, color=INK_2, va="top")
    fig.subplots_adjust(left=0.15, right=0.98, top=0.8, bottom=0.04)
    return fig


# Every dataset on our disk (docs/data_map.md, sections 0 and 6), in order of fit to the concept (D14).
# (name, fit, images on disk or None, quality-gate pass % or None, labels it gives, what it trains, licence).
# Gate %: data/processed/real_data_quality.csv (27 Oct 2026). PHI-Net counts: train + test from the npy files [53][55].
HELD = [
    ("Haiti 2010 survey", "high", 4_450, 95, "engineer rating per building:\nlight / moderate / severe (183)",
     "building risk score evidence,\nsanity check of levels", "not stated"),
    ("PHI-Net t1 scene level", "medium", 27_306, None, "pixel / object / structural", "scene check (S, O, P)",
     "CC BY-NC-SA 4.0"),
    ("PHI-Net t2 damage state", "medium", 13_271, None, "damaged / undamaged", "crack or no crack", "CC BY-NC-SA 4.0"),
    ("PHI-Net t3 spalling", "medium", 7_735, 62, "spalling / none", "pattern: spalling", "CC BY-NC-SA 4.0"),
    ("PHI-Net t4 material", "medium", 9_291, None, "steel / other\n(not concrete vs brick)", "not the material model",
     "CC BY-NC-SA 4.0"),
    ("PHI-Net t5 collapse mode", "medium", 1_372, None, "none / partial / global collapse", "building reading (S)",
     "CC BY-NC-SA 4.0"),
    ("PHI-Net t6 component", "medium", 5_334, 79, "beam / column / wall / other", "element model",
     "CC BY-NC-SA 4.0"),
    ("PHI-Net t7 damage level", "medium", 4_636, None, "undamaged / minor /\nmoderate / heavy",
     "second opinion only", "CC BY-NC-SA 4.0"),
    ("PHI-Net t8 damage type", "medium", 4_585, 78, "undamaged / flexural /\nshear / combined",
     "pattern model", "CC BY-NC-SA 4.0"),
    ("Darbhanga brick masonry", "medium", 4_259, 75, "crack boxes on brick walls", "crack detector", "CC BY 4.0"),
    ("Strathclyde brick facades", "medium", 700, 100, "crack / no crack", "crack or no crack, material",
     "CC BY 4.0"),
    ("METU building concrete", "low", 40_000, 21, "crack / no crack", "crack or no crack, material", "CC BY 4.0"),
    ("SDNET2018 walls", "low", 18_138, 16, "crack / no crack", "crack or no crack", "CC BY 4.0"),
    ("crack-seg (Ultralytics)", "low", 4_029, 97, "crack outlines (polygons)", "crack segmentation",
     "Public Domain"),
    ("CrackSeg9k Vol1", "low", None, None, "crack pixel masks", "crack segmentation", "CC0"),
    ("SDNET2018 pavement", "off-concept", 24_334, 87, "crack / no crack", "nothing (roads)", "CC BY 4.0"),
    ("SDNET2018 bridge decks", "off-concept", 13_620, 19, "crack / no crack", "nothing (bridges)", "CC BY 4.0"),
]
FIT_FILL = {"high": SEQ["covered"], "medium": SEQ["partial"], "low": SEQ["planned"], "off-concept": SURFACE}
HELD_MISSING = {"CrackSeg9k": "zip not extracted"}


def dataset_comparison():
    n = len(HELD)
    fig, ax = plt.subplots(figsize=((0.62 * n + 3.3) * 16 / 9, 0.62 * n + 3.3))
    # x layout: name | fit | images bar | gate bar | labels | trains | licence
    x_fit, x_img, w_img, x_gate, w_gate, x_lab, x_use, x_lic, x_end = 3.3, 4.6, 3.0, 8.3, 1.6, 10.6, 13.6, 16.4, 18.2
    top = 40_000
    heads = [(0, "Dataset", "left"), (x_fit, "Fit", "left"), (x_img, "Images on our disk", "left"),
             (x_gate, "Pass the quality gate", "left"), (x_lab, "Labels it gives", "left"),
             (x_use, "What it can train for us", "left"), (x_lic, "Licence", "left")]
    for x, h, ha in heads:
        ax.text(x, -0.75, h.upper(), ha=ha, va="center", fontsize=15.5, weight="bold", color=INK)
    ax.axhline(-0.42, color=INK, lw=1.4)  # one line, under the column titles only
    for r, (name, fit, images, gate, labels, use, licence) in enumerate(HELD):
        ax.text(0, r, name.upper(), va="center", fontsize=12, weight="bold", color=INK)
        ax.add_patch(FancyBboxPatch((x_fit, r - 0.2), 1.05, 0.4, boxstyle="round,pad=0,rounding_size=0.08",
                                    facecolor=FIT_FILL[fit], edgecolor=GRID if fit == "off-concept" else "none"))
        ax.text(x_fit + 0.525, r, fit, ha="center", va="center", fontsize=8.5,
                color="white" if fit in ("high", "medium") else INK)
        missing = next((v for k, v in HELD_MISSING.items() if name.startswith(k)), "not on disk")
        if images is None:
            ax.text(x_img, r, missing, va="center", fontsize=9, color=MUTED, style="italic")
        else:
            w = w_img * min(images, top) / top
            ax.add_patch(FancyBboxPatch((x_img, r - 0.17), w, 0.34, boxstyle="round,pad=0,rounding_size=0.04",
                                        facecolor=BLUE, edgecolor="none"))
            ax.text(x_img + w + 0.08, r, f"{images:,}", va="center", fontsize=9.5, color=INK)
        if gate is None:
            ax.text(x_gate, r, "not run",
                    va="center", fontsize=9, color=MUTED, style="italic")
        else:
            ax.add_patch(FancyBboxPatch((x_gate, r - 0.17), w_gate, 0.34, boxstyle="round,pad=0,rounding_size=0.04",
                                        facecolor=GRID, edgecolor="none"))
            ax.add_patch(FancyBboxPatch((x_gate, r - 0.17), w_gate * gate / 100, 0.34,
                                        boxstyle="round,pad=0,rounding_size=0.04", facecolor=BLUE, edgecolor="none"))
            ax.text(x_gate + w_gate + 0.08, r, f"{gate}%", va="center", fontsize=9.5, color=INK)
        ax.text(x_lab, r, labels, va="center", fontsize=8.8, color=INK_2, linespacing=1.05)
        ax.text(x_use, r, use.upper(), va="center", fontsize=12, weight="bold", color=INK, linespacing=1.1)
        ax.text(x_lic, r, licence, va="center", fontsize=8.8, color=INK_2)
    ax.set_xlim(-0.1, x_end)
    ax.set_ylim(n - 0.4, -1.2)
    ax.set_axis_off()
    H, W = fig.get_figheight(), fig.get_figwidth()  # title as in the other figures: 38 pt on a 24 in figure
    fig.text(0.01, 1 - 0.6 / H, "DATASET LABELS AND TRAINING OBJECTIVES", fontsize=38 * W / 24, weight="bold", color=INK, va="center")
    fig.text(0.01, 1 - 1.25 / H, "Every dataset we hold, compared: fit to the concept, size, quality, labels, use",
             fontsize=18 * W / 24, weight="bold", color=INK_2, va="center")
    fig.text(0.01, 1 - 1.6 / H,
             "Ordered by fit to the concept (D14). Image bars capped at 40,000. Low gate rates on METU and SDNET "
             "come from smooth concrete read as blur (open issue D12), not from bad photos. PHI-Net is post-earthquake "
             "damage, research use only;\nits images are 224 px crops. Sources: docs/data_map.md, sections 0 and 6.",
             fontsize=9.5, color=INK_2, va="top")
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - 2.2 / H, bottom=0.02)
    return fig


# Pipeline v3 as a task flow (same content as 01_pipeline.png; spec/features/pipeline.md), in black, white and grey.
# Colour says who does the work: white = photos and people, grey = written rules (code), black = learned models.
# Shape says what it is: pill = start, rectangle = step, diamond = check, stacked = model,
# parallelogram = result shown to a person, striped block = the person who acts.
FLOW_INK, FLOW_RULE, FLOW_MODEL, FLOW_TEXT, FLOW_SOFT = "#111111", "#d2d2d2", "#111111", "#2b2b2b", "#6b6b6b"
FLOW_LANES = [  # (lane y, code, name, what, start pill, who takes the photo; spec: "Users and devices")
    (46, "S", "STRUCTURAL", "the building", "BUILDING", "field worker, first visit\n(resident answers questions)"),
    (32, "O", "OBJECT", "one element", "ELEMENT", "field worker, first visit\nevery element with a crack"),
    (18, "P", "PIXEL", "the crack", "CRACK", "field worker, first visit\nresident, at every recheck"),
]
FLOW_COLUMNS = [(11.5, None, "START"), (26.5, "1", "CAPTURE\nPHOTOS AND ANSWERS"), (43.5, "2", "CHECK\nWRITTEN RULES"),
                (66, "3", "PERCEIVE\nMODELS + CONFIDENCE"), (88.5, "4", "DECIDE\nWRITTEN RULES, AUDITABLE"),
                (109.5, "5", "ACT\nPEOPLE")]
FLOW_SEPARATORS = [17.5, 35.5, 51.5, 79.5, 98.5]
# The models each Perceive box needs (spec/constitution/tech_stack.md); architectures beyond U-Net not chosen yet.
FLOW_MODELS = {"S": "FACADE\nCLASSIFIER\n(later)", "O": "MATERIAL\n+ ELEMENT\nCLASSIFIERS",
               "P": "U-NET\n+ PATTERN\nCLASSIFIER"}
# Check diamonds per lane: (question, failure label or None)
FLOW_CHECKS = {"S": ("ALL\nANSWERED?", None), "O": ("CARD +\nANGLE OK?", "RETAKE"),
               "P": ("SHARP, LIT,\n≥ 3 PX?", "RETAKE")}


def _flow_tools(ax):
    """Shapes of the task-flow figures (01b, 02): white = photos and people, grey = written rules, black = models."""
    def label(x, y, s, size=11, colour=FLOW_INK, weight="normal", **kw):
        ax.text(x, y, s, ha=kw.pop("ha", "center"), va=kw.pop("va", "center"), fontsize=size, color=colour,
                weight=weight, linespacing=1.25, zorder=kw.pop("zorder", 6), **kw)

    def photo_box(cx, cy, w, h, title):  # white: photos and people
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.5",
                                    facecolor="white", edgecolor=FLOW_INK, lw=1.8, zorder=4))
        label(cx, cy, title, weight="bold")

    def rule_box(cx, cy, w, h, title):  # grey: written rules
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.5",
                                    facecolor=FLOW_RULE, edgecolor=FLOW_RULE, lw=1.4, zorder=4))
        label(cx, cy, title, weight="bold")

    def model_box(cx, cy, w, h, title):  # black: learned models, stacked
        for k in (2, 1):
            ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2 - 0.55 * k), w, h, zorder=3 - k * 0.1,
                                        boxstyle="round,pad=0,rounding_size=0.5", facecolor="#4a4a4a",
                                        edgecolor="white", lw=1.4))
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.5",
                                    facecolor=FLOW_MODEL, edgecolor="white", lw=1.4, zorder=4))
        label(cx, cy, title, colour="white", weight="bold")

    def diamond(cx, cy, hw, hh, text):  # grey: a written check
        ax.add_patch(matplotlib.patches.Polygon([(cx - hw, cy), (cx, cy + hh), (cx + hw, cy), (cx, cy - hh)],
                                                closed=True, facecolor=FLOW_RULE, edgecolor="none", zorder=4))
        label(cx, cy, text, size=10, weight="bold")

    def para(cx, cy, w, h, text, size=10.5):  # white: result shown to a person
        k = 0.8
        ax.add_patch(matplotlib.patches.Polygon([(cx - w / 2 - k, cy - h / 2), (cx + w / 2 - k, cy - h / 2),
                                                 (cx + w / 2 + k, cy + h / 2), (cx - w / 2 + k, cy + h / 2)],
                                                closed=True, facecolor="white", edgecolor=FLOW_INK, lw=1.8,
                                                zorder=4))
        label(cx, cy, text, size=size, weight="bold")

    def pill(cx, cy, w, h, text):  # white: where a person starts
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={h / 2}",
                                    facecolor="white", edgecolor=FLOW_INK, lw=1.8, zorder=4))
        label(cx, cy, text, size=10.5, weight="bold")

    def terminal(cx, cy, w, h, text):  # white: the person who acts
        ax.add_patch(matplotlib.patches.Rectangle((cx - w / 2, cy - h / 2), w, h, facecolor="white",
                                                  edgecolor=FLOW_INK, lw=1.8, zorder=4))
        for dx in (-w / 2 + 0.6, w / 2 - 0.6):
            ax.plot([cx + dx] * 2, [cy - h / 2, cy + h / 2], color=FLOW_INK, lw=1.8, zorder=5)
        label(cx, cy, text, size=10, weight="bold")

    def mark(cx, cy, ok, r=0.95):
        ax.add_patch(plt.Circle((cx, cy), r, facecolor="white", edgecolor=FLOW_INK, lw=1.6, zorder=5))
        k = r * 0.45
        if ok:
            ax.plot([cx - k, cx - k * 0.2, cx + k], [cy, cy - k * 0.7, cy + k * 0.8], color=FLOW_INK, lw=2.1,
                    zorder=7, solid_capstyle="round")
        else:
            for sgn in (1, -1):
                ax.plot([cx - k, cx + k], [cy - sgn * k, cy + sgn * k], color=FLOW_INK, lw=2.1, zorder=7,
                        solid_capstyle="round")

    def line(points, ls="-", lw=1.8, head=True, dot=True):
        xs, ys = zip(*points)
        ax.plot(xs[:-1] if head else xs, ys[:-1] if head else ys, color=FLOW_TEXT, lw=lw, ls=ls, zorder=2,
                solid_capstyle="round")
        if head:
            ax.annotate("", xy=points[-1], xytext=points[-2], zorder=2,
                        arrowprops=dict(arrowstyle="-|>", color=FLOW_TEXT, lw=lw, ls=ls, shrinkA=0, shrinkB=0,
                                        mutation_scale=14))
        if dot:
            ax.plot(*points[0], "o", color=FLOW_TEXT, ms=5, zorder=3)

    def status(cx, top, key):
        text_, fill, edge, ls, ink = {"code": ("IN CODE", FLOW_RULE, FLOW_INK, "-", FLOW_INK),
                                      "partly": ("PARTLY IN CODE", "white", FLOW_INK, "-", FLOW_INK),
                                      "spec": ("SPECIFIED", "white", "#b5b5b5", (0, (2, 2)), FLOW_SOFT)}[key]
        w = 0.55 * len(text_) + 1.4
        ax.add_patch(FancyBboxPatch((cx - w / 2, top + 0.35), w, 1.5, boxstyle="round,pad=0,rounding_size=0.7",
                                    facecolor=fill, edgecolor=edge, lw=1.1, ls=ls, zorder=4))
        label(cx, top + 1.1, text_, size=8.2, colour=ink, weight="bold")

    def detail(cx, y_top, text):
        label(cx, y_top, text, size=10, colour=FLOW_TEXT, va="top")

    return SimpleNamespace(label=label, photo_box=photo_box, rule_box=rule_box, model_box=model_box, diamond=diamond, para=para, pill=pill, terminal=terminal, mark=mark, line=line, status=status, detail=detail)


def pipeline_flow():
    """Pipeline v3 drawn as a task flow: phases as columns, scales as lanes; colour = photo, rule or model."""
    W = 17.6 * 16 / 9  # 16:9 like every figure; the 120-unit drawing sits in the middle
    pad = (W / 0.2 - 120) / 2
    fig = plt.figure(figsize=(W, 17.6), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-pad, 120 + pad)
    ax.set_ylim(0, 88)
    ax.set_axis_off()

    t = _flow_tools(ax)
    label, photo_box, rule_box, model_box, diamond, para, pill, terminal, mark, line, status, detail = (t.label, t.photo_box, t.rule_box, t.model_box, t.diamond, t.para, t.pill, t.terminal, t.mark, t.line, t.status, t.detail)

    def wrap(title):
        if "\n" in title or len(title) <= 12:
            return title
        if " + " in title:
            return title.replace(" + ", "\n+ ", 1)
        mid, spaces = len(title) / 2, [i for i, ch in enumerate(title) if ch == " "]
        cut = min(spaces, key=lambda i: abs(i - mid))
        return title[:cut] + "\n" + title[cut + 1:]

    def with_icon(draw, cx, cy, w, h, title, icon, ink=FLOW_INK, paper="white", size_text=10.5):
        """A box with its symbol on the left and its title on the right."""
        draw(cx, cy, w, h, "")
        size = min(h * 0.7, 3.6)
        _handoff_icon(ax, icon, cx - w / 2 + 0.9 + size / 2, cy, s=size, ink=ink, paper=paper)
        label(cx + (size + 0.9) / 2, cy, wrap(title), colour=ink, weight="bold", size=size_text)

    def ipill(cx, cy, title, icon):
        pill(cx, cy, 11.0, 4.2, "")
        _handoff_icon(ax, icon, cx - 3.8, cy, s=2.8)
        label(cx + 1.4, cy, title, size=15 if "\n" not in title else 12.5, weight="bold")

    def iterminal(cx, cy, title, icon):
        terminal(cx, cy, 7.6, 5.6, "")
        _handoff_icon(ax, icon, cx, cy + 1.3, s=2.4)
        label(cx, cy - 1.4, title, size=14, weight="bold")

    # title, section heading, colour key, columns
    label(1.5 - pad, 85.5, "SYSTEM ARCHITECTURE", size=38 * W / 24, weight="bold", ha="left")
    label(1.5 - pad, 79.2, "The triage mechanism: from a resident's photo to an engineer's decision", size=18 * W / 24,
          colour=FLOW_TEXT, weight="bold", ha="left")
    _flow_key(fig)
    for x in FLOW_SEPARATORS:
        ax.plot([x, x], [3.6, 60], color="#cfcfcf", lw=1.2, ls=(0, (5, 4)), zorder=0)
    edges = FLOW_SEPARATORS + [120.0]  # each numbered phase sits midway between its two dotted lines
    for cx, num, head in FLOW_COLUMNS:
        if num:
            mid = (edges[int(num) - 1] + edges[int(num)]) / 2
            ax.add_patch(plt.Circle((mid, 58.2), 1.45, color=FLOW_INK, zorder=4))
            label(mid, 58.2, num, size=14, colour="white", weight="bold")
            label(mid, 55.6, head.replace("\n", " · "), size=10.5)
        else:
            label(cx, 57.2, head, size=15, weight="bold")

    # step 0: report and dispatch, a written rule decides when a field worker visits (spec/features/dispatch.md, D27)
    ry = 71.0
    label(2.6, ry, "REPORT (0)\nresident's phone", size=12, weight="bold", rotation=90)
    ipill(11.5, ry, "CRACK\nREPORTED", "person")
    with_icon(photo_box, 26.5, ry, 13, 5, "CRACK PHOTO\n+ ANSWERS", "camera")
    status(26.5, ry + 2.5, "spec")
    line([(17.0, ry), (20.0, ry)])
    diamond(43.5, ry, 4.4, 3.2, "USABLE?")
    status(43.5, ry + 3.2, "partly")
    line([(33.0, ry), (39.1, ry)])
    line([(43.5, ry - 3.2), (43.5, ry - 4.9)], head=False, dot=False)
    mark(43.5, ry - 5.8, False)
    label(45.0, ry - 5.8, "RETAKE\n(after 2: V2)", size=10, weight="bold", ha="left")
    line([(47.9, ry), (55.9, ry)], dot=False)
    mark(50.2, ry, True, r=0.8)
    with_icon(model_box, 62.5, ry, 13, 5, "U-NET\nSEGMENTATION", "network", ink="white", paper=FLOW_MODEL)
    status(62.5, ry + 2.5, "spec")
    line([(69.0, ry), (82.0, ry)])
    with_icon(rule_box, 88.5, ry, 13, 5, "DISPATCH RULE", "rulebook", paper=FLOW_RULE)
    status(88.5, ry + 2.5, "code")
    outcomes = [(74.3, "V3 · NO VISIT YET\nrecheck in 1 month"), (71.0, "V2 · VISIT\nWITHIN 1 MONTH"),
                (67.7, "V1 · VISIT WITHIN 1 WEEK\nengineer told today")]
    for oy, text_ in outcomes:
        para(106.2, oy, 10.6, 2.9, text_, size=9)
        line([(95.0, ry), (100.6, oy)], dot=False)
    # V3: the resident's recheck is a new report
    line([(112.3, 74.3), (114.0, 74.3), (114.0, 77.0), (11.5, 77.0), (11.5, ry + 1.9)], ls="--", dot=False)
    label(47.5, 77.9, "V3: the recheck in 1 month is a new report", size=10, weight="bold")
    # V1 and V2: the field worker's first visit starts the three scales below
    for oy in (71.0, 67.7):
        line([(112.3, oy), (117.8, oy)], head=False, dot=False)
    line([(117.8, 71.0), (117.8, 61.8), (5.2, 61.8), (5.2, 46.0), (6.0, 46.0)], lw=2.4, dot=False)
    # the same visit also photographs every cracked element and every crack
    lane_y = {code: y for y, code, *_ in FLOW_LANES}
    ax.plot([5.2, 5.2], [46.0, lane_y["P"] + 0.9], color=FLOW_TEXT, lw=2.4, zorder=2)
    line([(5.2, lane_y["O"]), (6.0, lane_y["O"])], lw=2.4, dot=False)
    line([(5.2, lane_y["P"] + 0.9), (6.0, lane_y["P"] + 0.9)], lw=2.4, dot=False)
    label(62.0, 60.7, "V1 OR V2: A FIELD WORKER VISITS with a reliable phone and the scale card, and photographs "
          "the building once, every cracked element and every crack  (also 1 in 10 V3 reports, as an audit)",
          size=10.5, weight="bold")
    for r, (y, code, name, what, start, who) in enumerate(FLOW_LANES):
        label(2.6, y, f"{name} ({code})\n{what}", size=12, weight="bold", rotation=90)
        ipill(11.5, y, start, {"S": "house", "O": "column", "P": "crack"}[code])
        cap = PIPE_BOXES[(r, 0)]
        with_icon(photo_box, 26.5, y, 13, 5, cap[1].upper(), {"S": "clipboard", "O": "camera", "P": "ruler"}[code])
        status(26.5, y + 2.5, cap[3])
        line([(17.0, y), (20.0, y)])
        # check diamond and its failure branch
        question, fail = FLOW_CHECKS[code]
        diamond(43.5, y, 4.4, 3.4, question)
        status(43.5, y + 3.4, PIPE_BOXES[(r, 1)][3])
        line([(33.0, y), (39.1, y)])
        if fail:
            line([(43.5, y - 3.4), (43.5, y - 5.6)], head=False, dot=False)
            mark(43.5, y - 6.5, False)
            label(45.0, y - 6.5, fail, size=10.5, weight="bold", ha="left")
        line([(47.9, y), (55.5, y)], dot=False)
        mark(50.2, y, True, r=0.8)
        # perceive: learned model, then 'sure?' for the models that exist this semester
        per = PIPE_BOXES[(r, 2)]
        with_icon(model_box, 62.5, y, 13, 5, FLOW_MODELS[code], "network", ink="white", paper=FLOW_MODEL,
                  size_text=9.8)
        status(62.5, y + 2.5, per[3])
        dec = PIPE_BOXES[(r, 3)]
        if code == "S":
            line([(69.0, y), (82.0, y)])
        else:
            diamond(73.6, y, 2.9, 2.6, "SURE?")
            line([(69.0, y), (70.7, y)], dot=False)
            line([(73.6, y - 2.6), (73.6, y - 4.6)], head=False, dot=False)
            mark(73.6, y - 5.5, False)
            label(75.0, y - 5.5, "EXPERT\nNEEDED", size=10.5, weight="bold", ha="left")
            line([(76.5, y), (82.0, y)], dot=False)
            mark(78.4, y, True, r=0.8)
        with_icon(rule_box, 88.5, y, 13, 5, dec[1].upper(), "bars" if code == "S" else "rulebook", paper=FLOW_RULE)
        status(88.5, y + 2.5, dec[3])
        if (r, 4) in PIPE_BOXES:
            act = PIPE_BOXES[(r, 4)]
            out = "PRIORITY LIST\n+ DOSSIER" if code == "S" else "LEVEL, REASON,\nNEXT STEP"
            para(105.0, y, 9.4, 5, out)
            status(105.0, y + 2.5, act[3])
            iterminal(115.6, y, act[1].upper(), "hardhat" if code == "S" else "person")
            line([(95.0, y), (99.5, y)])
            line([(110.5, y), (111.9, y)], dot=False)

    yS, yO, yP = (lane[0] for lane in FLOW_LANES)
    # scene check band between Check and Perceive, all scales
    ax.add_patch(FancyBboxPatch((52.4, yP - 6.8), 2.6, yS - yP + 10.6, boxstyle="round,pad=0,rounding_size=0.8",
                                facecolor="white", edgecolor=FLOW_INK, lw=1.4, ls=(0, (4, 3)), zorder=3))
    label(53.7, (yS + yP) / 2 - 1.2, "SCENE CHECK (PHI-Net task 1): is this photo S, O or P?   now the guided "
          "capture sets it · later a scene model", size=9.6, weight="bold", rotation=90)
    # between the lanes: material + element down to the crack level; crack results up to the building score
    line([(82.0, yO - 1.6), (81.0, yO - 1.6), (81.0, yP + 1.6), (82.0, yP + 1.6)])
    label(80.5, yO - 8.6, "material,\nelement", size=10, weight="bold", ha="right")
    line([(95.0, yP + 1.4), (96.8, yP + 1.4), (96.8, yS - 1.4), (95.0, yS - 1.4)])
    label(97.3, yO + 0.2, "crack levels,\ngrowth, damage\nspread across\nelements", size=10, weight="bold", ha="left")
    # learning loop: engineer -> the learned models of phase 3
    ly = 51.6
    line([(115.6, yS + 2.8), (115.6, ly), (62.5, ly), (62.5, yS + 4.3)], ls="--", dot=False)
    label(89.0, ly + 1.5, "LEARNING LOOP: engineer corrections and on-site ratings become labels and retrain the "
          "models (phase 3)\neach new version is checked on held-out photos before use; written rules change only by "
          "a reviewed decision", size=10.5, weight="bold")
    # monitoring loop: the resident's recheck date brings the same crack back
    my = 7.9
    line([(115.6, yP - 2.8), (115.6, my), (4.2, my), (4.2, yP - 0.9), (6.0, yP - 0.9)], ls="--", dot=False)
    label(63.5, my - 1.8, "MONITORING LOOP: on the recheck date (Category 1: 6 months, Category 2: 1 month) the same crack ID "
          "and stickers are photographed again, with the last photo as an overlay\ngrowth larger than the measurement "
          "uncertainty raises the crack level, and through it the building risk score", size=10.5, weight="bold")

    # legend of shapes
    ly0, x = 1.9, 1.5
    items = [("pill", "START"), ("rect", "STEP"), ("diamond", "CHECK"), ("stack", "MODEL"),
             ("para", "RESULT SHOWN TO A PERSON"), ("term", "PERSON WHO ACTS"), ("ok", "CONTINUE"),
             ("no", "STOP HERE")]
    for kind, text_ in items:
        if kind == "pill":
            pill(x + 1.3, ly0, 2.6, 1.3, "")
        elif kind == "rect":
            photo_box(x + 1.3, ly0, 2.6, 1.5, "")
        elif kind == "diamond":
            diamond(x + 1.3, ly0, 1.3, 0.95, "")
        elif kind == "stack":
            ax.add_patch(FancyBboxPatch((x, ly0 - 0.75), 2.6, 1.3, boxstyle="round,pad=0,rounding_size=0.3",
                                        facecolor="#4a4a4a", edgecolor="white", lw=1.2, zorder=3))
            ax.add_patch(FancyBboxPatch((x, ly0 - 0.45), 2.6, 1.3, boxstyle="round,pad=0,rounding_size=0.3",
                                        facecolor=FLOW_MODEL, edgecolor="white", lw=1.2, zorder=4))
        elif kind == "para":
            para(x + 1.3, ly0, 1.8, 1.5, "")
        elif kind == "term":
            terminal(x + 1.3, ly0, 2.6, 1.5, "")
        else:
            mark(x + 1.3, ly0, kind == "ok", r=0.7)
        label(x + 3.2, ly0, text_, size=10, ha="left")
        x += 4.2 + 0.52 * len(text_)
    label(118.5, ly0, "Source: spec/features/pipeline.md, 27 Oct 2026 · thresholds illustrative until calibrated",
          size=9, colour=FLOW_SOFT, ha="right")
    return fig


# Figure 01b as a summary: the same rows (report, building, element, crack), phases, stops and loops, with one
# symbol and one or two words per box. Same colour logic: white = photos and people, grey = written rules,
# black = learned models. Statuses, thresholds and the long notes stay in 01b.
SIMPLE_COLS = {"cap": 52, "chk": 88, "per": 124, "dec": 162, "act": 206}
SIMPLE_HEADS = [("cap", "1", "CAPTURE"), ("chk", "2", "CHECK"), ("per", "3", "PERCEIVE"), ("dec", "4", "DECIDE"),
                ("act", "5", "ACT")]
SIMPLE_ROWS = [  # (y, pill icon, pill word, capture, check (word, stop), model (word, stop), decide, act)
    (67, "house", "BUILDING", ("clipboard", "QUESTIONS"), ("ANSWERED?", None), ("FACADE\nMODEL (later)", None),
     ("bars", "BUILDING\nSCORE"), ("hardhat", "ENGINEER")),
    (46, "column", "ELEMENT", ("camera", "ELEMENT\nPHOTO"), ("CARD +\nANGLE?", "RETAKE"),
     ("MATERIAL +\nELEMENT", "EXPERT NEEDED"), ("rulebook", "ELEMENT\nRULES"), None),
    (25, "crack", "CRACK", ("ruler", "CLOSE-UP"), ("SHARP?", "RETAKE"), ("U-NET +\nPATTERN", "EXPERT NEEDED"),
     ("rulebook", "CATEGORY\n1 · 2 · 3"), ("person", "RESIDENT")),
]


def pipeline_simple():
    """Figure 01b in summary: report and dispatch, then building, element and crack lanes; stops and loops."""
    fig = plt.figure(figsize=(24, 13.5), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])  # units of 0.1 inch
    ax.set_xlim(0, 240)
    ax.set_ylim(0, 135)
    ax.set_axis_off()
    t = _flow_tools(ax)
    K, GREY, C = FLOW_INK, FLOW_RULE, SIMPLE_COLS
    w, h = 26, 13

    def arrow(points, ls="-", lw=2.4):
        xs, ys = zip(*points)
        ax.plot(xs[:-1], ys[:-1], color=K, lw=lw, ls=ls, zorder=2, solid_capstyle="round")
        ax.annotate("", xy=points[-1], xytext=points[-2], zorder=2,
                    arrowprops=dict(arrowstyle="-|>", color=K, lw=lw, ls="-", shrinkA=0, shrinkB=0,
                                    mutation_scale=20))

    def card(cx, cy, kind, icon, word):
        fill, edge, ink = {"photo": ("white", K, K), "rule": (GREY, GREY, K), "model": (K, K, "white")}[kind]
        if kind == "model":  # stacked, as in 01b
            ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2 - 1.0), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                        facecolor="#4a4a4a", edgecolor="white", lw=1.4, zorder=3))
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                    facecolor=fill, edgecolor=edge, lw=2.0, zorder=4))
        _handoff_icon(ax, icon, cx - w / 2 + 5.5, cy, s=7.5, ink=ink, paper=fill)
        t.label(cx + 3.6, cy, word, size=13, colour=ink, weight="bold")

    def check(cx, cy, word):
        ax.add_patch(matplotlib.patches.Polygon([(cx - 10, cy), (cx, cy + 6.8), (cx + 10, cy), (cx, cy - 6.8)],
                                                closed=True, facecolor=GREY, edgecolor="none", zorder=4))
        t.label(cx, cy, word, size=12, weight="bold")

    def stop(cx, top, word):
        ax.plot([cx, cx], [top, top - 2.4], color=K, lw=2.0, zorder=2)
        t.mark(cx, top - 4.0, False, r=1.6)
        t.label(cx + 2.4, top - 4.0, word, size=12, weight="bold", ha="left")

    def start(cy, icon, word):
        t.pill(19, cy, 26, 10, "")
        _handoff_icon(ax, icon, 11.5, cy, s=6.5)
        t.label(22.5, cy, word, size=17, weight="bold")

    t.label(2.4, 130.0, "SYSTEM ARCHITECTURE", size=38, weight="bold", ha="left")
    t.label(2.4, 119.5, "The triage mechanism in summary: from a resident's photo to an engineer's decision", size=18,
            colour=FLOW_TEXT, weight="bold", ha="left")
    _flow_key(fig)
    for key, num, word in SIMPLE_HEADS:
        ax.add_patch(plt.Circle((C[key] - 6.5, 111.5), 2.2, color=K, zorder=4))
        t.label(C[key] - 6.5, 111.5, num, size=15, colour="white", weight="bold")
        t.label(C[key] - 3.4, 111.5, word, size=15, weight="bold", ha="left")

    # row 0: the resident reports a crack; a written rule decides if and when a field worker visits (D27)
    ry = 96
    start(ry, "person", "REPORT")
    card(C["cap"], ry, "photo", "camera", "CRACK\nPHOTO")
    check(C["chk"], ry, "USABLE?")
    stop(C["chk"], ry - 6.8, "RETAKE")
    card(C["per"], ry, "model", "network", "U-NET")
    card(C["dec"], ry, "rule", "rulebook", "DISPATCH")
    for x0, x1 in (("pill", "cap"), ("cap", "chk"), ("chk", "per"), ("per", "dec")):
        a = 32 if x0 == "pill" else C[x0] + (10 if x0 == "chk" else w / 2)
        b = C[x1] - (10 if x1 == "chk" else w / 2)
        arrow([(a, ry), (b - 0.6, ry)])
    visits = [(ry + 6, "V3 · RECHECK IN 1 MONTH"), (ry, "V2 · VISIT IN 1 MONTH"), (ry - 6, "V1 · VISIT IN 1 WEEK")]
    for vy, text_ in visits:
        t.para(C["act"], vy, 24, 4.6, text_, size=10.5)
        arrow([(C["dec"] + w / 2, ry), (C["act"] - 13.4, vy)], lw=1.8)
    # V3: the resident's recheck is a new report
    arrow([(C["act"] + 13, ry + 6), (233, ry + 6), (233, 106.5), (19, 106.5), (19, ry + 5.6)], ls=(0, (4, 3)), lw=2.0)
    # V1 and V2: a field worker visits and photographs the building, every cracked element and every crack
    for vy in (ry, ry - 6):
        ax.plot([C["act"] + 12.6, 236.5], [vy, vy], color=K, lw=2.4, zorder=2)
    ax.plot([236.5, 236.5, 3.5, 3.5], [ry, 81.5, 81.5, 25], color=K, lw=2.8, zorder=2, solid_joinstyle="round")
    for y, *_ in SIMPLE_ROWS:
        arrow([(3.5, y), (5.4, y)], lw=2.8)
    ax.add_patch(FancyBboxPatch((36, 79.2), 48, 4.6, boxstyle="round,pad=0,rounding_size=2.3", facecolor="white",
                                edgecolor=K, lw=2.0, zorder=5))
    _handoff_icon(ax, "worker", 40, 81.5, s=4.0)
    t.label(62, 81.5, "FIELD VISIT  (V1, V2)", size=13, weight="bold")

    # rows 1-3: building, element, crack
    for y, icon, word, cap, chk, per, dec, act in SIMPLE_ROWS:
        start(y, icon, word)
        card(C["cap"], y, "photo", *cap)
        check(C["chk"], y, chk[0])
        if chk[1]:
            stop(C["chk"], y - 6.8, chk[1])
        card(C["per"], y, "model", "network", per[0])
        if per[1]:
            stop(C["per"], y - 7.5, per[1])
        card(C["dec"], y, "rule", *dec)
        arrow([(32, y), (C["cap"] - w / 2 - 0.6, y)])
        arrow([(C["cap"] + w / 2, y), (C["chk"] - 10.6, y)])
        arrow([(C["chk"] + 10, y), (C["per"] - w / 2 - 0.6, y)])
        arrow([(C["per"] + w / 2, y), (C["dec"] - w / 2 - 0.6, y)])
        if act:
            card(C["act"], y, "photo", *act)
            arrow([(C["dec"] + w / 2, y), (C["act"] - w / 2 - 0.6, y)])
    yB, yE, yC = (row[0] for row in SIMPLE_ROWS)
    # between the lanes: material and element feed the crack category; crack categories feed the building score
    arrow([(C["dec"] - 6, yE - h / 2), (C["dec"] - 6, yC + h / 2 + 0.6)], lw=2.0)
    t.label(C["dec"] - 7.5, (yE + yC) / 2, "material,\nelement", size=10.5, weight="bold", ha="right")
    arrow([(C["dec"] + w / 2, yC + 3), (180, yC + 3), (180, yB - 3), (C["dec"] + w / 2 + 0.6, yB - 3)], lw=2.0)
    t.label(181.5, yE, "crack\ncategories", size=10.5, weight="bold", ha="left")

    # loops: learning (engineer -> models) and monitoring (resident -> the same crack again)
    arrow([(C["act"], yB + h / 2), (C["act"], 75.5), (C["per"], 75.5), (C["per"], yB + h / 2 + 0.6)],
          ls=(0, (4, 3)), lw=2.0)
    t.label(165, 77.4, "LEARN: corrections retrain the models", size=12, weight="bold")
    arrow([(C["act"], yC - h / 2), (C["act"], 7.5), (19, 7.5), (19, yC - 5.6)], ls=(0, (4, 3)), lw=2.0)
    t.label(70, 9.6, "RECHECK: the same crack again; growth raises the category", size=12, weight="bold")

    t.label(237.6, 2.6, "Summary of figure 01b (statuses, scene check, thresholds). Never a safety clearance. "
            "Times illustrative until calibrated.", size=10, colour=FLOW_SOFT, ha="right")
    return fig


# Figure 01d: the two-minute explanation of the pipeline, drawn as a pipeline with its own words.
# Same colour logic as 01b and 01c: white = photos and people, grey = written rules, black = learned models.
def pipeline_story():
    """The spoken pipeline in three rows: report, field visit, decision; stops, categories and loops."""
    fig = plt.figure(figsize=(24, 13.5), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])  # units of 0.1 inch
    ax.set_xlim(0, 240)
    ax.set_ylim(0, 135)
    ax.set_axis_off()
    t = _flow_tools(ax)
    K, GREY = FLOW_INK, FLOW_RULE
    styles = {"photo": ("white", K, K), "rule": (GREY, GREY, K), "model": (K, K, "white")}

    def arrow(points, ls="-", lw=2.4):
        xs, ys = zip(*points)
        ax.plot(xs[:-1], ys[:-1], color=K, lw=lw, ls=ls, zorder=2, solid_capstyle="round")
        ax.annotate("", xy=points[-1], xytext=points[-2], zorder=2,
                    arrowprops=dict(arrowstyle="-|>", color=K, lw=lw, ls="-", shrinkA=0, shrinkB=0,
                                    mutation_scale=20))

    def card(cx, cy, w, h, kind, icon, head, body, num=None):
        fill, edge, ink = styles[kind]
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.4",
                                    facecolor=fill, edgecolor=edge, lw=2.0, zorder=4))
        left = cx - w / 2
        if icon:
            _handoff_icon(ax, icon, left + 5.2, cy, s=6.5, ink=ink, paper=fill)
            left += 9.6
        else:
            left += 2.4
        t.label(left, cy + h / 2 - 3.6, head, size=15, colour=ink, weight="bold", ha="left")
        t.label(left, cy + h / 2 - 7.0, body, size=12, colour=ink, ha="left", va="top")
        if num:
            ax.add_patch(plt.Circle((cx - w / 2, cy + h / 2), 2.3, color=K, zorder=6))
            t.label(cx - w / 2, cy + h / 2, num, size=13, colour="white", weight="bold", zorder=7)

    def chip(cx, cy, w, h, kind, text, size=11.5, bold=True):
        fill, edge, ink = styles[kind]
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.0",
                                    facecolor=fill, edgecolor=edge, lw=1.6, zorder=4))
        t.label(cx, cy, text, size=size, colour=ink, weight="bold" if bold else "normal")

    def stop(cx, top, word, why):
        ax.plot([cx, cx], [top, top - 2.4], color=K, lw=2.0, zorder=2)
        t.mark(cx, top - 4.2, False, r=1.8)
        t.label(cx + 2.8, top - 4.2, word, size=14, weight="bold", ha="left")
        t.label(cx + 2.8 + 1.3 * len(word) + 1.5, top - 4.2, why, size=12, colour=FLOW_TEXT, ha="left")

    _house_title(fig, "SYSTEM ARCHITECTURE")
    t.label(2.4, 119.5, "Second Look in plain words. The system never says \"safe\": it decides who looks next, and how "
            "soon.", size=18, colour=FLOW_TEXT, weight="bold", ha="left")
    _flow_key(fig)

    # row 1: the report
    y1, h1 = 97, 24
    card(25, y1, 40, h1, "photo", "person", "RESIDENT", "Notices a crack, opens the app,\ntakes one photo and answers a\n"
         "few simple questions. Never\nasked the material, the element\nor the type of crack.", "1")
    card(67, y1, 34, h1, "rule", "check", "PHOTO CHECK", "Blurred or too dark?\nWe never guess from\na bad photo.", "2")
    stop(67, y1 - h1 / 2, "RETAKE", "")
    card(105, y1, 26, h1, "model", "network", "MODEL", "Finds the\ncrack.", "3")
    card(144, y1, 36, h1, "rule", "rulebook", "WRITTEN RULE", "Not the AI: decides\nwhether someone\nneeds to go.", "4")
    arrow([(45.6, y1), (49.4, y1)])
    arrow([(84.6, y1), (91.4, y1)])
    arrow([(118.6, y1), (125.4, y1)])
    visits = [(106.5, "V1", "a field worker within a week,\nthe engineer is told today"),
              (97, "V2", "a visit within a month"),
              (87.5, "V3", "no visit yet; the resident\nrechecks in a month")]
    for vy, code_, text_ in visits:
        ax.add_patch(FancyBboxPatch((180, vy - 4.0), 54, 8.0, boxstyle="round,pad=0,rounding_size=1.0",
                                    facecolor="white", edgecolor=K, lw=1.6, zorder=4))
        t.label(182.2, vy, code_, size=15, weight="bold", ha="left")
        t.label(190, vy, text_, size=11.5, ha="left")
        arrow([(162.6, y1), (179.4, vy)], lw=1.8)
    t.label(203, 80.8, "We can't send someone to every crack, so this rule decides where people go first.", size=11.5,
            colour=FLOW_TEXT, style="italic")

    # row 2: the field visit (V1 or V2)
    y2, h2 = 58, 24
    ax.plot([234.6, 237.5, 237.5], [106.5, 106.5, 76.5], color=K, lw=2.6, zorder=2)
    ax.plot([234.6, 237.5], [97, 97], color=K, lw=2.6, zorder=2)
    arrow([(237.5, 76.5), (25, 76.5), (25, y2 + h2 / 2 + 2.6)], lw=2.6)
    t.label(130, 78.3, "V1 OR V2", size=12, weight="bold")
    card(25, y2, 40, h2, "photo", "worker", "FIELD WORKER", "Brings a reliable phone and\na scale card, and looks at\n"
         "three scales:", "5")
    for cx, icon, head, body in ((70, "house", "BUILDING", "A short case\nfile, answered\nonce."),
                                 (104, "column", "ELEMENT", "A photo of each\ncracked beam,\ncolumn or wall."),
                                 (138, "crack", "CRACK", "A close-up\nof each one.")):
        card(cx, y2, 30, h2, "photo", icon, head, body)
    arrow([(45.6, y2), (54.4, y2)])
    arrow([(85.6, y2), (88.4, y2)])
    arrow([(119.6, y2), (122.4, y2)])
    arrow([(153.6, y2), (163.4, y2)])
    ax.add_patch(FancyBboxPatch((164, y2 - h2 / 2), 70, h2, boxstyle="round,pad=0,rounding_size=1.4",
                                facecolor="white", edgecolor=K, lw=2.0, ls=(0, (4, 3)), zorder=3))
    t.label(199, y2 + 7.5, "EVERY SCALE, THE SAME FIVE STEPS", size=14, weight="bold")
    for k, (kind, word) in enumerate((("photo", "CAPTURE"), ("rule", "CHECK"), ("model", "PERCEIVE"),
                                      ("rule", "DECIDE"), ("photo", "ACT"))):
        chip(171.5 + k * 13.75, y2 - 2.5, 12.0, 8.0, kind, word, size=10.5)

    # row 3: perceive, decide, act, and the engineer at the end
    y3, h3 = 24, 22
    arrow([(199, y2 - h2 / 2), (199, 40), (31, 40), (31, y3 + h3 / 2 + 0.6)], lw=2.6)
    card(31, y3, 46, h3, "model", "network", "MODELS PERCEIVE", "They find the crack, measure\nits width in mm and "
         "recognise\nits pattern, each with a\nconfidence.", "6")
    stop(14, y3 - h3 / 2, "EXPERT NEEDED", "")
    t.label(16.8, 5.2, "if a model is unsure: never \"fine\"", size=12, colour=FLOW_TEXT, ha="left")
    card(80, y3, 40, h3, "rule", "rulebook", "DECISION", "A written, auditable\nrisk matrix gives\nCategory 1, 2 or 3.",
         "7")
    arrow([(54.6, y3), (59.4, y3)])
    cats = [(31.0, "CATEGORY 1 · monitor", "photo"), (24.0, "CATEGORY 2 · low-cost advice + recheck", "rule"),
            (17.0, "CATEGORY 3 · an engineer must confirm", "model")]
    for cy, text_, kind in cats:
        chip(129, cy, 44, 6.0, kind, text_, size=10.5)
        arrow([(100.6, y3), (104, y3), (104, cy), (106.4, cy)], lw=1.8)
    card(176, y3, 38, h3, "rule", "bars", "BUILDING SCORE", "Crack categories add\nup to a building score:\n"
         "the engineer's\npriority list.", "8")
    arrow([(151.6, y3), (156.4, y3)])
    card(218, y3, 36, h3, "photo", "hardhat", "ENGINEER", "Confirms every\nCategory 3 and\nevery Expert\nneeded, on site.",
         "9")
    arrow([(195.6, y3), (199.4, y3)])

    # learn: the engineer's corrections go back to the models (dashed)
    arrow([(218, y3 - h3 / 2), (218, 4.2), (47, 4.2), (47, y3 - h3 / 2 - 0.6)], ls=(0, (4, 3)), lw=2.2)
    t.label(132, 6.6, "LEARN: every correction the engineer makes retrains the models, for better accuracy and "
            "precision", size=12.5, weight="bold")
    # recheck: categories 1 and 2 bring the same crack back to the resident (dashed)
    arrow([(129, 34.0), (129, 37.9), (2.6, 37.9), (2.6, y1), (4.4, y1)], ls=(0, (4, 3)), lw=2.2)
    t.label(4.8, 43.2, "RECHECK: the same crack is photographed again; if it grows, its category goes up", size=11.5,
            weight="bold", ha="left")
    return fig


FIGURES = {
    "01_pipeline.png": pipeline,
    "01b_pipeline_flow.png": pipeline_flow,
    "01c_pipeline_simple.png": pipeline_simple,
    "01d_pipeline_story.png": pipeline_story,
    "02_decisions_and_players.png": decisions_and_players,
    "03_risk_matrix.png": risk_matrix,
    "04_segmentation_tree.png": segmentation_tree,
    "05_data_coverage.png": data_coverage,
    "06_dataset_sizes.png": dataset_sizes,
    "07_confusion_template.png": confusion_template,
    "09_timeline.png": timeline,
    "10_dataset_comparison.png": dataset_comparison,
}


def main(out_dir: Path = Path(__file__).resolve().parents[2] / "figures") -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, make in FIGURES.items():
        fig = make()
        path = out_dir / name
        fig.savefig(path, dpi=SLIDE_PX / fig.get_figwidth())  # 16:9 figures come out at 4800 x 2700 px
        plt.close(fig)
        paths.append(path)
    return paths


if __name__ == "__main__":
    for p in main():
        print(p)
