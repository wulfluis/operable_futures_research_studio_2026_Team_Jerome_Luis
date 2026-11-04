"""Real photos of each crack pattern of the risk matrix (figure 03), landscape, in the same design.

Spec: spec/features/figures.md. Run from the repository root:  py src/viz/pattern_examples.py
Needs PHI-Net in data/raw/phinet (src/experiments/phinet_prepare.py); downloads the Wikimedia Commons photos into
data/processed/pattern_examples/commons on first run. Writes figures/02b_real_examples.png (shown right before
the risk matrix), which is git-ignored:
the photos keep their own licences (PHI-Net CC BY-NC-SA 4.0; Commons as listed below), so the figure is not published.
"""

import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from viz.figures import PATTERN_WORD, _icon

ROOT = Path(__file__).resolve().parents[2]
PHINET = ROOT / "data" / "raw" / "phinet"
COMMONS_DIR = ROOT / "data" / "processed" / "pattern_examples" / "commons"
OUT = ROOT / "figures" / "02b_real_examples.png"
T3, T8 = "task3_spalling_condition", "task8_damage_type"

# Wikimedia Commons files: key -> (file name, author, licence)
COMMONS = {
    "asr_pillar": ("ASR concrete pillar National Gallery of Canada 02.jpg", "Shinkolobwe", "CC BY-SA 4.0"),
    "asr_barrier": ("ASR cracks concrete step barrier FHWA 2006.jpg", "US FHWA", "public domain"),
    "rust_crack": ("Père-Lachaise - Division 65 - Pagandet-Leclerc 11.jpg", "Coyau", "CC BY-SA 3.0"),
    "bar_exposed": ("Betonkorrosion.jpg", "Mailtosap", "public domain"),
    "imperial_column": ("Imperial County Services building 1979.tif", "USGS", "public domain"),
    "burst_column": ("Burst column.JPG", "unknown author", "public domain"),
    "rust_soffit": ("Rust-reo-steel.jpg", "Bill Bradley", "CC BY 2.5"),
    "bridge_spalling": ("Betonkorrosion unter Autobahnbruecke (01).JPG", "Karl-Heinz Wellmann", "CC BY-SA 3.0"),
}
# Columns follow the rows of the risk matrix. Photo: ("commons", key) or (PHI-Net task, image index).
# basis: an engineer label from PHI-Net, or the team's pick (not yet confirmed by an engineer).
EXAMPLES = [
    ("surface_map", [("commons", "asr_pillar"), ("commons", "asr_barrier"), (T3, 5374)],
     "clear examples + team pick"),
    ("flexural", [(T8, 1433), (T8, 1562), (T8, 1195)], "engineer label"),
    ("diagonal_shear", [(T8, 4048), (T8, 3825), (T8, 3393)], "engineer label"),
    ("along_bar", [("commons", "rust_crack"), ("commons", "bar_exposed"), (T3, 6004)], "team pick, to confirm"),
    ("column_horizontal", [("commons", "imperial_column"), (T8, 1253), (T8, 1611)], "engineer label + team pick"),
    ("spalling_exposed_bar", [("commons", "burst_column"), ("commons", "rust_soffit"), ("commons", "bridge_spalling")],
     "clear examples"),
    ("joint", [(T8, 61)], "team pick, to confirm"),
]
HEAD = {"User-Agent": "SecondLook-studio-research/0.1 (IAAC student project)"}
K = "#111111"


def commons_photo(key):
    name, author, licence = COMMONS[key]
    path = COMMONS_DIR / f"{key}.jpg"
    if not path.exists():
        COMMONS_DIR.mkdir(parents=True, exist_ok=True)
        url = "https://commons.wikimedia.org/wiki/Special:FilePath/" + urllib.parse.quote(name) + "?width=1000"
        time.sleep(1)
        path.write_bytes(urllib.request.urlopen(urllib.request.Request(url, headers=HEAD), timeout=60).read())
    return np.asarray(Image.open(path).convert("RGB")), f"Commons · {author} · {licence}"


_cache = {}


def phinet_photo(task, i):
    if task not in _cache:
        _cache[task] = np.load(PHINET / task / "X_train.npy", mmap_mode="r", allow_pickle=False)
    return np.asarray(_cache[task][i]), f"PHI-Net · {task.split('_')[0]} #{i}"


def square(img):
    h, w = img.shape[:2]
    s = min(h, w)
    return img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]


def pattern_examples():
    fig = plt.figure(figsize=(24, 13.5))
    ax = fig.add_axes([0, 0, 1, 1])  # drawing layer in units of 0.1 inch
    ax.set_xlim(0, 240)
    ax.set_ylim(0, 135)
    ax.set_axis_off()
    ax.text(2.4, 128.5, "CRACK FEATURES AND TYPES", fontsize=38, weight="bold", color=K, va="center")
    ax.text(2.4, 121.5, "Real photos of each crack pattern in the risk matrix (figure 03). 'Engineer label': PHI-Net's "
            "label; 'team pick': chosen by the team, to confirm with the engineer.", fontsize=13, color="#444444",
            va="center")
    col_w, size, top = 33.6, 30.0, 100.0
    for k in range(1, len(EXAMPLES)):  # a separator line between patterns
        x = 2.4 + col_w * k
        ax.plot([x, x], [6.5, 117.5], color="#8a8a8a", lw=1.6, zorder=0)
    for k, (pattern, photos, basis) in enumerate(EXAMPLES):
        xc = 2.4 + col_w * (k + 0.5)
        ax.add_patch(FancyBboxPatch((xc - 15, 105.5), 30, 10, boxstyle="round,pad=0,rounding_size=0.8",
                                    facecolor="white", edgecolor="#cfcfcf", lw=1.2))
        _icon(ax, pattern, xc - 9.5, 110.5, s=7.5)
        ax.text(xc - 4.3, 110.5, PATTERN_WORD[pattern].upper(), fontsize=17, weight="bold", color=K, va="center")
        ax.text(xc, 103.0, basis, fontsize=10.5, color="#444444", ha="center", va="center", style="italic")
        for r, (src, ref) in enumerate(photos):
            img, caption = commons_photo(ref) if src == "commons" else phinet_photo(src, ref)
            y0 = top - r * (size + 1.2) - size
            pax = fig.add_axes([(xc - size / 2) / 240, y0 / 135, size / 240, size / 135])
            pax.imshow(square(img))
            pax.set_axis_off()
            pax.text(0.02, 0.03, caption, transform=pax.transAxes, fontsize=7.5, color="white",
                     bbox=dict(facecolor=K, edgecolor="none", pad=2))
    ax.text(2.4, 2.6, "Photos: Wikimedia Commons, each under the licence printed on it (links: data/processed/"
            "pattern_examples/credits.md); PHI-Net (PEER, UC Berkeley; Gao & Mosalam [53][55]), CC BY-NC-SA 4.0, "
            "research and teaching only. Not for publication.", fontsize=9.5, color="#555555", va="center")
    return fig


def main(out: Path = OUT) -> Path:
    fig = pattern_examples()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(main())
