"""Deck backgrounds: the app's crack, with the red reaching further along it on each slide (slide 2 to 10).
The red spreads like the app's flash: down the main crack, into each branch when it reaches the branch.
Like the app, both ends of the crack fade into the black instead of stopping at the slide edge; crack.svg (cover and
last slide) is the app's crack with its flash, faded the same way."""
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\User\operable_futures_research_studio_2026_Team_Jerome_Luis\src")
from app.background import CRACK_LIGHTNESS, FLASH_RED, H, W, _lengths, crack_lines, crack_svg, flash_times

OUT = Path(__file__).parent / "assets"
lines = crack_lines(7)
times = flash_times(lines)
total = max(s + t for s, t in times)

# A 1920 x 1080 slide shows y 50 to 950 of the 1600 x 1000 crack (cover, cropped top and bottom). The crack is
# hidden in the first and last 2 % of that height and fully shown from 14 %, as in the app.
SHOWN, HIDE, FADE = (50, 950), 0.02, 0.14


def fade_ends(svg: str) -> str:
    top, bottom = SHOWN
    stops = [(top + HIDE * (bottom - top), 0), (top + FADE * (bottom - top), 1),
             (bottom - FADE * (bottom - top), 1), (bottom - HIDE * (bottom - top), 0)]
    grad = "".join(f'<stop offset="{y / H:.4f}" stop-color="#fff" stop-opacity="{o}"/>' for y, o in stops)
    head, body = svg.split(">", 1)
    body = body.rsplit("</svg>", 1)[0]
    return (f'{head}><defs><linearGradient id="ends" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="0" y2="{H}">'
            f'{grad}</linearGradient><mask id="fade" maskUnits="userSpaceOnUse" x="-100" y="-100" '
            f'width="{W + 200}" height="{H + 200}"><rect x="-100" y="-100" width="{W + 200}" height="{H + 200}" '
            f'fill="url(#ends)"/></mask></defs><g mask="url(#fade)">{body}</g></svg>')


(OUT / "crack.svg").write_text(fade_ends(crack_svg(7)), encoding="utf-8")

N = 12  # slides in the deck
for slide in range(2, N + 1):
    reach = total * (slide - 1) / (N - 1)  # last slide: the whole crack is red
    paths = []
    for (pts, width, opacity), (start, travel) in zip(lines, times):
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        paths.append(f'<path d="{d}" fill="none" stroke="#ffffff" stroke-width="{width:.1f}" '
                     f'stroke-opacity="{opacity * CRACK_LIGHTNESS:.2f}" stroke-linejoin="miter" stroke-linecap="round"/>')
        share = min(max((reach - start) / travel, 0), 1)
        if share > 0:
            length = _lengths(pts)[-1]
            paths.append(f'<path d="{d}" fill="none" stroke="{FLASH_RED}" stroke-width="{width + 1:.1f}" '
                         f'stroke-linejoin="miter" stroke-linecap="round" '
                         f'stroke-dasharray="{length * share:.1f} {length + 10:.0f}"/>')
    svg = fade_ends(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid slice">'
                    + "".join(paths) + "</svg>")
    (OUT / f"crack12_{slide:02d}.svg").write_text(svg, encoding="utf-8")
    print(slide, len(svg))
