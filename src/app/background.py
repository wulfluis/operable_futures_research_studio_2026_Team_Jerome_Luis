"""The app's background: one big white crack, drawn as an SVG (no photo, so no licence and sharp at any size).
Spec: spec/features/class_demo.md."""

import base64
import html
import random

W, H = 1600, 1000  # viewBox; the image is scaled to cover the page
CRACK_LIGHTNESS = 0.5  # the white crack is drawn at this share of its opacity, so it stays in the background
FLASH_RED = "#d62728"  # the red of the crack overlay on the photos
FLASH_SPEED = 500  # viewBox units per second along the crack
FLASH_DASH = 160  # viewBox units, length of the red flash
FLASH_PAUSE = 2.5  # s of plain white crack between two flashes
MODEL_LINE_INSET = 1.1  # vw, distance of the model line from the screen edges
PAGE = "#0b0b0b"  # the page colour, for the fades at both ends of the crack
TOP_FADE = (2, 14)  # vh: the crack is hidden down to the first, fully shown from the second


def _jag(a: tuple[float, float], b: tuple[float, float], rough: float, rng: random.Random, depth: int = 7) -> list:
    """Midpoint displacement: a straight line made crack-like, sharper at small scales."""
    if depth == 0:
        return [a, b]
    (x1, y1), (x2, y2) = a, b
    dx, dy = x2 - x1, y2 - y1
    length = (dx * dx + dy * dy) ** 0.5 or 1
    shift = rng.uniform(-rough, rough) * length
    mid = ((x1 + x2) / 2 - dy / length * shift, (y1 + y2) / 2 + dx / length * shift)
    return _jag(a, mid, rough, rng, depth - 1)[:-1] + _jag(mid, b, rough, rng, depth - 1)


def crack_lines(seed: int = 7) -> list[tuple[list, float, float]]:
    """(points, stroke width, opacity): the main crack, its branches and hairlines."""
    rng = random.Random(seed)
    main = _jag((W * 0.22, -20), (W * 0.58, H + 20), 0.12, rng, depth=8)
    lines = [(main, 6.0, 0.7)]
    fork_at = len(main) // 4  # a second strand that splits off near the top, as in a real crack
    fx, fy = main[fork_at]
    lines.append((_jag((fx, fy), (fx + W * 0.10, -20), 0.14, rng, depth=7), 3.5, 0.6))
    step = (len(main) - 40) // 6  # branches spread along the crack, not bunched
    for i in range(30, len(main) - 10 - step // 2, step):
        x, y = main[i + rng.randrange(step // 2)]
        angle = rng.choice((-1, 1)) * rng.uniform(0.35, 1.1)
        reach = rng.uniform(90, 300)
        end = (x + reach * angle, y + reach * rng.uniform(0.3, 0.9))
        branch = _jag((x, y), end, 0.2, rng, depth=6)
        lines.append((branch, rng.uniform(1.6, 2.8), 0.55))
        if rng.random() < 0.6:  # a hairline off the branch
            j = rng.randrange(len(branch) // 3, len(branch))
            bx, by = branch[j]
            lines.append((_jag((bx, by), (bx + rng.uniform(-120, 120), by + rng.uniform(40, 140)), 0.25, rng, 5), 1.0, 0.45))
    return lines


def _lengths(points: list) -> list[float]:
    """Distance along the line to each point."""
    out = [0.0]
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        out.append(out[-1] + ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
    return out


def flash_times(lines: list) -> list[tuple[float, float]]:
    """(start, travel) in seconds of the red flash on each line: it runs down the main crack at FLASH_SPEED
    and enters each branch or hairline when it reaches the point the branch leaves from."""
    reached: list[tuple[float, float, float]] = []  # (x, y, time the flash is there)
    times = []
    for points, _, _ in lines:
        x0, y0 = points[0]
        start = min(reached, key=lambda p: (p[0] - x0) ** 2 + (p[1] - y0) ** 2)[2] if reached else 0.0
        along = _lengths(points)
        reached += [(x, y, start + d / FLASH_SPEED) for (x, y), d in zip(points, along)]
        times.append((start, along[-1] / FLASH_SPEED))
    return times


def crack_svg(seed: int = 7, flash: bool = True) -> str:
    """The white crack; with flash, a red dash runs along every line in turn and the crack goes white again."""
    lines = crack_lines(seed)
    times = flash_times(lines)
    cycle = max(s + t for s, t in times) + FLASH_PAUSE
    paths = []
    for (points, width, opacity), (start, travel) in zip(lines, times):
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in points)
        paths.append(f'<path d="{d}" fill="none" stroke="#ffffff" stroke-width="{width:.1f}" '
                     f'stroke-opacity="{opacity * CRACK_LIGHTNESS:.2f}" stroke-linejoin="miter" stroke-linecap="round"/>')
        if flash:  # the dash waits before the line's start, runs to past its end, then waits for the next cycle
            length = _lengths(points)[-1]
            dash = FLASH_DASH
            run = (f'stroke-linejoin="miter" stroke-linecap="round" stroke-dasharray="{dash} {length + dash:.0f}" '
                   f'stroke-dashoffset="{dash}"><animate attributeName="stroke-dashoffset" '
                   f'values="{dash};{-length:.0f};{-length:.0f}" keyTimes="0;{(travel + dash / FLASH_SPEED) / cycle:.4f};1" '
                   f'dur="{cycle:.2f}s" begin="{start:.2f}s" repeatCount="indefinite"/></path>')
            paths.append(f'<path d="{d}" fill="none" stroke="{FLASH_RED}" stroke-width="{width + 1:.1f}" {run}')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid slice">'
            + "".join(paths) + "</svg>")


def model_line(text: str) -> str:
    """The text stands still in one line along the bottom of the screen, from the left edge to the right edge:
    the letter size follows from the letter count (assumption: a monospace letter is at most 0.6 em wide),
    and the last line is justified so the words spread to fill what is left. A band in the page colour behind it,
    fading out at its top, hides the crack where it runs under the letters, so the line always reads cleanly."""
    size_vw = (100 - 2 * MODEL_LINE_INSET) / (len(text) * 0.6)
    css = ("<style>.wall-model { position: fixed; left: 0; right: 0; bottom: 0; z-index: 40; "
           f"padding: {2 * MODEL_LINE_INSET}vw {MODEL_LINE_INSET}vw {MODEL_LINE_INSET}vw; "
           f"background: linear-gradient(to bottom, {PAGE}00, {PAGE}f2 55%); "
           "pointer-events: none; white-space: nowrap; text-align: justify; text-align-last: justify; "
           f"font: 700 {size_vw:.3f}vw/1 Consolas, 'Courier New', monospace; "
           f"color: rgba(255, 255, 255, {0.7 * CRACK_LIGHTNESS:.2f}); }}</style>")  # the same grey as the main crack
    return f'{css}<div class="wall-model" aria-label="{html.escape(text)}">{html.escape(text.upper())}</div>'


UNDERLINE_W, UNDERLINE_H, UNDERLINE_Y = 1600, 80, 70  # viewBox, and the height of the line in it
UNDERLINE_REM = 3.4  # height of the drawn strip; the branches rise into the bottom of the buttons above it


def crack_underline_svg(colour: str = "#ffffff", seed: int = 3, parts: int = 4, opacity: float = 1.0) -> str:
    """One continuous cracked line near the bottom, with short hairlines and one branch rising toward each of the
    `parts` buttons above it: off the button's centre, each by a different amount, so they are not symmetrical."""
    rng = random.Random(seed)
    w, h, y = UNDERLINE_W, UNDERLINE_H, UNDERLINE_Y
    knots = [(4, y)] + [(x + rng.uniform(-2, 2), y + rng.uniform(-3, 3)) for x in range(30, w - 20, 30)] + [(w - 4, y)]
    line = [knots[0]]
    for p, q in zip(knots, knots[1:]):  # sharp in short pieces
        line += _jag(p, q, 0.18, rng, depth=3)[1:]
    paths = [(line, 2.2)]
    for _ in range(w // 160):  # hairlines off the line, up or down, as off a real crack
        x, y0 = line[rng.randrange(6, len(line) - 6)]
        end = (x + rng.uniform(-25, 25), y0 + rng.choice((-1, 1)) * rng.uniform(6, h - y - 2))
        paths.append((_jag((x, y0), end, 0.25, rng, 3), 1.0))
    piece = w / parts
    for i in range(parts):  # a branch up toward each button, inside that button's piece of the line
        side = 1 if i % 2 else -1
        x0 = piece * (i + 0.5) + side * piece * rng.uniform(0.08, 0.3)
        y0 = min(line, key=lambda pt: abs(pt[0] - x0))[1]
        tip = (x0 + side * rng.uniform(15, 60), rng.uniform(4, 20))
        paths.append((_jag((x0, y0), tip, 0.2, rng, 4), 1.6))
    body = "".join(f'<path d="M{" L".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="none" stroke="{colour}" '
                   f'stroke-opacity="{opacity:.2f}" stroke-width="{width}" vector-effect="non-scaling-stroke" '
                   'stroke-linejoin="miter"/>' for pts, width in paths)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" preserveAspectRatio="none">'
            f"{body}</svg>")


def crack_underline_css(selector: str, parts: int, lit: str) -> str:
    """One cracked line across the element, cut into `parts` equal pieces (child divs) that line up seamlessly.
    Piece n turns red while the selector `lit.format(n=n)` matches an ancestor (for example a hovered button).
    The strip is pulled up so its branches reach the buttons; it never takes the clicks meant for them."""
    def url(colour: str, opacity: float = 1.0) -> str:
        data = base64.b64encode(crack_underline_svg(colour, parts=parts, opacity=opacity).encode("utf-8")).decode("ascii")
        return f'url("data:image/svg+xml;base64,{data}")'
    grey = url("#ffffff", 0.7 * CRACK_LIGHTNESS)  # the same grey as the main crack
    lift = UNDERLINE_REM * UNDERLINE_Y / UNDERLINE_H  # the line itself stays where the element starts
    css = (f"{selector} {{ display: flex; height: {UNDERLINE_REM}rem; margin-top: -{lift:.2f}rem; "
           "pointer-events: none; position: relative; z-index: 1; }"
           f"{selector} > div {{ flex: 1; background: {grey} 0 0 / {parts * 100}% 100% no-repeat; }}")
    red = url(FLASH_RED)
    for n in range(1, parts + 1):  # each piece shows its own share of the one long line
        css += (f"{selector} > div:nth-child({n}) {{ background-position: {(n - 1) / (parts - 1) * 100:.4f}% 0; }}"
                f"{lit.format(n=n)} {selector} > div:nth-child({n}) {{ background-image: {red}; }}")
    return css


def background_css(seed: int = 7) -> str:
    """The crack behind the page. A fade in the page colour over its top end, like the band behind the model line
    at the bottom, so the crack grows out of the dark instead of being cut off at the screen edge."""
    data = base64.b64encode(crack_svg(seed).encode("utf-8")).decode("ascii")
    return (f'.stApp {{ background-color: {PAGE}; background-image: '
            f'linear-gradient(to bottom, {PAGE} 0, {PAGE} {TOP_FADE[0]}vh, {PAGE}00 {TOP_FADE[1]}vh), '
            f'url("data:image/svg+xml;base64,{data}"); '
            "background-size: 100% 100%, cover; background-position: center; background-attachment: fixed; }")
