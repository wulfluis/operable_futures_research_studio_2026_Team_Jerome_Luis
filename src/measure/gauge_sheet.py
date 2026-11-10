"""Printable width gauge, A4: four corner markers, a 100 mm bar, lines of known width and a frame for a bank card.
Spec: spec/features/measurement.md ("Width check").

Usage, from the repository root (writes a PNG, a PDF and a truth file at 600 dpi):
    py src/measure/gauge_sheet.py docs/width_gauge

Print at 100 % ("actual size", never "fit to page") and check the bar with a ruler. Every line is a whole number of
printer pixels wide, so its true printed width is known exactly (the truth file); ink spread adds a few hundredths
of a millimetre (assumption: not measured). The four ArUco markers let the check find every line and the sheet's own
scale in a photo, independently of the bank card. Photograph the sheet with a dark bank card inside the frame, flat.
"""

import json
import sys
from pathlib import Path

import cv2
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

SHEET_MM = (210.0, 297.0)  # A4
DPI = 600
LINES_MM = (0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0)  # nominal widths; 0.3, 1 and 3 mm are band edges of the risk matrix
LINE_X_MM, LINE_LEN_MM = 20.0, 90.0
FIRST_LINE_Y_MM, LINE_STEP_MM = 78.0, 17.0
BAR_X_MM, BAR_Y_MM, BAR_MM = 55.0, 42.0, 100
CARD_FRAME_MM = (35.0, 215.0, 35.0 + 85.6, 215.0 + 53.98)  # a bank card, ISO/IEC 7810 ID-1
DICTIONARY = cv2.aruco.DICT_4X4_50
MARKER_MM = 16.0
MARKER_ORIGINS_MM = {0: (8.0, 8.0), 1: (186.0, 8.0), 2: (186.0, 273.0), 3: (8.0, 273.0)}  # clockwise from top left
VERSION = "v2"


def marker_corners_mm() -> dict[int, list[tuple[float, float]]]:
    """Each marker's four corners on the sheet in mm, in OpenCV's order (top left, top right, bottom right, bottom left)."""
    s = MARKER_MM
    return {i: [(x, y), (x + s, y), (x + s, y + s), (x, y + s)] for i, (x, y) in MARKER_ORIGINS_MM.items()}


def printed_px(nominal_mm: float, dpi: int = DPI) -> int:
    return max(1, round(nominal_mm * dpi / 25.4))


def printed_width_mm(nominal_mm: float, dpi: int = DPI) -> float:
    """The width that really comes out of the printer: a whole number of device pixels."""
    return printed_px(nominal_mm, dpi) * 25.4 / dpi


def _font(size_px: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal"))
    return ImageFont.truetype(path, size_px)


def render(dpi: int = DPI) -> Image.Image:
    k = dpi / 25.4  # pixels per mm
    img = Image.new("L", (round(SHEET_MM[0] * k), round(SHEET_MM[1] * k)), 255)
    d = ImageDraw.Draw(img)
    px = lambda mm: round(mm * k)  # noqa: E731
    aruco = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    for i, (x, y) in MARKER_ORIGINS_MM.items():
        img.paste(Image.fromarray(cv2.aruco.generateImageMarker(aruco, i, px(MARKER_MM))), (px(x), px(y)))
    d.text((px(30), px(10)), f"SECOND LOOK  ·  WIDTH GAUGE {VERSION}", fill=0, font=_font(px(5.5)))
    d.text((px(30), px(19)), "Print at 100 % (actual size, not fit to page). The bar must measure 100 mm.", fill=70,
           font=_font(px(2.8), False))
    d.text((px(30), px(25)), "Photograph it flat and upright, a dark bank card in the frame, 20 to 40 cm away.", fill=70,
           font=_font(px(2.8), False))
    # the 100 mm bar: blocks of 10 mm, ticks every 1 mm, labels every 10 mm
    for j in range(10):
        d.rectangle([px(BAR_X_MM + 10 * j), px(BAR_Y_MM), px(BAR_X_MM + 10 * j + 10) - 1, px(BAR_Y_MM + 2.5)],
                    fill=0 if j % 2 == 0 else 255, outline=0, width=max(1, px(0.15)))
    for t in range(BAR_MM + 1):
        tick = 4.0 if t % 10 == 0 else 2.5 if t % 5 == 0 else 1.5
        d.line([px(BAR_X_MM + t), px(BAR_Y_MM + 2.5), px(BAR_X_MM + t), px(BAR_Y_MM + 2.5 + tick)], fill=0,
               width=max(1, px(0.15)))
        if t % 10 == 0:
            f = _font(px(2.4))
            d.text((px(BAR_X_MM + t) - f.getlength(str(t)) / 2, px(BAR_Y_MM + 7.5)), str(t), fill=0, font=f)
    # lines of known width, each a whole number of printer pixels
    for i, nominal in enumerate(LINES_MM):
        y = FIRST_LINE_Y_MM + i * LINE_STEP_MM
        n = printed_px(nominal, dpi)
        top = px(y) - n // 2
        d.rectangle([px(LINE_X_MM), top, px(LINE_X_MM + LINE_LEN_MM) - 1, top + n - 1], fill=0)
        d.text((px(LINE_X_MM + LINE_LEN_MM + 5), px(y) - px(1.8)), f"{nominal:g} mm", fill=0, font=_font(px(3.6)))
    # the frame for a bank card, dashed
    x0, y0, x1, y1 = CARD_FRAME_MM
    dash, gap, w = 4.0, 3.0, max(1, px(0.3))
    for (ax, ay, bx, by) in ((x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)):
        length = max(abs(bx - ax), abs(by - ay))
        pos = 0.0
        while pos < length:
            end = min(pos + dash, length)
            f0, f1 = pos / length, end / length
            d.line([px(ax + (bx - ax) * f0), px(ay + (by - ay) * f0), px(ax + (bx - ax) * f1), px(ay + (by - ay) * f1)],
                   fill=0, width=w)
            pos += dash + gap
    d.text((px(x0 + 6), px((y0 + y1) / 2) - px(3)), "PUT A BANK CARD HERE, FLAT", fill=110, font=_font(px(3.2)))
    d.text((px(30), px(280)), "Printed line widths are exact in printer pixels (truth file); ink spread adds", fill=110,
           font=_font(px(2.4), False))
    d.text((px(30), px(284.5)), "a little. Not a calibrated instrument.", fill=110, font=_font(px(2.4), False))
    return img


def truth(dpi: int = DPI) -> dict:
    return {"version": VERSION, "dpi": dpi, "sheet_mm": list(SHEET_MM), "bar_mm": BAR_MM,
            "card_frame_mm": [round(v, 2) for v in CARD_FRAME_MM],
            "markers": {"dictionary": "DICT_4X4_50", "size_mm": MARKER_MM,
                        "origins_mm": {str(i): list(v) for i, v in MARKER_ORIGINS_MM.items()}},
            "lines": {f"{n:g}": round(printed_width_mm(n, dpi), 4) for n in LINES_MM},
            "note": "nominal width -> width in mm that the printer lays down at this dpi (whole device pixels)"}


def save(folder: Path, dpi: int = DPI) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    img = render(dpi)
    png, pdf, js = (folder / f"width_gauge_{VERSION}.png", folder / f"width_gauge_{VERSION}.pdf",
                    folder / f"width_gauge_{VERSION}_truth.json")
    img.save(png, dpi=(dpi, dpi))
    img.convert("RGB").save(pdf, resolution=dpi)  # the PDF keeps the physical size
    js.write_text(json.dumps(truth(dpi), indent=1), encoding="utf-8")
    return [png, pdf, js]


if __name__ == "__main__":
    for p in save(Path(sys.argv[1] if len(sys.argv) > 1 else "docs/width_gauge")):
        print(p)
