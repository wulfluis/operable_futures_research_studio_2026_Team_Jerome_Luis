"""Which crack widths a phone can see and measure at which distance (spec/features/capture_visibility.md).

A planning calculation from camera geometry, for midterm action item 7. Camera values are assumptions (typical
phones); the pixel rule is the measurement's own (MIN_PX_ACROSS, D24). Never a statement about safety.

    python3 src/measure/visibility.py            # table in the terminal
    python3 src/measure/visibility.py --markdown # the table for docs/experiments
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from measure.crack import GOOD_PX_ACROSS, MIN_PX_ACROSS  # noqa: E402

FRAME_LONG_MM = 34.62  # long side of a 4:3 photo within the 43.27 mm diagonal of the 35 mm-equivalent frame
CARD_MM = 85.6  # ISO/IEC 7810 ID-1, the bank card's long side
CARD_MIN_SHARE = 0.012  # smallest card the card bridge searches for (D59, element.locate.LOOK_MIN_SHARE)
SEEN_PX = 1.0  # assumption: a dark, clean crack on a light wall can show at about one pixel

# (name, 35 mm-equivalent focal length in mm, pixels along the long side): typical values, assumptions
CAMERAS = [("Main 12 MP", 26.0, 4032), ("Main 48 MP", 26.0, 8064), ("Tele 2x 12 MP", 52.0, 4032),
           ("Tele 3x 12 MP", 77.0, 4032)]
DISTANCES_M = [0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]

# widths that mean something: our matrix bands and the engineering classes (spec "Width references")
REFERENCES_MM = [(0.2, "JBDPA class I/II"), (0.3, "matrix band"), (1.0, "matrix band, JBDPA II/III"),
                 (1.6, "FEMA 306 1/16 in."), (2.0, "JBDPA III/IV"), (3.0, "matrix band"),
                 (3.2, "FEMA 306 1/8 in.")]


@dataclass(frozen=True)
class Visibility:
    camera: str
    distance_m: float
    mm_per_px: float
    seen_mm: float  # about one pixel (assumption)
    measurable_mm: float  # MIN_PX_ACROSS pixels across
    good_mm: float  # GOOD_PX_ACROSS pixels across
    card_px: float
    card_found: bool  # the card bridge (D59) searches down to CARD_MIN_SHARE of the long side

    def finest_reference(self) -> str:
        """The finest reference width that can still be measured, or none."""
        ok = [(w, n) for w, n in REFERENCES_MM if w >= self.measurable_mm]
        return f"{ok[0][0]:g} mm ({ok[0][1]})" if ok else "none of the references"


def mm_per_px(distance_m: float, focal_eq_mm: float, px_long: int) -> float:
    """Size of one pixel on a wall held square to the camera."""
    return distance_m * 1000.0 * FRAME_LONG_MM / focal_eq_mm / px_long


def visibility(camera: str, focal_eq_mm: float, px_long: int, distance_m: float) -> Visibility:
    s = mm_per_px(distance_m, focal_eq_mm, px_long)
    card_px = CARD_MM / s
    return Visibility(camera, distance_m, s, SEEN_PX * s, MIN_PX_ACROSS * s, GOOD_PX_ACROSS * s, card_px,
                      card_px >= CARD_MIN_SHARE * px_long)


def table(cameras=CAMERAS, distances=DISTANCES_M) -> list[Visibility]:
    return [visibility(name, f, px, d) for name, f, px in cameras for d in distances]


def markdown(rows: list[Visibility]) -> str:
    head = ("| Camera | Distance | mm per px | Seen from (about 1 px) | Measurable from (3 px) | Good from (6 px) "
            "| Card length | Card bridge finds it | Finest reference width measurable |\n"
            "|---|---|---|---|---|---|---|---|---|\n")
    return head + "".join(
        f"| {r.camera} | {r.distance_m:g} m | {r.mm_per_px:.3f} | {r.seen_mm:.2f} mm | {r.measurable_mm:.2f} mm "
        f"| {r.good_mm:.2f} mm | {r.card_px:.0f} px | {'yes' if r.card_found else 'no'} | {r.finest_reference()} |\n"
        for r in rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--markdown", action="store_true")
    rows = table()
    if ap.parse_args().markdown:
        print(markdown(rows))
        return
    for r in rows:
        print(f"{r.camera:14s} {r.distance_m:5.1f} m  {r.mm_per_px:6.3f} mm/px  seen {r.seen_mm:5.2f} mm  "
              f"measurable {r.measurable_mm:5.2f} mm  card {r.card_px:6.0f} px {'found' if r.card_found else '-'}")


if __name__ == "__main__":
    main()
