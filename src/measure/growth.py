"""Has the crack grown since the last photo? (D99) Spec: spec/features/crack_kinematics.md ("Growth").

E. Corres Sojo's thesis [46] (ch. 5.3, D23): a change counts as growth only when it is larger than the measurement's
uncertainty; photos should be taken in similar conditions. Both photos must be measured with a bank card (millimetres):
pixels of two photos taken from different distances cannot be compared. The two uncertainties are combined as
independent errors (root of the sum of squares): an assumption. Shown only (rule 5)."""

from __future__ import annotations

import math


def compare(before: dict | None, now: dict, before_number: int | None = None) -> dict:
    """before, now: two results (to_dict). Returns {grown, change_mm, uncertainty_mm, ..., reason}; grown None when the
    two cannot be compared."""
    tag = f"#{before_number}" if before_number is not None else "the last photo"
    out = {"before": before_number, "grown": None, "change_mm": None, "uncertainty_mm": None}
    if before is None:
        out["reason"] = f"Result {tag} is no longer on the wall: nothing to compare with."
        return out
    if not (before.get("measured") and now.get("measured")):
        out["reason"] = (f"Not compared with {tag}: both photos need a bank card next to the crack, so both widths are "
                         "in millimetres. Pixels of two photos from different distances cannot be compared.")
        return out
    b, n = before["crack"], now["crack"]
    change = n["width_max_mm"] - b["width_max_mm"]
    u = math.hypot(b["width_uncertainty_mm"], n["width_uncertainty_mm"])
    out.update(change_mm=round(change, 3), uncertainty_mm=round(u, 3), width_before_mm=round(b["width_max_mm"], 3),
               width_now_mm=round(n["width_max_mm"], 3), length_before_mm=b.get("length_main_mm"),
               length_now_mm=n.get("length_main_mm"))
    if change > u:
        out["grown"] = True
        out["reason"] = (f"Grown since {tag}: widest {b['width_max_mm']:.2f} → {n['width_max_mm']:.2f} mm, "
                         f"+{change:.2f} mm, more than the measurement's error (± {u:.2f} mm). An engineer should look.")
    elif change < -u:
        out["grown"] = False
        out["reason"] = (f"Narrower than {tag} by {-change:.2f} mm, more than the error (± {u:.2f} mm): more likely a "
                         "different spot, light or distance than a closing crack. Retake both in the same way.")
    else:
        out["grown"] = False
        out["reason"] = (f"No change beyond the measurement's error since {tag}: widest {b['width_max_mm']:.2f} → "
                         f"{n['width_max_mm']:.2f} mm ({change:+.2f} mm, error ± {u:.2f} mm). Take the next photo in "
                         "the same light, from the same distance.")
    return out
