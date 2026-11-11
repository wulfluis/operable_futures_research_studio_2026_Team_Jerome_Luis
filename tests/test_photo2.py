"""What the second photo adds (spec/features/photo2.md, D61): the crack's place and extent on the element, spalling,
and the level from both photos. No model file is needed."""

import re

import numpy as np
import pytest

from app.core import Wall, auto_triage, run_measured
from element import photo2, spalling
from element.photo2 import CrackOnElement
from gh3d import layout
from test_app_measure import DEMO_MODEL, _demo
from triage.element_rule import ElementAnswer


@pytest.fixture
def measured():
    r = run_measured(_demo(1.0)[0], DEMO_MODEL, card_placed=True)
    r.element = ElementAnswer("column", False, 0.95, "")
    return r


def test_place_in_element_coordinates():
    path = np.array([[0, 99], [50, 49], [99, 0]], float)  # bottom-left... a diagonal across a 100 x 100 photo
    on = photo2.place(path, (100, 100))
    assert on.found and on.path_uv[0] == [0.0, 0.0] and on.path_uv[-1] == [1.0, 1.0]  # v up
    assert on.extent == pytest.approx(1.0) and on.centre_uv == pytest.approx((0.5, 0.5), abs=0.01)


@pytest.mark.parametrize("centre, element, expected", [
    ((0.5, 0.9), "column", "end_zone"), ((0.5, 0.1), "load_bearing_wall", "end_zone"),
    ((0.5, 0.5), "column", "middle"), ((0.1, 0.5), "beam", "end_zone"), ((0.5, 0.9), "beam", "middle")])
def test_end_zone_or_middle_per_element(centre, element, expected):
    on = CrackOnElement(True, [[0, 0], [1, 1]], 0.5, centre)
    assert on.position(element) == expected


def test_no_crack_on_photo_2_means_unknown():
    on = photo2.place(None, (100, 100))
    assert not on.found and on.position("column") == "unknown" and "not known" in on.reason


def test_photo_2_position_reaches_the_matrix(measured):
    measured.test = True  # a TEST case: photo 2 alone may say "middle" (D73)
    measured.on_element = CrackOnElement(True, [[0.5, 0.4], [0.5, 0.6]], 0.2, (0.5, 0.5))
    auto_triage(measured)
    assert measured.triage["position"] == "middle" and "photo of the element" in measured.triage["why"]


def test_a_real_upload_gets_middle_only_from_the_linked_photos(measured):
    """D73: photo 2 alone assumes the element fills the photo, too weak to lower a real level; an end zone counts."""
    measured.on_element = CrackOnElement(True, [[0.5, 0.4], [0.5, 0.6]], 0.2, (0.5, 0.5))
    auto_triage(measured)
    assert measured.triage["position"] == "unknown"
    measured.on_element = CrackOnElement(True, [[0.5, 0.85], [0.5, 0.95]], 0.1, (0.5, 0.9))
    auto_triage(measured)
    assert measured.triage["position"] == "end_zone"
    measured.on_element = CrackOnElement(True, [[0.5, 0.4], [0.5, 0.6]], 0.2, (0.5, 0.5))
    measured.location = {"position": "middle", "reason": "both ends seen"}  # the strict link (D66)
    auto_triage(measured)
    assert measured.triage["position"] == "middle" and measured.triage["position_reason"] == "both ends seen"


def test_spalling_on_photo_2_makes_the_type_spalling(measured):
    measured.spalling, measured.spalling_p = True, 0.93
    auto_triage(measured)
    assert measured.triage["pattern"] == "spalling_exposed_bar"
    assert "Spalling" in measured.triage["why"] and measured.triage["level"] in ("Category 2", "Category 3")


def test_no_spalling_model_raises_nothing(tmp_path):
    assert spalling.load(tmp_path / "none.pt") is None
    assert spalling.seen(None, None) == (None, None)


def test_spalling_threshold_catches_the_target_share():
    p = np.array([0.95, 0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.15, 0.05])
    c = spalling.choose_spall_min(p, np.ones(10, bool), target=0.8)
    assert np.mean(p >= c) >= 0.8 and np.mean(p >= c + 0.05) < 0.8


def test_the_3d_model_uses_photo_2_place_and_proportion(measured):
    measured.on_element = CrackOnElement(True, [[0.3, 0.2], [0.4, 0.5], [0.35, 0.8]], 0.6, (0.35, 0.5))
    auto_triage(measured)
    m = layout.build({"number": 1, **measured.to_dict()})
    assert m["placed_from_photo2"]
    line = np.asarray(next(o for o in m["objects"] if o["name"] == "crack_line")["points"])
    (dx, _, dz), z0 = layout.SIZES["column"]
    assert line[:, 2].min() == pytest.approx(z0 + 0.2 * dz) and line[:, 2].max() == pytest.approx(z0 + 0.8 * dz)
    assert line[:, 0].min() == pytest.approx((0.3 - 0.5) * dx)  # 60 % of the element's height, as on photo 2
    assert any("photo 2" in s for s in m["label"]) and not any(o["name"] == "card" for o in m["objects"])


def test_without_photo_2_crack_the_model_is_as_before(measured):
    m = layout.build({"number": 1, **measured.to_dict()})
    assert not m["placed_from_photo2"] and any(o["name"] == "card" for o in m["objects"])


def test_never_safe_with_both_photos(measured):
    measured.on_element = CrackOnElement(True, [[0.5, 0.1], [0.5, 0.9]], 0.8, (0.5, 0.5))
    measured.spalling, measured.spalling_p = False, 0.1
    auto_triage(measured, Wall())
    assert not re.search(r"\bsafe\b|clearance|permit", str(measured.to_dict()), re.I)


# ---------- D74: every branch, and the element's own outline as the frame ----------

def test_every_branch_of_a_crack_is_kept():
    m = np.zeros((200, 200), bool)
    m[20:180, 99:101] = True  # a vertical crack ...
    m[100:102, 100:180] = True  # ... with a branch to the right
    branches = photo2.trace_network(m, 10)
    assert len(branches) >= 2
    on = photo2.place(branches[0], (200, 200), None, branches)
    assert len(on.network_uv) == len(branches)
    xs = [p[0] for b in on.network_uv for p in b]
    assert max(xs) > 0.85  # the branch reaches the right, not only the main line


def test_the_crack_is_read_relative_to_the_element_outline():
    path = np.array([[50, 25], [50, 75]], float)  # middle of a 100 x 100 photo
    on = photo2.place(path, (100, 100), frame=(0.0, 0.0, 1.0, 0.5))  # the element is only the top half
    assert on.path_uv[0] == pytest.approx([0.5, 0.5], abs=0.02)  # from its middle ...
    assert on.path_uv[-1] == pytest.approx([0.5, 0.0], abs=0.02)  # ... down to its bottom edge (clipped)
    whole = photo2.place(path, (100, 100))
    assert whole.centre_uv[1] == pytest.approx(0.5, abs=0.01) and on.centre_uv[1] < 0.3


def test_the_3d_draws_every_branch():
    stats = {"number": 1, "triage": {"element": "column", "level": "Category 2"},
             "estimate": {"width_max_mm": 5, "width_median_mm": 4, "uncertainty_mm": 1, "length_mm": 300},
             "photo2": {"crack_on_element": {"found": True, "path_uv": [[0.5, 0.2], [0.5, 0.8]],
                                             "network_uv": [[[0.5, 0.2], [0.5, 0.8]], [[0.5, 0.5], [0.9, 0.5]]],
                                             "frame_source": "SAM 3.1's outline of the column"}}}
    m = layout.build(stats)
    assert sum(o["name"] == "crack_line" for o in m["objects"]) == 2
    assert "SAM 3.1's outline" in " ".join(m["label"])


# ---------- D75: photo 2's crack is photo 1's crack, carried by the link ----------

def test_not_linked_a_real_photo_never_takes_photo_2s_own_guess(measured):
    from PIL import Image

    from app.core import crack_from_link
    measured.on_element = photo2.place(np.array([[10, 10], [10, 90]], float), (100, 100))  # e.g. a plant's leaf
    measured.location = {"linked": False}
    crack_from_link(measured, Image.new("RGB", (100, 100)))
    assert not measured.on_element.found and "not known" in measured.on_element.reason


def test_a_test_case_without_a_card_keeps_photo_2s_reading_when_not_linked(measured):
    from PIL import Image

    from app.core import crack_from_link
    measured.card, measured.test = None, True
    measured.on_element = photo2.place(np.array([[10, 10], [10, 90]], float), (100, 100))
    measured.location = {"linked": False}
    crack_from_link(measured, Image.new("RGB", (100, 100)))
    assert measured.on_element.found


def test_no_estimate_when_photo_1_has_a_card(measured):
    from app.core import estimate_from_photo2
    measured.test = True
    measured.crack = None  # e.g. a tilted card: nothing measured
    measured.on_element = CrackOnElement(True, [[0.5, 0.2], [0.5, 0.8]], 0.6, (0.5, 0.5), size_px=(624, 832),
                                         width_px_max=10.0, width_px_median=8.0, length_px=500.0)
    assert measured.card is not None and estimate_from_photo2(measured) is None
