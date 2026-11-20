"""The crack to explore (D83): its numbers come from the measurement, never invented; mm only with a bank card."""
from types import SimpleNamespace

import numpy as np
from PIL import Image

from app.crack_view import mask_image, view_data, viewer_html


def _crack(widths, mm_per_px=0.1, unc=0.05):
    n = len(widths)
    ys = np.linspace(100, 900, n)
    prof = np.column_stack([np.full(n, 500.0), ys])
    edges = np.stack([prof - [2, 0], prof + [2, 0]], axis=1)
    return SimpleNamespace(found=True, profile_px=prof, profile_mm=np.asarray(widths, float), profile_edges_px=edges,
                           main_path_px=prof.copy(), px_across=4.0, mm_per_px=mm_per_px, width_uncertainty_mm=unc,
                           length_main_mm=80.0)


def _result(crack, measured=True, pixel=None):
    return SimpleNamespace(crack=crack, measured=measured, pixel_crack=pixel, card=None,
                           photo=Image.new("RGB", (1000, 1000)))


def test_widest_and_narrowest_come_from_the_profile():
    w = [0.05] + [0.4] * 20 + [0.25] + [0.4] * 20 + [0.9] + [0.4] * 20 + [0.05]  # thin tips at both ends
    d = view_data(_result(_crack(w)))
    assert d["unit"] == "mm" and d["widest"]["w"] == 0.9 and d["widest"]["u"] == 0.05
    assert d["narrowest"]["w"] == 0.25  # the tips, where every crack thins to nothing, are left out
    assert d["length"] == 80.0  # the measured length, as in the reason line
    assert 0 <= d["widest"]["x"] <= 1 and len(d["widest"]["e"]) == 2


def test_no_card_means_pixels_never_mm():
    px = _crack([3.0] * 30, mm_per_px=None)
    d = view_data(_result(None, measured=False, pixel=px))
    assert d["unit"] == "px" and d["mm_per_x"] is None


def test_untrusted_card_width_is_shown_in_pixels():
    d = view_data(_result(_crack([0.2] * 30, mm_per_px=0.1), measured=False))
    assert d["unit"] == "px" and abs(d["widest"]["w"] - 2.0) < 1e-6


def test_nothing_to_draw():
    assert view_data(_result(None, measured=False)) is None


def test_page_cannot_be_ended_by_data():
    html = viewer_html(b"\xff\xd8", {"unit": "mm", "line": [], "note": "</script><b>"})
    assert "</script><b>" not in html


def test_mask_drops_the_text_band():
    im = Image.new("RGB", (100, 140))
    im.info["photo_height"] = 100
    assert mask_image(im).size == (100, 100)


def test_camera_angle_is_90_minus_the_tilt():
    from app.crack_view import camera_angle
    card = SimpleNamespace(tilt_deg=4.2, tilt_left_right_deg=-3.0, tilt_up_down_deg=2.5, status="straight")
    assert camera_angle(card) == {"to_wall": 85.8, "left_right": 3.0, "up_down": 2.5, "status": "straight"}


def test_shape_row_carries_the_measured_shape_and_says_when_it_is_in_the_photo():
    c = _crack([0.4] * 30)
    c.angle_to_vertical_deg, c.straightness, c.coherence, c.n_loops, c.branch_ratio = 28.04, 0.861, 0.523, 0, 1.0
    d = view_data(_result(c))
    assert d["shape"] == {"angle": 28.0, "lean": d["shape"]["lean"], "in_photo": False, "straightness": 0.86,
                          "coherence": 0.52, "loops": 0, "branches": 1.0}
    px = _crack([3.0] * 30, mm_per_px=None)
    px.angle_to_vertical_deg, px.n_loops = 10.0, 3
    assert view_data(_result(None, measured=False, pixel=px))["shape"]["in_photo"] is True  # no card: the photo's angle
    assert view_data(_result(_crack([0.4] * 30)))["shape"] is None  # no angle measured: no row
    assert "Closed cells" in viewer_html(b"", d)  # the chip explains what a cell is
