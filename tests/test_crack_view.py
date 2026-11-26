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


def test_no_card_means_no_width_shown_and_never_pixels():
    """D116: pixels stay inside the app; without a scale no width number is shown, only relative colours."""
    px = _crack([3.0] * 30, mm_per_px=None)
    d = view_data(_result(None, measured=False, pixel=px))
    assert d["unit"] == "none" and d["mm_per_x"] is None and d["length"] is None
    assert d["scale"]["source"] is None
    html = viewer_html(b"", d)
    assert 'return "no scale"' in html and '" px"' not in html


def test_untrusted_card_width_is_not_shown():
    d = view_data(_result(_crack([0.2] * 30, mm_per_px=0.1), measured=False))
    assert d["unit"] == "none" and abs(d["widest"]["w"] - 2.0) < 1e-6  # relative only, for the colours


def test_a_door_in_photo_2_gives_estimated_mm_with_its_error():
    """D116: no card, a door's scale: ≈ mm, the widest point's error includes the scale's share."""
    px = _crack([3.0] * 30, mm_per_px=None)
    r = _result(None, measured=False, pixel=px)
    r.scale_estimate = {"source": "door", "name": "a door in photo 2", "mm_per_px": 0.2, "rel": 0.16,
                        "basis": "door assumed", "crack": {"uncertainty_mm": 0.3}}
    d = view_data(r)
    assert d["unit"] == "mm" and d["approx"] is True and abs(d["widest"]["w"] - 0.6) < 1e-6
    assert d["widest"]["u"] == 0.3 and d["scale"] == {"source": "door", "name": "a door in photo 2", "rel": 0.16,
                                                      "measured": False, "basis": "door assumed"}
    assert d["mm_per_x"] == 200.0


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
    # D117: the angle is read on the traced line itself (vertical here), against the photo without photo 2
    assert d["shape"] == {"angle": 0.0, "lean": d["shape"]["lean"], "in_photo": True, "straightness": 0.86,
                          "coherence": 0.52, "loops": 0, "branches": 1.0, "ref": d["ref"]["source"],
                          "from_horizontal": False}
    px = _crack([3.0] * 30, mm_per_px=None)
    px.angle_to_vertical_deg, px.n_loops = 10.0, 3
    assert view_data(_result(None, measured=False, pixel=px))["shape"]["in_photo"] is True  # no card: the photo's angle
    assert view_data(_result(_crack([0.4] * 30)))["shape"] is None  # no angle measured: no row
    assert "Closed cells" in viewer_html(b"", d)  # the chip explains what a cell is


def _branchy(angle_deg: float):
    """A straight crack at angle_deg from vertical (in the photo) with one side branch."""
    t = np.radians(angle_deg)
    n = 60
    s = np.linspace(-300, 300, n)
    prof = np.column_stack([500 + s * np.sin(t), 500 + s * np.cos(t)])
    c = _crack([0.4] * n)
    c.profile_px, c.main_path_px = prof, prof.copy()
    c.profile_edges_px = np.stack([prof - [2, 0], prof + [2, 0]], axis=1)
    side = np.column_stack([np.linspace(500, 800, 20), np.full(20, 500.0)])  # horizontal branch from the middle
    c.branch_paths_px = [prof.copy(), side]
    return c


def test_every_segment_has_its_own_angle_and_branches_are_numbered_from_the_centre():
    """D117: segments with their angles, branches each with its angle and number, counted from the centre."""
    d = view_data(_result(_branchy(30.0)))
    br = d["cracks"][0]["branches"]
    assert len(br) == 2 and sorted(b["n"] for b in br) == [1, 2]
    main, side = br
    assert abs(main["angle"] - 30.0) < 1.0 and abs(side["angle"] - 90.0) < 1.0
    assert len(main["segs"]) >= 5 and all(abs(sg[2] - 30.0) < 2.0 for sg in main["segs"])
    assert main["length"] is not None and side["length"] is not None  # mm with the card


def test_on_a_beam_the_angle_is_read_from_horizontal():
    r = _result(_branchy(0.0))
    r.triage = {"element": "beam"}
    d = view_data(r)
    assert d["ref"]["from_horizontal"] is True and abs(d["cracks"][0]["branches"][0]["angle"] - 90.0) < 1.0


def test_against_photo_2_the_angle_follows_the_room_vertical():
    """D117: a crack vertical in photo 1 but with the phone turned 40°: photo 2's room vertical gives the angle."""
    import cv2
    from app.crack_view import reference
    r = _result(_branchy(0.0))
    th = np.radians(40.0)
    rot = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    r.link_map, r.link_down, r.link_k = (rot, 1.0, 1.0), np.array([0.0, 1.0]), 1.0
    to_ref, down, horizontal, source = reference(r)
    assert source.startswith("photo 2")
    d = view_data(r)
    assert abs(d["cracks"][0]["branches"][0]["angle"] - 40.0) < 1.0
