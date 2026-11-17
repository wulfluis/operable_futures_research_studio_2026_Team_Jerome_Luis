"""Photo 1 seen three more ways (D81): width heatmap (bank card only), confidence heatmap, enhanced view; the width
all along the crack, kept by the measurement."""

import numpy as np
from PIL import Image

from app import views
from app.core import Wall, run_measured
from test_app_measure import DEMO_MODEL, _demo


def test_a_measured_crack_keeps_its_width_all_along():
    r = run_measured(_demo(1.0)[0], DEMO_MODEL, card_placed=True)
    c = r.crack
    assert c.profile_px is not None and len(c.profile_px) == len(c.profile_mm) == len(c.profile_edges_px) > 10
    assert np.max(c.profile_mm) == np.float64(c.width_max_mm)  # the widest is one of the points along it


def test_views_with_and_without_a_card():
    grey = _demo(1.0)[0]
    measured = run_measured(grey, DEMO_MODEL, card_placed=True)
    assert list(measured.views) == ["Width heatmap", "Confidence heatmap", "Enhanced view (not an X-ray)"]
    plain = run_measured(grey, DEMO_MODEL, card_placed=False)  # no mm: no width heatmap
    assert list(plain.views) == ["Confidence heatmap", "Enhanced view (not an X-ray)"]


def test_the_enhanced_view_says_it_is_not_an_x_ray():
    assert "Not an X-ray" in views.XRAY_NOTE and "cannot see inside" in views.XRAY_NOTE
    img = views.enhanced_view(Image.new("RGB", (200, 150), "grey"))
    assert img.width == 200 and img.height > 150  # the note is written under the picture


def test_the_wall_keeps_the_views():
    r = run_measured(_demo(1.0)[0], DEMO_MODEL, card_placed=True)
    wall = Wall()
    n = wall.add(r.statement, r.overlay, r.to_dict(), r.views)
    e = wall.get(n)
    assert set(e.views) == set(r.views) and all(v[:2] == b"\xff\xd8" for v in e.views.values())  # JPEGs
