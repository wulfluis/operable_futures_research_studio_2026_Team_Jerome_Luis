"""Which crack widths a phone can see and measure at which distance (spec/features/capture_visibility.md)."""

import pytest

from element.locate import LOOK_MIN_SHARE
from measure.crack import MIN_PX_ACROSS
from measure.visibility import CARD_MIN_SHARE, markdown, mm_per_px, table, visibility


def test_pixel_size_follows_distance_focal_length_and_pixel_count():
    base = mm_per_px(1.0, 26.0, 4032)
    assert base == pytest.approx(1000 * 34.62 / 26 / 4032)
    assert mm_per_px(2.0, 26.0, 4032) == pytest.approx(2 * base)
    assert mm_per_px(1.0, 52.0, 4032) == pytest.approx(base / 2)
    assert mm_per_px(1.0, 26.0, 8064) == pytest.approx(base / 2)


def test_the_measurable_width_is_the_measurements_own_pixel_rule():
    v = visibility("Main 12 MP", 26.0, 4032, 1.0)
    assert v.measurable_mm == pytest.approx(MIN_PX_ACROSS * v.mm_per_px)
    assert CARD_MIN_SHARE == LOOK_MIN_SHARE  # the card bridge's own limit (D59)


def test_seo_et_al_case_puts_0_3_mm_at_the_edge_of_reading():
    """Seo et al. 2021 [105]: 0.3, 0.5 and 1 mm told apart at 2 m with 64 MP and a 2x lens."""
    s = mm_per_px(2.0, 52.0, 9248)
    assert 1.0 < 0.3 / s < 3.0


def test_the_card_is_found_at_1_m_and_not_at_10_m():
    assert visibility("Main 12 MP", 26.0, 4032, 1.0).card_found
    assert not visibility("Main 12 MP", 26.0, 4032, 10.0).card_found


def test_closer_means_finer_cracks_and_the_table_never_says_safe():
    rows = [r for r in table() if r.camera == "Main 12 MP"]
    widths = [r.measurable_mm for r in rows]
    assert widths == sorted(widths)
    text = markdown(table()).lower()
    assert "safe" not in text and "| main 12 mp | 1 m |" in text
