"""Real cases per risk tab (D79): the picker counts by the tab the app's own level gives; the simulator sends them."""

import json

import pytest
from PIL import Image

from app import pick_by_level, simulate


def test_tally_counts_each_risk_tab_and_nothing_else():
    picks = [{"level_tab": t} for t in ("Category 3", "Category 3", "Expert needed", "Category 1")]
    assert pick_by_level.tally(picks) == {"Category 3": 2, "Expert needed": 1, "Category 2": 0, "Category 1": 1}


def test_the_simulator_sends_the_picked_cases_crop_and_whole(tmp_path):
    photo = tmp_path / "p.jpg"
    Image.new("RGB", (400, 300), "grey").save(photo)
    picks = tmp_path / "by_level.json"
    picks.write_text(json.dumps({"cases": [{"photo": str(photo), "crop": [10, 20, 110, 120], "element": "wall",
                                            "level_tab": "Category 2"}]}), encoding="utf-8")
    simulate.ROOT = tmp_path  # the photo path is read relative to the repository root
    try:
        (close_up, whole, meta), = simulate.real_cases(10, picks)
    finally:
        simulate.ROOT = simulate.Path(simulate.__file__).resolve().parents[2]
    assert close_up.size == (100, 100) and whole.size == (400, 300) and meta["card_placed"] is False


def test_by_level_without_picks_says_how_to_make_them(tmp_path, monkeypatch):
    monkeypatch.setattr(simulate, "BY_LEVEL", tmp_path / "none.json")
    monkeypatch.setattr(simulate, "INBOX", tmp_path / "inbox")
    with pytest.raises(SystemExit, match="pick_by_level"):
        simulate.main(["--by-level"])


def test_a_synthetic_filler_is_regenerated_and_says_it_is_synthetic(tmp_path):
    """D79: a tab the real photos never reach is filled with synthetic cases, never passed off as real."""
    picks = tmp_path / "by_level.json"
    picks.write_text(json.dumps({"cases": [{"synthetic": True, "width_mm": 0.2, "shape": "vertical", "element": "wall",
                                            "k": 1, "seed": 3, "tilt": [2, -1], "level_tab": "Category 1"}]}),
                     encoding="utf-8")
    if not simulate.PHINET6.exists():
        pytest.skip("PHI-Net task 6 is not on this laptop")
    (crack, element, meta), = simulate.real_cases(10, picks)
    assert meta["card_placed"] is True and meta["kind"].startswith("SYNTHETIC")
    assert crack.size == (1200, 900) and element.size == (672, 672)
