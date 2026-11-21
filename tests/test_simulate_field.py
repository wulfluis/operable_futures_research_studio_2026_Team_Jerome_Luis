"""The team's field photos are always in the simulated population (D87). Blank images stand in for the photos."""
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from app.simulate import field_cases  # noqa: E402


def _img(path: Path, size=(40, 30)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path)


def test_every_pair_is_sent_with_its_element_photo_and_the_card_tick(tmp_path):
    for n in (1, 2):
        _img(tmp_path / "team" / f"pair_{n}_crack.jpg")
        _img(tmp_path / "team" / f"pair_{n}_element.jpg", (30, 40))
    cases = list(field_cases(tmp_path))
    assert len(cases) == 2
    for crack, element, meta in cases:
        assert crack.size == (40, 30) and element.size == (30, 40)
        assert meta["card_placed"] is True and meta["kind"].startswith("FIELD")
    assert "pair 1" in cases[0][2]["kind"] and "pair 2" in cases[1][2]["kind"]


def test_a_crack_photo_without_its_element_photo_is_sent_alone(tmp_path):
    _img(tmp_path / "team" / "pair_3_crack.jpg")
    (case,) = list(field_cases(tmp_path))
    assert case[1] is None


def test_no_folder_means_no_field_cases(tmp_path):
    assert list(field_cases(tmp_path / "missing")) == []


def test_the_photographer_is_not_named_on_the_wall(tmp_path):
    _img(tmp_path / "someone" / "pair_1_crack.jpg")
    (case,) = list(field_cases(tmp_path))
    assert "someone" not in case[2]["kind"]
