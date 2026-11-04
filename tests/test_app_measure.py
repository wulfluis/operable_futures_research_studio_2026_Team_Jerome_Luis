"""Bank-card measurement in the class demo (spec/features/app_measure.md, D46). No model file is needed."""

import io
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from streamlit.testing.v1 import AppTest

from app.core import NOT_A_SAFETY_CHECK, STATEMENTS, Wall, analyse, run_measured
from measure.synthetic import make_demo
from segmentation.predict import run

ROOT = Path(__file__).resolve().parents[1]
PAGE = str(ROOT / "src" / "app" / "second_look.py")


class FakeModel(torch.nn.Module):
    """Stands in for the U-Net: 'sees' a crack wherever the photo is dark (as in test_segmentation_predict.py)."""

    def forward(self, x):
        return (x.mean(1, keepdim=True) < self.level).float() * 20 - 10

    def __init__(self, level=-1.0):
        super().__init__()
        self.level = level


DEMO_MODEL = FakeModel(-0.4)  # the synthetic demo crack is lighter than the plain test line


def _plain_wall_with_line():
    rng = np.random.default_rng(0)
    a = rng.integers(110, 190, (900, 1200, 3)).astype(np.uint8)
    a[448:452, 100:1100] = 10
    return Image.fromarray(a)


def _demo(width_mm=1.0):
    grey, truth = make_demo(width_mm)
    return Image.fromarray(np.repeat(grey.astype(np.uint8)[..., None], 3, axis=2)), truth


def test_without_a_card_the_answer_is_the_original_pixel_for_pixel():
    photo = _plain_wall_with_line()
    mine, original = run_measured(photo, FakeModel()), run(photo, FakeModel())
    assert mine.card is None and mine.crack is None
    assert (mine.statement, mine.reason, mine.confidence) == (original.statement, original.reason, original.confidence)
    assert np.array_equal(np.asarray(mine.overlay), np.asarray(original.overlay))


def test_with_a_card_the_crack_is_measured_in_mm():
    photo, truth = _demo(1.0)
    r = run_measured(photo, DEMO_MODEL)
    assert r.statement == "Crack found" and r.card is not None and r.measured
    assert abs(r.crack.width_max_mm - truth["width_mm"]) < 0.25
    assert abs(r.crack.length_main_mm - truth["length_mm"]) / truth["length_mm"] < 0.05
    assert abs(r.card.tilt_deg - truth["tilt_deg"]) < 2
    assert "mm (bank card)" in r.reason and "scale card" not in r.reason
    assert r.band() is not None and r.statement in STATEMENTS  # never a new statement, never a level


def test_card_edges_are_not_read_as_the_crack():
    """A card on a plain wall: the card's dark outline must not become 'Crack found'."""
    photo, _ = _demo(1.0)
    a = np.asarray(photo).copy()
    a[:, 700:] = 200  # wipe the crack (the demo card sits on the left)
    r = run_measured(Image.fromarray(a), DEMO_MODEL)
    assert r.card is not None and r.statement == "No crack found" and not r.measured
    assert "no crack to measure" in r.measure_note


def test_result_never_says_safe():
    photo, _ = _demo(1.0)
    d = run_measured(photo, DEMO_MODEL).to_dict()
    import re
    assert not re.search(r"\bsafe\b|clearance|permit", str(d), re.I)
    assert "not a safety check" in NOT_A_SAFETY_CHECK


def test_only_shared_results_reach_the_wall():
    photo, _ = _demo(1.0)
    import io
    buf = io.BytesIO()
    photo.save(buf, "PNG")
    wall = Wall()
    analyse(buf.getvalue(), True, False, wall, DEMO_MODEL)
    assert wall.items() == []
    analyse(buf.getvalue(), True, True, wall, DEMO_MODEL)
    assert wall.counts()["Crack found"] == 1


@pytest.fixture
def page(monkeypatch):
    monkeypatch.setenv("SECOND_LOOK_WALL_KEY", "right-key")
    monkeypatch.setenv("SECOND_LOOK_URL", "https://cranium-bladder-tapioca.ngrok-free.dev")
    return AppTest.from_file(PAGE, default_timeout=60)


def test_phone_page_keeps_the_original_layout_and_adds_the_card_hint(page):
    at = page.run()
    assert not at.exception
    assert len(at.checkbox) == 1 and at.checkbox[0].label == "Share results"
    text = " ".join(m.value for m in at.markdown)
    assert "temporary disguise" in text and "By sending a photo you confirm you took it" in text
    assert "bank-card-sized card" in text


def test_wall_refuses_a_wrong_key(page):
    page.query_params["wall"] = "wrong"
    at = page.run()
    assert at.error and "presenter" in at.error[0].value


def test_one_app_on_the_teams_link():
    """D46: the measurement lives in the class demo itself; no second copy of the app, the launcher is unchanged."""
    assert not (ROOT / "src" / "app_measure").exists()
    page = (ROOT / "src" / "app" / "second_look.py").read_text(encoding="utf-8")
    assert "from app.core import" in page and "import app_measure" not in page and "from app_measure" not in page


def test_length_comes_from_the_unet_line_when_the_edges_are_found_only_in_pieces(monkeypatch):
    """Field case of 4 Nov: on a faint crack the classical finder sees only pieces; the length must still be the
    whole crack (the U-Net line), and the width comes from the pieces."""
    import app.core as cmod
    from measure.crack import segment
    photo, truth = _demo(1.0)

    def pieces(*a, **k):
        m = segment(*a, **k)
        ys, xs = np.nonzero(m)
        keep = np.zeros_like(m)
        top = ys < np.percentile(ys, 30)  # only the upper third of the crack is found
        keep[ys[top], xs[top]] = True
        return keep

    monkeypatch.setattr(cmod, "segment", pieces)
    r = run_measured(photo, DEMO_MODEL)
    assert r.measured
    assert abs(r.crack.length_main_mm - truth["length_mm"]) / truth["length_mm"] < 0.08
    assert abs(r.crack.width_max_mm - truth["width_mm"]) < 0.3


def test_wall_keeps_number_large_copy_and_stats_for_each_result():
    wall = Wall()
    img = Image.new("RGB", (2000, 1500), "grey")
    a = wall.add("Crack found", img, {"measured": True})
    b = wall.add("Retake", img)
    assert (a, b) == (1, 2) and [e.number for e in wall.entries()] == [2, 1]
    e = wall.get(1)
    assert e.stats == {"measured": True} and max(Image.open(io.BytesIO(e.large)).size) == 1400
    assert wall.get(99) is None and wall.items()[0][0] == "Retake"  # items() still as the original


MODEL_FILE = ROOT / "models" / "unet_crack.pt"


@pytest.mark.skipif(not MODEL_FILE.exists(), reason="the wall loads models/unet_crack.pt (team release, not in git)")
def test_result_page_shows_the_stats_large(page, monkeypatch):
    photo, _ = _demo(1.0)
    r = run_measured(photo, DEMO_MODEL)
    from app.core import WallEntry
    buf = io.BytesIO()
    r.overlay.convert("RGB").save(buf, "JPEG")
    entry = WallEntry(7, r.statement, buf.getvalue(), buf.getvalue(), r.to_dict())
    monkeypatch.setattr(Wall, "get", lambda self, n: entry if n == 7 else None)
    page.query_params["wall"] = "right-key"
    page.query_params["item"] = "7"
    at = page.run()
    assert not at.exception and not at.error
    text = " ".join(m.value for m in at.markdown)
    assert "Width, widest point" in text and f'{r.crack.width_max_mm:.2f}' in text and "Camera tilt" in text
    assert "Back to the wall" in text and "#7" in text
    page.query_params["item"] = "8"
    assert "Result gone" in " ".join(m.value for m in page.run().markdown)
