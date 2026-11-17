from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PAGE = str(Path(__file__).resolve().parents[1] / "src" / "app" / "second_look.py")


@pytest.fixture
def page(monkeypatch):
    monkeypatch.setenv("SECOND_LOOK_WALL_KEY", "right-key")
    monkeypatch.setenv("SECOND_LOOK_URL", "https://example.trycloudflare.com")
    return AppTest.from_file(PAGE, default_timeout=60)


def test_phone_page_has_one_tick_and_states_the_consent_terms(page):
    at = page.run()
    assert not at.exception
    labels = [c.label for c in at.checkbox]  # the card tick (D56) and the share tick; consent is the upload (D34)
    assert labels == ["I put a bank card next to the crack", "Share results"]
    assert not any(c.value for c in at.checkbox)  # both off: no card search, nothing shared, unless ticked
    text = " ".join(m.value for m in at.markdown)
    assert "temporary disguise" not in text  # the quote was removed (D78)
    assert "By sending a photo you confirm you took it" in text and "no faces" in text  # consent is the upload (D34)


def test_wall_refuses_a_wrong_key(page):
    page.query_params["wall"] = "wrong"
    at = page.run()
    assert at.error and "presenter" in at.error[0].value


def test_wall_shows_version_model_and_call_to_action(page):
    page.query_params["wall"] = "right-key"
    at = page.run()
    text = " ".join(m.value for m in at.markdown)
    assert not at.error and "v1.0" not in text and "Human in the loop" in text and "U-Net" in text and "your turn" not in text
    assert "temporary disguise" not in text and "Reading cracks" not in text  # no quote (D78); nothing moves (D76)
    assert "data:image/png;base64" not in text  # no QR on the wall: it has its own page (1 Nov)


def test_qr_page_shows_the_qr_alone(page):
    page.query_params["qr"] = ""
    at = page.run()
    text = " ".join(m.value for m in at.markdown)
    assert not at.error and 'class="qr-alone"' in text and "data:image/png;base64" in text
    assert "temporary disguise" not in text and len(at.checkbox) == 0  # nothing else on it


def _entries(spec):
    """Wall entries from (statement, level or None), numbered from 1."""
    import io

    from PIL import Image

    from app.core import WallEntry
    buf = io.BytesIO()
    Image.new("RGB", (40, 30), "grey").save(buf, "JPEG")
    return [WallEntry(i, s, buf.getvalue(), buf.getvalue(),
                      {"statement": s, "reason": f"reason {i}",
                       "triage": None if lv is None else {"level": lv, "element_name": "column"}})
            for i, (s, lv) in enumerate(spec, 1)]


def test_wall_sorts_by_risk_and_shows_photos_only_when_a_tab_is_clicked(page, monkeypatch):
    """D78: tabs by risk, highest first; no Crack found / No crack found / Retake tabs."""
    from app.core import Wall
    fake = _entries([("Crack found", "Category 3")] * 2 + [("Crack found", "Category 2")] * 3
                    + [("Crack found", "Category 1")] + [("Crack found", "Expert needed")]
                    + [("Crack found", None), ("Expert needed", None)]  # no level yet, or unsure: Expert needed
                    + [("Retake", None), ("No crack found", None), ("Crack found", "Retake")])  # not on the tabs
    monkeypatch.setattr(Wall, "entries", lambda self: fake)
    page.query_params["wall"] = "right-key"
    at = page.run()
    assert not at.exception and not at.error
    assert [b.label for b in at.button] == ["Category 3 `2`", "Expert needed `3`", "Category 2 `3`", "Category 1 `1`"]
    text = " ".join(m.value for m in at.markdown)
    assert "Crack found" not in text and "Retake" not in text
    assert not any("crack-rule" in m.value for m in at.markdown)  # GitHub's tabs, no cracked line (D76)
    assert _photos(at) == 0  # no photo until a tab is clicked
    at.button[0].click().run()
    assert _photos(at) == 2  # Category 3 opened
    at.button[1].click().run()
    assert _photos(at) == 3  # Expert needed replaces it, with the two still waiting for photo 2
    assert sum(m.value.count("Waiting for photo 2") for m in at.markdown) == 2
    at.button[1].click().run()
    assert _photos(at) == 0  # a second click closes it


def test_risk_group_never_puts_doubt_in_a_lower_tab():
    from app.core import risk_group
    assert risk_group({"statement": "Crack found", "triage": {"level": "Category 1"}}) == "Category 1"
    assert risk_group({"statement": "Crack found", "triage": None}) == "Expert needed"  # no level yet (rule 2)
    assert risk_group({"statement": "Expert needed"}) == "Expert needed"
    assert risk_group({"statement": "Retake"}) is None and risk_group({"statement": "No crack found"}) is None
    assert risk_group({"statement": "Crack found", "triage": {"level": "Retake"}}) is None


def _photos(at) -> int:
    """Wall photos in the opened collection: links to each result's page (D46)."""
    return sum(m.value.count('class="wall-photo"') for m in at.markdown)


def test_a_card_opens_its_result_over_the_wall_on_the_same_page(page, monkeypatch):
    """D77: a click on a card keeps the wall (blurred behind) and opens the result in a window beside the crack;
    the close button brings the wall back. No new page."""
    import io

    from PIL import Image

    from app.core import Wall, WallEntry
    buf = io.BytesIO()
    Image.new("RGB", (40, 30), "grey").save(buf, "JPEG")
    fake = _entries([("Crack found", None)] * 2)  # no level yet: on the Expert needed tab
    monkeypatch.setattr(Wall, "entries", lambda self: fake)
    monkeypatch.setattr(Wall, "get", lambda self, n: next((e for e in fake if e.number == n), None))
    page.query_params["wall"] = "right-key"
    at = page.run()
    at.button[1].click().run()
    assert not any('class="sl-crack"' in m.value for m in at.markdown)
    at.button(key="open-item-2").click().run()
    assert not at.exception and at.session_state["item"] == 2
    assert any('class="sl-crack"' in m.value and "#2" in m.value for m in at.markdown)  # the clicked crack
    assert any("reason 2" in m.value for m in at.markdown)  # its window
    assert _photos(at) == 2 and dict(page.query_params).get("item") is None  # the wall stays; same page
    at.button(key="sl-close").click().run()
    assert not any('class="sl-crack"' in m.value for m in at.markdown) and _photos(at) == 2


def test_the_wall_keeps_the_photo_without_the_statement_band():
    """D78: the wall sorts by risk; the "CRACK FOUND" band under the photo stays on the sender's phone only."""
    import io

    import numpy as np
    from PIL import Image

    from app.core import Wall
    from segmentation.predict import draw_overlay
    img = draw_overlay(Image.new("RGB", (200, 100), "grey"), np.zeros((100, 200), bool), "Crack found", "a reason")
    assert img.height > 100
    wall = Wall()
    n = wall.add("Crack found", img, {})
    assert Image.open(io.BytesIO(wall.get(n).large)).size == (200, 100)
