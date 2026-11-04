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
    assert len(at.checkbox) == 1 and at.checkbox[0].label == "Share results" and not at.checkbox[0].disabled
    text = " ".join(m.value for m in at.markdown)
    assert "temporary disguise" in text  # the team's quote under the title
    assert "By sending a photo you confirm you took it" in text and "no faces" in text  # consent is the upload (D34)


def test_wall_refuses_a_wrong_key(page):
    page.query_params["wall"] = "wrong"
    at = page.run()
    assert at.error and "presenter" in at.error[0].value


def test_wall_shows_version_model_and_call_to_action(page):
    page.query_params["wall"] = "right-key"
    at = page.run()
    text = " ".join(m.value for m in at.markdown)
    assert not at.error and "v1.0" in text and "U-Net" in text and "your turn" not in text
    assert "temporary disguise" in text and "Reading cracks..." in text  # quote under the title
    assert "data:image/png;base64" not in text  # no QR on the wall: it has its own page (1 Nov)


def test_qr_page_shows_the_qr_alone(page):
    page.query_params["qr"] = ""
    at = page.run()
    text = " ".join(m.value for m in at.markdown)
    assert not at.error and 'class="qr-alone"' in text and "data:image/png;base64" in text
    assert "temporary disguise" not in text and len(at.checkbox) == 0  # nothing else on it


def test_wall_shows_photos_only_when_a_result_is_clicked(page, monkeypatch):
    import io

    from PIL import Image

    from app.core import STATEMENTS, Wall
    buf = io.BytesIO()
    Image.new("RGB", (40, 30), "grey").save(buf, "JPEG")
    from app.core import WallEntry
    fake = [WallEntry(i, st_, buf.getvalue(), buf.getvalue(), {}) for i, st_ in enumerate(
        [s for s, n in zip(STATEMENTS, (7, 3, 1, 2)) for _ in range(n)], 1)]
    monkeypatch.setattr(Wall, "entries", lambda self: fake)
    monkeypatch.setattr(Wall, "counts", lambda self: {s: n for s, n in zip(STATEMENTS, (7, 3, 1, 2))})
    page.query_params["wall"] = "right-key"
    at = page.run()
    assert not at.exception and not at.error
    assert len(at.button) == 4 and "7" in at.button[0].label
    assert not any("Reading cracks" in b.label for b in at.button)  # it sits by the title, not the buttons
    assert not any("Click a result" in m.value for m in at.markdown)
    assert sum('class="crack-rule"' in m.value for m in at.markdown) == 1  # one line under all four buttons
    assert _photos(at) == 0  # no photo until a result is clicked
    at.button[0].click().run()
    assert _photos(at) == 7  # "Crack found" opened: its 7 photos, each a link to its page
    at.button[1].click().run()
    assert _photos(at) == 3  # another result replaces it
    at.button[1].click().run()
    assert _photos(at) == 0  # a second click closes it


def _photos(at) -> int:
    """Wall photos in the opened collection: links to each result's page (D46)."""
    return sum(m.value.count('class="wall-photo"') for m in at.markdown)
