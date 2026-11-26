"""The studio review of 26 Nov 2026 (D115 to D123): exposed steel first, millimetres without a card from a door,
the questions, the knock, the advice, the photo in steps, no author's name in the app."""
import io
import math
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from triage.matrix import assess  # noqa: E402


# ---- D115: exposed steel first ----------------------------------------------------------------------------------

def test_exposed_steel_is_category_3_whatever_the_crack_and_the_confidence():
    for conf in (0.1, 0.99, float("nan")):
        r = assess("surface_map", "infill_wall", 0.05, conf, True, exposed_steel=True)  # Jerome's rule (D112)
        assert r.level == "Category 3" and r.engineer_review and "critical" in r.reason


def test_exposed_steel_never_hides_a_retake():
    assert assess("none", "unknown", float("nan"), 0.9, False, exposed_steel=True).level == "Retake"


def test_steel_gives_the_level_at_once_before_photo_2(monkeypatch):
    from app import core
    from test_app_measure import DEMO_MODEL, _demo
    photo, _ = _demo(1.0)
    monkeypatch.setattr(core, "_check_steel", lambda full: {"seen": True, "score": 0.8, "prompt": "rebar",
                                                             "defects": [], "causes": []})
    r = core.run_measured(photo, DEMO_MODEL, card_placed=True)
    assert r.triage is None and core.steel_first(r).level == "Category 3"
    assert r.triage["level"] == "Category 3" and r.triage["engineer_review"]


def test_causes_before_the_crack_from_sam():
    from element.steel import decide_causes
    m = np.zeros((100, 100), bool)
    m[:20, :20] = True
    out = decide_causes([("water stain", m, 0.7), ("pipe", m, 0.4), ("plant roots", m, 0.6), ("rebar", m, 0.9)])
    assert [c["kind"] for c in out] == ["roots"]  # water is not asked any more; the pipe's score is too low


def test_roots_and_pipes_are_shown_only_and_never_change_the_level():
    """D115: water was left out (it marked plain wall); roots and pipes are shown first but never set a level."""
    from app import core
    from test_app_measure import DEMO_MODEL, _demo
    photo, _ = _demo(1.0)
    r = core.run_measured(photo, DEMO_MODEL, card_placed=True)
    assert r.measured
    plain = core.level_for(r, "infill_wall", "middle", "surface_map", confidence=0.95)
    r.exposed_steel = {"seen": False, "causes": [{"kind": "roots", "score": 0.7}, {"kind": "pipe", "score": 0.6}]}
    assert core.level_for(r, "infill_wall", "middle", "surface_map", confidence=0.95).level == plain.level
    from element.steel import CAUSES
    assert "water" not in CAUSES

# ---- D116: millimetres without a card ------------------------------------------------------------------------------

def _door(h=0.6, w=0.25, score=0.8, top=0.2, shape=(200, 200)):
    m = np.zeros(shape, bool)
    y0, x0 = int(top * shape[0]), 20
    m[y0:y0 + int(h * shape[0]), x0:x0 + int(w * shape[1])] = True
    return ("door", m, score)


def test_a_whole_upright_door_gives_the_scale():
    from measure.photo2_scale import door_in, scale_from_door, DOOR_MM
    d = door_in([_door()], (1000, 1000))
    assert d is not None and abs(d["height_px"] - 600) < 1
    s = scale_from_door(d, 0.5)  # photo 2 has half as many pixels per photo-1 pixel at the crack
    assert abs(s["mm_per_px_photo2"] - DOOR_MM / 600) < 1e-6 and abs(s["mm_per_px"] - DOOR_MM / 600 * 0.5) < 1e-6
    assert 0.1 < s["rel"] < 0.3 and "assumption" in s["basis"]


def test_a_cut_wide_or_unsure_door_is_not_used():
    from measure.photo2_scale import door_in
    assert door_in([_door(top=0.0)], (1000, 1000)) is None  # cut by the top edge: its height is not seen
    assert door_in([_door(h=0.3, w=0.6)], (1000, 1000)) is None  # wider than tall: not a door
    assert door_in([_door(h=0.6, w=0.05)], (1000, 1000)) is not None  # an open door seen edge-on
    assert door_in([_door(score=0.3)], (1000, 1000)) is None
    assert door_in([("person", _door()[1], 0.9)], (1000, 1000)) is None  # people are never a scale (rule 9)


def test_estimated_widths_carry_the_scale_error():
    from measure.photo2_scale import widths
    w = widths({"mm_per_px": 0.1, "rel": 0.16, "source": "door", "basis": "b"}, 5.0, 3.0, 400.0)
    assert w["width_max_mm"] == 0.5 and w["length_mm"] == 40.0
    assert math.isclose(w["uncertainty_mm"], 0.5 * 0.16 + 0.1)


def test_an_estimated_width_never_gives_category_1():
    """Rule 3: Category 1 is earned, never from an assumed door height."""
    from app import core
    r = SimpleNamespace(crack=None, measured=False, test=False, estimate=None, confidence=0.95, statement="Crack found",
                        growth=None, exposed_steel=None, steel_element=None, answers={},
                        scale_estimate={"crack": {"width_max_mm": 0.05, "uncertainty_mm": 0.02}})
    assert core.level_for(r, "infill_wall", "middle", "surface_map", confidence=0.95).level == "Expert needed"
    r.scale_estimate["crack"] = {"width_max_mm": 4.0, "uncertainty_mm": 0.5}
    assert core.level_for(r, "infill_wall", "middle", "diagonal_shear", confidence=0.95).level == "Category 3"


# ---- D120: questions and the knock ---------------------------------------------------------------------------------

def test_a_yes_to_a_sign_raises_and_nothing_else_does():
    from app.questions import QUESTIONS, signs, next_question
    assert signs({"sticking": "yes", "earthquake": "no", "water": "dont_know", "floor": "ground"}) == \
        ("sticking_openings",)
    assert signs({"sagging": "yes"}) == ("sagging",)
    from triage.matrix import ESCALATORS
    assert all(q["sign"] in ESCALATORS for q in QUESTIONS if q["sign"])
    assert next_question({}) is QUESTIONS[0] and next_question({q["key"]: "dont_know" for q in QUESTIONS}) is None
    # most questions ask where in the building the crack is
    assert sum(q["sign"] is None for q in QUESTIONS) >= 6


def test_answers_raise_the_level_through_the_matrix():
    from app import core
    from test_app_measure import DEMO_MODEL, _demo
    photo, _ = _demo(1.0)
    r = core.run_measured(photo, DEMO_MODEL, card_placed=True)
    calm = core.level_for(r, "infill_wall", "middle", "surface_map", confidence=0.95)
    r.answers = {"sagging": "yes"}
    assert core.level_for(r, "infill_wall", "middle", "surface_map", confidence=0.95).level == "Category 3"
    r.answers = {"sagging": "no", "earthquake": "dont_know"}
    assert core.level_for(r, "infill_wall", "middle", "surface_map", confidence=0.95).level == calm.level


def _wav(signal: np.ndarray, rate=16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(signal, -1, 1) * 32000).astype(np.int16).tobytes())
    return buf.getvalue()


def test_the_knock_gives_its_pitch_and_ringing_but_no_hollow_or_solid():
    from app.questions import knock
    rate = 16000
    t = np.arange(rate) / rate
    x = 0.002 * np.random.default_rng(0).standard_normal(rate)
    for start in (0.2, 0.6):  # two knocks: 900 Hz, ringing about 30 ms
        k = (t >= start)
        x += k * np.sin(2 * np.pi * 900 * (t - start)) * np.exp(-(t - start) / 0.013)
    f = knock(_wav(x, rate))
    assert f is not None and abs(f["pitch_hz"] - 900) < 30 and 15 < f["ring_ms"] < 60
    assert f["knocks"] == 2 and f["reading"] is None
    assert knock(_wav(0.002 * np.random.default_rng(1).standard_normal(rate), rate)) is None  # no knock heard
    assert knock(b"not a wav") is None


# ---- D121: advice -----------------------------------------------------------------------------------------------

def test_advice_is_concrete_never_says_safe_and_puts_steel_first():
    from triage.advice import advice, LEAVE
    a3 = advice("Category 3")
    assert any("fridge" in s for s in a3) and LEAVE in a3
    st = advice("Category 1", steel=True)
    assert "steel" in st[0] and LEAVE in st
    for lv in ("Category 1", "Category 2", "Category 3", "Expert needed", None):
        assert not any(" safe" in s.lower() for s in advice(lv))
    assert advice("Retake") == []
    assert any("water" in s for s in advice("Category 2", answers={"water": "yes"}))
    assert any("column" in s for s in advice("Category 2", element="column"))


# ---- D122: the photo in steps -------------------------------------------------------------------------------------

def test_the_photo_in_steps_ends_with_the_crack_alone_on_black():
    from app.sequence import steps
    photo = Image.new("RGB", (400, 300), (150, 150, 150))
    m = np.zeros((150, 200), bool)
    m[70:80, 20:180] = True
    r = SimpleNamespace(photo=photo, decision=SimpleNamespace(mask=m), anomaly=None)
    s = steps(r)
    assert [c.split(" ·")[0] for c, _ in s] == ["1", "2", "3"]
    alone = np.asarray(Image.open(io.BytesIO(s[2][1])))
    assert alone[:100].mean() < 10 and alone[140:160, 60:340].mean() > 80  # black, the crack bright
    assert len(steps(SimpleNamespace(photo=photo, decision=None, anomaly=None))) == 1


def test_growth_per_day_only_between_dated_photos_a_day_apart():
    from app.crack_view import growth_data
    from measure.growth import compare
    before = {"measured": True, "crack": {"width_max_mm": 0.5, "width_uncertainty_mm": 0.05}, "received_at": 0}
    now = {"measured": True, "crack": {"width_max_mm": 0.8, "width_uncertainty_mm": 0.05}, "received_at": 10 * 86400}
    g = compare(before, now, 3)
    assert g["grown"] and g["days"] == 10
    assert growth_data(g)["per_day_mm"] == pytest.approx(0.03)
    assert growth_data({**g, "days": 0.01}) is None  # minutes apart: no rate
    assert growth_data({**g, "grown": False}) is None


# ---- D123: no author's name in the app ----------------------------------------------------------------------------

def test_no_author_name_in_what_the_app_shows():
    root = Path(__file__).resolve().parents[1] / "src" / "app"
    from app.crack_view import PAGE
    from app import engineer
    shown = PAGE + engineer.html({"statement": "Crack found", "measured": True, "crack": {"width_uncertainty_mm": 0.1}})
    for name in ("Corres", "Pantoja", "Rezaie"):
        assert name not in shown
    assert "Corres" not in (root / "second_look.py").read_text() and "Corres" not in (root / "result_blocks.py").read_text()


def test_photo_2_colours_the_element_the_crack_lies_on():
    from app.core import ELEMENT_TINT, tint_element
    img = Image.new("RGB", (300, 200), (120, 120, 120))
    m = np.zeros((200, 300), bool)
    m[40:160, 100:200] = True
    out = np.asarray(tint_element(img, m)).astype(int)
    inside, outside = out[100, 150], out[10, 10]
    assert list(outside) == [120, 120, 120]  # the rest of the photo as it was
    assert inside[2] > 120 and inside[0] < 120  # light blue over the element
    assert list(out[40, 150]) == list(ELEMENT_TINT)  # its outline
    assert np.array_equal(np.asarray(tint_element(img, np.zeros((5, 5), bool))), np.asarray(img))  # no mask: unchanged


def test_resident_viewer_has_no_corrections_and_no_unusual_surface_the_engineer_has_both():
    """Review 26 Nov 2026: adding and removing cracks and the unusual-surface layer are the engineer's, on the wall."""
    import json as _json
    from app.crack_view import viewer_html
    data = {"unit": "mm", "line": [], "heat": "data:image/png;base64,AA"}
    res, eng = viewer_html(b"", data, engineer=False), viewer_html(b"", data)
    assert '"engineer": false' in res and '"engineer": true' in eng
    assert 'if (!HAS || !ENG) $("tedit").style.display = "none"' in res  # the edit button hides for residents
    assert "C.heat && ENG" in res  # and the unusual-surface switch


def test_the_wall_shows_the_steps_from_what_it_keeps():
    import base64 as b64
    from app.sequence import steps_from_entry
    photo = Image.new("RGB", (400, 300), (150, 150, 150))
    buf = io.BytesIO(); photo.save(buf, "JPEG")
    area = Image.new("RGBA", (400, 300), (255, 77, 106, 0))
    area.paste((255, 77, 106, 160), (40, 140, 360, 160))
    abuf = io.BytesIO(); area.save(abuf, "PNG")
    s = steps_from_entry(buf.getvalue(), {"area": "data:image/png;base64," + b64.b64encode(abuf.getvalue()).decode()})
    assert [c.split(" ·")[0] for c, _ in s] == ["1", "2", "3"]
    assert steps_from_entry(None, None) == []


def _photo_with_record(lat=None, lon=None, when="2026:11:26 10:15:00", fmt="JPEG"):
    from PIL import Image as _I
    im = _I.new("RGB", (64, 48), (120, 120, 120))
    ex = im.getexif()
    if when:
        ex.get_ifd(0x8769)[36867] = when
    if lat is not None:
        def dms(v):
            v = abs(v); d = int(v); m = int((v - d) * 60); s = round(((v - d) * 60 - m) * 60, 4)
            return (d, m, s)
        g = ex.get_ifd(0x8825)
        g[1], g[2], g[3], g[4] = ("N" if lat >= 0 else "S"), dms(lat), ("E" if lon >= 0 else "W"), dms(lon)
    buf = io.BytesIO()
    im.save(buf, fmt, exif=ex)
    return buf.getvalue()


def test_the_photo_gives_its_time_and_place():
    from app.place import read
    m = read(_photo_with_record(41.39682, 2.19455))
    assert m["taken_at"] == "2026-11-26 10:15"
    assert abs(m["lat"] - 41.39682) < 1e-4 and abs(m["lon"] - 2.19455) < 1e-4
    south = read(_photo_with_record(-12.0464, -77.0428))
    assert south["lat"] < 0 and south["lon"] < 0  # south and west keep their sign


def test_a_photo_without_a_record_has_no_time_or_place():
    from app.place import read, valid
    assert read(_photo_with_record(when=None)) == {"taken_at": None, "lat": None, "lon": None}
    assert read(b"not a photo") == {"taken_at": None, "lat": None, "lon": None}
    assert not valid(0, 0) and not valid(None, 2) and valid(41.4, 2.19)


def test_the_place_is_kept_only_when_the_sender_shares_it():
    from app import core
    from test_app_measure import DEMO_MODEL, _demo
    photo, _ = _demo(1.0)
    buf = io.BytesIO()
    ex = Image.new("RGB", (1, 1)).getexif()
    g = ex.get_ifd(0x8825); g[1], g[2], g[3], g[4] = "N", (41, 23, 48.5), "E", (2, 11, 40.4)
    ex.get_ifd(0x8769)[36867] = "2026:11:26 10:15:00"
    photo.save(buf, "JPEG", exif=ex)
    wall = core.Wall()
    private = core.analyse(buf.getvalue(), True, True, wall, DEMO_MODEL)
    assert private.place is None and private.taken_at == "2026-11-26 10:15"
    shared = core.analyse(buf.getvalue(), True, True, wall, DEMO_MODEL, share_location=True)
    assert shared.place["source"] == "photo" and abs(shared.place["lat"] - 41.39681) < 1e-3
    assert shared.to_dict()["place"] == shared.place
