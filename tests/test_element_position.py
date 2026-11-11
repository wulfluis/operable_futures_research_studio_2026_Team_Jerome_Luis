"""Where on the element the crack is, from the two photos linked (spec/features/element_position.md, D58).
Synthetic photo pairs: one high-resolution textured scene with a column; the element photo is it 4 times smaller,
the close-up a cut-out at full resolution (4 times closer). No model file is needed."""

import io
import re

import cv2
import numpy as np
import pytest
from PIL import Image

from app.core import Wall, analyse_element, auto_triage, run_measured
from element.locate import locate
from test_app_measure import FakeModel
from test_element_predict import FakeElementModel

CLOSER = 4
COLUMN = (1400, 1800)  # x, high resolution: 100 px wide in the element photo
ENDS = (400, 2000)  # y: the column's ends, both inside the photo


CARD = (1520, 200, 126)  # x, size in high-resolution px (about 0.43 mm per px): on the column, below the crack


def _scene(ends=ENDS, seed=0, h=2400, w=3200, plain=False):
    """Plaster-like texture at several scales (so both photos have key points), and a lighter column. plain: a
    smooth wall with only fine grain, where key points cannot link the photos."""
    rng = np.random.default_rng(seed)
    tex = np.zeros((h, w), np.float32)
    if plain:
        return 140 + rng.normal(0, 6, (h, w)).astype(np.float32) + np.where(
            (np.arange(h)[:, None] >= ends[0]) & (np.arange(h)[:, None] < ends[1])
            & (np.arange(w) >= COLUMN[0]) & (np.arange(w) < COLUMN[1]), 45, 0)
    for cell, amp in ((64, 16), (24, 12), (8, 8)):
        small = rng.normal(0, 1, (h // cell + 2, w // cell + 2)).astype(np.float32)
        big = cv2.resize(small, (w + 2 * cell, h + 2 * cell), interpolation=cv2.INTER_CUBIC)
        tex += amp * big[cell:cell + h, cell:cell + w]
    img = 140 + tex
    img[ends[0]:ends[1], COLUMN[0]:COLUMN[1]] += 45
    return img


def _card(hi, x, y, plain=False, upright=False, dark=False):
    """A light card with a dark stripe near one long side (its look; a plain one has none). upright: long side
    vertical, as when the card is laid on the wall on its short side."""
    _, w, h = CARD
    c = np.full((h, w), 35 if dark else 225, np.float32)
    if dark:  # a black card: only a small chip and faint print (field test of 11 Nov)
        c[45:72, 28:60] = 150
        c[90:96, 120:180] = 55
    elif not plain:
        c[14:36] = 40
        c[70:100, 20:70] = 120
    if upright:
        c = np.rot90(c)
    hi[y:y + c.shape[0], x:x + c.shape[1]] = c


def pair(crack_y, ends=ENDS, turn_close=0.0, seed=0, beam=False, card=False, decoy=False, plain=False,
         thin=False, upright=False, dark=False, clutter=False):
    """(close-up grey, its crack line in close-up pixels, element photo RGB). The crack runs across the column.
    card: a bank card under the crack; decoy: a plain card-shaped thing elsewhere; thin: a crack 1 px wide, which
    vanishes in the element photo."""
    hi = _scene(ends, seed, plain=plain)
    x0, x1 = COLUMN[0] + 40, COLUMN[1] - 40
    half = 1 if thin else 2
    hi[crack_y - half:crack_y + half, x0:x1] = 20
    if card:
        _card(hi, CARD[0] + (40 if upright else 0), crack_y + 100, upright=upright, dark=dark)
    if clutter:  # dark things elsewhere in the room: patches, a sofa, a plant's leaves
        rng = np.random.default_rng(11)
        for _ in range(25):
            x, y = int(rng.integers(0, 1200)), int(rng.integers(0, 2300))
            hi[y:y + int(rng.integers(30, 220)), x:x + int(rng.integers(30, 220))] = float(rng.integers(20, 70))
        hi[2050:2400, 2300:3200] = 30  # a sofa
    if decoy:
        _card(hi, 500, 1500, plain=True)
    hi = np.clip(hi, 0, 255).astype(np.uint8)
    cx = (COLUMN[0] + COLUMN[1]) // 2
    close = hi[crack_y - 450:crack_y + 450, cx - 600:cx + 600].copy()
    crack = np.array([[x0 - (cx - 600), 450.0], [x1 - (cx - 600), 450.0]])
    if turn_close:  # the phone held at a slant for the close-up
        m = cv2.getRotationMatrix2D((600, 450), turn_close, 1.0)
        close = cv2.warpAffine(close, m, (1200, 900), borderMode=cv2.BORDER_REFLECT)
        crack = (m[:, :2] @ crack.T).T + m[:, 2]
    element = cv2.resize(hi, (hi.shape[1] // CLOSER, hi.shape[0] // CLOSER), interpolation=cv2.INTER_AREA)
    if beam:  # the same scene turned a quarter: a beam with the crack across it
        close, element = close.T.copy(), element.T.copy()
        crack = crack[:, ::-1].copy()
    return close, crack, np.repeat(element[..., None], 3, axis=2)


def test_the_close_up_is_found_where_it_was_cut_out():
    close, crack, element = pair(1200)
    loc = locate(close, crack, element, "column")
    assert loc.linked and loc.inliers >= 20
    assert abs(loc.depth_px - 100) < 5  # the column is 100 px wide in the element photo
    assert loc.axis_to_vertical_deg < 3 and loc.crack_to_axis_deg > 85


def test_middle_end_zone_and_cut_by_the_photo():
    assert locate(*pair(1200), "column").position == "middle"
    assert locate(*pair(520), "column").position == "end_zone"  # 120 px high-res = 30 px from the end: < 1 depth
    assert locate(*pair(1200, ends=(0, 2000)), "column").position == "unknown"  # top end cut by the photo


def test_an_unrelated_photo_is_not_linked():
    close, crack, _ = pair(1200)
    loc = locate(close, crack, pair(1200, seed=5)[2], "column")
    assert not loc.linked and loc.position == "unknown" and "could not be linked" in loc.reason


def test_a_slanted_close_up_still_reads_the_crack_across_the_column():
    loc = locate(*pair(1200, turn_close=20), "column")
    assert loc.linked and loc.crack_to_axis_deg > 85 and loc.angle_for_rules("column") > 85


def test_a_slanted_element_photo_measures_against_the_column_not_the_phone():
    close, crack, element = pair(1200)
    h, w = element.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), 8, 1.0)
    turned = cv2.warpAffine(element, m, (w, h), borderMode=cv2.BORDER_REFLECT)
    loc = locate(close, crack, turned, "column")
    assert loc.linked and 5 < loc.axis_to_vertical_deg < 11 and loc.crack_to_axis_deg > 85


def test_a_beam_reads_the_same_way():
    loc = locate(*pair(1200, beam=True), "beam")
    assert loc.position == "middle" and loc.crack_to_axis_deg > 85
    assert loc.angle_for_rules("beam") < 5  # across a beam = upright in the photo: the rules' 'vertical'


@pytest.mark.parametrize("element", ["load_bearing_wall", None])
def test_walls_and_unclear_elements_keep_unknown(element):
    loc = locate(*pair(1200), element)
    assert loc.linked and loc.position == "unknown"


def _png(a: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "PNG")
    return buf.getvalue()


def test_through_the_app_the_position_and_the_axis_reach_the_level():
    close, _, element = pair(1200, turn_close=20)
    r = run_measured(Image.fromarray(np.repeat(close[..., None], 3, axis=2)), FakeModel())
    assert r.statement == "Crack found" and r.link_grey is not None and r.link_line_px is not None
    wall = Wall()
    r.wall_number = wall.add(r.statement, r.overlay, r.to_dict())
    analyse_element(_png(element), r, wall, FakeElementModel("column"))
    assert r.location["position"] == "middle" and r.location_preview is not None
    assert r.suggested_pattern().pattern == "column_horizontal"  # across the column, though the phone was slanted
    assert r.triage["position"] == "middle" and r.triage["level"] == "Expert needed"  # no card: no width
    stats = wall.get(r.wall_number).stats
    assert stats["location"]["position"] == "middle" and stats["triage"]["position"] == "middle"
    assert not re.search(r"\bsafe\b|clearance|permit", str(stats), re.I)


def test_when_the_photos_cannot_be_linked_the_level_is_as_before():
    close, _, _ = pair(1200)
    other = pair(1200, seed=5)[2]
    r = run_measured(Image.fromarray(np.repeat(close[..., None], 3, axis=2)), FakeModel())
    analyse_element(_png(other), r, Wall(), FakeElementModel("column"))
    assert r.location["position"] == "unknown" and r.triage["position"] == "unknown"
    assert auto_triage(r).level == "Expert needed"


def _close_card(close):
    """What the app keeps of the card in the close-up (core._keep_for_link)."""
    from element.locate import card_patch, card_view
    from measure.card import detect_card
    rgb = np.repeat(close[..., None], 3, axis=2)
    det = detect_card(rgb)
    assert det is not None
    patch, q = card_patch(rgb, det.corners_px)
    return (q, patch, card_view(rgb, q)), float(np.linalg.norm(q[1] - q[0])) / max(close.shape)


def test_the_card_links_a_plain_wall_where_key_points_cannot():
    close, crack, element = pair(1200, card=True, plain=True, thin=True)
    assert not locate(close, crack, element, "column").linked  # no card: nothing to hold on to
    card, size = _close_card(close)
    loc = locate(close, crack, element, "column", card, size)
    assert loc.linked and loc.via == "bank card" and loc.position == "middle"
    assert loc.crack_to_axis_deg > 85 and abs(loc.depth_px - 100) < 6


def test_the_card_puts_the_crack_where_it_is_in_the_element_photo():
    close, crack, element = pair(1200, card=True, plain=True, thin=True)
    card, size = _close_card(close)
    from element.locate import card_link, to_link_size
    H, q, _ = card_link(card, size, element, 1.0, to_link_size(element[..., 0])[1])
    mid = cv2.perspectiveTransform(np.float32(crack.mean(axis=0))[None, None], H)[0, 0]
    assert np.allclose(mid, [1600 / CLOSER, 1200 / CLOSER], atol=3)  # the crack's centre in the element photo


def test_a_card_shaped_thing_that_does_not_look_like_the_card_is_not_taken():
    close, crack, element = pair(1200, card=True, decoy=True, plain=True)
    card, size = _close_card(close)
    loc = locate(close, crack, element, "column", card, size)
    assert loc.linked and loc.via == "bank card" and loc.position == "middle"
    close2, crack2, element2 = pair(1200, card=True, plain=True)
    element2 = pair(1200, decoy=True, plain=True, seed=3)[2]  # only the decoy, not the card
    loc2 = locate(close2, crack2, element2, "column", *_close_card(close2))
    assert not loc2.linked and loc2.position == "unknown"


def test_a_card_near_the_end_gives_the_end_zone():
    close, crack, element = pair(560, card=True, plain=True)
    loc = locate(close, crack, element, "column", *_close_card(close))
    assert loc.via == "bank card" and loc.position == "end_zone"


def test_through_the_app_the_card_links_the_photos():
    close, _, element = pair(1200, card=True, plain=True)
    r = run_measured(Image.fromarray(np.repeat(close[..., None], 3, axis=2)), FakeModel(), card_placed=True)
    assert r.card is not None and r.link_card is not None and r.statement == "Crack found"
    analyse_element(_png(element), r, Wall(), FakeElementModel("column"))
    assert r.location["via"] == "bank card" and r.location["position"] == "middle"
    assert r.triage["position"] == "middle"


def test_a_card_laid_upright_is_searched_for_upright():
    close, crack, element = pair(1200, card=True, plain=True, thin=True, upright=True)
    loc = locate(close, crack, element, "column", *_close_card(close))
    assert loc.via == "bank card" and loc.position == "middle" and loc.diag["card_upright_in_photo"]


def test_what_each_step_found_is_kept_for_testing():
    close, crack, element = pair(1200, card=True, plain=True)
    d = locate(close, crack, element, "column", *_close_card(close)).diag
    assert d["card_in_close_up"] and d["card_link"] == "linked" and d["element_edges"] and d["look_scores"][0] > 0.5
    d2 = locate(close, crack, pair(1200, plain=True, seed=3)[2], "column", *_close_card(close)).diag
    assert d2["card_link"] != "linked" and "look_scores" in d2


def test_a_black_card_with_little_print_is_found_among_dark_things():
    """Field test of 11 Nov: a black card; the look of its inside alone matched a plant. With the wall round it, it is
    a dark card on a light wall."""
    close, crack, element = pair(1200, card=True, plain=True, thin=True, dark=True, clutter=True)
    loc = locate(close, crack, element, "column", *_close_card(close))
    assert loc.via == "bank card" and loc.position == "middle", loc.diag


def _room(turn_close=0.0, turn_room=0.0):
    """A plain wall with a door frame, a lintel and a ceiling line; a crack at 45 degrees up and to the right above
    the door's right corner; the card under it (field test of 11 Nov, D61)."""
    rng = np.random.default_rng(2)
    hi = (150 + rng.normal(0, 6, (2400, 3200))).astype(np.float32)
    hi[150:170, :] = 90  # ceiling line
    hi[900:2400, 600:640] = 70  # door frame, left
    hi[900:2400, 1300:1340] = 70  # door frame, right
    hi[860:900, 600:1340] = 70  # lintel
    cv2.line(hi, (1450, 800), (1750, 500), 30, 3)  # the crack, 45 degrees
    _card(hi, 1500, 850, dark=True)
    hi = np.clip(hi, 0, 255).astype(np.uint8)
    close = hi[200:1100, 1000:2200].copy()
    crack = np.array([[450.0, 600.0], [600.0, 450.0], [750.0, 300.0]])
    if turn_close:
        m = cv2.getRotationMatrix2D((600, 450), turn_close, 1.0)
        close = cv2.warpAffine(close, m, (1200, 900), borderMode=cv2.BORDER_REFLECT)
        crack = (m[:, :2] @ crack.T).T + m[:, 2]
    room = cv2.resize(hi, (800, 600), interpolation=cv2.INTER_AREA)
    if turn_room:
        m = cv2.getRotationMatrix2D((400, 300), turn_room, 1.0)
        room = cv2.warpAffine(room, m, (800, 600), borderMode=cv2.BORDER_REFLECT)
    return close, crack, np.repeat(room[..., None], 3, axis=2)


def test_the_crack_direction_is_read_against_the_rooms_vertical_lines():
    close, crack, room = _room(turn_close=20, turn_room=8)
    loc = locate(close, crack, room, "load_bearing_wall", *_close_card(close))
    assert loc.via == "bank card" and loc.position == "unknown"  # a wall: no ends read
    assert abs(loc.crack_to_vertical_deg - 45) < 6 and loc.direction == "diagonal", loc.diag
    assert loc.diag["true_vertical_found"]


def test_without_room_lines_there_is_no_true_vertical():
    close, crack, element = pair(1200, card=True, plain=True)
    room = element.copy()
    room[:] = 150  # nothing straight in the room photo but the card
    room[250:400, 300:500] = element[250:400, 300:500]
    loc = locate(close, crack, room, "load_bearing_wall", *_close_card(close))
    assert loc.crack_to_vertical_deg is None and loc.direction is None


def test_a_diagonal_crack_in_a_wall_gets_its_type_through_the_app_although_the_phone_was_slanted():
    from app.core import axis_angle
    from triage.pattern_rules import suggest
    close, crack, room = _room(turn_close=20, turn_room=8)
    loc = locate(close, crack, room, "load_bearing_wall", *_close_card(close)).to_dict()
    angle, source = axis_angle(loc, "load_bearing_wall")
    assert "room" in source
    shape = {"angle_to_vertical_deg": angle, "straightness": 0.98, "coherence": 0.9, "n_loops": 0,
             "branch_ratio": 1.0}
    assert suggest(shape, "load_bearing_wall").pattern == "diagonal_shear"
    phone = {**shape, "angle_to_vertical_deg": 25.0}  # what the slanted phone alone would have read
    assert suggest(phone, "load_bearing_wall").pattern is None
