import csv

import numpy as np

from viz.segmentation_video import H, PER_OUTCOME, W, _wrap, frame, pick


def test_frame_is_full_hd_and_shows_red_only_when_faded_in():
    photo = np.full((416, 416, 3), 150, np.uint8)
    mask = np.zeros((416, 416), bool); mask[200:210, :] = True
    before = frame(photo, mask, 0.0, "Crack found", "x", "Crack\nCORRECT", 1, 5)
    after = frame(photo, mask, 1.0, "Crack found", "x", "Crack\nCORRECT", 1, 5)
    assert before.shape == (H, W, 3) and after.shape == (H, W, 3)
    redness = lambda f: int(((f[..., 2] > 180) & (f[..., 1] < 110)).sum())  # BGR
    assert redness(before) == 0 and redness(after) > 1000


def test_wrap_keeps_every_word():
    text = "The model marks a continuous crack line in the photo."
    assert all(len(line) <= 22 for line in _wrap(text, 22).splitlines())
    assert _wrap(text, 22).split() == text.split()


def test_pick_takes_each_outcome_up_to_its_quota(tmp_path):
    rows = [dict(file=f"{o}_{i}.png", outcome=o, has_crack="True") for o in PER_OUTCOME for i in range(10)]
    rows = [r for r in rows if not (r["outcome"] == "false_alarm" and r["file"] != "false_alarm_0.png")]
    path = tmp_path / "photos.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    chosen = pick(path)
    assert len(chosen) == sum(PER_OUTCOME.values()) - PER_OUTCOME["false_alarm"] + 1
    assert chosen == pick(path)  # repeatable
