"""D101: the simulator's --everything sends each real case once, and only the EPFL photos reviewed by eye."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from app import simulate  # noqa: E402


def test_each_real_case_is_sent_once(tmp_path, monkeypatch):
    a = {"photo": "p1.jpg", "crop": [0, 0, 10, 10], "element": "wall"}
    b = {"photo": "p1.jpg", "crop": [5, 5, 15, 15], "element": "wall"}  # D113: same photo, another close-up: left out
    c = {"photo": "p2.jpg", "crop": [0, 0, 10, 10], "element": "column"}
    d = {"photo": "p3.jpg", "crop": [0, 0, 10, 10], "element": "beam"}  # reviewed and dropped (D113)
    for name, cases in (("t.json", [a, b]), ("l.json", [a, {"synthetic": True}, d]), ("p.json", [b, c])):
        (tmp_path / name).write_text(json.dumps({"cases": cases}), encoding="utf-8")
    (tmp_path / "dropped.json").write_text(json.dumps({"cases": [{"photo": "p3.jpg", "crop": [0, 0, 10, 10]}]}),
                                          encoding="utf-8")
    monkeypatch.setattr(simulate, "TEST_SET", tmp_path / "t.json")
    monkeypatch.setattr(simulate, "BY_LEVEL", tmp_path / "l.json")
    monkeypatch.setattr(simulate, "PICKS", tmp_path / "p.json")
    monkeypatch.setattr(simulate, "DROPPED", tmp_path / "dropped.json")
    assert simulate.all_real_cases() == [a, c]


def test_no_epfl_folder_sends_no_epfl_case(tmp_path):
    assert list(simulate.topo_cases(tmp_path / "missing")) == []


def test_only_reviewed_epfl_photos_are_kept():
    assert "DSC_0123.png" not in simulate.TOPO_KEEP and simulate.TOPO_KEEP == {"AD0021.png", "IMG_1400.png"}  # D113
