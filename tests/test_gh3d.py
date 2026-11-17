"""Each crack in 3D (spec/features/rhino_3d.md, D57): the export, the layout, the folder, the Grasshopper script (with a
stand-in for Rhino) and the wall's viewer page. No Rhino and no browser are needed."""

import json
import math
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from app.core import Wall, auto_triage, mesh_name, run_measured
from app.viewer3d import viewer_html
from gh3d import layout
from test_app_measure import DEMO_MODEL, FakeModel, _demo, _plain_wall_with_line
from triage.element_rule import ElementAnswer

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def measured():
    photo, truth = _demo(1.0)
    return run_measured(photo, DEMO_MODEL, card_placed=True), truth


def _length(points):
    p = np.asarray(points, float)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


# ---------- the export ----------

def test_the_crack_path_in_mm_matches_the_measured_length(measured):
    r, truth = measured
    g = r.to_dict()["geometry"]
    assert 2 <= len(g["crack_path_mm"]) <= 200
    assert abs(_length(g["crack_path_mm"]) - r.crack.length_main_mm) / r.crack.length_main_mm < 0.05
    assert abs(_length(g["crack_path_mm"]) - truth["length_mm"]) / truth["length_mm"] < 0.05


def test_the_card_is_a_bank_card_in_mm(measured):
    c = np.asarray(measured[0].to_dict()["geometry"]["card_corners_mm"])
    sides = sorted(np.linalg.norm(np.roll(c, -1, 0) - c, axis=1))
    assert sides[0] == pytest.approx(53.98, abs=0.5) and sides[-1] == pytest.approx(85.6, abs=0.5)
    assert np.allclose(c.mean(axis=0), 0, atol=0.5)  # origin at the card's centre


def test_up_on_the_wall_is_up_in_the_export(measured):
    """The direction of the crack in the export agrees with the angle the pattern rules read (phone upright)."""
    r, _ = measured
    p = np.asarray(r.geometry["crack_path_mm"])
    d = p[-1] - p[0]
    angle = math.degrees(math.atan2(abs(d[0]), abs(d[1])))  # 0 = vertical
    assert abs(angle - r.crack.angle_to_vertical_deg) < 15


def test_no_card_no_geometry():
    r = run_measured(_plain_wall_with_line(), FakeModel())
    assert r.geometry is None and r.to_dict()["geometry"] is None


# ---------- the layout ----------

def _stats(r, number=7):
    return {"number": number, **r.to_dict()}


def test_the_crack_lies_on_the_front_face_of_the_element(measured):
    r, _ = measured
    m = layout.build(_stats(r))
    objs = {o["name"]: o for o in m["objects"]}
    assert set(objs) >= {"element", "element_edges", "crack", "crack_line", "card"}
    ys = [p[1] for p in objs["crack_line"]["points"]]
    assert all(y == -layout.OFFSET_MM for y in ys)  # on the face, just in front of it
    zs = [p[2] for p in objs["crack_line"]["points"]]
    lo, hi = m["bounds"]["min"][2], m["bounds"]["max"][2]
    assert lo < min(zs) and max(zs) < hi  # inside the element's height


def test_typical_size_per_element_and_unknown_is_a_wall(measured):
    r, _ = measured
    assert layout.build(_stats(r))["element"] == "wall" and not layout.build(_stats(r))["element_known"]
    r.element = ElementAnswer("column", False, 0.95, "")
    m = layout.build(_stats(r))
    assert m["element"] == "column" and m["element_known"]
    (dx, dy, dz), _ = layout.SIZES["column"]
    b = m["bounds"]
    assert b["max"][0] - b["min"][0] == dx and b["max"][2] - b["min"][2] == dz
    r.element = None


def test_the_ribbon_is_as_wide_as_the_measured_crack(measured):
    r, _ = measured
    m = layout.build(_stats(r))
    rib = next(o for o in m["objects"] if o["name"] == "crack")["vertices"]
    a, b = np.asarray(rib[10]), np.asarray(rib[11])  # the two sides at one point of the line
    assert np.linalg.norm(a - b) == pytest.approx(r.geometry["width_median_mm"], rel=0.01)


def test_the_label_says_what_is_assumed_and_never_safe(measured):
    r, _ = measured
    r.element = ElementAnswer("beam", False, 0.95, "")
    auto_triage(r)
    lab = " ".join(layout.build(_stats(r))["label"])
    assert "assumed" in lab and "measured" in lab and "Level:" in lab
    assert not re.search(r"\bsafe\b|clearance|permit", lab, re.I)
    r.element, r.triage = None, None


def test_no_card_no_model():
    r = run_measured(_plain_wall_with_line(), FakeModel())
    assert layout.build(_stats(r)) is None


# ---------- the folder ----------

def test_shared_results_are_written_as_numbers_and_rewritten_on_update(measured, tmp_path):
    r, _ = measured
    wall = Wall(export_dir=tmp_path)
    n = wall.add(r.statement, r.overlay, r.to_dict())
    f = tmp_path / f"{n:03d}.json"
    data = json.loads(f.read_text(encoding="utf-8"))
    assert data["number"] == n and data["geometry"]["crack_path_mm"]
    assert not any(isinstance(v, (bytes, bytearray)) for v in data.values())  # numbers, never the photo
    wall.update(n, {**r.to_dict(), "triage": {"level": "Category 2"}})
    assert json.loads(f.read_text(encoding="utf-8"))["triage"]["level"] == "Category 2"


def test_the_folder_is_emptied_when_the_wall_starts(tmp_path):
    (tmp_path / "001.json").write_text("{}")
    (tmp_path / "001.mesh.json").write_text("{}")
    (tmp_path / "notes.txt").write_text("keep")  # not ours: kept
    Wall(export_dir=tmp_path)
    assert not list(tmp_path.glob("*.json")) and (tmp_path / "notes.txt").exists()


def test_without_an_export_folder_nothing_is_written(measured, tmp_path):
    r, _ = measured
    Wall().add(r.statement, r.overlay, r.to_dict())
    assert not list(tmp_path.iterdir())


# ---------- the Grasshopper script, with a stand-in for Rhino ----------

def _fake_rhino(monkeypatch, units="mm"):
    class Point3d(tuple):
        def __new__(cls, x, y, z):
            return tuple.__new__(cls, (x, y, z))

    class _List(list):
        def Add(self, *a):
            self.append(a)

        def AddFace(self, *a):
            self.append(a)

        def ComputeNormals(self):
            return True

    class Mesh:
        def __init__(self):
            self.Vertices, self.Faces, self.Normals, self.moved = _List(), _List(), _List(), None

        def Transform(self, t):
            self.moved = t

    class Curve:
        def __init__(self, *pts):
            self.pts, self.moved = pts, None

        def Transform(self, t):
            self.moved = t

    class Polyline(list):
        def ToNurbsCurve(self):
            return Curve(*self)

    geo = types.SimpleNamespace(Point3d=Point3d, Mesh=Mesh, LineCurve=Curve, Polyline=Polyline,
                                Vector3d=lambda *v: v, Plane=lambda o, a, b: ("plane", o, a, b),
                                Transform=types.SimpleNamespace(Translation=lambda x, y, z: (x, y, z),
                                                                Scale=lambda origin, s: ("scale", s)))
    rhino = types.SimpleNamespace(Geometry=geo, UnitSystem=types.SimpleNamespace(Millimeters="mm"),
                                  RhinoMath=types.SimpleNamespace(UnitScale=lambda a, b: {"mm": 1.0, "m": 0.001}[b]),
                                  RhinoDoc=types.SimpleNamespace(ActiveDoc=types.SimpleNamespace(ModelUnitSystem=units)))
    monkeypatch.setitem(sys.modules, "Rhino", rhino)
    monkeypatch.setitem(sys.modules, "Rhino.Geometry", geo)


def _run_script(folder, run=True):
    g = {"folder": str(folder), "run": run, "__name__": "gh"}
    exec(compile((ROOT / "src" / "gh3d" / "second_look_gh.py").read_text(encoding="utf-8"), "second_look_gh.py",
                 "exec"), g)
    return g


def test_the_grasshopper_script_builds_each_crack_and_writes_its_model(measured, tmp_path, monkeypatch):
    _fake_rhino(monkeypatch)
    r, _ = measured
    folder = tmp_path / "data" / "results" / "cracks"  # the script finds src/ three levels up, as in the repo
    folder.mkdir(parents=True)
    monkeypatch.syspath_prepend(str(ROOT / "src"))
    wall = Wall(export_dir=folder)
    n1 = wall.add(r.statement, r.overlay, r.to_dict())
    n2 = wall.add(r.statement, r.overlay, r.to_dict())
    no_card = run_measured(_plain_wall_with_line(), FakeModel())
    wall.add(no_card.statement, no_card.overlay, no_card.to_dict())
    g = _run_script(folder)
    assert g["status"].startswith("2 cracks in 3D, 2 new"), g["status"]
    assert (folder / mesh_name(n1)).exists() and (folder / mesh_name(n2)).exists()
    assert len(g["labels"]) == 2 and g["cracks"] and g["meshes"]
    moves = {m.moved for m in g["meshes"]}
    assert (0, 0, 0) in moves and (8000, 0, 0) in moves  # side by side, room for each drawing sheet (D81)
    again = _run_script(folder)
    assert "0 new" in again["status"]  # already built: not rebuilt


def test_the_grasshopper_script_reports_a_wrong_folder(monkeypatch, tmp_path):
    _fake_rhino(monkeypatch)
    assert "Folder not found" in _run_script(tmp_path / "nope")["status"]
    assert _run_script(tmp_path, run=False)["status"].startswith("Off")


# ---------- the wall's viewer ----------

def test_the_viewer_turns_by_itself_until_dragged_and_carries_the_model(measured):
    r, _ = measured
    page = viewer_html(layout.build(_stats(r)))
    assert "three@0.128.0/build/three.min.js" in page and "OrbitControls.js" in page
    assert "autoRotate = true" in page and 'addEventListener("start"' in page and "autoRotate = false" in page
    assert '"crack_line"' in page and "dblclick" in page
    assert "</script>" not in page.split("const M = ")[1].split(";\n")[0]  # no text can end the script early


def test_the_grasshopper_script_draws_at_real_size_in_a_metre_file(measured, tmp_path, monkeypatch):
    """11 Nov: in a Rhino file in metres the column came out hundreds of metres wide. The model is in mm; the script
    scales it to the file's units."""
    _fake_rhino(monkeypatch, units="m")
    r, _ = measured
    folder = tmp_path / "data" / "results" / "cracks"
    folder.mkdir(parents=True)
    monkeypatch.syspath_prepend(str(ROOT / "src"))
    wall = Wall(export_dir=folder)
    wall.add(r.statement, r.overlay, r.to_dict())
    wall.add(r.statement, r.overlay, r.to_dict())
    g = _run_script(folder)
    assert "units m" in g["status"] and "1 mm = 0.001" in g["status"]
    assert {m.moved for m in g["meshes"]} >= {(0.0, 0, 0), (8.0, 0, 0)}  # 8000 mm apart = 8 m in a metre file


def test_a_red_frame_marks_where_the_crack_is_and_the_crack_keeps_its_real_size(measured):
    """11 Nov: at real size the cracks were hard to find on their elements. A frame marks the place; the crack itself
    is not enlarged (rule 7), and the label says so."""
    r, _ = measured
    m = layout.build(_stats(r))
    objs = {o["name"]: o for o in m["objects"]}
    zone = np.asarray(objs["crack_zone"]["points"])
    line = np.asarray(objs["crack_line"]["points"])
    assert zone[:, 0].min() < line[:, 0].min() and zone[:, 0].max() > line[:, 0].max()
    assert zone[:, 2].min() < line[:, 2].min() and zone[:, 2].max() > line[:, 2].max()
    assert abs(_length(line) - r.crack.length_main_mm) / r.crack.length_main_mm < 0.05  # real size
    assert any("real size" in s for s in m["label"])


def test_the_viewer_opens_on_the_crack_and_switches_to_the_whole_element(measured):
    page = viewer_html(layout.build(_stats(measured[0])))
    assert "frame(crack, 2.2)" in page and "frame(scene, 1.25)" in page and "crack / whole element" in page


# ---------- D81: the detailed drawing ----------

def test_the_detailed_drawing_has_sections_tension_and_a_sheet(measured):
    """Stations every 50 mm with the measured width, tension arrows across the crack (direction only), and a sheet
    of enlarged sections whose depth is drawn as not measured; the crack's ribbon follows the width point by point."""
    r, _ = measured
    g = r.to_dict()["geometry"]
    assert g["stations"] and all(s["width_mm"] > 0 for s in g["stations"])
    assert [s["name"] for s in g["stations"]][:2] == ["A", "B"]
    assert len(g["width_along_mm"]) == len(g["crack_path_mm"])
    m = layout.build(_stats(r))
    names = {o["name"] for o in m["objects"]}
    assert {"stations", "tension", "sheet", "dimensions", "not_measured", "width_chart"} <= names
    texts = [o["text"] for o in m["objects"] if o["kind"] == "text"]
    assert any(t.endswith(" mm") and t[0].isdigit() for t in texts)  # a width as a dimension
    assert "depth ?" in texts and not any("stress" in t.lower() for t in texts)  # no invented depth or stress
    sheet = [o for o in m["objects"] if o.get("group") == "sheet"]
    xs = [p[0] for o in sheet if o["kind"] == "segments" for s in o["segments"] for p in s]
    assert min(xs) > m["bounds"]["max"][0]  # the sheet stands beside the element, not on it


def test_the_viewer_draws_text_and_offers_the_sheet(measured):
    page = viewer_html(layout.build(_stats(measured[0])))
    assert "textSprite" in page and "drawing sheet" in page
