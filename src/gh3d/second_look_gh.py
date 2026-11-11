"""Second Look, each crack in 3D (D57): script for a Grasshopper **Python 3 Script** component, Rhino 8.
Spec: spec/features/rhino_3d.md.

Setup, once:
  1. Grasshopper: double-click the canvas, type "Python 3 Script", Enter. Double-click it, paste this whole file.
  2. Inputs (zoom in on the component, use the + to add them; right-click each to rename):
       folder : a Panel with the full path of the app's folder, ...\\data\\results\\cracks   (Type hint: str)
       run    : a Boolean Toggle, True                                                   (Type hint: bool)
  3. Outputs (same way): meshes, cracks, outlines, labels, status.
  4. A Timer on the canvas (double-click, type "Timer"); right-click it, Interval 2 seconds; drag its output onto
     this component. New cracks now appear by themselves.
Each NNN.json the app writes becomes NNN.mesh.json (read by the wall's 3D view) and a model in the Rhino preview,
models side by side along X, at real size in the file's units (mm, cm, m or inches). Nothing is drawn by hand.
"""

import importlib
import json
import os
import sys

import Rhino
import Rhino.Geometry as rg

SPACING_MM = 6000  # between models in the preview, along X


def _src_from(folder):
    """The repository's src folder, three levels above data/results/cracks."""
    return os.path.normpath(os.path.join(folder, "..", "..", "..", "src"))


def unit_scale():
    """The model is in millimetres; this file may be in metres, centimetres or inches. Factor from mm to the file's
    units (1.0 for a millimetre file, or when it cannot be read)."""
    try:
        doc = Rhino.RhinoDoc.ActiveDoc
        return float(Rhino.RhinoMath.UnitScale(Rhino.UnitSystem.Millimeters, doc.ModelUnitSystem)), str(doc.ModelUnitSystem)
    except Exception:
        return 1.0, "Millimeters (assumed)"


def _to_rhino(model, dx, s=1.0):
    """The model's objects as Rhino geometry, scaled from mm to the file's units (s) and moved dx (mm) along X."""
    scale = rg.Transform.Scale(rg.Point3d(0, 0, 0), s)
    move = rg.Transform.Translation(dx * s, 0, 0)
    meshes, cracks, outlines = [], [], []
    for o in model["objects"]:
        if o["kind"] == "mesh":
            m = rg.Mesh()
            for p in o["vertices"]:
                m.Vertices.Add(*p)
            for a, b, c in o["faces"]:
                m.Faces.AddFace(a, b, c)
            m.Normals.ComputeNormals()
            m.Transform(scale)
            m.Transform(move)
            meshes.append(m)
        elif o["kind"] == "segments":
            for a, b in o["segments"]:
                line = rg.LineCurve(rg.Point3d(*a), rg.Point3d(*b))
                line.Transform(scale)
                line.Transform(move)
                outlines.append(line)
        else:
            curve = rg.Polyline([rg.Point3d(*p) for p in o["points"]]).ToNurbsCurve()
            curve.Transform(scale)
            curve.Transform(move)
            (cracks if o["name"] in ("crack_line", "crack_zone") else outlines).append(curve)
    return meshes, cracks, outlines


def run_once(folder):
    src = _src_from(folder)
    if src not in sys.path:
        sys.path.insert(0, src)
    from gh3d import layout
    importlib.reload(layout)  # picks up a new version of layout.py without restarting Rhino

    names = sorted(n for n in os.listdir(folder) if n.endswith(".json") and not n.endswith(".mesh.json")
                   and n[:3].isdigit())
    all_meshes, all_cracks, all_outlines, all_labels = [], [], [], []
    made = 0
    s, units = unit_scale()
    for i, name in enumerate(names):
        src_path = os.path.join(folder, name)
        mesh_path = src_path[:-5] + ".mesh.json"
        try:
            with open(src_path, encoding="utf-8") as f:
                stats = json.load(f)
        except (OSError, ValueError):
            continue  # being written: next tick
        model = layout.build(stats)
        if model is None:
            continue  # no bank card: nothing measured to draw
        if not os.path.exists(mesh_path) or os.path.getmtime(mesh_path) < os.path.getmtime(src_path):
            tmp = mesh_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(model, f)
            os.replace(tmp, mesh_path)
            made += 1
        m, c, o = _to_rhino(model, i * SPACING_MM, s)
        all_meshes += m
        all_cracks += c
        all_outlines += o
        all_labels.append(" | ".join(model["label"]))
    status = (f"{len(all_labels)} cracks in 3D, {made} new this tick; file units {units}, drawn at real size "
              f"(1 mm = {s:g} {units.lower()}) (folder: {folder})")
    return all_meshes, all_cracks, all_outlines, all_labels, status


meshes, cracks, outlines, labels = [], [], [], []
if not run:  # noqa: F821  (Grasshopper input)
    status = "Off: set run to True."
elif not folder or not os.path.isdir(folder):  # noqa: F821  (Grasshopper input)
    status = f"Folder not found: {folder!r}. Paste the app's data\\results\\cracks path into the Panel."  # noqa: F821
else:
    try:
        meshes, cracks, outlines, labels, status = run_once(folder)  # noqa: F821
    except Exception as e:  # shown on the component, so it can be pasted back to us
        status = f"Error: {type(e).__name__}: {e}"
