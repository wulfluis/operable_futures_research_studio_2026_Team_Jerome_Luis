# Feature: each crack in 3D, built by Grasshopper, turned with the mouse on the wall (`src/gh3d/`, `src/app/`)

Status: first version, 11 Nov 2026, decision D57. The Grasshopper script is written for Rhino 8 (Python 3 script component); its first run inside Rhino is Jerome's.

## Purpose
An engineer reads an axonometry of the element with the crack on it faster than a table of numbers. Every shared crack measured with a bank card gets a schematic 3D model: the element, the measured crack on its face, the card for scale and the level. Grasshopper builds it; the wall's result page shows it, turning by itself until someone drags it with the mouse.

## Flow (automatic)
```
phone -> app analyses the crack -> data/results/cracks/007.json   (numbers only)
      -> Grasshopper (opened once, a Timer re-reads the folder) -> 007.mesh.json   (the 3D model)
      -> the wall's page for result 7 shows it: turns by itself, drag to turn, scroll to zoom
```
- The app writes one file per **shared** result (`NNN.json`, the result's numbers as in "Result (JSON)") and rewrites it when the element photo or the level arrives. Never the photo (rule 9).
- The folder is emptied when the app starts, so, like the wall, it forgets everything when the app stops. Default `data/results/cracks/` (ignored by git); `SECOND_LOOK_EXPORT_DIR` changes it.
- Grasshopper writes `NNN.mesh.json` next to it. Until it exists the page says "3D model being drawn (Grasshopper)" and checks again every 3 seconds.

## What is measured and what is assumed (rule 7)
| In the model | Source |
|---|---|
| Crack path (main line), its length, widest and median width | **measured** with the bank card (only with the card tick and a card found, D56) |
| The card's outline | **measured**, same wall plane |
| Element type | element model (D52), or "unknown" (drawn as a wall panel) |
| Element size | **assumption**, typical sizes below |
| Crack position on the element | **assumption**: centred on the front face (the position is not known; the matrix counts it as an end zone) |
The label always says: "Schematic: element size and crack position assumed; crack shape and width measured."

Typical sizes (assumption, illustrative; mm): column 300 wide x 300 deep x 3000 high; beam 4000 long x 250 wide x 500 deep; wall (load-bearing, infill or unknown) 3000 wide x 200 thick x 3000 high; slab 4000 x 4000 x 150, crack on its underside.

## The export (`geometry` in the result's numbers)
- `frame`: wall plane, mm; x to the right and y up as the phone saw it (phone upright); origin at the card's centre.
- `crack_path_mm`: the main crack line, at most 200 points, in order along the crack.
- `widest_point_mm`, `card_corners_mm`, and the widths and length.

## The model (`src/gh3d/layout.py`, plain Python, tested; used by the Grasshopper script)
In mm, Z up. Objects, each with a colour in the figure style (black, white, grey; the crack red):
- `element`: the box (grey);
- `crack`: a ribbon as wide as the measured median width, 1 mm in front of the face (red), and the crack line itself (red, always visible however small the crack);
- `widest`: a small ring at the widest point (white);
- `card`: the card outline (white);
- `label`: lines of text (number, element, widths, level, the schematic note).

## Grasshopper (`src/gh3d/second_look_gh.py`)
A Python 3 script component. Inputs: `folder` (text), `run` (boolean), plus a Timer of 2 s on the component. For each `NNN.json` without a newer `NNN.mesh.json` it builds the model with `layout.py`, writes `NNN.mesh.json`, and outputs the meshes for the Rhino preview (`meshes`, `lines`, `labels`). Nothing to draw by hand.

## The viewer (wall result page)
A three.js view (version 0.128.0 from jsDelivr, an exact version) inside the page: dark background, the model turning slowly; it stops turning as soon as it is dragged; scroll zooms. It needs the internet for the library, as the fixed link does.

## Limits
- Main crack line only, no branches; one crack on one face.
- No card: no model (the page says why).
- Rhino must be open with the definition for new models to appear.

## Tests
`tests/test_gh3d.py`: export (path length within 5 % of the measured length, card 85.6 x 54 mm, up is up), layout (crack on the face, typical sizes, label says assumed and never "safe", no card means no model), the folder (written on share, rewritten on update, emptied at start, no photo), the viewer page (three.js, auto-rotate that stops on drag, the model's data).
