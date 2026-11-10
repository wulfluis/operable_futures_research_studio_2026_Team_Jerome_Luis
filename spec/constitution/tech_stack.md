# Tech stack

Status on 10 Nov 2026: partly built; each line says what runs now. Full pipeline: `spec/features/pipeline.md`.

- Language: Python.
- Quality gate: written rules on sharpness and brightness, measured at a fixed working size (`spec/features/quality_gate.md`).
- Material model: image classifier, concrete / not concrete / unsure, with confidence; trained first on public concrete and brick images. Residents are never asked the material (D25).
- Element model: image classifier on the element photo (column, beam, slab, wall), pre-trained on PHI-Net task 6 [55]. YOLO only if element boxes are needed later; the `ultralytics` package is AGPL-3.0 (`spec/decisions.md` D9).
- Segmentation: U-Net with a ResNet-34 encoder (`segmentation-models-pytorch`), model v4 trained on 12,477 public photos (D48), with a written crack rule on top (`src/segmentation/decide.py`). Close-ups only: wide views are read wrongly.
- Measurement: plain code (`src/measure/`): a bank card of ISO size (85.60 x 53.98 mm) next to the crack is found automatically, the camera tilt is checked, and the width (widest point and median) and length come in mm with an uncertainty of ±1 pixel plus the scale error (D41, D42, D46). The coin is not used for measurement (D42). The printed scale card (D24) is still to be designed. Widths are not calibrated yet.
- Pattern model: classifies the crack pattern and reports its probability; pre-trained on PHI-Net tasks 3 and 8, the other patterns need our labels.
- Agent: adds the number of floors, type, year and footprint of a building from OpenStreetMap over MCP (the official `mcp` Python SDK; `spec/features/mcp_building_data.md`). MCP data is untrusted input: it adds context and never lowers a level (AGENTS.md rule 8).
- Rules: written triage rules in `src/triage/` (auditable, not learned): risk matrix v3 per crack; building risk score per building (`spec/features/building_score.md`).
- Monitoring: crack IDs, marker stickers and repeat photos (`spec/features/monitoring.md`).
- Interfaces: resident app and engineer review screen (Streamlit prototype). The class demo runs now (`src/app/`: phone page, live wall, a page per result) on a fixed ngrok link (D40).
- Learning loop: engineer corrections stored as labels; batch retraining of the learned models, checked on held-out data before use (human-in-the-loop learning).
- Tests: pytest.

## Data
See `docs/data_map.md`.
