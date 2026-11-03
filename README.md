# Second Look

Second Look helps people living in ageing reinforced-concrete buildings decide whether a crack needs an engineer. A resident photographs a crack in a concrete beam, column, slab or wall. The app checks the photo, measures the crack's width in millimetres and recognises its pattern. Written, auditable rules then sort it into one of three categories: monitor, low-cost advice and a recheck, or an engineer must confirm. Whenever the system is unsure, it sends the case to an expert, and every correction the engineer makes improves the models. It never says a building is safe: it decides who looks next, and how soon.

IAAC Barcelona, MaAI, Research Studio "Operable Futures", 2026-27.
Team: Jerome Eid (@jeromgeid-ui), Luis Sebastian Wulf (@wulfluis).
Project stage: Stage 4, Operation and End-of-Life.

## Problem
Residents of old buildings see cracks every day but cannot tell a harmless one from a sign of structural failure, and engineers are too few to inspect every building.
Tripoli, February 2026: two blocks collapsed on 8 February (14 to 15 dead, per AFP and Reuters), and 114 buildings were ordered evacuated.
Rana Plaza, Dhaka, 2013: cracks were visible the day before the collapse.

## What it does
The unit is the **building**, photographed at three scales as in PHI-Net: **S** the building, **O** the element, **P** the crack.
- **S:** a short building questionnaire (age, storeys, floors added, typology and its changes) and a facade photo give a **building risk score** that ranks buildings for engineers (P1, P2, P3).
- **O:** models identify the **material** (concrete or not) and the **element** (column, beam, slab, wall); residents are never asked.
- **P:** the crack close-up with a printed scale card is checked (sharp, at least 3 pixels across the crack), segmented and measured (width in mm with its uncertainty); the written risk matrix gives Category 1, Category 2, Category 3, "Expert needed" or "Retake", each with its confidence and a one-sentence reason.
- **Monitoring:** the same crack is photographed again on its recheck date; growth larger than the measurement uncertainty raises its level.
- **Engineers** review every Category 3 and every referral; their corrections retrain the models (human-in-the-loop learning).

Full pipeline: `spec/features/pipeline.md` and `figures/01_pipeline.png`.
It never gives a permit or a clearance. Any doubt goes to an expert, and an engineer confirms every Category 3.

## Data
The concept is cracks in buildings in slum-like, informal neighbourhoods. See `docs/data_map.md`: every source is ranked by how close it is to that case.
- Closest public data: engineer surveys of damaged buildings after the Haiti 2010 and Nepal 2015 earthquakes, rated light / moderate / severe (one rating per building, no scale).
- Building and masonry crack images: METU, Darbhanga, Strathclyde (crack / no crack, no scale).
- Generic crack pixels for pre-training: CrackSeg9k, crack-seg. Bridge and pavement sets are off-concept.
- Still to collect: our own phone photos with the printed scale card, rated by our engineering partners (Engineers Without Borders and a third-party structural engineer).

**Can we train with it?** Partly. There is enough public data to train the step that finds cracks (segmentation and detection). There is not enough to train or prove the step that rates danger: no public dataset has our seven crack patterns, a scale for width in mm, or an engineer's rating per crack. Those come only from our own photos and the engineering partners (`docs/data_map.md`, section 0b).

Raw data is not committed; `data/` stays empty in the repository.

## Method and status
The risk matrix and decision rules are implemented in `src/triage/` with tests, and so is the dispatch rule that decides when a field worker visits after a resident's report (`spec/features/dispatch.md`, D27). Thresholds are illustrative until calibrated with a structural engineer.
The confusion tables (true and false positives and negatives, with missed Category 3 as the key error) are defined in `spec/features/evaluation.md` and coded in `src/triage/evaluate.py`.
The quality gate (`src/quality/`) ran as a first test on real images from five public datasets. It found and fixed a resolution bug in the blur check, and showed that the blur check confuses smooth plaster and concrete with blur (decision D12) (`docs/experiments/`). Since 30 Oct a photo counts as blurred only when two blur measures agree, which stopped most sharp crack close-ups being sent to retake (D33; thresholds still to calibrate).
First model trained: crack or no crack with a U-Net, tested on 721 unseen public photos (`docs/experiments/2026-10-29_unet_first_training.md`). Model v3 (30 Oct) adds no-crack brick photos (D32). Class demo: `py src/app/launch.py` (`spec/features/class_demo.md`). The other models are not trained yet. Decisions are logged in `spec/decisions.md`. The plan to the final presentation (17 Dec) is in `spec/constitution/roadmap.md`; this week's tasks in `docs/weekly_todo.md`.

## Results
None yet. Success means beating the non-AI baseline (an engineer's visual check) on held-out, engineer-labelled photos, while missing almost no Category 3 cases (a target, not a result).
We stop or redesign if the tool misses Category 3 cases that engineers would flag.

## Limitations
A surface photo cannot show rebar, concrete strength or foundations. Concrete and brick crack differently and need separate data.
Public crack data is mostly concrete and pavement.

## Repository map
- `AGENTS.md`: system constitution for coding agents
- `spec/constitution/`: mission, roadmap, tech stack
- `spec/features/`: triage and evaluation specs; `spec/decisions.md`: decisions log
- `docs/`: segmentation tree, data map, sources, experiments, literature notes (`docs/literature/`), weekly to-do (`docs/weekly_todo.md`)
- `src/triage/`: risk matrix, decision logic, dispatch rule and evaluation tables
- `tests/`: tests (the comparison against the non-AI baseline runs once engineer-labelled photos exist)
- `notebook/`, `figures/`, `models/`, `data/`: notebooks, plots, model files, local data (large files are not committed)
- `branding/logos/`: logo ideas (SVG)
- `requirements.txt`: Python packages

## AI usage disclosure
This repository was set up with help from Claude (Anthropic). Review everything before relying on it.

## Run the tests
    pip install -r requirements.txt
    pytest

## License
Code under the MIT License (see `LICENSE`). Datasets follow their own terms.
