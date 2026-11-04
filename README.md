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
**Crack model, in use (v3, 30 Oct):** a U-Net with a ResNet-34 encoder (pre-trained on ImageNet) marks the crack pixels in a photo. It was trained for 25 rounds on 8,241 photos (8,082 public crack close-ups from CrackSeg9k [25] and crack-seg [33], plus 159 brick walls without cracks from Darbhanga [30]), on a laptop GPU. A written rule then says "Crack found", "No crack found" or "Expert needed" (`src/segmentation/decide.py`). Details: `docs/experiments/2026-10-29_unet_first_training.md`, decisions D31 and D32.
**App, working (class demo):** a phone page where anyone sends a crack photo and gets the answer with the crack marked in red, and a live wall for the projector that collects the shared results (`src/app/`, `spec/features/class_demo.md`). See "Run the app" below.
**Not trained yet:** the material model (concrete or not), the element model (column, beam, slab, wall), the crack-pattern model and a "close-up of a surface?" check. Width in mm is not measured yet, so the risk categories are not in the app.
Decisions are logged in `spec/decisions.md`. The plan to the final presentation (17 Dec) is in `spec/constitution/roadmap.md`; this week's tasks in `docs/weekly_todo.md`.

## Results
**Crack model v3, on 1,203 public test photos it never saw in training:** 538 of 550 cracks found (12 missed), 7 false alarms on 653 photos without a crack (none on brick), pixel F1 0.63 (`docs/experiments/2026-10-29_unet_first_training.md`).
**Known failure:** it only knows close-ups. On 30 photos of whole damaged buildings (Haiti 2010, local only) it still said "Crack found" 28 times: it reads window frames, wires and edges as cracks. Next: a "close-up of a surface?" check before the model.
These are scores on public close-ups, not on our own photos, and not yet against the non-AI baseline. Success means beating the baseline (an engineer's visual check) on held-out, engineer-labelled photos, while missing almost no Category 3 cases (a target, planned for 3 Dec).
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
- `src/quality/`: photo quality gate; `src/segmentation/`: crack model (prepare, train, predict, decide, evaluate)
- `src/app/`: the app (phone page, live wall, QR page, launcher, word mark)
- `docs/presentation/midterm/`: midterm deck source (6 Nov)
- `tests/`: tests (the comparison against the non-AI baseline runs once engineer-labelled photos exist)
- `notebook/`, `figures/`, `models/`, `data/`: notebooks, plots, model files, local data (large files are not committed)
- `branding/logos/`: logo ideas (SVG)
- `requirements.txt`: Python packages

## AI usage disclosure
This repository was set up with help from Claude (Anthropic). Review everything before relying on it.

## Run the tests
    pip install -r requirements.txt
    pytest

## Run the app
1. Python 3.10 or newer. With an NVIDIA GPU, install torch first with the CUDA build named in `requirements.txt`; without one the app runs on the CPU, slower (a few seconds per photo).
2. `pip install -r requirements.txt`
3. Download the model file [unet_crack.pt](https://github.com/iaac-maai/operable_futures_research_studio_2026_Team_Jerome_Luis/releases/download/model-v3/unet_crack.pt) (v3, about 98 MB, release `model-v3`) and put it in `models/`.
4. On this laptop only: `py -m streamlit run src/app/second_look.py`, then open `http://localhost:8501` (phone page) and `http://localhost:8501/?wall` (live wall).
5. With a public link for phones (class demo): `py src/app/launch.py`. It opens the team's fixed link https://cranium-bladder-tapioca.ngrok-free.dev through ngrok (install it, then `ngrok config add-authtoken <your token>` once; never commit the token), so the QR is always the same (`docs/presentation/midterm/assets/qr_fixed.png`). Only one laptop can serve that link at a time. Without ngrok it falls back to a random Cloudflare link (`winget install --id Cloudflare.cloudflared`). It prints the phone link, the wall link for a teammate and the QR page (`http://localhost:8501/?qr`). The link only answers while the app runs; the free ngrok plan shows a one-time "You are about to visit" page first.

Photos are analysed in memory and never saved; the wall forgets them when the app stops.

## License
Code under the MIT License (see `LICENSE`). Datasets follow their own terms.
