# Feature: full pipeline, version 3 (27 Oct 2026)

Status: specified. Supersedes v2 (26 Oct). Answers the professor's feedback of 27 Oct: engineers do not judge a crack alone, they read the pattern across the building and, above all, whether it is growing; building data (age, added floors, typology and its changes) produce a building risk score; cheap phones are not reliable and scale matters; PHI-Net as the reference for data and machine learning (decisions D21 to D25).
Figure: `figures/01_pipeline.png`.

## Unit of analysis: the building, seen at three scales
The building is the unit. Cracks are evidence inside it. Every photo belongs to one of three scales, defined as in PHI-Net task 1, scene level [53][55]:

| Code | Scale | PHI-Net definition [55] | In Second Look |
|---|---|---|---|
| **S** | structural | taken from far away; shows most of the structure or its outline | the building: one facade photo per building |
| **O** | object | taken from mid-range; shows one or more components | one element (column, beam, slab or wall), photographed whole with the scale card |
| **P** | pixel | taken from very close range, or shows only part of a component | the crack close-up, with the scale card |

PHI-Net puts the scene level first because it "can help decrease the doubt of scale issue" [55]; its later tasks are applied by level: collapse mode at S (task 5), component type at O (task 6), damage level at P and O (task 7), damage type (flexural, shear, combined) and spalling (tasks 8 and 3).
Note: PHI-Net's material task (task 4) is steel vs other materials, so it cannot separate concrete from brick; our material check needs its own data (`docs/data_map.md`).

## Users and devices
- **Field worker:** first visit of a building, sent by the dispatch rule of step 0 (D27), with a reference phone and the printed scale card. Cheap phones are not reliable enough to measure crack width (professor, 27 Oct; at least 3 pixels across the crack are needed [46]). Trained to confirm the material, the element and the crack pattern when a model is unsure.
- **Resident:** reports a crack (step 0) and does the follow-up photos on the recheck dates, with any phone; the app guides the capture. A resident is **never asked** the material, the element or the crack pattern: residents cannot tell concrete from other materials, or a column from a wall (D25).
- **Engineer:** reviews on the engineer screen and on site.

## Step 0: report and dispatch (D27)
A resident reports a crack with any phone: a close-up with the scale card or a coin, and answers they can know. The quality gate, a crack model and a rough width feed a written dispatch rule (`spec/features/dispatch.md`, in code: `src/triage/dispatch.py`) that returns a visit priority: **V1** visit within 1 week, **V2** within 1 month, **V3** no visit yet (recheck in 1 month; 1 in 10 audited), or Retake. Only V1 and V2 (and audited V3) start the field visit below. These are visit priorities, not triage levels.

## Stages: three scales x five phases
Kinds: learned model (reports a confidence), written rule (auditable), people. Status on 27 Oct 2026: **in code**, **partly in code**, **specified** (designed here, not coded yet).

| Scale | Phase | Box | Kind | Output | If it fails or is unsure | Status |
|---|---|---|---|---|---|---|
| S | 1 Capture | Building case (once per building) | people | Questionnaire: age, storeys, floors added, typology and its changes, open ground floor; facade photo; map data over MCP | - | specified |
| S | 2 Check | Completeness check | rule | Each answer known or "unknown" | Unknown counts as a risk flag and is shown as unknown | specified |
| S | 3 Perceive | Building reading | learned (later) | Now: the questionnaire answers. Later: a model on the facade photo (storeys, typology; PHI-Net task 5) | - | specified |
| S | 4 Decide | Building risk score | rule | Priority P1, P2 or P3 with its reasons (`spec/features/building_score.md`) | - | specified |
| S | 5 Act | Engineer | people | Ranked list of buildings and a dossier per building; confirms or corrects every Category 3 and every referral; rates on site | - | specified |
| O | 1 Capture | Element photo | people | The whole element with the scale card; the resident is not asked what it is | - | specified |
| O | 2 Check | Scale and angle | rule | Scale card visible; phone parallel to the wall | Retake | specified |
| O | 3 Perceive | Material + element | learned | Two models, each with a confidence: **material** (concrete or not concrete) and **element** (column, beam, slab, wall; pre-trained on PHI-Net task 6) | Either unsure: Expert needed. A confirmation by a field worker or an engineer is stored for that element and reused at follow-ups | specified |
| O | 4 Decide | Element rules | rule | Not concrete or unsure: Expert needed. Pattern does not fit the element: Expert needed. Column: +1 level | - | **in code** (`src/triage/matrix.py`) |
| P | 1 Capture | Crack close-up | people | Scale card and two marker stickers in the photo; a burst of photos with and without flash (`spec/features/photo_protocol.md`) | - | specified |
| P | 2 Check | Photo checks | rule | Sharp and lit; at least 3 pixels across the crack [46] | Not sharp or lit: Retake. Under 3 pixels: retake closer; still under 3: width band read on the card by eye | **partly in code** (sharpness, brightness: `src/quality/gate.py`) |
| P | 3 Perceive | Crack reading | learned | Crack pixels (segmentation), then width in mm with an uncertainty of ±1 pixel [46], length, orientation. Pattern: now suggested by written rules from the crack's direction and form and the element (`spec/features/pattern_rules.md`, D43), confirmed by a field worker; later a model where data exists (PHI-Net tasks 3 and 8, IDEA [64]); the resident is never asked | No crack found with confidence, or pattern unknown: Expert needed | specified |
| P | 4 Decide | Crack level + growth | rule | Matrix v3: level, reason, next step. Growth counts only if the change is larger than the measurement uncertainty (`spec/features/monitoring.md`) | - | **partly in code** (matrix v3; growth check not yet) |
| P | 5 Act | Resident | people | Level, reason, next step, recheck date; never "safe", never a clearance | - | **partly in code** (outputs of `assess()`; app screen not yet) |

**Scene check** (between Check and Perceive, all scales): is this photo S, O or P? Now the app's guided capture sets the scale (step 1 building, step 2 element, step 3 crack). Later a scene model (PHI-Net task 1) confirms it; a mismatch asks for a retake. Status: specified.

**Between the rows:** the element rules (O) pass the material and the element to the crack level (P); the crack levels, their growth and the spread of damage across elements pass up to the building risk score (S).

## Two loops
- **Monitoring loop:** on the recheck date (Category 1: 6 months, Category 2: 1 month, from the risk matrix) the same crack ID and marker stickers are photographed again, with the previous photo shown as an overlay. Growth larger than the measurement uncertainty raises the crack level and, through it, the building priority (`spec/features/monitoring.md`). Status: specified.
- **Learning loop:** engineer corrections and on-site ratings are stored as labels; the learned models (phase 3) are retrained in batches and each new version is checked on held-out engineer-rated photos before use. Written rules change only by a reviewed decision in `spec/decisions.md` (AGENTS.md rule 5). This semester: one manual retraining as a demonstration. This is human-in-the-loop learning, not reinforcement learning (D18). Status: specified.

## Who answers what (D25)
- **Residents** answer only what they can know: the building questionnaire (age, floors added, changes, open ground floor; "unknown" is always allowed), whether doors or windows have started to stick, and whether the crack is new.
- **Models** identify what residents cannot: the material (concrete or not), the element (column, beam, slab, wall) and the crack pattern, from the photos, each with a confidence.
- **The guided capture** sets the scale (S, O, P).
- **When a model is unsure,** the case goes to an engineer (Expert needed). Confirmations by field workers and engineers are stored with the photo as labels; they retrain the models (learning loop), and a confirmed material or element is reused for that element at follow-ups.

Three models are trained this semester, where public data exists (`docs/data_map.md`): **material** (concrete vs not concrete), **element** (pre-trained on PHI-Net task 6, research licence) and **crack segmentation**. The pattern model needs our own labels; until then a field worker chooses the pattern, and a report without a field worker goes to an engineer.
Known limit: plaster and paint hide the material. Until our own labelled photos exist, expect many "material unclear" referrals.

## Designed but not built this semester
Building reading from the facade photo; open-ground-floor (soft storey) detection; satellite comparison over time to find added floors; PHI-Net scene and damage models; the crack pattern model; crack sliding measurement; automatic retraining.

## Thresholds
All thresholds are illustrative until calibrated with the engineering partners.
