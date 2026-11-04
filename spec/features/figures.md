# Feature: figures (`src/viz/figures.py`)

Status: implemented. Run `py src/viz/figures.py` from the repository root; PNGs go to `figures/`.

Every figure is drawn from the repo's own rules and data map, so it changes when they change. None shows a measured result.

| File | Shows | Drawn from |
|---|---|---|
| `01_pipeline.png` | Pipeline v3: three scales (S, O, P as in PHI-Net task 1) x five phases, scene check, monitoring and learning loops, status of each box | `spec/features/pipeline.md` |
| `01b_pipeline_flow.png` | Step 0, report and dispatch (when a field worker visits: V1, V2, V3, D27), then the same pipeline v3 drawn as a task flow: phases as numbered columns, scales as lanes; diamonds for written checks with stop (Retake, Expert needed) and continue marks, stacked boxes for learned models, both loops. Black, white and grey only: white = photos and people, grey = written rules (code), black = learned models | `spec/features/pipeline.md`, `spec/features/dispatch.md` |
| `01c_pipeline_simple.png` | System architecture for a talk: the story of 01b in five big steps (photo, check, measure, decide, act), one symbol and one word each; the two stops (Retake, Expert needed), the three categories, the engineer, and the recheck and learn loops. Same colour logic as 01b; details (dispatch, scales, scene check) stay in 01b | `spec/features/pipeline.md`, `triage.matrix.assess()` |
| `01d_pipeline_story.png` | System architecture in plain words: the two-minute explanation of the pipeline drawn as a pipeline, in its own sentences. Nine numbered boxes in three rows (report: resident, photo check, model, written dispatch rule with V1 / V2 / V3; field visit: field worker, building, element, crack, the same five steps; decision: models only perceive, risk matrix, Category 1 / 2 / 3, building score, engineer), the Retake and Expert needed stops; recheck loop (categories back to the resident) and learn loop (engineer corrections back to the models). Legend on one line, top right. Same colour logic as 01b | `spec/features/pipeline.md`, `spec/features/dispatch.md` |
| `02_decisions_and_players.png` | Decisions and players: the five players as lifelines, time going down, 14 numbered hand-offs, each with a symbol and one word (key below); decisions only on the written rules and the engineer; a 'who decides what' panel. Black, white and grey | `triage.dispatch.screen()`, `triage.matrix.assess()`, `spec/features/building_score.md` |
| `02b_real_examples.png` | Real examples: up to three real photos of each crack pattern of the risk matrix, in the matrix's order and design (symbol + one word); each photo credited; engineer label or team pick per column. **Local only (git-ignored):** the photos keep their own licences (PHI-Net CC BY-NC-SA 4.0, Wikimedia Commons) | `src/viz/pattern_examples.py` (needs `data/raw/phinet`) |
| `03_risk_matrix.png` | Matrix v3 in black, white and grey: each crack pattern as a drawn symbol with one word (key below), base category per pattern x width band (Category 1 white, 2 grey, 3 black), and what raises it, in words, with new and assumed values tagged | `src/triage/risk_matrix_v3.json` (read at run time) |
| `04_segmentation_tree.png` | Tree v3: crack path (yellow) into the atomic problem, building path (navy) into the building risk score, context (grey); scale badge and evidence under each node | `docs/segmentation_tree.md` |
| `05_data_coverage.png` | Which data source feeds which pipeline stage, by scale | `docs/data_map.md`, section 1 |
| `06_dataset_sizes.png` | What we found, and how it could help: image counts and licences of every dataset and of our own data, and what each can be used for: train a model, test a model or the quality gate, measure crack widths in mm (only photos with a scale); black and white | `docs/data_map.md`, sections 0b, 1, 2 and 6 |
| `07_confusion_template.png` | TP / FN / FP / TN and the full 3 x 5 confusion matrix, no counts | `spec/features/evaluation.md` |
| `08_real_data_quality_gate.png` | Sharpness of public dataset images against the blur limit (made by `src/experiments/real_data_quality.py`, not by this script) | Haiti 2010, Darbhanga, Strathclyde, PHI-Net tasks 8, 3 and 6, METU, SDNET2018, crack-seg; panels ordered by fit to the concept |
| `09_timeline.png` | Plan to the final presentation (17 Dec) with syllabus milestones and tasks at risk | `spec/constitution/roadmap.md` |
| `10_dataset_comparison.png` | Every dataset on our disk side by side, in order of fit: images, quality gate pass rate, labels, what it trains, licence; one row per PHI-Net task | `docs/data_map.md`, sections 0 and 6; gate rates from `data/processed/real_data_quality.csv` |

Colour rules: every figure uses black, white and greys only (28 Oct 2026); differences are also carried by labels, line styles or hatching, never by grey alone. Flow figures (01b, 01c, 02): white = photos and people, grey = written rules, black = learned models. Category 1 light, 2 mid grey, 3 black, always with the number. Exceptions: the real photos in 02b keep their own colours, 08 (quality gate histograms) stays in colour, and the roadmap (09) uses black, grey, white, blue, green, red and one hatch, one per workstream, all by the user's choice. Titles: bold capitals, 38 pt on a 24 in wide figure (scaled by width), no figure number: numbering is done in the presentation (`_house_title` in `src/viz/figures.py`). Legend: one line, top right, level with the title, same size on every figure (`_house_legend`; flow figures share `_flow_key`: people and photos, written rules, learned models). Keys that carry meanings (03 categories, 08 per-panel series) stay with their content. Size: every figure is 16:9 and saved at 4800 x 2700 px (`SLIDE_PX`), so it drops straight into a slide; tested in `tests/test_figures.py`.

When `docs/` changes, update the data constants in `figures.py` (TREE, COVERAGE, DATASETS, HELD, PLAN, MILESTONES) and rerun.
Later, once engineer-labelled photos exist: a filled confusion matrix and the measured threshold curve (missed Category 3 vs referral rate).

## Symbol key of the risk matrix (`03_risk_matrix.png`, also used in `02b_real_examples.png`)
The seven crack patterns are drawn as symbols, each with one word; everything else on the figure is in words.

| Symbol | Word | Pattern |
|---|---|---|
| Network of fine cracks on a panel | Map | surface map cracking |
| Beam with short vertical cracks from the bottom | Bending | flexural crack |
| Beam with a diagonal crack near its support | Shear | diagonal shear crack |
| Crack running over a dashed bar | Bar | crack along a bar |
| Column with horizontal cracks | Column | horizontal crack in a column |
| Missing patch with a bar showing | Spalling | spalling with exposed bar |
| Cross of beam and column with an X at the joint | Joint | beam-column joint crack |

## Arrow key of decisions and players (`02_decisions_and_players.png`)
Each arrow shows its step number, a symbol and one word; the full meaning is here.

| # | Symbol | Word | From → to | What passes |
|---|---|---|---|---|
| 1 | camera | Report | resident → app models | photo of the crack + answers |
| 2 | ruler | Measure | app models → app written rules | crack found? rough width · confidence |
| 3 | clock | Wait | app written rules → resident | Retake, or V3: not assessed yet, recheck in 1 month |
| 4 | house | Visit | app written rules → field worker | V1: visit within 1 week · V2: within 1 month |
| 5 | bell | Alert | app written rules → engineer | V1 only: the engineer is told the same day |
| 6 | clipboard | Survey | field worker → resident | asks the building questions, places stickers + crack ID |
| 7 | camera | Photos | field worker → app models | reliable photos: the building, every cracked element, every crack |
| 8 | bar chart | Readings | app models → app written rules | material, element, width, pattern + confidence |
| 9 | pencil | Pattern | field worker → app written rules | pattern picked from drawings while the model is unsure |
| 10 | document | Result | app written rules → resident | Category 1 or 2: level, reason, recheck date |
| 11 | warning triangle | Escalate | app written rules → engineer | Category 3, Expert needed (any doubt), buildings P1 · P2 |
| 12 | check mark | Confirmed | engineer → resident | confirmed level, reason, next step · never 'safe' |
| 13 | stacked layers | Retrain | engineer → app models | corrections become labels → retraining |
| 14 | calendar | Recheck | resident → app models | recheck photo of the same crack ID → back to step 8 |
