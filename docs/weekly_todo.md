# Weekly to-do

Newest week on top. Every Monday: tick what is done, move what is not into the new week, add the new week from `spec/constitution/roadmap.md`.
Owner: J = Jerome, L = Luis, C = coding agent (Claude). Fill in the blanks.
**Every week:** update the deck for Thursday's studio presentation, in the layout set on 26 Oct.

---

## Week 3: 9 to 15 Nov (studio Thu 12 Nov: full SDD constitution, first feature with tests, first MCP attempt)

### Done
- [x] Crack model v4 in use: the reference thesis's building-concrete crack masks (Özgenel [49]) added; better on building concrete, slightly worse on pavement close-ups (C, 9 Nov, D48)
- [x] Old logo ideas removed (D47) (C, 4 Nov)

### Before Thursday 12 Nov
- [x] Constitution refresh: mission unchanged; roadmap, tech stack and the AGENTS.md map brought up to date (C, 10 Nov)
- [x] First MCP connection built: `src/mcp_osm/`, spec and 68 tests (C, 10 Nov, D49). **Live check pending**: the public Overpass servers did not answer on 10 Nov; retry and record the result
- [x] Width check kit built: printed gauge with markers, `width_check.py`, 22 tests (C, 10 Nov, D50)
- [ ] **Take the photos for the width check:** print `docs/width_gauge/width_gauge_v2.pdf` at 100 % (check the 100 mm bar), about 10 photos of the gauge and about 10 real cracks with 3 photos each, as in `spec/features/measurement.md`, "Width check" (J + L; one hour)
- [ ] Run `py src/measure/width_check.py <photos> <truth.csv>` and read the report together; agree with the engineer what error the matrix can accept (C, J + L)
- [x] Element model, method B run on PHI-Net task 6 after a bug fix: not accurate enough alone (C, 10 Nov). Open: method A, slabs and beams, a rule for a hidden or unknown element for the engineer
- [ ] Decide the proposal of 10 Nov: a wide-view model (cracks, windows, elements), element classification and video (would supersede D30 in part) (J + L)
- [ ] Deck for Thursday's studio (J + L)

---

## Week 2: 2 to 8 Nov (midterm: submit Wed 5 Nov on Google Drive, present Fri 6 Nov)

### Done
- [x] Midterm deck, 12 slides, with Surfside and Rana Plaza as 3D animations (J + L, C, 31 Oct to 4 Nov)
- [x] App: QR on its own page (D37), wall link for a teammate, word mark from the deck cover (D38), "Share results" tick (C, 1 to 4 Nov)
- [x] Repo pushed with the app, model v3 and its results; README brought up to date (C, 3 and 4 Nov)
- [x] Width in mm with a bank card: card found automatically, camera tilt, width (widest and median), length; in the class app and on the wall's result pages (L, merged by C, 4 Nov, D41 to D46)
- [x] Deck fixes in Luis's copy: Surfside title "Documented in 2018. Collapsed in 2021.", "12 missed" no longer described with v1's analysis, "Share results" (C, 4 Nov)
- [x] PowerPoint with every animation as an autoplay video, one click per slide, speaker notes (C, 4 Nov; in Downloads, not in git)

### Before the submission (Wed 5 Nov)
- [ ] Choose the final deck (Luis's copy with the risk matrix) and finish the text (J + L)
- [ ] Decide: Rana Plaza slide stays or goes (J + L)
- [ ] Upload to Google Drive: the PowerPoint (animations as videos), the deck's HTML download and a PDF backup; README note asking professors to use the PowerPoint (J)
- [ ] Deck: the model slide and the app demo do not show the width in mm yet; add a line or a slide if wanted (J + L, C)
- [ ] Rehearse once against 7 minutes, with the animations playing (J + L)

### On the day (Fri 6 Nov)
- [ ] Start the app shortly before (`py src/app/launch.py`); keep the laptop plugged in and awake (J)
- [ ] Bring a bank-card-sized card (not the real bank card on stage: an old or loyalty card) for the live measurement (J + L)
- [ ] After the talk: QR full screen (`localhost:8501/?qr`), then the live wall (`localhost:8501/?wall`) (J)

### Open from week 1
- [ ] Calibrate the width: photograph known widths (printed 0.1 to 3 mm lines or a crack-width gauge) with the bank card and compare (L, C, 12 Nov)
- [ ] The "close-up of a surface?" check; pilot photos with a bank card (C, J + L)
- [ ] Engineering partners: requests to Engineers Without Borders and the structural engineer (__)
- [x] Old logo ideas removed from `branding/` (D47) (J, C, 4 Nov)
- [ ] Decide what to do with the deleted `figures/01_pipeline.png` (still on GitHub) (J)

---

## Week 1: 26 Oct to 1 Nov (studio Thu 29 Oct: tree, data map, repo, first real-data test)

### Done
- [x] Segmentation tree, data map, repo, AGENTS.md (C, 24 to 25 Oct)
- [x] First test on real data: quality gate on crack-seg, Darbhanga, SDNET2018 (C, 25 Oct)
- [x] Plan aligned with the syllabus dates (C, 25 Oct)

### Professor's feedback (26 Oct): answer before contacting the expert
- [x] Pipeline v2 specified: material check, confidence on every result, engineer review, human-in-the-loop learning (C, 26 Oct)
- [x] Figures updated: pipeline, decision flow, matrix, tree, data coverage, timeline (C, 26 Oct)
- [x] Professor's update (27 Oct) answered in the specs: pipeline v3 with three scales S, O, P (PHI-Net), building risk score, monitoring, scale card, models identify material and element (C, 27 Oct)
- [x] Pipeline and segmentation tree figures redrawn to match (C, 27 Oct)
- [ ] Material model: concrete vs not concrete, first training test (C)
- [x] First model, day 1: PyTorch on the RTX 5080, crack data prepared and split by source photo (C, 29 Oct)
- [x] First model, day 2: U-Net trained, tested once on 721 unseen photos: 95.8 % of cracks found, 3.1 % missed, 2.9 % false alarms; overlays and a 53 s video in `data/results/segmentation/` (C, 29 Oct)
- [x] Class demo: scan a QR, send a photo, live wall (C, 30 Oct); try it once with a phone before class (J)
- [x] Model v3: brick false alarms 15 -> 0, missed cracks 17 -> 12 on the test photos (C, 30 Oct)
- [ ] "Close-up of a surface?" check before the model: whole scenes are read as cracks (C)
- [ ] First model, day 3: width in mm from the mask + `predict.py` (C); ~10 photos of real cracks with a ruler or scale card (J + L)
- [x] Reference thesis read; notes and proposed features in `docs/literature/corres_sojo_2024.md` (C, 26 Oct)
- [ ] State-of-the-art review based on the reference thesis (C drafts Wed 28 Oct, J + L correct)
- [ ] Engineer review screen mock-up and one-page pitch (C, Fri 30 Oct to Sun 1 Nov)
- [ ] Send the expert package to the professor (J + L, Mon 2 Nov)

### Before Thursday 29 Oct
- [ ] Open the repo on GitHub and check that the figures show in the docs and README (__)
- [ ] Read `docs/experiments/2026-10-25_quality_gate_first_test.md`; be able to explain the two findings: the resolution bug and the smooth-wall blur problem (J + L)
- [ ] Mon 26 Oct: set the deck layout and style once, as a template for every weekly presentation (feedback: it looks too AI-generated); restyle the figures to match (J + L, C)
- [ ] Thursday's slides in the new layout: tree, data map, first real-data test and its two findings (J + L)
- [ ] Decide who presents what on Thursday (J + L)

### Engineering partners (send early, replies take time)
- [ ] Send a dated request to Engineers Without Borders: rate about 150 to 200 crack photos Category 1 / Category 2 / Category 3, at least 30 Category 3, by 20 Nov; put the scale card (or a coin) next to cracks in their own inspection photos (__)
- [ ] Same request to the third-party structural engineer; confirm the agreement (__)

### Data
- [x] Haiti 2010 Earthquake Database: 4,450 photos and the ratings of 183 buildings recovered from a broken download, run through the quality gate (C, 25 Oct)
- [ ] Email santiago.pujol@canterbury.ac.nz (subject "datacenterhub inquiry") to confirm the Haiti licence before using any photo in slides or training; ask about the Türkiye 2023 survey too (draft in the chat of 25 Oct) (__)
- [ ] Nepal 2015: download only the Nepal folder in a browser from https://datacenterhub.org/post-nees.html into `data/raw/nepal2015/` (__)
- [ ] Decide how many images to label by hand with the seven crack patterns, and who labels (J + L)

### Photos
- [ ] Set up the iPhone: Most Compatible (JPEG), grid on, Camera location off (J + L)
- [ ] Print the crack-width card (C makes the PDF)
- [ ] Width test at the desk: about 36 photos, 6 line widths x 3 distances x 2 lights (__)
- [ ] Pilot: about 30 cracks on public facades, 3 shots each, plus about 6 deliberately bad shots (`spec/features/photo_protocol.md`) (J + L)
- [ ] Run `py src/pilot/prepare.py <phone folder> data/pilot/photos` and fill `data/pilot/labels.csv` (__)

### Decisions to take
- [ ] One material for the hackathon beta: brick or concrete walls (J + L)
- [ ] Building-data source for the first MCP connection (due 12 Nov) (J + L)
- [ ] Five or six crack patterns (`spec/decisions.md` D1) (J + L)

### Midterm (submit Wed 5 Nov, present Thu 6 Nov)
- [ ] Outline of the midterm deck in the new layout, reusing the content of figures 01 to 09 (__)
- [ ] Problem & data brief: answer the syllabus questions (where is the problem, what data, what AI can generate, limits, who is affected) (__)

### Repo (coding agent)
- [ ] Crack-width card PDF and one-page field sheet for the pilot (C)
- [ ] Sample-size rule ("rule of three") in `spec/features/evaluation.md` (C)

---

## Coming up (from the roadmap)
- **Week 2, 2 to 8 Nov:** midterm (5/6 Nov); pilot photos labelled; quality gate retuned on phone photos; label mapping to the patterns.
- **Week 3, 9 to 15 Nov:** first feature with tests against the baseline (scale card + width with its uncertainty); first MCP: OpenStreetMap building data (12 Nov).
- **Week 4, 16 to 20 Nov:** hackathon, beta version end to end on pilot photos.
