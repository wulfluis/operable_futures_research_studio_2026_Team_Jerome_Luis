# Decisions log

Newest last. Each entry: date, decision, reason, status.

## D1 (2026-10-24): five crack patterns, not six
The deck (slide 5) names six crack types; the matrix (slide 7) and the code use five.
Decision: keep five. "Diagonal crack" and "stair-step crack in brick" both map to `diagonal_step`; "column crack" maps to `column_or_beam`.
Reason: the matrix rates both the same in every width band, and fewer classes need less labelled data.
Status: provisional, to confirm with the engineering partner in Phase 1.

## D2 (2026-10-24): boundary widths go to the wider band
Width bands are lower-inclusive: 1.0 mm is in `1_to_3`, 3.0 mm is in `over_3`.
Reason: any doubt goes towards more caution (AGENTS.md rule 2). Previously 3.0 mm fell in `1_to_3`.
Status: illustrative until calibrated.

## D3 (2026-10-24): failed measurement means Expert needed
A missing, NaN, negative or infinite width or confidence returns Expert needed, never a matrix level.
Reason: before this, NaN confidence and negative width returned Low, breaking rules 2 and 3.

## D4 (2026-10-24): confidence threshold is fixed in code
`CONFIDENCE_THRESHOLD = 0.8` is a module constant; callers cannot lower it.
Reason: a caller could pass `threshold=0` and turn any guess into Low.
Status: 0.8 is illustrative; the real value comes from the missed-High vs sent-to-expert curve on a labelled test set (deck slide 7).

## D5 (2026-10-24): every result carries a reason and a review flag
`assess()` returns level, reason and `engineer_review` (true for High and Expert needed). `triage()` still returns the level only.
Reason: rule 4 (an engineer confirms every High and referral) and the atomic problem ("with the reason").

## D6 (2026-10-25): non-commercial datasets are for research only
PHI-Net (CC BY-NC-SA 4.0) and CODEBRIM (non-commercial; trained models may not be used commercially) may be used to explore and pre-train in the studio.
Any model meant for real deployment is retrained without them, on CC0 / CC BY data and our own photos.
Reason: their licences forbid commercial use, and a deployed tool may later be run by a company or a funded programme.

## D7 (2026-10-25): SDNET2018 is a code test, not training data for thresholds
Public sets (crack-seg, Darbhanga, SDNET2018, METU, Strathclyde, run on 25 Oct) are used for the 29 Oct first test on real data. They do not set any threshold, because they are not phone photos and have no scale reference.
Thresholds for the quality gate come from the pilot phone photos; triage thresholds come from the engineering partner.

## D8 (2026-10-25): measure sharpness at a fixed working size
The quality gate scales every image down to a long side of 512 px before measuring sharpness and brightness.
Reason: the first real-data test rejected 67% of the Darbhanga brick images (768 px) as blurred; the same images scaled to 416 px failed only 11% (sample of 107 images). Laplacian variance depends on resolution, so phone photos (about 4000 px) would almost all fail.
Status: 512 px is illustrative; check on the pilot phone photos.

## D9 (2026-10-25): Ultralytics YOLO is AGPL-3.0
The `ultralytics` package (YOLO) is licensed AGPL-3.0, or under a paid Enterprise licence [35]. Using it in the studio is fine; any tool offered to others over a network must publish its source under AGPL-3.0, or buy the Enterprise licence, or swap the detector.
Decision: use it for research now; decide before any deployment. Keep the detector behind one function in `src/` so it can be swapped.
Status: open, to revisit before Phase 3.

## D10 (2026-10-25): engineering partners
The engineering partners are Engineers Without Borders and a third-party structural engineer. The agreement with the structural engineer is under construction.
They label photos, give the baseline ratings (an engineer's visual check), and sign off the risk matrix and thresholds.
Status: Engineers Without Borders named; structural engineer to confirm.

## D11 (2026-10-25): plan follows the syllabus dates
The plan runs to the final presentation on 17 Dec and the blog post by 24 Dec, with the midterm on 6 Nov, the hackathon (beta) on 16 to 20 Nov and the baseline comparison on 3 Dec (`spec/constitution/roadmap.md`). It replaces the earlier plan, which assumed a winter break and ran to 3 Jan.

## D12 (2026-10-25): the blur check is not trusted on smooth walls yet
Laplacian variance rejected 75 to 89% of SDNET2018 wall and deck tiles, most of them in focus but smooth (visual check of 8). Hairline cracks on smooth walls would be sent to retake again and again.
Decision: keep the check (a retake is never a Low, so it is safe), but do not tune it on public data. Choose the measure and threshold on the pilot photos, using the deliberately blurred shots as ground truth.
Status: open; confirmed on METU building walls (61% of cracked tiles rejected); results in `docs/experiments/2026-10-25_quality_gate_first_test.md`.

## D13 (2026-10-25): the name is Second Look
The project is called Second Look everywhere: repo, decks, logo, blog post. "Crack triage" is no longer used as a name; "triage" stays only as the word for what the tool does.
Reason: the 22 Oct deck said "Crack triage" while the repo said "Second Look"; one name for the logo and the weekly decks.

## D14 (2026-10-25): the concept is buildings in informal neighbourhoods
Second Look is for cracks in buildings in slum-like, informal or ageing unsupervised neighbourhoods, not bridges, roads or pavement.
Data is ranked by how close it is to that case (`docs/data_map.md`, section 0). Bridge and pavement sets (CODEBRIM, SDNET2018 decks and pavement) are off-concept; road-heavy sets (crack-seg, CrackSeg9k) are generic pre-training only.
Reason: the team's concept; earlier data choices leaned on bridge and pavement sets because they are the largest public ones.

## D15 (2026-10-25): train crack finding now; danger rating waits for our own data
Public data is enough to train crack segmentation, crack detection and crack / no-crack (`docs/data_map.md`, section 0b). It is not enough for the crack pattern (no dataset uses our five classes), width in mm (no scale anywhere), or Low / Medium / High per crack (Haiti and Nepal rate whole buildings).
Decision: the hackathon beta trains the crack-finding step on public data. Pattern labels, width and danger ratings come from hand labelling, the pilot photos and the engineering partners; those tasks stay on the critical path.

## D16 (2026-10-26): the segmentation tree shows only what the tool works on
Out-of-scope items (unauthorised construction, earthquake triggers, rebar and foundations, overridden warnings) and the Phase 3 item (soil erosion) are removed from the tree. The "Warnings overridden" branch is gone; "Data is scarce" becomes branch D. What the tool does not cover is stated once, in `spec/constitution/mission.md`.
The tree figure is redrawn as a flowchart: rounded pills for the macro problem and the four branches, letter tags on the connectors, circles for the micro items, then the atomic problem and the final output. Yellow marks the path into the atomic problem (B, C, B1, B2, B3, C1), navy the supporting context.
Reason: team feedback; out-of-scope nodes were unnecessary detail and the old box layout was hard to read.

## D17 (2026-10-26): scope is reinforced concrete; a material check comes first
Following the professor's feedback (26 Oct): the tool assesses cracks in reinforced concrete elements only (walls, columns, beams, slabs). A material check runs right after the quality check: concrete continues; not concrete or unsure goes to an engineer ("Expert needed: material not supported"). Brick image sets (Darbhanga, Strathclyde) become the "not concrete" examples for training this check.
Consequence: the brick stair-step pattern leaves the matrix; the diagonal pattern now means a diagonal (shear) crack in concrete.
Reason: fits the timeline, the professor's advice, the reference thesis (reinforced concrete) and the Haiti and Nepal surveys (concrete-frame buildings).

## D18 (2026-10-26): two users and a human-in-the-loop learning loop
The resident takes the photo; the engineer reviews every High and every referral on a review screen and confirms or corrects it. Corrections are stored as labels and the learned models (material, crack, pattern) are retrained in batches, each new version checked on held-out engineer-rated photos before use.
We call it human-in-the-loop learning, not reinforcement learning: it is supervised retraining with expert feedback. The risk matrix never changes automatically (AGENTS.md rule 5).

## D19 (2026-10-26): every result carries confidence, not only a colour
Each stage reports its output with a confidence: material, crack found, width in mm with an uncertainty range, pattern probability, and an overall confidence with the level. Full list in `spec/features/pipeline.md`.

## D20 (2026-10-26): risk matrix v2, with elements, warning signs and next steps
The matrix now has three parts: a base level from seven reinforced-concrete patterns x four width bands; element criticality (a column raises one level); and seven warning signs that can only raise the level (growing, rust staining, sticking openings, recent earthquake, added floors: +1; offset and sagging: straight to High). The width used is the upper end of the measurement. Every result adds a next step with a recheck interval. The values live in `src/triage/risk_matrix_v2.json` (version 2.0-illustrative), readable and editable by an engineer; each change is a new version recorded here.
Supersedes D1 (five patterns): surface map cracking replaces plaster crazing; vertical wall crack and column-or-beam are replaced by the element axis and by flexural, horizontal-column, along-bar and joint patterns.
Calibration plan in `spec/features/risk_matrix.md`: literature anchors (to verify in the sources), a sense check on the Haiti definitions, an engineer card session, a test on engineer-rated photos, and versioning. All values stay illustrative until then.

## D21 (2026-10-27): the building is the unit, seen at three scales (pipeline v3)
Following the professor's feedback (27 Oct: engineers read the pattern across the building and whether cracks grow; building data; cheap phones and scale; PHI-Net), the unit of analysis is the building and cracks are evidence inside it. Every photo is at one of three scales defined as in PHI-Net task 1: S (structural, the building), O (object, the element), P (pixel, the crack) [53][55]. Pipeline v3: three scales x five phases (capture, check, perceive, decide, act), a scene check, a monitoring loop and a learning loop (`spec/features/pipeline.md`, `figures/01_pipeline.png`). Supersedes the single-crack pipeline v2 (D17 to D19 still hold).

## D22 (2026-10-27): building risk score and priority P1 to P3
A written rule ranks buildings for engineers: base tier from the worst crack; one tier higher with two or more building risk flags (older than 50 years, floors added, structure or use changed, open ground floor; unknown counts as a flag); one tier higher when Medium or worse appears in two or more elements; never lowers a crack level (`spec/features/building_score.md`). Illustrative until calibrated.

## D23 (2026-10-27): crack monitoring with IDs, stickers and repeat photos
Every crack gets an ID and two marker stickers; it is photographed again on its recheck date with the previous photo as an overlay; growth counts only when it is larger than the measurement uncertainty, and then raises the crack level (`spec/features/monitoring.md`).

## D24 (2026-10-27): printed scale card, pixel rule and device roles
A printed scale card replaces the coin (the coin stays as a backup). A photo must show at least 3 pixels across the crack (6 to 8 recommended) [46]; otherwise retake closer, or read the width band on the card. Field workers make the first visit with a reference phone; residents do the follow-ups (`spec/features/photo_protocol.md`).

## D25 (2026-10-27): residents are never asked the material, the element or the pattern
Residents cannot tell concrete from other materials, or a column from a wall (team, 27 Oct). Models identify the material, the element and the crack pattern from the photos, each with a confidence; when unsure, the case goes to an engineer. Residents answer only what they can know (building history, sticking doors, whether the crack is new). Confirmations by field workers and engineers become labels. Three models this semester: material, element (pre-trained on PHI-Net task 6) and crack segmentation.

## D26 (2026-10-27): no validation site yet; Tripoli is a motivating example only
The architecture is general. Tripoli was an example in the problem statement; we will not validate there now (possibly later, once the tool works). The Tripoli building lists are removed from the data map and the coverage figure. Cited examples in the problem statement and the segmentation tree stay as evidence of the problem, not as data.

## D27 (2026-10-27): a written dispatch rule decides when a field worker visits
A resident's report cannot send a field worker to every crack, and a resident's phone cannot give a triage level (D24). Step 0 of the pipeline, report and dispatch, turns the resident's photo and answers into a visit priority with a written rule (`src/triage/dispatch.py`, `spec/features/dispatch.md`): V1 visit within 1 week (offset or sagging, width of 1 mm or more, growing plus another sign), V2 within 1 month (any warning sign, doubt, no scale, 0.3 mm or more, two or more building risk flags, repeated unusable photos), V3 no visit yet (confident hairline crack with nothing else; the resident rechecks in 1 month and 1 report in 10 is visited anyway as an audit). V1 to V3 are visit priorities, not triage levels: the five levels are given only after the field visit, so the "only five outputs" rule still holds for triage. All values illustrative until calibrated.

## D28 (2026-10-27): risk matrix v3, location on the element and three new warning signs
After reading chapters 2 to 4 of the reference thesis [46] (notes, section D), the matrix adds what an element photo can show and that changes how serious a crack is: the **location on the element** (end zone near a joint or support, where bars are anchored or lapped: +1 for cracks along bars and diagonal shear cracks; an unknown location counts as an end zone), and the warning signs **thin cover or no stirrups** where the bar is exposed (+1), **a crack meeting a crack along a bar or stirrup** (+1) and **damp, water marks or moss** (+1, an assumption to check against the Eurocode 2 exposure classes). Every location and warning sign now records its basis in `src/triage/risk_matrix_v3.json`. Crack count and spacing, bar diameter and cover, and opening and sliding are recorded for the engineer, not used as rules. The base table and bands are unchanged; the thesis gives no width threshold. All values illustrative until calibrated; the list of what we need from the engineer is in `spec/features/risk_matrix.md`. Supersedes D20 for the matrix values (v2 kept in `risk_matrix_v2.json`).

## D29 (2026-10-27): the three triage levels are renamed Category 1, 2 and 3
Low, Medium and High are replaced by **Category 1** (routine monitoring), **Category 2** (low-cost advice, recheck) and **Category 3** (an engineer must confirm), everywhere: AGENTS.md outputs, code, risk matrix v3 values, specs, tests and figures (team choice, 27 Oct: more professional, neutral codes whose meaning lives in the legend). Retake and Expert needed keep their names, so the outputs are still the only five. Earlier decisions keep their original wording as a record. The risk matrix figure (03) now uses black, white and grey only; the crack patterns are drawn as symbols with one word each (key in `spec/features/figures.md`).

## D30 (2026-10-28): scope to the final presentation follows the pipeline story
Two people and seven weeks cannot build the whole system. We build the crack line of the pipeline story (figure 01d): the photo check [2], crack segmentation and width in mm from the scale card [6], the written rules (dispatch, risk matrix, building score) [4] [7] [8], the engineer check and the baseline comparison [9], and a prototype around them (MCP building data, harness, recheck and learn loops, UI), on the syllabus dates. The pattern model is a stretch goal. Designed but not built this semester: the field worker's app, the facade model and the element model; the field worker picks the element in the capture form instead (D25 still holds: residents are never asked). Supersedes the "three models this semester" line of D25. Plan in `spec/constitution/roadmap.md` and figure 09.

## D31 (2026-10-29): first model is a U-Net on public crack sets, split by source photo
The first working model (box [6], D30) is a U-Net with a ResNet-34 encoder (`segmentation-models-pytorch`, MIT; no Ultralytics, D9), pre-trained on CrackSeg9k Vol1 [25] and crack-seg [33], 9,441 image and mask pairs at 416 x 416 px. CrackSeg9k's published train/test lists put tiles of the same source photo in both (116 Rissbilder photos), which would inflate the test score, so we re-split it 80/10/10 by source photo; crack-seg keeps its own split (no photo crosses it). crack-seg polygons are wider than the cracks, so they are not used to calibrate width in mm. Spec: `spec/features/segmentation.md`.

## D32 (2026-10-30): model v3 in use; class demo over a temporary link
v1 read brick mortar joints as cracks (15 of 22 brick test photos). v3 is trained with no-crack brick photos added (Darbhanga [30]); on the same 1,203 test photos it missed 12 cracks against 17 and raised 7 false alarms against 21, 0 on brick, so it replaced v1 under the rule written before training. On the val photos it missed 5 against 1, so the evidence is mixed: the pilot photos decide next, and v1 is kept. Neither model knows whole scenes (28 to 30 of 30 building photos read as "Crack found"); the next step is a "close-up of a surface?" check before the model. For the studio, a class demo (`spec/features/class_demo.md`) lets students scan a QR and send a photo; it runs on the laptop through a temporary Cloudflare link that the presenter starts and stops, photos stay in memory, and the live wall shows only shared results.

## D33 (2026-10-30): "blurred" needs two blur measures to agree
In the class demo, sharp phone photos of cracks came back "Retake: blurred" (D12 in practice). The blur check now calls a photo blurred only when the Laplacian variance is below 100 and the blur effect [56] is above 0.45 (both illustrative until calibrated).
Check on 400 test photos and 30 Haiti 2010 photos (local only), sharp and with Gaussian blur added at the 512 px working size: sharp photos rejected, old rule against new: CrackSeg9k 69% to 0%, crack-seg 3% to 0%, SDNET2018 100% to 6%, METU 100% to 61% (227 px tiles, not phone photos), Haiti 0% to 0%. Blurred copies caught: Haiti blur 1.5 px 100% to 100%; blur 3 px 100% to 93 to 100% on every set. Mild blur (0.8 px, 1.5 px on small tiles) now mostly passes.
Assumption: added Gaussian blur stands in for a shaky or out-of-focus phone. Status: open, as D12: calibrate on the pilot photos with deliberately blurred shots. The gate now also measures only the photo, not the grey padding added for the model.

## D34 (2026-10-30): class demo, one tick; sending the photo is the consent
The phone page had two ticks: consent (required) and "Show my result on the class screen". The team asked for one tick only. The consent terms (took it yourself, no faces, house numbers or addresses, agree it is analysed) now stand right above the upload, and sending the photo counts as agreeing; the one tick left decides whether the result goes to the wall. Photos are still analysed in memory and never saved (rule 9). For the pilot with households, written consent stays a separate, explicit step: this applies to the class demo only.

## D35 (2026-10-31): the model line stands still along the bottom
On the class screen the model line ran clockwise around the four edges. The team found the motion distracting. After trying a ticker and a flicker, they chose a line that does not move: along the bottom, from the left edge of the screen to the right edge, in the same grey. Removing the "Reading cracks..." typing was discussed; the team keeps it for now. The line keeps naming the model and its training data on screen, so anyone can see what is running.

## D36 (2026-10-31): the red flash on the crack stays plain
The team tried two "electric" versions of the red flash on the background crack: first a red glow, a near-white core and a faster, longer flash, then a plain red that flickered as it ran. They kept neither. The red runs down the crack as before (same red, length and speed, about 5 s per cycle), with no glow and no flicker.

## D37 (2026-11-01): the QR leaves the wall and gets its own page
The team took the QR off the live wall so the wall shows only the title, the results and the model line. The QR now has its own page, `/?qr`, with the code alone in the middle of the screen, and is shown on the projector when students should scan it. The launcher prints that link with the others.

## D38 (2026-11-04): the deck cover's word mark goes into the app
The midterm deck cover shows "SECOND LOOK" with its two O's drawn as scan frames: the second slides out of the first and a red crack draws through it. The team wanted the same mark on the phone page and the live wall, so the app and the deck share one identity. It plays once on load, like the cover; the wall's refresh does not replay it, because the title sits outside the refreshing part.

## D39 (2026-11-04): the crack model is published as a GitHub Release file
So that Luis, and anyone else, can run the app without the training data, the team published `unet_crack.pt` (v3) on the public repository as the file of release `model-v3`, not committed to git. The release notes give the training sets and their licences (CrackSeg9k CC0 1.0, Darbhanga CC BY 4.0, crack-seg Public Domain on its page but AGPL-3.0 in its file header: unclear, chosen knowingly), the test result, the SHA-256 and that it is a research prototype that never says a building is safe.

## D40 (2026-11-04): a fixed link and QR through ngrok
The Cloudflare quick tunnel gave a new random link at every start, so the QR in the slides went stale. The launcher now uses the team's free ngrok dev domain, https://cranium-bladder-tapioca.ngrok-free.dev, the same at every start; the QR is made once (`docs/presentation/midterm/assets/qr_fixed.png`). Cloudflare stays as the fallback. Trade-offs: the laptop must still run the app; only one laptop can serve the link at a time; the free plan shows a one-time warning page that visitors click through; the authtoken lives only in ngrok's own config on the laptop, never in the repository.

## D41 (2026-11-04): crack width and length measured from a reference object
`src/measure/` measures the crack width (widest point, with ±1 pixel [46] plus the scale error) and the length in mm, using a reference object of known size in the photo. A 4-corner rectangle (scale card square, bank card, A4) gives a homography, so tilt is corrected; a 2-point length (50 mm bar, 1 euro coin) gives one scale. The width is taken perpendicular to the centre line, from the grey profile with sub-pixel precision, after [46] ch. 5.2 and [48]. Under 3 pixels across, the width is not passed to the matrix (rule 2). Crack pixels come from a mask file or, until the U-Net exists, a classical black-hat finder. Tested on synthetic images only; accuracy on real photos is checked next against the width lines of the scale card (`spec/features/measurement.md`). New dependency: scipy.

## D42 (2026-11-04): the bank card is the reference, found automatically, with a tilt check
The coin is not used for measurement after all (team, 4 Nov): one circle gives no full perspective correction and does not show small tilts. The default reference is a bank-card-sized card (ISO/IEC 7810 ID-1), flat on the wall next to the crack, card number covered. `src/measure/card.py` finds it automatically (OpenCV contours, then a camera pose check that a card at most 35 degrees off straight explains the shape), places its corners to sub-pixel precision, and gives the camera tilt: up to 10 degrees straight, 10 to 20 a warning, over 20 a Retake (illustrative). `src/measure/analyse.py` joins the quality gate, the card, the tilt and the measurement into one outcome (Retake, Width not measured, Measured), and a local Streamlit app (`src/measure/app.py`) takes an uploaded photo and shows the result. Tested on synthetic photos with a camera model only. Assumption: focal length about 0.75 x the long side, photo not cropped. New dependencies: opencv-python-headless, streamlit. The printed scale card (D24) is still to be designed. The rectangle presets for it stay in the command line.

## D43 (2026-11-04): the crack pattern is suggested by written rules, confirmed by a field worker
No public dataset labels our seven patterns (`docs/data_map.md`, [71]); PHI-Net and IDEA label only flexural, shear and a few others [55][64]. Engineers tell most patterns apart by the crack's direction against the element, its position and its form, which the measurement now gives (angle to vertical, straightness, coherence, closed cells, branch ratio). So `src/triage/pattern_rules.py` suggests the pattern with written rules (`spec/features/pattern_rules.md`): spalling seen, a network of cracks, a joint, then the angle to the element axis (column vertical, beam horizontal; walls only diagonal; slabs none). A vertical crack in a column without rust stays unclear (it can be crushing). The suggestion never enters the matrix by itself: a field worker confirms it until the rules are checked on engineer-labelled photos. Models stay for what has data: crack mask, element (PHI-Net task 6, IDEA [64], under its licence), spalling. Angles assume an upright phone. All values illustrative. IDEA, Wang's two RC sets and dacl10k are recorded in the data map; IDEA's CC BY-NC-ND licence is an open item.

## D44 (2026-11-04): first element-model test, an untrained detector against a trained classifier
No open code recognises building elements in photos (searched 4 Nov; only papers, and bridge work without code). Two open routes are compared on the same PHI-Net task 6 test images (`spec/features/element_model.md`, `src/experiments/element_baseline.py`): A, Grounding DINO tiny (Apache 2.0) without training, asked for "a column. a beam. a wall."; B, a ResNet-18 pre-trained on ImageNet and retrained on PHI-Net task 6. Both report accuracy on confident answers, coverage, and how often a column is confidently called something else; below the threshold the field worker decides (rule 2). Not run yet: PHI-Net is on Jerome's PC. New dependencies for experiments only: torch, torchvision, transformers. PHI-Net has no slab class; IDEA does, under its licence question (D43).

## D45 (2026-11-04): bank-card measurement goes into a copy of the class demo, not the original
Luis wants to try the width in mm inside the class demo without changing the team's app. A copy in `src/app_measure/` reuses the original's look and pipeline functions unchanged; with no bank card in the photo its answer is the original's, with a card it adds the width, the length and the camera tilt (`spec/features/app_measure.md`). The U-Net mask is about twice as wide as the crack, so the width is read from the classical edges inside the U-Net's area (synthetic sample: 1.48 mm against 1.40 mm true; U-Net mask alone 1.89 mm). It runs on port 8503 with its own QR on the local Wi-Fi, not the team's ngrok link (D40). Status: experiment, not team-decided; thresholds illustrative until calibrated.

## D46 (2026-11-04): Luis's bank-card measurement goes into the class demo itself
Jerome took Luis's `measurement` branch (D41 to D45) into the team's app instead of keeping a second copy: the measuring logic of `src/app_measure/core_measure.py` became `src/app/core.py` and its page became `src/app/second_look.py`. The interface stays the team's (word mark, quote, crack background, version "v1.0") and so do the launcher, the fixed ngrok link and the QR in the slides (D40); the copy's own launcher (port 8503, self-signed certificate on the local Wi-Fi) was removed so there is one app on one link. Without a bank card the answer is as before; with one the phone page adds the measurement block and each wall photo opens a page with the numbers. Supersedes D45's "copy, not the original". Thresholds stay illustrative until calibrated on real photos.

## D47 (2026-11-04): the early logo ideas are removed
The four logo ideas in `branding/logos/` (two lenses, stairstep, coin lens, crack path) and their generator are deleted: the team's logo is the word mark from the deck cover, "SECOND LOOK" with the two O's as scan frames and a red crack (D38), in `src/app/logo.py`. They stay in the git history.

## D48 (2026-11-09): crack model v4 replaces v3, judged on all test photos
v4 adds 4,236 tiles from 353 photos of the Özgenel Concrete Crack Segmentation Dataset [49] (concrete of buildings with crack masks, the data of the reference thesis [46]) to v3's training (`spec/features/segmentation.md`). Tested on photos it never saw: on real building concrete (588 Özgenel tiles) it misses 17 cracks against 30 for v3 and raises 5 false alarms against 21 (pixel F1 0.85 against 0.78); on the old public test photos without METU (906, mostly pavement and generic close-ups) it misses 18 against 12, with 6 false alarms each (pixel F1 0.64 against 0.63). Over all 1,494 test photos: 35 missed against 42, 11 false alarms against 27. The rule of D32 (no more misses and no more false alarms on the same test photos) compared only the old test photos, where v4 fails by 6 misses. The team chose to judge on all test photos, missed cracks first, because building concrete is closer to the concept (D14) than pavement. v4's 6 extra misses on the old close-ups are recorded as its known weakness. v3 is kept as `models/unet_crack_v3.pt` and release model-v3. A first v4 run on masks misaligned by EXIF rotations was discarded.

## D49 (2026-11-10): the first MCP connection reads building facts from OpenStreetMap, as suggestions
For the studio session of 12 Nov, `src/mcp_osm/` is an MCP server (the official `mcp` Python SDK, version 2) with one tool, `building_at(lat, lon, radius_m)`, that asks the public Overpass API for the building outline at a place and returns storeys, year, type, material, height and footprint (`spec/features/mcp_building_data.md`). Three rules shape it. (1) Outside data is untrusted (AGENTS.md rule 8): only numbers and words from fixed lists leave the server, never names, addresses or any free tag text, because tag text can carry instructions meant for an agent. (2) The facts are suggestions for the field worker and may raise a risk flag of the building score (a year more than 50 years ago may raise the age flag) but never clear one: unknown already counts as a flag, so a recent or missing year changes nothing. (3) It never raises into the agent: no building in range, a failing source or bad input give a result with a reason. Status on 10 Nov: the code and 68 tests (no network) pass; the live check against the real servers failed that day (the main Overpass instance answered 502 and three other public instances timed out, while another OpenStreetMap service answered), so it is retried and the result recorded. Limits: coverage is likely thin in informal neighbourhoods (assumption, not measured); the agent harness that calls it, more sources and the building score code come next (26 Nov).

## D50 (2026-11-10): the width check uses a printed gauge with markers, and says what it does not show
The first feature against the baseline (12 Nov) is the width in mm. `docs/width_gauge/width_gauge_v2.pdf` has four ArUco markers, a 100 mm bar and lines of 0.1 to 5 mm that are whole printer pixels wide, so their printed width is known exactly (`spec/features/measurement.md`, "Width check"). `src/measure/width_check.py` compares the app's width with that known width, with the by-eye reading on the gauge (the non-AI baseline) and between repeated photos of one crack, and compares the bank card's scale with the sheet's own. The markers, not the card, find the lines: the first design found them through the card and the card finder took the lower half of the white sheet for a card (a box made of paper edges and a thick line), so the corners now come from markers and the card finder sees only the frame region. A card as light as the paper is refused, since it cannot be told from the empty printed frame (a test without a card showed the check would otherwise have run on the frame). What it shows: the card scale, the tilt handling and the width method on known widths. What it does not: the U-Net on real walls (a line on paper is not a crack), or any truth for real cracks, which only a microscope or feeler gauge would give. No pass mark is set; the acceptable error is to be agreed with the structural engineer.

