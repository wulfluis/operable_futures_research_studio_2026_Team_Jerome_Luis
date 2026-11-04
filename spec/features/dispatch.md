# Feature: report and dispatch (`src/triage/dispatch.py`)

Status: rule in code on 27 Oct 2026; app screens not built. Every value is illustrative until calibrated with the engineering partners (decision D27).

## Why
A field worker cannot visit every crack a resident photographs. The professor's point (27 Oct) is that cheap phones are not reliable enough to measure a crack, so a resident's photo cannot give a triage level. It can still say **how soon someone must look**. This step decides, from the resident's report, whether a field worker visits and how soon. It is a written rule (AGENTS.md rule 5), applied before the three scales of the pipeline (`spec/features/pipeline.md`).

## What the resident sends (any phone)
- A close-up of the crack, with the printed scale card if they have one, or a coin (backup).
- Answers they can know (D25): is the crack new? has it grown? have doors or windows started to stick? has there been an earthquake since? is one side of the crack shifted out of line (offset)? does a beam or ceiling sag? The last two are asked with drawings (assumption: residents can recognise them from drawings; to test in the pilot).
- Once per building: the building questions (age, floors added, changes, open ground floor; "unknown" allowed).

## What the app measures (models, each with a confidence)
- The quality gate (`src/quality/gate.py`): is the photo usable?
- Crack found, with a confidence.
- A rough width, upper end of the measurement, only when the scale is visible. A resident's phone gives a wide uncertainty, so the upper end is large: the rule leans towards visits.

## The rule (first match wins)

| # | If | Then | Why |
|---|---|---|---|
| 1 | Offset or sagging reported | **V1: visit within 1 week**, engineer told today | These go straight to Category 3 in the risk matrix |
| 2 | Photo not usable | **Retake**; after 2 failed retakes, **V2** | A resident who cannot take the photo is not left alone |
| 3 | Width upper end ≥ 1 mm | **V1** | From 1 mm the matrix can reach Category 3 |
| 4 | "Growing" plus any other warning sign | **V1** | Growth is what engineers look at first |
| 5 | Any other warning sign (growing, rust staining, sticking doors or windows, recent earthquake, damp, or any sign of the risk matrix) | **V2: visit within 1 month** | Each raises the level in the risk matrix; no reported sign is ever ignored |
| 6 | Crack not found with confidence | **V2** | Any doubt goes to a person (rule 2) |
| 7 | No scale in the photo, so no width | **V2** | Without a width, Category 1 cannot be earned |
| 8 | Width upper end ≥ 0.3 mm | **V2** | Above the first matrix band |
| 9 | 2 or more building risk flags (unknown counts) | **V2** | Same flags as the building risk score (D22) |
| 10 | Otherwise: hairline under 0.3 mm, crack found with confidence, no sign, fewer than 2 flags | **V3: no visit yet** | The only case that can wait |

Times (1 week, 1 month), widths (0.3 and 1 mm, the risk matrix band edges) and the confidence threshold (0.8, as in `matrix.py`) are illustrative.

## What the resident sees
- **V1:** "A field worker will visit within a week. An engineer has been told today."
- **V2:** "A field worker will visit within a month. If the crack changes before then, photograph it again."
- **V3:** "Not assessed yet: no field worker has seen this crack. Photograph it again in 1 month, and at once if it grows, if doors or windows stick, or after an earthquake."
- **Retake:** "Retake the photo with the scale card, following the guide."

V1, V2 and V3 are visit priorities, not triage levels. The five levels (Retake, Expert needed, Category 1, Category 2, Category 3) are only given after the field visit, from the field worker's photos. V3 is never Category 1 and never "safe" (rules 1 and 3).

## After the decision
- **Visit (V1, V2):** the field worker does the first visit: the building (once), every cracked element, every crack (pipeline phases 1 to 5), with a reference phone and the scale card. Visits are grouped: one visit covers every open report in the same building.
- **Queue:** V1 before V2; within a tier, more building risk flags first, then the oldest report.
- **No visit yet (V3):** the resident rechecks in 1 month. Each recheck is a new report through the same rule, so growth or a new sign moves it to V1 or V2.
- **Audit:** 1 V3 report in 10 (illustrative), picked by a fixed rule on the report ID, is visited anyway. This measures how many serious cracks the rule leaves waiting (its false negatives, `spec/features/evaluation.md`).

## Limits
- With no scale card, every report becomes a visit (rule 7). Since 4 Nov a bank card (or any card of that size) is the default reference (D42, D46), so most residents already have one; the printed card stays for field workers.
- The width from a resident's phone is rough; the rule only uses it to decide how soon to look, never for the level.
- How many visits this creates per field worker is unknown until the pilot.

## Tests (`tests/test_dispatch.py`)
- Offset or sagging gives V1, even when the photo is unusable.
- A bad photo asks for a retake; after 2 failed retakes it becomes a visit.
- A wide crack (≥ 1 mm) gives V1; growing plus another sign gives V1.
- Any one warning sign, low confidence, no scale, 0.3 mm or more, or 2 building flags gives V2.
- Only a confident hairline crack with no sign and fewer than 2 flags gives V3, and V3 never mentions Category 1 or "safe".
- Unknown sign names are rejected; the audit picks about 1 in 10 and always the same reports.
