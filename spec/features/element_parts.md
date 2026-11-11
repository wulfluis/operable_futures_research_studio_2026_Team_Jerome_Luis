# Feature: which part of the building the crack lies on, from SAM 3.1 (`src/element/parts.py`)

Status: first version, 11 Nov 2026, decision D62. Luis, local branch `element-position`. Tested with a fake segmenter and on one real room photo with a drawn crack line. Every limit below is illustrative until checked on real photo pairs.

## Purpose
The element model (D52, D53) reads one word for the whole element photo and was unsure on room photos (wall 72 %, 11 Nov). SAM 3.1 [81][82] segments the photo by name: it returns a mask for each column, wall, beam and so on that it finds. Laid on the crack's line, the masks tell which part the crack lies on. Luis, 11 Nov: "SAM dient wirklich der Segmentierung zwischen Column, Beam, Slab".

## The parts (Luis's list, 11 Nov 2026)
| Part | Words SAM 3.1 is asked with | In the risk matrix |
|---|---|---|
| Column | column / concrete column | yes |
| Wall | concrete wall | yes (load-bearing or infill not known from a photo) |
| Beam | beam / concrete beam | yes |
| Slab | ceiling slab / concrete ceiling | yes |
| Balcony | balcony | no |
| Stairs | staircase / stairs | no |
| Window | window | no |
| Expansion joint | expansion joint | no |

A thing found by two words of one part counts once (masks overlapping by more than 60 %, illustrative).

## How
1. The element photo, long side 1600 px, is read by SAM 3.1 (Ultralytics, 644 px inside, on the laptop's graphics chip). Finds with SAM's score below 0.5 are dropped (illustrative; a mirror was read as a window at 0.41 on 11 Nov).
2. The crack's line in the element photo comes from the existing link of the two photos (bank card or key points, D58, D59). That step is unchanged.
3. The line is drawn about 0.4 % of the photo's side thick (at least 3 px). For each part, all its masks together, the share of the line inside is measured.
4. Answer:
   - **On a part**: at least 50 % of the line lies on it, and no second part holds 25 % or more.
   - **On the border of two parts**: two parts each hold at least 25 % (for example a crack along the joint of wall and ceiling).
   - **Not clear**: no part holds 25 %, or the best holds less than 50 %.
   - **Not placed**: the two photos were not linked; the parts found in the photo are listed.

## What it does not change
Shown only. The level, the element the level uses (element model and "doubt means column"), the position (end zone, middle) and the crack's angle (element edges, true vertical, D58, D61) stay as they were (rule 5). Whether SAM's part should feed the level is a later decision, with Jerome and the engineer, after field tests.

## In the app
- Phone page: a fourth cell "Lies on" (the part, with "SAM 3.1, score" or "border with ..."), a line with SAM's reason ("... Shown only: the level does not use it."), and a small copy of the element photo with each part's mask in its colour and the crack in red with a ring. The element photo stays off the wall (rule 9).
- Result page of the website: the same "Lies on" cell.
- Without the weights on the machine (`models/sam3.1/`, gated, SAM License, never in git) the app runs as before and shows no cell. If SAM fails on a photo (memory), the app says so and goes on.
- About 4 s per element photo on the laptop after the first (the first loads the model, about 15 s).

## Limits
- One real photo so far; column, beam, balcony, stairs and expansion joint not yet seen by SAM on a real photo.
- A part not in the list (cornice, door frame) gives "not clear" when the crack lies on it.
- SAM's score is its own number, not a probability of being right.
- Memory: SAM 3.1 needs about 4 to 5 GB besides the app's models on a 16 GB laptop; 1008 px ran out of memory.

## Tests (`tests/test_element_parts.py`)
Fake segmenter, no weights: crack in a column lies on the column; along two parts gives the border; off every part gives not clear; without the link the parts are listed; two walls count together and two words for one thing once; a find at 0.41 is not used; the answer names SAM and never says "safe"; no weights gives no model; through the app the part is shown and the level, element, position and location are exactly as without SAM; a failing SAM does not stop the triage.
