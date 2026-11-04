# Feature: crack pattern from written rules (`src/triage/pattern_rules.py`)

Status: draft v0 (4 Nov 2026), in code, tested on synthetic shapes only. Decision D43. Every angle and limit below is illustrative until a structural engineer sets it. The basis is standard reading of crack patterns in reinforced concrete (assumption: to be checked against FEMA 306 and ACI 224R, not yet read, and with the engineer).

## Why rules and not a model
No public dataset labels our seven patterns, and the closest ones (PHI-Net task 8 [53][55], IDEA [64]) label only flexural, shear and a few others (`docs/data_map.md`). Engineers tell most of the patterns apart by **the crack's direction relative to the element, its position on the element, and its form**. Those can be measured, so a written rule can suggest the pattern. That keeps the decision auditable (AGENTS.md rule 5), and a model is needed only for what has data: the crack mask, the element, and spalling.

## What it returns
A **suggestion**: a pattern key of the risk matrix (or none), `clear` true or false, the rule that fired, and a one-sentence reason. **It never feeds the matrix by itself.** Until the rules are checked against engineer-labelled photos, a field worker confirms or corrects every suggestion, and only a confirmed pattern goes into `triage.matrix.assess()`. An unclear suggestion, or none, means the field worker chooses from the drawings; without a field worker the case goes to an engineer (Expert needed, rule 2).

## Inputs
| Input | From | Values |
|---|---|---|
| Crack shape | `src/measure/crack.py` | angle to vertical (0 = vertical, 90 = horizontal, on the wall), straightness, coherence, closed loops, branch ratio |
| Element | element model or field worker (never the resident, D25) | column, beam, slab, load_bearing_wall, infill_wall, or unknown |
| At a beam-column joint | field worker (element photo) | yes, no, unknown |
| Spalling or exposed bar | spalling model (PHI-Net task 3, IDEA) or field worker | yes, no, unknown |
| Rust stain | field worker, resident answer | yes, no |
"Vertical" is the image's up-down direction carried onto the wall plane, so it assumes the **phone was held upright** (assumption). A slab photographed from below has no up-down direction.

## Element axis
Column: vertical. Beam: horizontal. Walls and slabs: no axis. The angle φ between the crack and the element axis is 0 to 90 degrees.

## Rules, first match wins (illustrative)
| # | Condition | Suggestion | Clear |
|---|---|---|---|
| 1 | No crack measured | none | no |
| 2 | Spalling or exposed bar seen | `spalling_exposed_bar` | yes |
| 3 | A network: at least 2 closed loops, or coherence under 0.35 with all branches at least 1.8 x the main length | `surface_map` | yes |
| 4 | Element unknown | none | no |
| 5 | At a beam-column joint, on a column or beam | `joint` | yes |
| 6 | Straightness under 0.6 (too winding to read a direction) | none | no |
| 7 | Slab | none (no up-down direction from below) | no |
| 8 | Wall: crack 35 to 55 degrees from vertical | `diagonal_shear` | yes |
| 9 | Wall: any other direction | none | no |
| 10 | Column: φ at least 65 (horizontal) | `column_horizontal` | yes |
| 11 | Column or beam: φ 35 to 55 | `diagonal_shear` | yes |
| 12 | Beam: φ at least 65 (across the beam) | `flexural` | yes |
| 13 | Column or beam: φ at most 25 (along the element) with a rust stain | `along_bar` | yes |
| 14 | Column or beam: φ at most 25 without a rust stain | `along_bar` | **no**: on a column, a vertical crack can also be crushing under load; an engineer decides |
| 15 | φ between bands (25 to 35, 55 to 65) | the nearer band's pattern | no |

## Limits
- One crack, one pattern. A crack that changes direction (flexural turning into shear) is read by its main direction only, and its straightness drops.
- The phone must be upright; a sideways photo swaps vertical and horizontal. The app does not yet read the phone's orientation (prepare.py strips the metadata that holds it).
- Spalling and the element come from models that are not trained yet; until then, from the field worker.
- No location along the element (end zone, mid-span) is used yet, though it separates flexural cracks at mid-span from shear cracks near supports. Next version, with the engineer.

## Tests (`tests/test_pattern_rules.py`)
Synthetic shapes: horizontal and diagonal cracks on beams and columns, vertical cracks in columns with and without rust, a crack network, spalling, a joint, a slab, an unknown element, angles in the margins, and a measured synthetic crack end to end. The suggestion never carries a level.

## What we need from the engineer
The bands (25, 35, 55, 65 degrees), whether a vertical crack in a column may ever be suggested without rust, the network limits, how to use the location on the element, and a set of photos rated by pattern to measure how often the rules are right.
