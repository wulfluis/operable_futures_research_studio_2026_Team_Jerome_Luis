# Feature: quality gate (`src/quality/gate.py`)

Status: first version, written rules (not learned). Thresholds are illustrative until tuned on our own phone photos.

## Purpose
Decide whether a photo is usable before any model sees it. Unusable means Retake (`spec/features/triage.md`).

## Checks
| Check | Measure | Fails when (illustrative) |
|---|---|---|
| Sharpness | Variance of the 4-neighbour Laplacian of the grey image (0 to 255 scale), and the blur effect [56] (0 sharp to 1 blurred; how little the image changes when blurred again) | Laplacian variance below 100 **and** blur effect above 0.45: both must agree (D33) |
| Too dark | Mean grey level | below 40 |
| Too bright | Mean grey level | above 220 |
| Scale card visible | Not implemented yet (needs a scale-card detector; D24) | `scale_visible` is not `True` |

Each blur measure alone sends some sharp walls to retake: Laplacian variance fails smooth plaster with one thin crack (D12), the blur effect fails sharp walls with soft, mottled plaster. Both together: see D33 for the numbers.
Sharpness and brightness are measured on a working copy whose long side is scaled down to 512 px (illustrative); smaller images are measured as they are.
Reason: Laplacian variance falls as resolution rises, so without this a sharp phone photo would fail as blurred (`spec/decisions.md` D8).

`image_ok` is true only when every check passes. A missing scale-card result counts as a fail: any doubt means retake.
The result lists every failed check, so the app can tell the user what to fix.

## First test on real data (29 Oct)
Run the sharpness and brightness checks on public datasets (`src/experiments/real_data_quality.py`): crack-seg, Darbhanga and SDNET2018, done on 25 Oct.
Report the distributions and pass rates per dataset and crack / no-crack. Results: `docs/experiments/2026-10-25_quality_gate_first_test.md`.
Limits: these are not phone photos and have no scale card, so this tests the code on real images; it does not calibrate thresholds.
