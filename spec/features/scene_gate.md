# Feature: scene check, is the photo a close-up? (`src/quality/scene.py`, `src/experiments/scene_gate.py`)

Status: trained and in the app, 11 Nov 2026, decision D54: test 93.4 % right, whole buildings let through 0 %, close-ups sent to retake 4.0 % (`docs/experiments/2026-11-11_scene_gate_phinet1.md`). Pipeline step [2], the photo check, at the three scales S, O, P (D21).

## Why
The crack model was trained only on close-ups of surfaces. On anything else (a room, a window, a whole building) it reads frames, blinds, wires and corners as cracks (`spec/features/class_demo.md`, Limits; the app test of 11 Nov). A written rule per shape (D51) cannot cover every room. A model that tells a close-up from a wider view stops the crack model from looking at photos it cannot read.

## Data
PHI-Net task 1, scene level [53][55]: 224 x 224 px images in three classes, exactly our three scales: **P** pixel level (a close-up of a surface), **O** object level (a whole element), **S** structural level (a whole building). Train 24,309 (O 8,111, P 7,690, S 8,508), test 2,997 (O 962, P 965, S 1,070). CC BY-NC-SA 4.0, research only (D6). Earthquake-damage photos, not informal homes: the test measures the method.

## Model
ConvNeXt-tiny pre-trained on ImageNet, trained as the element model's second run (D52: reshuffled batches, crops, light and colour changes, cosine learning rate, label smoothing, test-time flip), 8 rounds; 10 % of train picks the best round. Probabilities calibrated with a temperature fitted on the validation images.

## The rule (written, `src/quality/scene.py`)
- **Crack photo:** if the probability of P is under `CLOSE_UP_MIN`, the answer is **Retake**: "This looks like a wider view, not a close-up of the wall: hold the phone 20 to 40 cm from the crack." The crack model does not run.
- `CLOSE_UP_MIN` is chosen on the validation images: the highest value at which at most 5 % of real close-ups are sent to retake (illustrative target: a needless retake costs one more photo; a wide view let through gives false cracks).
- **Element photo:** the element model runs only if the scene model does not read it as a close-up (P highest): a close-up cannot show which element it is. Otherwise "This is a close-up: step back so the whole column, beam or wall is in the photo."
- The scene answer and its probability are part of the result (JSON), like every other step.

## Reported (`docs/experiments/2026-11-11_scene_gate_phinet1.md`)
The confusion table on the test images; at `CLOSE_UP_MIN`: the share of close-ups sent to retake, and the share of O and S photos let through as close-ups (the false cracks it would still allow).

## Limits
- PHI-Net close-ups are mostly damaged concrete after earthquakes; a clean painted wall at home may look different (to check on our own photos).
- A close-up of a window frame or a socket is still a close-up: this check does not stop it. Step 3 of the plan (D54) teaches the crack model those.

## Tests
`tests/test_scene_gate.py`: the rule with a stand-in model (close-up passes, wide view gets Retake with its reason, threshold choice on validation, missing model file passes everything to the crack model with a note), the app: a wide view never reaches the crack model.
