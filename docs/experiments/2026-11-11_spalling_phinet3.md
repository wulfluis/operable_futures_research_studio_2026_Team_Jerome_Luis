# Spalling model on PHI-Net task 3, and the ten simulated cases (11 Nov 2026)

Experiment of D61 (`spec/features/photo2.md`, `src/experiments/spalling_model.py`).

## Spalling model
ConvNeXt-tiny pre-trained on ImageNet, trained as the scene model (D54), 8 rounds, 5 minutes on the laptop GPU. PHI-Net task 3 [53][55]: 6,898 training photos (10 % to pick the round, the temperature and the threshold), 837 test photos used once (spalling 310, no spalling 527). CC BY-NC-SA 4.0. Numbers: `data/processed/spalling_model.json` (local).

Best validation accuracy 0.858. Temperature 1.0. Threshold chosen on validation to catch at least 90 % of spalling: **0.40**.

| Test (837) | Seen | Not seen |
|---|---|---|
| spalling (310) | 278 | 32 |
| no spalling (527) | 101 | 426 |

Spalling caught **89.7 %**; false alarms **19.2 %**; accuracy 84.1 %. A missed spalling can lower a level; a false one sends the case up to an engineer, so the threshold favours catching.

## The ten simulated cases (`src/app/simulate.py`)
Photo 2: real PHI-Net task 6 test photos of damaged columns, beams and walls on which the crack model finds a large crack (chosen also because the element model reads the element right). Photo 1: synthetic with a bank card, its crack in the same direction. Dry run with the real models, before the live run:

- 10 of 10: crack found, card found, width within 0.06 mm of the drawn width, length 133 to 153 mm, element right, crack found on photo 2 (it spans 33 % to 100 % of the element), 3D model placed from photo 2.
- Spalling seen on 6 of 10 photo 2s (post-earthquake damage).
- Levels: Category 3 nine times, Category 2 once (a 0.64 mm crack across a beam, no spalling).

What this shows: the flow from two photos to a level and a 3D model. It does not show the models' accuracy (selected photos; see the experiments of D52, D54 and this one), nor anything about informal homes (PHI-Net is post-earthquake damage).
