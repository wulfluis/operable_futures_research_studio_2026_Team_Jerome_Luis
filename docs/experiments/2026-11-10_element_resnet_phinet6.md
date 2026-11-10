# Element classifier on PHI-Net task 6, method B (10 Nov 2026)

Experiment of D44 (`spec/features/element_model.md`, `src/experiments/element_baseline.py`), method B only: ResNet-18 pre-trained on ImageNet, retrained on PHI-Net task 6 (component type, 224 px crops: beam, column, wall, others), 12 rounds, on the laptop GPU (about 2 minutes). Method A (Grounding DINO, no training) was not run: the `transformers` package is not installed here.
Data: 4,755 training images (10 % of them to pick the best round), 579 test images used once (beam 60, column 205, wall 265, others 49). CC BY-NC-SA 4.0, research only (D6). The images mix real buildings, laboratory specimens and timber columns, so this tests the method, not performance on informal concrete homes.

## A bug found and fixed first
The validation accuracy compared predictions in sorted order with labels in shuffled order, so the "best round" was picked at random (the first run, 8 rounds, had a validation accuracy of 0.31 to 0.41, near chance for four classes). `predict_resnet` now puts the rows back in the order asked for, with a test (`tests/test_element_baseline.py`). The first run was discarded; the numbers below come from the fixed code.

## Result (threshold 0.8: below it the answer is "unsure")
| True \ predicted | beam | column | wall | others | unsure |
|---|---|---|---|---|---|
| beam (60) | 29 | 0 | 8 | 2 | 21 |
| column (205) | 3 | 90 | 29 | 10 | 73 |
| wall (265) | 5 | 2 | 209 | 3 | 46 |
| others (49) | 6 | 9 | 10 | 5 | 19 |

- Coverage (answered with confidence): 72.5 %. Accuracy on the confident answers: 79.3 %. The best validation round reached 0.758.
- **A column called something else with confidence: 20.5 %** (42 of 205: 29 walls, 10 others, 3 beams). This is the costly error: a column called a wall loses the +1 raise of the risk matrix.
- Images that are not an element of the matrix ("others") called an element: 51 %.

## What it means
Not accurate enough alone: one in five columns would be confidently read as another element, and a quarter of the photos would go to a person anyway. Used as designed (D30), it can only suggest, and the field worker confirms. It also cannot see a hidden beam, and it has no slab class. Slabs, beams and the "element unknown" case are under discussion (`spec/constitution/roadmap.md`, open items). Weights: `models/element_resnet18_phinet6.pt` and the numbers `data/processed/element_baseline.json` (both local, not committed).
