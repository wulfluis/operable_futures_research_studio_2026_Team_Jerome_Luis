# Element classifier on PHI-Net task 6, method B, first and second run (10 Nov 2026)

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

## Second run, same evening: better training and "doubt means column" (D52)
Same data, same 579 test images, same tables (`spec/features/element_model.md`, "Second run"). Command: `--method second-run --epochs 20`, then `--method calibrate`. Logs in `data/results/element/`, numbers in `data/processed/element_second_run.json` and `element_calibrated.json` (local).

**Training.** Batches reshuffled every round, per-image crops and light and colour changes, warm-up then cosine learning rate, label smoothing 0.1, test-time flip; 20 rounds. Validation accuracy: ResNet-18 0.783, ResNet-50 0.819, **ConvNeXt-tiny 0.823 (chosen on validation)**, against 0.758 for the first run. Top-1 accuracy on the test images (every image answered, no threshold): 0.758, 0.751, 0.788.

**A side effect, found and fixed.** Label smoothing makes a network under-confident: with the 0.8 threshold the three networks answered only 23 to 31 % of the test images (30 % on validation too). Temperature scaling, one number fitted on the validation images by log loss (T = 0.78, 0.80, 0.76), gives the probabilities back their meaning; no retraining. Note: this step was added after the uncalibrated test numbers had been seen; the problem shows on the validation images alone, and the temperature uses only them.

**Result, ConvNeXt-tiny, calibrated, threshold 0.8** (test images, used once):

| True \ predicted | beam | column | wall | others | unsure |
|---|---|---|---|---|---|
| beam (60) | 35 | 2 | 1 | 1 | 21 |
| column (205) | 2 | 140 | 7 | 1 | 55 |
| wall (265) | 8 | 10 | 154 | 2 | 91 |
| others (49) | 9 | 7 | 3 | 5 | 25 |

| | First run (ResNet-18) | Second run, ConvNeXt-tiny, calibrated |
|---|---|---|
| Answered with confidence | 72.5 % | 66.8 % |
| Right, of those answered | 79.3 % | 86.3 % |
| **Column confidently called something else** | **20.5 %** (42 of 205) | **4.9 %** (10 of 205) |
| Not an element ("others") called an element | 51 % | 39 % |

ResNet-18 and ResNet-50 calibrated: answered 50.1 % and 58.5 %, right 88.6 % and 87.3 %, column confidently wrong 3.9 % and 5.4 %.

**"Doubt means column" had almost no effect.** On validation, the chosen network already kept columns confidently called something else at 3.0 %, under the 5 % target, so the rule's `COLUMN_DOUBT` came out at 0.5, where it practically never acts. Tried lower for the record: down to 0.10 nothing changes; at 0.05 one validation column more goes to a person and nothing changes on test (4.9 %). The columns it still misses, it misses with near certainty (column probability under 5 %), so there is no doubt for a rule to catch. The rule stays as a guard, but the gain came from the training.

**What it means now.** The dangerous error fell from one column in five to one in twenty, while the share answered fell a little. Still not for use alone: one in twenty columns is still confidently wrong, most non-elements are called an element, there is no slab class, and PHI-Net is not informal concrete homes. As designed, the model suggests and a person confirms. Weights: `models/element_convnext_tiny_phinet6_v2.pt` (local, not committed).

## What the first run meant
Not accurate enough alone: one in five columns would be confidently read as another element, and a quarter of the photos would go to a person anyway. Used as designed (D30), it can only suggest, and the field worker confirms. It also cannot see a hidden beam, and it has no slab class. Slabs, beams and the "element unknown" case are under discussion (`spec/constitution/roadmap.md`, open items). Weights: `models/element_resnet18_phinet6.pt` and the numbers `data/processed/element_baseline.json` (both local, not committed).
