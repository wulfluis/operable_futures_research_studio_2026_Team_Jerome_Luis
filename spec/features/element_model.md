# Feature: element model, first test (`src/experiments/element_baseline.py`)

Status: experiment specified and coded (4 Nov 2026). Method B (trained) ran on 10 Nov on this laptop after a bug fix: 79 % accurate on the answers it was confident about, 72 % of the photos answered, a column confidently called something else 20 % of the time. Second run the same evening (D52): ConvNeXt-tiny, calibrated, 86 % right, 67 % answered, a column confidently called something else 4.9 % of the time (`docs/experiments/2026-11-10_element_resnet_phinet6.md`). Method A (no training) not run yet. Decision D44. Pipeline step O3 (`spec/features/pipeline.md`).

## Question
Can a model tell which element an element photo (scale O) shows, well enough that the field worker only has to confirm it? Is an off-the-shelf model without training enough, or do we need to train one?

## Data
PHI-Net task 6, component type [53][55]: 224 x 224 px images, classes beam, column, wall, others; 4,755 train and 579 test images (beam 511, column 1,618, others 358, wall 2,268 in train). Licence CC BY-NC-SA 4.0, research only (D6). No slab class. The images mix real buildings, laboratory specimens and timber columns (`docs/data_map.md`), so the test measures the code and the method, not performance on informal concrete homes.
Class mapping to the risk matrix: beam -> `beam`, column -> `column`, wall -> wall (load-bearing or infill cannot be told from the photo: the field worker decides), others -> not an element of the matrix.

## Two methods, same test images
| | A. Without training | B. Trained |
|---|---|---|
| Model | Grounding DINO tiny (IDEA Research, Apache 2.0, via Hugging Face transformers): finds objects named in a text prompt | ResNet-18 pre-trained on ImageNet (torchvision), last layer retrained on PHI-Net task 6 |
| Input | prompt "a column. a beam. a wall." | the image |
| Score per class | highest box score for that word | softmax probability |
| Unsure | best score under its threshold, or no box | probability under 0.8 (`CONFIDENCE_THRESHOLD`, `src/triage/matrix.py`) |
| Training | none | 90 % of train to learn, 10 % to pick the best epoch; test used once at the end |

## What we report (`docs/experiments/`)
Per method: the confusion table (true class x predicted class, with "unsure" as its own column); **accuracy on confident answers**; **coverage** (the share of images answered with confidence); recall per class; and the share of `others` images wrongly called beam, column or wall. A wrong confident answer is the costly error: a column called a wall loses the +1 raise of the risk matrix. So we report how often a column is confidently called something else.

## Rules for using it later (not now)
- Below the threshold: the field worker chooses, or Expert needed without one (rule 2).
- The element model never sees a resident's answer about the element (D25).
- Before it is used on real cases: test on our own element photos, rated by the engineer.

## Second run: better training, and "doubt means column" (10 Nov, D52)
The first run is not good enough alone: one column in five confidently called something else. Two steps, same data, same test images, same tables.

**Step 1, better training** (`--arch`, `src/experiments/element_baseline.py`):
- Networks compared: ResNet-18 (as before), ResNet-50 and ConvNeXt-tiny, all pre-trained on ImageNet (torchvision, BSD-3 code; ImageNet weights, research use).
- Training: the batches are reshuffled every round; each image is flipped left-right at random, cropped (75 to 100 % of its side, resized back) and given small changes of brightness, contrast and colour (±20 %, illustrative); no vertical flip or rotation (a beam turned upright looks like a column). Learning rate with a short warm-up, then a cosine decay; label smoothing 0.1; class weights as before; 20 rounds.
- The answer is the mean of the image and its mirror image (test-time flip).
- **The network is chosen on the validation images (10 % of train), never on the test images.** The test images are then used once per network and all results are reported, the ones not chosen too.

**Step 2, "doubt means column"** (a written rule, `src/triage/element_rule.py`, rule 5):
The costly error is a column called a wall: the column's +1 raise is lost. So the probabilities become an answer in this order:
1. The column has the highest probability and reaches the threshold (0.8, `CONFIDENCE_THRESHOLD`): **column**.
2. Another class is highest, but the column's probability is at least `COLUMN_DOUBT`: **unsure, could be a column**. A person decides; without one it is "Expert needed" (rule 2).
3. Otherwise the highest class if it reaches the threshold, else **unsure**.
- `COLUMN_DOUBT` is chosen on the validation images: the largest value (fewest photos sent to a person) at which at most 5 % of the columns are confidently called something else (target illustrative, to agree with the engineer). It is then applied once to the test images.
- What it costs is reported with it: the coverage (share of photos answered without a person) falls.

**Reported for each network**, with and without the rule: coverage, accuracy on confident answers, column confidently wrong, others called an element, the confusion table (`docs/experiments/2026-11-10_element_resnet_phinet6.md`).

**Calibration (added after the first second-run result).** Label smoothing made the networks under-confident (23 to 31 % answered at 0.8). A temperature fitted on the validation images by log loss (`fit_temperature`, `--method calibrate`) is applied to the probabilities before the threshold and the rule.

**Result (10 Nov):** ConvNeXt-tiny, chosen on validation, calibrated: 66.8 % answered, 86.3 % right, **column confidently called something else 4.9 %** (first run 20.5 %). "Doubt means column" barely acted: the remaining wrong columns are confident ones. Full tables: `docs/experiments/2026-11-10_element_resnet_phinet6.md`.

**Still true after both steps:** PHI-Net has no slab class; its images are not informal concrete homes; the element needs a photo of the whole element, not the crack close-up. The app does not use the model yet (step 3, hackathon).

## Tests (`tests/test_element_baseline.py`)
Class mapping and names from the PHI-Net README; the metrics on a hand-made confusion (accuracy on confident answers, coverage, recall, unsure handling); the loader on small fake arrays. No model is downloaded in the tests.
