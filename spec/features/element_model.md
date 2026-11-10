# Feature: element model, first test (`src/experiments/element_baseline.py`)

Status: experiment specified and coded (4 Nov 2026). Method B (trained) ran on 10 Nov on this laptop after a bug fix: 79 % accurate on the answers it was confident about, 72 % of the photos answered, a column confidently called something else 20 % of the time (`docs/experiments/2026-11-10_element_resnet_phinet6.md`). Method A (no training) not run yet. Decision D44. Pipeline step O3 (`spec/features/pipeline.md`).

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

## Tests (`tests/test_element_baseline.py`)
Class mapping and names from the PHI-Net README; the metrics on a hand-made confusion (accuracy on confident answers, coverage, recall, unsure handling); the loader on small fake arrays. No model is downloaded in the tests.
