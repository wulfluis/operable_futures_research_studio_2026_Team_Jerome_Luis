# Reading list: models and datasets for crack, element, damage and material (checked 11 Nov 2026)

These links come from a Claude chat (Luis, "Second Look project"). Each page was opened on 11 Nov 2026, and the notes below say only what the page says. "Not read" means the page refused automated reading (login or HTTP 403), so it is listed but not judged. The numbers in brackets refer to `docs/sources.md`.

**Rating** (fit to our case: phone photos of cracks in reinforced-concrete buildings in informal neighbourhoods, D14):
- **High**: use or test it soon.
- **Medium**: useful later or for one part.
- **Low**: background only, or off-concept (bridges, roads).

## Corrections to the chat summary
- **OmniCrack30k:** the dataset is **not openly downloadable** ("due to license issues"). You email the author from a university address for non-commercial use. The trained nnU-Net weights are public.
- **"CrackESS":** the arXiv link 2312.04233 is **CrackSAM** (Ge et al.), which fine-tunes SAM with adapters and LoRA. The abstract does not mention YOLOv8 prompting SAM.
- **Our crack model:** pixel F1 is 0.85 on building concrete and 0.64 on public close-ups (model v4, D48), not 0.63.
- **RF-DETR:** the Roboflow comparison page lists instance segmentation as "not listed". The benchmark is Roboflow's own, and Roboflow makes RF-DETR.
- **SAM 3:** the weights are gated (request access on Hugging Face). The model is large: 473.6 M parameters, 3.45 GB, 2.9 s per image on a workstation GPU per Ultralytics. The weights' licence was not stated on either page we read.

## Summary: what helps us now
1. **SAM 3** [81][82] is the strongest candidate for the open problem: finding the bank card and the column or beam in the far element photo by a text prompt ("credit card", "concrete column"). Its masks could replace our edge search for the element. Cost: a 3.45 GB model, gated weights, licence to check, and likely tens of seconds per photo on a laptop CPU (assumption, to measure). Test it offline first, not in the app.
2. **Crack type has no ready model or dataset.** Every source found labels only crack / no crack, or bridge damage classes. That confirms our written pattern rules (D43, D57, D58).
3. **OmniCrack30k** [72] is the best route to a better crack model, if the author grants access. Its nnU-Net weights can be compared with our U-Net v4 on our own test tiles now.
4. **Material** ("is it concrete?"): Apple DMS [90][91] has Concrete, Brickwork and Paint/plaster/enamel classes. That makes it the only large public source for the material model. Its licence is Apple's own and must be read first.

## 1. Crack pixels (segmentation)

| Source | What it is | What the page says | Fit to us | Rating |
|---|---|---|---|---|
| **OmniCrack30k** [72], Benz and Rodehorst, CVPRW 2024; repo `ben-z-original/omnicrack30k` | Crack segmentation dataset compiled from many sets, plus the best nnU-Net model | Repo: GPL-3.0 code; nnU-Net weights downloaded automatically (Google Drive); **dataset by email only**, non-commercial. Includes Masonry, CSSC (concrete), S2DS (structural defects) and road sets. Image count and scores were not on the repo page; the CVPR page refused reading (403) | The largest crack set and a ready model to compare with ours. Mixed with roads | **High** |
| **Segment Any Crack (SAC)** [73], Rostami, Chen, Hosseini, arXiv Apr 2025 | Fine-tunes only the normalisation parameters of SAM and 5 other models | 61.22 % F1, 44.13 % IoU on OmniCrack30k; best on 3 zero-shot sets. Code: not stated | A cheap way to adapt SAM to cracks; no code yet | Medium |
| **CrackSAM** [74], Ge et al., arXiv Dec 2023, rev. Apr 2024 | SAM fine-tuned with adapters and LoRA | Two new open datasets, **road and exterior-wall cracks**, 810 images in total; "far exceeds" 12 models on noisy and unseen data (no numbers in the abstract). Links: not stated | The only SAM crack work with **exterior-wall** cracks; its wall set could be test data | Medium |
| **Crack detection review** [75], Zhang et al., Georgia Tech, arXiv Sep 2025 | Review of learning paradigms, about 40 datasets, new 3DCrack set | Project page `nantonzhang/Awesome-Crack-Detection`. Benchmark findings came from a model summary of the paper, not checked: ResUNet strongest with full labels; CrackSAM better with few labels; zero-shot models weak | Map of datasets and methods; supports keeping a U-Net | Medium |
| **SAM-based instance segmentation, MCrack1300** [76], Ye, Lovell, Faramarzi, Ninic, arXiv Jan 2024 | SAM with LoRA, 2 prompt-free variants; masonry dataset | 1,300 images (640 x 640), classes bricks, broken bricks, cracks; about +6 % on cracks over the best baseline; **crack size from known brick sizes** with a Hough-line orthographic map, within 10 % of laser scans. Links: not stated | Masonry, not RC; but its scale-from-bricks idea is close to our card scale | Medium |
| crack-seg [33] (Ultralytics) | Already in use for U-Net v1 to v4 | 4,029 images (train 3,717, val 200, test 112), one class "crack", roads and walls, Public Domain Mark per the page | Already known | (in use) |

## 2. Other damage (spalling, exposed bars)

| Source | What it is | What the page says | Fit to us | Rating |
|---|---|---|---|---|
| dacl10k [68] (already in `docs/sources.md`) | Bridge damage segmentation, WACV 2024 | Toolkit read: 19 classes (13 damages including Crack, Alligator Crack, Spalling, ExposedRebars, Rust; 6 bridge objects); train 6,935, val 975, test 1,012 + 998; data CC BY-NC 4.0. The WACV page refused reading (403) | Bridges (off-concept, D14); possible spalling pre-training | Low |
| CODEBRIM [28][29] (already in `docs/sources.md`) | Bridge concrete defects | 1,590 images, 30 bridges; crack, spallation, exposed bars, efflorescence, corrosion stain; non-commercial, no redistribution | Bridges; boxes not masks | Low |
| **StructDamage** [77], Ijaz et al., arXiv Mar 2026 | About 78,093 images from 32 public datasets, re-labelled | Nine surface types (walls, tile, stone, road, pavement, deck, concrete, brick); macro F1 above 0.96 for 12 of 15 classifiers. Labels by surface, not crack type. Licence and download: not stated | Large and includes walls; only crack / surface classes | Low |
| ArchitectEye [89], MBZUAI course project | FPN with EfficientNet-B4 on dacl10k | Weights on SharePoint; no scores, no licence | Bridges; a student project | Low |

## 3. Crack type and width

| Source | What it is | What the page says | Fit to us | Rating |
|---|---|---|---|---|
| Crack orientation classification [78], Scientific Reports 2026 | Classifies crack direction (per the chat) | **Not read** (login) | Would back up our angle rules; read via the university library | Medium (to read) |
| Crack characterisation in RC beams [79], Springer 2026 | Per the chat, crack characterisation in RC beams | **Not read** (login) | Possibly closest to our pattern rules; read via the library | Medium (to read) |
| Crack width on binary images [80], MATLAB Answers 2015 to 2025 | Q&A | Width = 2 x distance transform along the skeleton; the asker says it fails on irregular cracks; a 2024 answer links MATLAB width packages | We already do more: width read across the crack at each centre-line point, with sub-pixel edges (`src/measure/crack.py`) | Low |

## 4. Element (column, beam, wall, slab)

| Source | What it is | What the page says | Fit to us | Rating |
|---|---|---|---|---|
| **SAM 3** [81], Meta, released 19 Nov 2025 (SAM 3.1 update Mar 2026) | One model for detection, segmentation and tracking by **text**, exemplar or click prompts | Code, checkpoints and fine-tuning code at `facebookresearch/sam3`; weights on Hugging Face `facebook/sam3` (gated). 30 ms per image on an H200 GPU. Weights licence: not stated on the page | Find "concrete column", "beam" and the bank card in the element photo without training; pre-label our own photos | **High** (test offline first) |
| **SAM 3 in Ultralytics** [82] | How to run SAM 3 | `pip install -U ultralytics`; `SAM3SemanticPredictor` with `text=[...]`; `sam3.pt` must be downloaded by hand after access; 473.6 M parameters, 3,450 MB, 2,921 ms per image on an RTX PRO 6000. CPU: not stated | The easiest way to try SAM 3 here | **High** (with [81]) |
| Roboflow Construction_seg [83], Roboflow structure [84], Roboflow beam/column/slab [85] | Small element datasets | **Not read** (403). Roboflow Universe needs a browser | Possible element labels incl. slab; check classes and licences by hand | Medium (to check) |
| Turer, Bai, Sezen, Yilmaz [86], arXiv Oct 2025 | Post-earthquake framework: inside/outside and component classifier, YOLOv11 for cracking, spalling, exposed bars, a damage-level model | New images from the 2023 Turkey earthquakes; image counts, classes and scores not in the abstract; CC BY 4.0 paper; links not stated | Concrete buildings; the closest pipeline to ours; ask the authors about the data | Medium |
| Tauzowski et al. [87], Sensors 2025 (Tokaido) | Component and damage segmentation | 7,575 synthetic viaduct images; components slab, beam, column; damage spalling, exposed bars; 87 % mIoU on components (validation). Article CC BY 4.0 | Synthetic bridges; shows component + damage together works | Low |
| Liu, Wei, Chen, Noh [88], Smart Structures and Systems 2023 | Hierarchical segmentation: a damage only on the component where it can occur | No numbers in the abstract | The idea (damage tied to component) matches our element-then-pattern rules | Low |

## 5. Material ("is it concrete?")

| Source | What it is | What the page says | Fit to us | Rating |
|---|---|---|---|---|
| **Apple DMS** [90], Upchurch and Niu, ECCV 2022 | Dense material segmentation, indoor and outdoor | 44,560 images, 3.2 M segments; baseline 0.420 mIoU. Licence: Apple's own terms | Has **Concrete**, **Brickwork** and **Paint/plaster/enamel** classes (per [91]) | **High** for the material model (licence first) |
| DMS re-benchmark [91], Kazakov et al., arXiv May 2026; repo `AllanK24/Material_Seg_AppleDMS` | SegFormer-B5 and Mask2Former on DMS | 41,396 of 44,560 images recovered; SegFormer-B5 0.457 mIoU on the original split; splits on Hugging Face; no weights named; no code licence | A ready training recipe for the material model | Medium |

## 6. Licences of detection models

| Source | What the page says | Fit to us | Rating |
|---|---|---|---|
| RF-DETR vs YOLO26 [92], Roboflow | RF-DETR Nano to Large: Apache 2.0 (XL, 2XL: PML 1.0); YOLO26: AGPL-3.0; segmentation listed for YOLO26, "not listed" for RF-DETR; COCO numbers from Roboflow's own leaderboard | Only if we add a detector. We use none now (U-Net from segmentation-models-pytorch, ConvNeXt from torchvision) | Low |

## What this means for the current problem (card and element in the far photo, D59)
- Our card search by look found the wrong place in a room photo (11 Nov). SAM 3 with a text prompt could find the card and the column in the far photo directly. It would then hand our code a mask, and our code keeps the written rules (rule 5).
- Proposed next step, after the team agrees: a small offline test (`src/experiments/`) on our own photo pairs. Measure whether SAM 3 finds the card and the element, and how long it takes on the laptop. Nothing goes into the app before that test.

## 7. Engineering rules for crack patterns and angles (added 11 Nov 2026)
Searched to back or correct the angle bands of our crack-type rules (D43). Full notes: `docs/literature/crack_pattern_rules_sources.md` and `docs/literature/fema_306.md`.

| Source | What it gives | Rating |
|---|---|---|
| **FEMA 306** [93] (ATC for FEMA, 1998), https://nehrpsearch.nist.gov/static/files/NIST/PB2001100763.pdf | Flexure across the member, shear diagonal "typically 35° to 70° from the horizontal"; crack widths per damage severity for RC walls | **High** |
| FEMA 307 [95] (1998), https://nehrpsearch.nist.gov/static/files/FEMA/PB2001101733.pdf | Width alone is a poor damage indicator; location, orientation, number and distribution matter as much | Medium |
| Yang, Chang, Wu 2019 [96], Sensors, CC BY | Image analysis: within 22.5° of horizontal = flexural, else shear (their specimens only) | Medium |
| JBDPA guideline [98] via Quan et al. 2025 [97] | RC member damage classes by crack width: under 0.2, 0.2 to 1, 1 to 2, over 2 mm | **High** (for width bands) |
| Eurocode 2 [101] | Design strut angle 21.8 to 45° (design, not crack reading) | Low |
| NZ C5 [99], AeDES [100], ACI 224R [94] | Not read (download refused, no file, or sold) | To read |

## 8. Midterm feedback follow-up (added 11 Nov 2026)
Sources for the reviewers' open points (post-disaster scale, how engineers assess today, prior work, capture distance): `docs/literature/midterm_followup/README.md`, sources [102] to [107].

## 9. Baselines (added 11 Nov 2026)
Existing trained crack models to compare our U-Net against (OmniCrack30k, Rezaie et al. 2020 from the reference thesis, SAM 3.1 zero-shot, and others): `docs/literature/baselines/README.md`.
