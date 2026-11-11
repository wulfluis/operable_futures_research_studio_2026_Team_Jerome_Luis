# Baselines: existing crack models to compare ours against (links)

Midterm feedback, Reviewer 3: "trained crack models already exist ... normally you start from an existing model and change one thing at a time"; action item 5: "test the existing trained crack model as a comparison". This is the list of candidates, each with one or two sentences on what it is and why it is here. Numbers refer to `docs/sources.md`. Checked 11 Nov 2026 (repositories read through the GitHub API).

Our model, for reference: U-Net with a ResNet-34 encoder pre-trained on ImageNet and fine-tuned on 12,477 public crack photos (model v4, D48): pixel F1 0.85 on building concrete, 0.64 on public close-ups. **Pre-trained and fine-tuned**, as the reviewers asked us to say.

## 1. The non-AI baseline (what the project must beat)
- **An engineer's visual check of the same photo**, scored with the same tables (`spec/features/evaluation.md`). This is the mission's success test; it needs engineer-labelled photos, which we do not have yet.
- **A crack-width gauge or card held to the crack** (`docs/width_gauge/`, `src/measure/width_check.py`): the manual way to measure width, for the width feature due 12 Nov.

## 2. Trained crack models with public weights (ready to compare)
- **OmniCrack30k nnU-Net** (Benz and Rodehorst, CVPR Workshops 2024) [72]: https://github.com/ben-z-original/omnicrack30k
  A crack segmentation model (nnU-Net) trained on a benchmark pooled from many datasets, including masonry, concrete and structural-defect sets (the image count is not stated on the repository page). Trained weights download automatically (Google Drive, GPL-3.0); the dataset itself is by email only. **First choice**: the broadest existing model, to run on our test tiles next to U-Net v4.
- **Rezaie et al. (2020), EPFL** [47]: https://github.com/amirrezaie1415/Concrete-Crack-Segmentation
  A TernausNet16 crack segmentation network trained on concrete test images (digital-image-correlation speckle photos and the Özgenel crack set [49]); a pre-trained checkpoint is linked in the README (Google Drive; no licence stated). Here because the reference thesis (Corres Sojo 2024 [46]) uses exactly this network, so it is the baseline closest to our engineering source.
- **Ultralytics YOLO trained on crack-seg** [33]: https://docs.ultralytics.com/datasets/segment/crack-seg/
  The page shows how to train a YOLO segmentation model on the crack-seg set we also train on; no ready crack model is stated, so this would be a model we train (AGPL-3.0, D9). Here as a quick off-the-shelf architecture to compare, with its licence in mind.

## 3. Foundation models (no crack training of our own)
- **SAM 3.1 with the text "crack"** [81][82]: https://huggingface.co/facebook/sam3.1
  Already on the laptop (D62); found the crack and its branch in our sample photo at 0.71 (`docs/experiments/2026-11-11_sam3_first_run.md`). Here as the zero-shot baseline: what a general model does without any crack training.
- **CrackSAM** (Ge et al., 2024) [74]: https://arxiv.org/abs/2312.04233
  SAM fine-tuned for cracks with adapters and LoRA, with an exterior-wall crack set. Code and weights not stated in the abstract; later.
- **Segment Any Crack** (Rostami et al., 2025) [73]: https://arxiv.org/abs/2504.14138
  Adapts SAM by tuning only its normalisation layers; reports 61.22 % F1 on OmniCrack30k. No code stated; later.

## 4. After the mask: from crack shape to crack movement
- **Crack kinematics, Pantoja-Rosero et al. (2022), EPFL** [48]: https://github.com/eesd-epfl/crack_kinematics
  Computes crack opening and sliding from a binary crack mask (GPL-3.0, data on Zenodo doi:10.5281/zenodo.6632071). Here as the engineering-grade comparison for what we derive from the mask (width, direction), used in the reference thesis [46].

## 5. Map of the field
- **Review of crack detection** (Zhang et al., Georgia Tech, 2025) [75]: https://arxiv.org/abs/2508.10256
  Reviews learning methods and about 40 datasets; project page `nantonzhang/Awesome-Crack-Detection`. Here to answer R1's "how often has this been done before".

## 6. Recommended by Reviewer 3
To be added: the paper or PhD thesis Reviewer 3 shared at the midterm (Luis uploads it next).

## How to compare (proposal)
Same held-out test tiles for every model (our split by source photo, D31), the same metrics (pixel F1 and IoU, `src/segmentation/evaluate.py`), reported separately for building concrete and for pavement. Change one thing at a time: first the model, keeping our crack rule and measurement unchanged.
