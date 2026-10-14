# Buried Threat Detection in Ground Penetrating Radar with a CNN

A PyTorch reimplementation of the training practices from Reichman, Collins and Malof, [*Some Good Practices for Applying Convolutional Neural Networks to Buried Threat Detection in Ground Penetrating Radar*](https://doi.org/10.1109/IWAGPR.2017.7996100) (IWAGPR 2017).

Ground penetrating radar (GPR) produces B-scans: 2-D images where depth runs down and distance along the scan runs across. A buried object shows up as a hyperbola. The task is to classify small patches of a B-scan as **threat** or **non-threat**.

## Pipeline

```
annotated B-scans ──► gpr.patches ──► 32×32 patches + patches.csv
                                            │
grayscale CIFAR-10 ──► gpr.pretrain_cifar ──┤ (optional) transfer first N conv layers
                                            ▼
                                       gpr.train ──► best.pt + metrics.json
```

1. **Patch extraction** (`gpr/patches.py`). Each annotated threat box is cropped and resized to 32×32. With `--augment`, shifted copies (±4 and ±8 px vertically, ±4 px horizontally) are added, following the paper's shift-based augmentation. Non-threat patches are random windows that do not overlap any annotated threat.
2. **Pretraining** (`gpr/pretrain_cifar.py`). The same network is trained on grayscale CIFAR-10. CIFAR images are already 32×32, so they match the patch size without resizing.
3. **Transfer and training** (`gpr/train.py`). The first N convolutional layers (default 4 of 6) are copied from the pretrained network, then the whole network is trained on GPR patches with mini-batches, Adam and a class-weighted loss.

**Model** (`gpr/model.py`): three blocks of two 3×3 convolutions (16, 32, 64 channels) with batch norm and max pooling, then dropout and a linear classifier.

## Evaluation design

- **Split by scan, not by patch.** Shifted threat patches and nearby background windows from one B-scan are near-duplicates. Splitting at patch level would put copies of the same target in train and test and inflate the score, so all patches from a scan stay in one split.
- **Model selection on validation.** The saved checkpoint is the epoch with the best validation ROC AUC, with ties broken by validation loss. AUC alone ignores calibration, and an early epoch can rank perfectly while predicting "no threat" for everything.
- **Threshold from validation.** The decision threshold is chosen on the validation set (Youden's J), then applied unchanged to the test set.
- **Metrics.** Precision, recall, F1, ROC AUC and the confusion matrix, alongside accuracy, since non-threat patches outnumber threats.

## Usage

```bash
pip install -r requirements.txt

# 1. Cut patches from annotated scans
python -m gpr.patches --images data/scans --annotations data/labels.json --out data/patches --augment

# 2. (Optional) Pretrain on grayscale CIFAR-10
python -m gpr.pretrain_cifar --epochs 10 --out runs/cifar_gray.pt

# 3. Train and evaluate
python -m gpr.train --csv data/patches/patches.csv --img-dir data/patches/images \
    --pretrained runs/cifar_gray.pt --transfer-layers 4 --out runs/gpr
```

The annotation file maps each B-scan image to its threat boxes:

```json
{"frames": {"scan_01.png": [{"x1": 10, "y1": 40, "x2": 42, "y2": 72}]}}
```

## Status

The B-scan dataset used originally is not included in this repository, so no results on real GPR data are reported here. The full pipeline (patch extraction, pretraining, layer transfer, training, scan-level evaluation) is tested end to end on synthetic B-scans with injected hyperbolas:

```bash
pytest -q
```
