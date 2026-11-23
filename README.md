# Food-11 Image Classification with ResNet-50 (PyTorch)

> **Project forked from:** [JackGomina/TIP_Projet](https://github.com/JackGomina/TIP_Projet)

After making a first model from scratch using Pytorch and hitting 75% accuracy, we wanted to make a more developped model. 
End-to-end deep learning pipeline for multi-class food image classification across 11 categories (`food-11` dataset), built with PyTorch and Torchvision using Transfer Learning and Test-Time Augmentation (TTA).

## Architecture & Technical Highlights

* **Backbone & Transfer Learning:** Pre-trained `ResNet-50` (`IMAGENET1K_V2` weights) with a custom classification head (`BatchNorm1d` → `Dropout(0.3)` → `Linear`).
* **Differential Learning Rates:** Fine-tuning with `AdamW` (`weight_decay=1e-4`), applying a lower learning rate (`1e-4`) to pre-trained backbone layers and a higher rate (`1e-3`) to the newly initialized classification head.
* **Learning Rate Scheduling:** `CosineAnnealingLR` decay down to `1e-6` over 12 epochs.
* **Regularization & Augmentation:**
  * Data augmentation pipeline (`RandomResizedCrop`, `RandomHorizontalFlip`, `RandomRotation(15°)`, `ColorJitter`).
  * `CrossEntropyLoss` with `label_smoothing=0.1` to prevent overconfidence and improve generalization.
* **Inference Optimization:** Horizontal-flip **Test-Time Augmentation (TTA)** averaging logits across original and mirrored images during validation and test inference.
* **Hardware Agnostic:** Automatic device selection supporting Apple Silicon (`MPS`), NVIDIA GPUs (`CUDA`), and `CPU`.

## Dataset Structure

The model classifies images into **11 food categories**:
`bread`, `dairy`, `dessert`, `egg`, `fried`, `meat`, `noodles-pasta`, `rice`, `seafood`, `soup`, `vegetable-fruit`.

```text
food-11/
├── train/
│   ├── bread/
│   ├── dairy/
│   └── ... (11 class subdirectories, split 85% train / 15% validation)
└── test/
    ├── 0.jpg
    ├── 1.jpg
    └── ... (unlabeled test images for JSON export)
