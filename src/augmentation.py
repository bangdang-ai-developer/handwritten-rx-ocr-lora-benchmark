"""Data augmentation for LoRA fine-tuning TrOCR-large-handwritten on the Kaggle-Rx
handwritten-prescription train split (3,120 images - small, needs augmentation to avoid
overfitting). Standard HTR augmentation techniques (affine, elastic distortion, noise/blur,
morphological erosion-dilation). Small-dataset TrOCR fine-tuning on prescription-scale data is
independently shown to work by Ali et al. (arXiv:2412.18199, Mask R-CNN + TrOCR on ~1k
Pakistani prescriptions - NOT an augmentation-methodology paper, cited only as precedent that
fine-tuning TrOCR at this data scale reaches strong CER) and Aradillas et al. (arXiv:1804.01527,
ICFHR 2018, transfer learning for small HTR datasets); see docs/05-ke-hoach-Q2.md Section 3.2
for the full rationale and citation caveats.

Kept in sync with the copy embedded in notebooks/kaggle_benchmark.py for the actual
Kaggle training run (same reason src/metrics.py has a synced copy there).
"""
from __future__ import annotations

import albumentations as A
import cv2
import numpy as np
from PIL import Image


def build_train_augmentation(elastic: bool = True) -> A.Compose:
    """Training-time augmentation. `elastic=False` is the A/B ablation arm without elastic
    distortion (TrOCR-small literature suggests elastic isn't always helpful - Sec 3.2 notes
    TrOCR-large likely benefits, but we A/B test rather than assume)."""
    steps = [
        A.Affine(rotate=(-5, 5), shear=(-8, 8), scale=(0.95, 1.05), p=0.7),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.OneOf([
            A.Morphological(scale=(1, 2), operation="erosion", p=1.0),
            A.Morphological(scale=(1, 2), operation="dilation", p=1.0),
        ], p=0.3),
    ]
    if elastic:
        steps.insert(1, A.ElasticTransform(alpha=30, sigma=5, p=0.3))
    return A.Compose(steps)


def augment_pil(image: Image.Image, augmenter: A.Compose, seed: int | None = None) -> Image.Image:
    """Apply `augmenter` to a PIL image, returning a new PIL image (RGB)."""
    if seed is not None:
        np.random.seed(seed)  # Albumentations' random ops draw from numpy's global RNG
    arr = np.array(image.convert("RGB"))
    out = augmenter(image=arr)["image"]
    return Image.fromarray(out)


if __name__ == "__main__":
    # Self-test: run against a handful of real training images if available locally
    # (python -m src.augmentation, or `python src/augmentation.py` from repo root).
    import glob
    import os

    candidates = glob.glob(
        "data/kaggle_rx/*/Training/training_words/*.png"
    ) or glob.glob("data/kaggle_rx/**/Training/**/*.png", recursive=True)
    if not candidates:
        print("Khong tim thay anh training cuc bo de test - bo qua self-test hinh anh, "
              "chi kiem tra pipeline khoi tao duoc khong.")
        aug_with = build_train_augmentation(elastic=True)
        aug_without = build_train_augmentation(elastic=False)
        dummy = Image.fromarray((np.random.rand(64, 128, 3) * 255).astype("uint8"))
        out1 = augment_pil(dummy, aug_with, seed=0)
        out2 = augment_pil(dummy, aug_without, seed=0)
        assert out1.size[1] > 0 and out2.size[1] > 0
        print("Self-test (dummy image) passed: both pipelines run without error.")
    else:
        os.makedirs("results/augmentation_preview", exist_ok=True)
        aug_with = build_train_augmentation(elastic=True)
        aug_without = build_train_augmentation(elastic=False)
        for i, path in enumerate(candidates[:5]):
            img = Image.open(path)
            augment_pil(img, aug_with, seed=i).save(f"results/augmentation_preview/{i}_elastic.png")
            augment_pil(img, aug_without, seed=i).save(f"results/augmentation_preview/{i}_noelastic.png")
            img.save(f"results/augmentation_preview/{i}_original.png")
        print(f"Self-test passed: {len(candidates[:5])} anh thuc te da augment, "
              f"xem results/augmentation_preview/")
