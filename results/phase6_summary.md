# Phase 6 — Fine-tuning preparation (investigation results + frozen test set)

## 1. Investigating "780 vs. 936 images" — NOT a path-resolution bug

The Q2 plan (`docs/05-q2-research-plan.md`) previously speculated that the actual number of test images (780) being lower than the theoretical value (936 = 20%×4680) was due to a path-matching bug in `build_manifest_kaggle_rx()`. **The dataset was downloaded locally (`data/kaggle_rx/`) and counted directly to verify this — that speculation was WRONG.**

| Split | CSV row count | Image file count | Match? |
|---|---|---|---|
| Training | 3,120 | 3,120 | ✅ |
| Validation | 780 | 780 | ✅ |
| Testing | 780 | 780 | ✅ |
| **Total** | **4,680** | **4,680** | ✅ matches the dataset total exactly |

**Conclusion:** the code correctly resolved **100%** of the images in the CSV from the start (780/780, no images lost). The only issue is that the "60% training, 20% validation, 20% testing" description on the Kaggle page **does not exactly match** the true split ratio (actually ≈ 66.7% / 16.65% / 16.65%). There is nothing to fix in the code — only the planning documentation needs to be **corrected throughout**: the real training set is **3,120 images**, not 2,808 as previously written. A warning note has been added at the top of `docs/05-q2-research-plan.md`.

## 2. Freezing the test manifest (frozen test set)

Verified: all 7 models benchmarked in Phases 1-5 (Tesseract, EasyOCR, TrOCR, Donut×2, GOT-OCR2.0, Qwen2.5-VL) used **exactly the same set of 780 kaggle_rx images and 400 IAM images** (checked via set-equality over the full image filenames — 100% match, confirming seed=42 worked as designed). These have been extracted and saved as 2 immutable files so that every future fine-tuning evaluation (Phase 8) is required to reuse exactly this set (a necessary condition for the paired Wilcoxon signed-rank test):

- `results/frozen_test_manifest_kaggle_rx.csv` (780 rows: `image_filename`, `label`)
- `results/frozen_test_manifest_iam.csv` (400 rows: `image_filename`, `label`)

## 3. Augmentation pipeline

Wrote `src/augmentation.py` (Albumentations): Affine (rotation ±5°, shear ±8°, scale 0.95-1.05), ElasticTransform (optional, used for the A/B ablation), GaussNoise, GaussianBlur, RandomBrightnessContrast, Morphological erosion/dilation (simulating pen stroke thickness) — matching the list proposed in the Q2 plan, Section 3.2. Self-test run on real training images, preview output saved at `results/augmentation_preview/`.

## Remaining work before Phase 7 (LoRA fine-tuning)

- Synchronize `src/augmentation.py` into `notebooks/kaggle_benchmark.py` (the Kaggle-running version) — the same way `src/metrics.py` was previously synchronized.
- Write the fine-tuning script (`peft` + `LoraConfig`), using the correct 3,120 Training images (not 2,808).
