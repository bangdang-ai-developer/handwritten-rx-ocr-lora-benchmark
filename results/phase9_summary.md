# Phase 9 — Aggregate quantitative analysis (all models, zero-shot + fine-tuned)

> **⚠️ Update (17/09/2026):** The entirety of Phase 9 has been rerun after the ablation (Phase 7b) selected
> **r=32** as the official fine-tuning configuration (replacing the initial r=16) — every
> "trocr-lora-finetuned" figure below is **r=32**. No GPU is needed to redo this (only renaming the model in
> `results_master_combined.csv` and rerunning `src/analyze_aggregate.py`) — the overall conclusions are
> unchanged, only the specific numbers are stronger.

Script for reproduction: `src/analyze_aggregate.py`. 95% bootstrap CI (1,000 resamples, seed=0), paired
Wilcoxon signed-rank test on exactly matched images (every model was run on the same manifest, seed=42).

## 1. Full summary table — 8 models × 2 datasets

### Kaggle-Rx Testing (780 images, in-domain — prescriptions), sorted by ascending CER

| Model | CER (95% CI) | WER | Exact-match | Degenerate | Top-1 acc (78 classes) |
|---|---|---|---|---|---|
| **trocr-lora-finetuned (r=32)** | **0.114 [0.100–0.130]** | 0.319 | **68.1%** | 0.0% | **92.4%** |
| qwen2.5-vl-3b | 0.434 [0.372–0.501] | 0.912 | 28.2% | 0.0% | 81.5% |
| got-ocr2.0 | 0.479 [0.446–0.513] | 1.238 | 15.8% | 0.0% | 67.2% |
| easyocr | 0.552 [0.529–0.574] | 1.117 | 10.4% | 3.3% | 54.9% |
| trocr-large-handwritten (zero-shot) | 0.580 [0.555–0.606] | 1.587 | 8.1% | 0.0% | 79.5% |
| tesseract | 0.625 [0.601–0.648] | 1.304 | 7.3% | 1.4% | 49.2% |
| donut-base-synthdog | 1.000 [1.000–1.000] | 1.000 | 0.0% | 100.0% | 1.3% |
| donut-base-synthdog-padded | 4.201 [3.310–5.171] | 1.580 | 1.3% | 77.8% | 3.6% |

### IAM (400 images, out-of-domain — general handwriting), sorted by ascending CER

| Model | CER (95% CI) | WER | Exact-match | Degenerate |
|---|---|---|---|---|
| got-ocr2.0 | 0.386 [0.275–0.581] | 0.518 | 53.8% | 0.3% |
| trocr-large-handwritten (zero-shot) | 0.441 [0.364–0.515] | 0.570 | 57.3% | 0.0% |
| **trocr-lora-finetuned (r=32)** | **0.501 [0.453–0.552]** | 0.733 | 26.8% | 0.0% |
| easyocr | 0.736 [0.703–0.766] | 1.105 | 4.8% | 33.3% |
| tesseract | 0.836 [0.787–0.886] | 1.248 | 6.3% | 1.8% |
| donut-base-synthdog | 1.000 [1.000–1.000] | 1.000 | 0.0% | 99.8% |
| donut-base-synthdog-padded | 1.058 [0.993–1.184] | 0.998 | 0.3% | 96.5% |
| qwen2.5-vl-3b | 1.100 [0.471–1.960] | 0.613 | 59.3% | 3.5% |

*(The 3 ablation configurations — r=8, r=16, r=16-without-elastic — are not included in this main table to
keep the "specialized model vs. general-purpose VLM" comparison clean; see the full set in
`results/phase7b_ablation_summary.md`.)*

## 2. First (headline) finding: fine-tuned TrOCR surpasses EVERY zero-shot model on the target domain

In Phase 5, the best zero-shot model on kaggle_rx was **Qwen2.5-VL-3B** (CER 0.434), surpassing even
GOT-OCR2.0. After LoRA fine-tuning (r=32), **TrOCR-large-handwritten (558M) surpasses Qwen2.5-VL-3B (3B,
~5.4x the parameter count) by an even LARGER margin than the previous r=16 result**:

| Comparison (paired, kaggle_rx) | CER model A | CER model B | Wilcoxon p |
|---|---|---|---|
| trocr-lora-finetuned (r=32) vs qwen2.5-vl-3b | 0.114 | 0.434 | **p = 4.43×10⁻⁵⁹** |
| trocr-lora-finetuned (r=32) vs got-ocr2.0 | 0.114 | 0.479 | **p = 8.00×10⁻⁹³** |

Both are extremely statistically significant — the strongest evidence for the paper's central claim: **a
small specialized model, lightly fine-tuned (LoRA r=32, ~1-2% of parameters updated) on free hardware
(Kaggle T4) can surpass a much larger general-purpose VLM on the target domain**.

## 3. Domain-shift gap (CER on IAM minus CER on kaggle_rx) — all models, sorted ascending

| Model | CER kaggle_rx | CER iam | Gap (iam − kaggle_rx) |
|---|---|---|---|
| donut-base-synthdog-padded | 4.201 | 1.058 | −3.143 |
| trocr-large-handwritten (zero-shot) | 0.580 | 0.441 | −0.139 |
| got-ocr2.0 | 0.479 | 0.386 | −0.093 |
| donut-base-synthdog | 1.000 | 1.000 | 0.000 |
| easyocr | 0.552 | 0.736 | +0.184 |
| tesseract | 0.625 | 0.836 | +0.211 |
| **trocr-lora-finetuned (r=32)** | 0.114 | 0.501 | **+0.387** |
| qwen2.5-vl-3b | 0.434 | 1.100 | +0.667 (see the note in Section 4) |

**Interpretation unchanged from r=16**: the zero-shot models specialized for OCR/HTR (zero-shot TrOCR,
GOT-OCR2.0) perform slightly better on IAM than on kaggle_rx (negative gap). After fine-tuning, TrOCR's gap
reverses sharply (from −0.139 to +0.387) — quantitative evidence for specialization traded off against
generalization.

**Note (to avoid confusion with Section 4/`phase7_summary.md`)**: the absolute gap for r=32 (+0.387) is
actually **slightly higher** than the gap for r=16 (+0.376) — because r=32 improves kaggle_rx (CER dropping
from 0.149→0.114) MORE STRONGLY than the improvement on IAM. This does NOT contradict the conclusion that
"r=32 is better than r=16 on both criteria" in `phase7_summary.md` — r=32 has a lower ABSOLUTE CER than
r=16 on BOTH datasets individually (0.114<0.149 and 0.501<0.524); it is only the *difference* between the
two datasets (the gap) that is slightly larger, because the two sides did not improve by the same amount.
The two ways of reading the numbers (absolute CER vs. relative gap) answer two different questions — the
paper should prioritize reporting the absolute CER per dataset (Section 1) as the main conclusion, and use
the gap only to illustrate the "specialization" trend, not to compare between ranks.

## 4. Important methodological note: mean CER can be misleading with heavy-tailed distributions

*(Unchanged from the previous version — independent of the rank choice, unchanged for the non-LoRA models.)*

| Model (IAM) | Mean CER | Median CER |
|---|---|---|
| qwen2.5-vl-3b | 1.100 | **0.000** |
| got-ocr2.0 | 0.386 | **0.000** |
| trocr-large-handwritten (zero-shot) | 0.441 | **0.000** |
| trocr-lora-finetuned (r=32) | 0.501 | **0.400** |

Three zero-shot models have **median = 0** on IAM — the majority of images are recognized perfectly, and
the mean is skewed by a long tail (especially for Qwen2.5-VL). Wilcoxon between Qwen2.5-VL and GOT-OCR2.0
on IAM: **p = 0.330 — NOT significant**, even though the difference in means looks very large (1.10 vs.
0.39) — GOT-OCR2.0 cannot be claimed to be "better" based on the mean alone.

The **fine-tuned (r=32) model has median CER = 0.400** on IAM (not 0, and in fact **lower** than the median
0.429 for r=16) — the loss of generalization remains a **systematic, widespread** effect (not just
outliers), but **slightly milder** than for r=16 — consistent with r=32 having a lower mean CER than r=16
on IAM (0.501 vs. 0.524).

## 5. All Wilcoxon tests run

| model_a | model_b | dataset | n | CER a | CER b | p-value | Significant (α=0.05) |
|---|---|---|---|---|---|---|---|
| trocr-large-handwritten | trocr-lora-finetuned (r=32) | kaggle_rx | 780 | 0.580 | 0.114 | 6.97×10⁻¹¹¹ | Significant |
| trocr-large-handwritten | trocr-lora-finetuned (r=32) | iam | 400 | 0.441 | 0.501 | 3.18×10⁻⁴ | Significant |
| qwen2.5-vl-3b | got-ocr2.0 | kaggle_rx | 780 | 0.434 | 0.479 | 9.57×10⁻²⁵ | Significant |
| qwen2.5-vl-3b | got-ocr2.0 | iam | 400 | 1.100 | 0.386 | 0.330 | **Not** significant |
| trocr-lora-finetuned (r=32) | qwen2.5-vl-3b | kaggle_rx | 780 | 0.114 | 0.434 | 4.43×10⁻⁵⁹ | Significant |
| trocr-lora-finetuned (r=32) | got-ocr2.0 | kaggle_rx | 780 | 0.114 | 0.479 | 8.00×10⁻⁹³ | Significant |
| trocr-lora-finetuned (r=32) | got-ocr2.0 | iam | 400 | 0.501 | 0.386 | 2.34×10⁻¹³ | Significant |

## 6. Summary for the paper's Results section

1. **Main contribution**: LoRA fine-tuning (r=32) of TrOCR-large-handwritten (558M) on 3,120 images allows
   it to surpass every zero-shot model, including larger general-purpose VLMs (Qwen2.5-VL-3B, GOT-OCR2.0),
   on the target domain, with very strong statistical significance (p<10⁻⁵⁸ for every relevant comparison)
   — an 80% CER reduction relative to zero-shot.
2. **Limitation that must be stated honestly**: this improvement comes with a statistically significant
   (p<0.001) loss of generalization on out-of-domain handwriting (IAM), although it is milder than the
   original r=16 configuration.
3. **Methodological note for the Discussion**: some models (especially general-purpose VLMs) have
   heavy-tailed CER distributions — median should be reported alongside mean.
