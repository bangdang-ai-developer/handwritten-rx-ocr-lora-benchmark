# Phase 3 — Real results (GOT-OCR2.0)

Run on Kaggle (kernel `dangbang1/ocr-med-benchmark-phase1`, version 11, GPU T4 x2, 16/09/2026, ~32 minutes). Same 780 Kaggle-Rx test images + 400 IAM images.

## Technical issue resolved (important — affects confidence in the numbers)

`pip install transformers` (without pinning a version) auto-selected version **5.0.0** (a very recently released major version) — which produced completely nonsensical, multilingual-garbled output (CER 55-120!) even though the prompt/input tensors were confirmed to be correctly formatted. Root cause: GOT-OCR2.0 was merged into `transformers` on 31/01/2025 ([PR #34721](https://github.com/huggingface/transformers/pull/34721)), stable on the 4.x releases but apparently not yet compatible with the 5.0.0 major-version jump. **Fixed by pinning `transformers==4.57.0`** — debugging on the first 5 images immediately gave sensible results ("Acek", "Acd", "Ach" for the word "Aceta"), and the full run produced a reasonable CER (~0.38-0.48). Lesson applied to Phases 4-5 (PaddleOCR-VL, Qwen-VL): always pin the `transformers`/related-library version and run a 5-image debug pass with an early-stop mechanism before running the full 1180-image set.

## Full summary table (4 phases, 6 models)

| Model | Dataset | n | CER | WER | Exact-match | Top-1 acc (78 classes) | Degenerate |
|---|---|---|---|---|---|---|---|
| **got-ocr2.0** | **kaggle_rx** | 780 | **0.479** | 1.238 | **15.8%** | 67.2% | 0.0% |
| easyocr | kaggle_rx | 780 | 0.552 | 1.117 | 10.4% | 54.9% | 3.3% |
| trocr-large-handwritten | kaggle_rx | 780 | 0.580 | 1.587 | 8.1% | **79.5%** | 0.0% |
| tesseract | kaggle_rx | 780 | 0.625 | 1.304 | 7.3% | 49.2% | 1.4% |
| donut-base-synthdog (raw/padded) | kaggle_rx | 780 | 1.00 / 4.20 | – | 0.0% | 1.3% / 3.6% | ~89-100% |
| **got-ocr2.0** | **iam** | 400 | **0.386** | 0.518 | 53.8% | – | 0.3% |
| trocr-large-handwritten | iam | 400 | 0.441 | 0.570 | **57.3%** | – | 0.0% |
| easyocr | iam | 400 | 0.736 | 1.105 | 4.8% | – | 33.3% |
| tesseract | iam | 400 | 0.836 | 1.248 | 6.3% | – | 1.8% |
| donut-base-synthdog (raw/padded) | iam | 400 | 1.00 / 1.06 | – | 0.0% | – | ~96-100% |

## Observations

1. **GOT-OCR2.0 is the best model so far as of Phase 3** on BOTH datasets — as expected for a next-generation specialized VLM-OCR model ("OCR-2.0"), surpassing both TrOCR (which had a "home-field" advantage on IAM) and the classical engines.
2. On prescriptions, GOT-OCR2.0 leads on both CER and exact-match, but TrOCR still has the highest Top-1 classification accuracy (79.5%) — reaffirming the Phase 2 observation: different metrics tell different stories, and both need to be reported fully in the paper.
3. Donut-base (both variants) remains a clear failure outlier, already confirmed as a valid finding (Phase 2 summary) — and it stands out even more when contrasted with GOT-OCR2.0 (also an "OCR-free"/VLM architecture, but with appropriate task-tuning) succeeding dramatically. This is a good contrast for the Discussion: not every VLM-OCR architecture is equal — pretraining/task-alignment matters more than architecture alone.

## Data files
- `results_master_phase3_got_ocr2.csv`
- `results_master_combined.csv` (Phase 1+2+2b+3 combined, 7080 rows — used for statistical analysis/writing)
