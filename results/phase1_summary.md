# Phase 1 — Real results (classical OCR: Tesseract, EasyOCR)

Run on Kaggle (kernel `dangbang1/ocr-med-benchmark-phase1`, version 5, GPU T4 x2, 15/09/2026 → 16/09/2026), total execution time ~220s. Data: **780 images** from the test split of the Kaggle "Doctor's Handwritten Prescription BD dataset" (handwritten prescriptions, 78 drug classes), **400 images** randomly sampled (seed=42) from the IAM Handwriting Word Database (domain-shift control).

## Summary table

| Model | Dataset | n | CER (mean) | 95% CI (bootstrap) | WER (mean) | Exact-match | Top-1 classification acc (78 classes) | "Degenerate" output rate |
|---|---|---|---|---|---|---|---|---|
| Tesseract | kaggle_rx | 780 | 0.625 | [0.601, 0.648] | 1.304 | 7.3% | 49.2% | 1.4% |
| Tesseract | iam | 400 | 0.836 | [0.787, 0.886] | 1.248 | 6.3% | – | 1.8% |
| EasyOCR | kaggle_rx | 780 | 0.552 | [0.529, 0.574] | 1.117 | 10.4% | 54.9% | 3.3% |
| EasyOCR | iam | 400 | 0.736 | [0.703, 0.766] | 1.105 | 4.8% | – | **33.3%** |

*(Top-1 classification acc is computed only for kaggle_rx, since IAM is open-vocabulary and has no fixed set of 78 labels. CI = 95% confidence interval via 1,000-resample bootstrap, seed=0.)*

## Initial observations (not final conclusions — more models still needed)

1. **Both classical engines struggle substantially with handwriting** — CER 0.55–0.84 (i.e., on average, more than half the characters are wrong) — consistent with the findings already summarized earlier (Section 4.2 of `docs/04-public-medical-datasets.md`).
2. **EasyOCR clearly outperforms Tesseract** on both datasets (CER lower by ~0.07–0.10, CIs non-overlapping) — as expected (deep-learning CRNN vs. classical engine).
3. **Surprising finding:** both engines achieve LOWER CER on kaggle_rx (prescriptions) than on IAM (general handwriting) — contrary to the initial hypothesis that "the medical domain is harder." This warrants further investigation in Phase 5 (qualitative error analysis) — possibly because the kaggle_rx images have already been cropped/normalized (128×128 or similar), whereas IAM contains much greater handwriting variation (657 different writers).
4. **Top-1 classification accuracy** (mapping to the nearest of the 78 drug labels) is much higher than exact-match (54.9% vs. 10.4% for EasyOCR) — indicating that many OCR errors are "close" and could be corrected with a closed-vocabulary post-processing step.
5. **EasyOCR produces 33.3% "degenerate" output on IAM** (empty or repeated-token output) — notable, possibly because EasyOCR's decoder gets confused when handwriting is too illegible or falls outside the training distribution; Tesseract rarely "gives up" entirely (low degenerate rate) but still produces incorrect output (high CER) — two different failure modes, worth including in the Discussion.

## Raw data file

`results_master_phase1.csv` (2360 rows: model × dataset × per-image reference/hypothesis/metrics) — used for the qualitative error analysis in Phase 5 (Day 11 per the plan).
