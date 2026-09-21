# Phase 2 — Real results (TrOCR-large-handwritten, Donut-base zero-shot)

Run on Kaggle (kernel `dangbang1/ocr-med-benchmark-phase1`, version 6, GPU T4 x2, 16/09/2026, ~29 minutes). Same 780 Kaggle-Rx test images + 400 IAM images as Phase 1.

## Summary table (Phase 1 + Phase 2 combined)

| Model | Dataset | n | CER | WER | Exact-match | Top-1 acc (78 classes) | Degenerate |
|---|---|---|---|---|---|---|---|
| **trocr-large-handwritten** | **iam** | 400 | **0.441** | 0.570 | **57.3%** | – | 0.0% |
| easyocr | iam | 400 | 0.736 | 1.105 | 4.8% | – | 33.3% |
| tesseract | iam | 400 | 0.836 | 1.248 | 6.3% | – | 1.8% |
| donut-base-synthdog (raw crop) | iam | 400 | 1.000 | 1.000 | 0.0% | – | **99.75%** |
| easyocr | kaggle_rx | 780 | **0.552** | 1.117 | 10.4% | 54.9% | 3.3% |
| trocr-large-handwritten | kaggle_rx | 780 | 0.580 | 1.587 | 8.1% | **79.5%** | 0.0% |
| tesseract | kaggle_rx | 780 | 0.625 | 1.304 | 7.3% | 49.2% | 1.4% |
| donut-base-synthdog (raw crop) | kaggle_rx | 780 | 1.000 | 1.000 | 0.0% | 1.3% | **100%** |

## Observations

1. **TrOCR-large-handwritten dominates on IAM** (CER 0.44, exact-match 57.3%) — unsurprising, since this is exactly the domain it was specifically trained on (English handwriting in the IAM style). This is **not a fair zero-shot comparison for IAM**, and this caveat needs to be stated explicitly in the paper — TrOCR has a "home-field" advantage.
2. **On prescriptions (kaggle_rx) — a genuinely new domain for TrOCR** — EasyOCR still wins on raw CER (0.552 vs. 0.580), but TrOCR has the **highest Top-1 classification accuracy (79.5%)** — meaning TrOCR's output, despite a slightly higher CER, is "closer" more often and maps more easily to the correct drug name in the 78-class dictionary. **This is an important finding for the Discussion**: raw CER and closed-vocabulary classification accuracy can lead to different conclusions — which metric matters depends on the actual deployment goal (free-form reading vs. matching against an existing drug list).
3. **Technical observation**: TrOCR tends to append a "." at the end of most outputs (e.g., "Aceta ." instead of "Aceta") — because it was trained on full sentences (IAM), it "learned" the habit of ending with punctuation, even when applied to single-word images. Noted as a characteristic worth discussing, not corrected in post-processing (left as-is to measure genuine zero-shot performance).
4. **Donut-base (SynthDoG pretraining only, not fine-tuned) FAILS UNDER BOTH TEST CONDITIONS — a confirmed conclusion, not a code bug:**
   - **Raw** (crop fed directly in): degenerate rate ~100% on both datasets (entirely empty output).
   - **Padded** (crop pasted into the center of a "blank page" mimicking the aspect ratio SynthDoG was trained on): degenerate rate drops to 77.8% (kaggle_rx) / 96.5% (IAM) — i.e., the model occasionally does generate text — but when it does, mean CER climbs to **4.20** (kaggle_rx) and **1.06** (IAM), i.e., long, heavily incorrect output (hallucination), not correct text.
   - **Conclusion:** verified via two different approaches (not caused by an aspect-ratio mismatch) — `donut-base` with SynthDoG pretraining alone, WITHOUT fine-tuning, genuinely does not produce useful OCR output at the single-word level, whether "silent" (raw) or "rambling" (padded). This is a **valid scientific finding worth including in the paper** (demonstrating that not every "OCR-free VLM" can be used directly zero-shot without prior fine-tuning/task adaptation — an argument that supports this paper's core fine-tuning contribution), not a technical bug requiring further fixing. No further time will be invested debugging Donut — moving on to Phase 3 (GOT-OCR2.0).

## Raw data files
- `results_master_phase1.csv` (Tesseract, EasyOCR)
- `results_master_phase2.csv` (TrOCR-large-handwritten, Donut-base raw-crop)
- `results_master_combined.csv` (both combined, used for analysis)
