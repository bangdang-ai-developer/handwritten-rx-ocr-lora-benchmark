# Phase 5 — Real results (Qwen2.5-VL-3B-Instruct)

Run on Kaggle (kernel v13, GPU T4 x2, 16/09/2026, ~7 minutes — the fastest of all VLMs tested so far). Same 780 Kaggle-Rx images + 400 IAM images.

## Results

| Dataset | CER (mean) | CER (median) | WER | Exact-match | Top-1 acc (78 classes) |
|---|---|---|---|---|---|
| kaggle_rx | **0.434 (BEST)** | – | 0.912 | **28.2% (BEST, 2x the runner-up)** | **81.5% (BEST)** |
| iam | 1.100 (WORST) | **0.0 (the majority fully correct!)** | 0.613 | 59.3% (2nd, after TrOCR) | – |

## Phase 5's most important finding: mean CER can be misleading

On IAM, **median CER = 0.0** (more than half of the images are read with 100% accuracy) yet **mean CER = 1.10** (the worst of all 6 models tested so far) — a seemingly contradictory result. The cause: in a small number of cases (~3%), the model **does not follow the "output text only" instruction** and instead responds with a full chatbot-style sentence, for example:
- Reference `"."` → the model responds *"The text is not visible in the image provided. Please upload an image..."* (CER = 109!)
- Reference `"a"` → the model responds *"The text in the image is: \"1234567890\"..."* (CER = 78)

Since many words/characters in IAM are very short (punctuation, 1-2 letter connecting words), a long "refusal/explanation" response produces a huge edit distance relative to the short reference, driving the mean CER up sharply even though most predictions are perfect.

**Implication for the paper:** this is concrete evidence for the argument that "mean CER can be misleading when evaluating general-purpose VLMs — median CER, or a separate 'instruction-following failure' rate, should also be reported, not just the mean." An additional median-CER metric will be applied in Phase 9 (aggregate analysis) for all models, not just Qwen.

## Another key finding

**On prescriptions (kaggle_rx) — the paper's target domain — Qwen2.5-VL-3B (a general-purpose VLM with only 3B parameters) outperforms EVERY specialized OCR model tested so far (GOT-OCR2.0 580M, TrOCR-large 558M, EasyOCR, Tesseract)** on CER, exact-match, and top-1 accuracy alike. This is a surprising and highly notable result: a small general-purpose VLM, using a simple English prompt and no domain-specific training, reads handwritten prescriptions better than dedicated OCR engines — suggesting that contextual/world-knowledge reasoning (knowing that "Aceta" is a plausible drug name) helps Qwen "guess correctly" more often, despite not being trained specifically for OCR.

## Data file
`results_master_phase5_qwen.csv`, merged into `results_master_combined.csv` (8,260 rows, 6 models × 2 datasets, complete and ready for Phase 9).
