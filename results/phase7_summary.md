# Phase 7-8 — LoRA Fine-tuning of TrOCR-large-handwritten + Before/After Evaluation (OFFICIAL RESULT: r=32)

> **⚠️ Update (17/09/2026):** After running the full ablation (Phase 7b — see
> `results/phase7b_ablation_summary.md`), **r=32 was selected as the official configuration**, replacing the
> initial r=16, because it outperforms r=16 on BOTH criteria (lower in-domain CER AND better-preserved
> generalization, with no trade-off). This document has been rewritten entirely around the r=32 results.
> The original r=16 results (formerly the "main" configuration) are retained as one data point in the
> ablation table — see `phase7b_ablation_summary.md`.

## 1. Training configuration used (official)

- Base model: `microsoft/trocr-large-handwritten` (558M parameters), patched via `peft.LoraConfig`:
  **`r=32, lora_alpha=64`**, `lora_dropout=0.1, target_modules=["query","value","q_proj","v_proj"], bias="none"`.
- Training data: **3,120 images** (`Training` split, the correct real count — see `phase6_summary.md`), with
  Albumentations augmentation (Affine, ElasticTransform, GaussNoise, GaussianBlur, RandomBrightnessContrast,
  Morphological erosion/dilation) — confirmed via the ablation to be **not harmful** to in-domain performance
  and showing a trend (not reaching significance at 0.05) toward slightly reducing catastrophic forgetting,
  see Section 2 of `phase7b_ablation_summary.md`.
- Monitored during training on a subsample of 300/780 `Validation` images.
- `Seq2SeqTrainingArguments`: batch 8 × grad-accum 2 (effective 16), `fp16=True` (required on T4/Turing,
  bf16 not used), `lr=2e-4` cosine, `num_train_epochs=15`, `early_stopping_patience=5` on `eval_cer`.
- Kaggle T4×2 kernel (`dangbang1/ocr-med-benchmark-phase1`), **kernel v22** (`ABLATION_CONFIG="r32"`) — ran
  to completion successfully, using the same script/pipeline already stabilized since kernel v20 (all 4
  consecutive technical bugs in kernels v15-v19 had already been fixed: `decoder_start_token_id`, a
  `vocab_size` property monkeypatch for `peft`, the `generate()` keyword argument, and an `OverflowError`
  when decoding `-100` — see the git commit history for details of each bug).

## 2. Training history (kernel v22, r=32)

Early-stopped at **epoch 7** (patience of 5 with no further improvement after epoch 2):

| Epoch | eval_cer (300 validation images) |
|---|---|
| 1 | 0.1732 |
| **2 (best)** | **0.1022** |
| 3 | 0.1228 |
| 4 | 0.1628 |
| 5 | 0.1812 |
| 6 | 0.2095 |
| 7 | 0.1907 |

The final adapter saved is the **best checkpoint (epoch 2)**, thanks to `load_best_model_at_end=True`.

## 3. Results on the frozen test set (MAIN RESULT — used in the paper)

**Paired** comparison (exactly per image, n matched 100%) between TrOCR-large-handwritten **zero-shot**
(Phase 2) and **fine-tuned** (LoRA r=32), 95% bootstrap CI (1,000 resamples, seed=0), Wilcoxon signed-rank test.

### 3.1. Kaggle-Rx Testing (780 images, in-domain — prescriptions)

| Metric | Zero-shot | Fine-tuned (r=32) | Change |
|---|---|---|---|
| CER (mean, 95% CI) | 0.580 [0.555–0.606] | **0.114 [0.100–0.130]** | **−80.3%** |
| WER (mean) | 1.587 | 0.319 | −79.9% |
| Exact-match | 8.1% | **68.1%** | +60.0 percentage points |
| Top-1 accuracy (78 drug classes) | — | **92.4%** | (not measured at zero-shot) |

**Wilcoxon signed-rank (CER, paired, n=780): p = 6.97×10⁻¹¹¹** — extremely statistically significant.

### 3.2. IAM (400 images, out-of-domain — testing catastrophic forgetting)

| Metric | Zero-shot | Fine-tuned (r=32) | Change |
|---|---|---|---|
| CER (mean, 95% CI) | 0.441 [0.364–0.515] | **0.501 [0.453–0.552]** | **+13.7% (WORSE)** |
| WER (mean) | 0.570 | 0.733 | +28.6% (worse) |
| Exact-match | 57.3% | 26.8% | −30.5 percentage points |

**Wilcoxon signed-rank (CER, paired, n=400): p = 3.18×10⁻⁴** — still statistically significant, although the
degree of worsening is **smaller** than for r=16 (r=16: +18.8%, p=4.3×10⁻⁵ — see the history in
`phase7b_ablation_summary.md`).

## 4. Interpretation — two findings, both must be reported honestly

**(a) A very large in-domain improvement, even stronger than r=16.** LoRA r=32 (updating only ~1-2% of the
parameters) reduces CER by **80%**, raises exact-match from 8% to 68%, and reaches 92.4% top-1 accuracy when
mapped to the 78 real drug names.

**(b) Catastrophic forgetting remains statistically significant on IAM, but is LIGHTER than for r=16.** The
r=32 model still shows reduced ability to read general handwriting (p<0.001), but the degree of worsening
(+13.7%) is smaller than for r=16 (+18.8%) — **r=32 is not only better in-domain, it also preserves
generalization better than r=16, with no clear trade-off**. Median CER on IAM still rises markedly (0.000 at
zero-shot → 0.400 after fine-tuning) — confirming that this degradation is **systematic**, not driven by a
handful of outliers (see Section 4 of `phase9_summary.md`).

**Note on the reliability of the rank comparison**: only 1 seed was run per rank (8/16/32) — part of the
difference between ranks COULD partly stem from run-to-run training variance (not from rank alone). See
Section 1 of `phase7b_ablation_summary.md` for this limitation in full.

## 5. Remaining work

- The A/B augmentation test has been run (Phase 7b) — elastic distortion does not change in-domain
  performance, and shows a trend (not statistically significant) toward slightly reducing forgetting →
  keeping elastic=True is reasonable, confirmed with real data.
- The rank ablation has been run in full (not just a quick comparison as originally planned) — r=32 is
  confirmed as the best choice among the 3 ranks tested (8/16/32); r>32 has not been tested (outside the
  scope of the original ablation, a possible extension if desired, but not required for the current
  conclusions).
- Phase 9 (aggregate quantitative analysis across all models) and Phase 10 (qualitative error analysis) have
  been fully re-run with r=32 as the main model — see the corresponding files.
