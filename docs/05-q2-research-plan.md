# DESIGN REPORT: EXPANDING THE MEDICAL OCR RESEARCH PLAN TOWARD A Q2 JOURNAL (3-4 WEEKS)

*Built on: `docs/archive/05-two-week-plan-SUPERSEDED-benchmark-only.md` (the original plan), `docs/04-public-medical-datasets.md`, the real Phase 1 results (`results/phase1_summary.md`, `results/results_master_phase1.csv`), the Phase 2 code currently running (`notebooks/kaggle_benchmark.py`), and the 2 verification research blocks already provided (Scimago + LoRA feasibility).*

> **⚠️ DATA UPDATE (Phase 6, 16/09/2026):** The dataset was downloaded locally and counted directly — the figures "2,808/936/936" (inferred from the "60/20/20" description on the Kaggle page) used throughout the document below **DO NOT MATCH REALITY**. The real image counts are: **Training = 3,120, Validation = 780, Testing = 780** (total 4,680, which checks out). This is **not a path-resolution bug** as initially suspected — the code has resolved 100% correctly (780/780, 3120/3120) from the start; it's simply that the "60/20/20" description on the Kaggle page does not match the real split ratio (~66.7/16.65/16.65). Every occurrence of "2,808 training images"/"936 test images" in the sections below should be read as **3,120 training images / 780 test images** — see `results/phase6_summary.md` for full details.

The repo was re-checked before writing this report: Phase 1 (Tesseract, EasyOCR) has genuinely been completed with **780 images** of kaggle_rx (not 936 as the theoretical calculation 20%×4680 would suggest — the discrepancy is because some images failed to resolve their path in `build_manifest_kaggle_rx()`, see the Risks in Section 6) + 400 images of IAM. Phase 2 (TrOCR-large-handwritten + Donut-base zero-shot) has just been added to the code (`notebooks/kaggle_benchmark.py`, `CURRENT_PHASE = 2`), and is running/about to run on Kaggle. The Kaggle-Rx dataset's real structure is 3 folders, `Training/Testing/Validation` (60/20/20 stratified across 78 classes), each with its own CSV/XLSX label file — corresponding to train ≈ 2,808 images, validation ≈ 936 images, test ≈ 936 images (matching the figure of 2,808 you mentioned).

---

## 1. SELECTING THE TARGET Q2 JOURNAL

Based entirely on the SJR verification table already provided (with no further guessing), **there are 3 confirmed real Q2 journals that fit the topic** (Journal of Imaging, PeerJ Computer Science, Health Information Science and Systems), so **there is no need to fall back to Q3/a conference** — a genuine Q2 venue can be secured.

### Primary journal: **PeerJ Computer Science**
| Criterion | Assessment |
|---|---|
| Quartile | Confirmed Q2 (majority of sources agree) |
| Precedent | **Strongest in the group** — has published "Adapting multilingual vision language transformers for low-resource Urdu OCR" (2024, using TrOCR) — nearly the same type of paper (benchmark + adapt/fine-tune TrOCR) that this plan targets |
| Cost | APC $1,395 — the lowest in the group, important since this is independent research paid for out of pocket |
| Review time | Officially advertised as ~35 days, but an author survey (SciRev) records that the real figure can be ~14 weeks — **be mentally prepared for longer than the advertised number** |
| Why chosen as primary | Best combination of (a) confirmed Q2, (b) a direct precedent of publishing the same type of paper, (c) lowest cost |

### Backup option 1: **Journal of Imaging (MDPI)**
- Confirmed Q2 in *all 4* related categories (CV&PR, Radiology/Imaging, Electrical Eng., Computer Graphics) — **the most certain Q2 confirmation in the entire list**, matching "medical imaging" directly.
- Extremely fast review (~20-22 days median) — use if publication is urgently needed or if PeerJ CS rejects the paper and a fast resubmission round is wanted.
- Trade-off: higher APC (CHF 1,800-2,400 ≈ $2,000-2,700), 100% OA mandatory (no free route).

### Backup option 2: **Health Information Science and Systems (Springer)**
- Confirmed Q2 (best quartile 2024), scope "AI in medicine/medical image processing" matches directly with the "medical OCR" framing.
- **Has a free publication route (subscription, APC not mandatory)** — important if personal budget is a real constraint (this is independent research, with no CCI funding).
- Trade-off: slowest in the group (~8 weeks for the first decision + 3-4 weeks after revision) — use as an option if the deadline isn't tight or to avoid the APC.

**No need to resort to IJDAR/Q1** since there are already 3 genuine Q2 options; IJDAR should only be considered as a parallel submission *later on* if all 3 Q2 options above are rejected (IJDAR is Q1 in the most precise category — CV&PR — but has 2 very similar papers newly published in 2026, a good precedent).

**Recommended course of action:** Submit to arXiv first (unchanged from the original plan, to establish priority), then submit to **PeerJ Computer Science** as the first official submission; if rejected, revise per the reviewers' feedback and submit to **Journal of Imaging**; keep **Health Information Science and Systems** as a fallback if the APC budget becomes a real issue.

---

## 2. FINE-TUNING MODEL AND SPECIFIC TECHNICAL CONFIGURATION

### Model: `microsoft/trocr-large-handwritten` (558M parameters)
Rationale (already covered in the feasibility study): lowest technical risk, already has a public code precedent (`peft` + `VisionEncoderDecoderModel`), and **has already been benchmarked zero-shot in the currently-running Phase 2** → allowing a direct "before/after" comparison on the exact same model, precisely the methodological-contribution evidence that a Q2 journal needs.

### Specific LoRA library + architecture
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                    # rank — see the ablation below (8/16/32)
    lora_alpha=32,           # alpha = 2×r per common convention
    lora_dropout=0.1,
    target_modules=["query", "value", "q_proj", "v_proj"],
    bias="none",
    task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(trocr_model, lora_config)
model.print_trainable_parameters()   # MUST run this line to confirm the real number
```
- `"query"`, `"value"`: match the self-attention module names of the BEiT/ViT encoder (`BeitSelfAttention`/`ViTSelfAttention`) in `transformers`.
- `"q_proj"`, `"v_proj"`: match the BART-style attention module names of the TrOCR decoder (`TrOCRAttention`, used for both self-attention and decoder cross-attention) — so applying LoRA will automatically cover both types of attention.
- **Before finalizing**, run `for n, _ in model.named_modules(): print(n)` on the actual downloaded `microsoft/trocr-large-handwritten` checkpoint to confirm the correct module names — names can differ slightly between `transformers` versions (a small, easily fixed risk that should be handled early on Day 1 of the fine-tuning phase).
- Estimated trainable parameters: ~0.5–1% of the total (≈3–6M/558M) — consistent with the 0.7% figure from DLoRA-TrOCR (arXiv 2404.12734); **the real number must be confirmed via `print_trainable_parameters()`**, and this estimate should not be used in the final paper.
- **Rank ablation**: run 3 configurations, r=8, r=16, r=32 (keeping alpha=2r), on **the same small number of epochs first** to quickly select the best one, then run the full number of epochs with the best rank according to validation CER — there's no need to run full training for all 3 ranks.
- **Optional escalation if time allows**: enable `use_dora=True` in `LoraConfig` (peft ≥0.9) for the encoder only, keeping plain LoRA for the decoder — exactly following the DLoRA-TrOCR formula (achieving CER 4.02% on IAM, outperforming other PEFT methods) — marked as a **supplementary experiment**, not a mandatory requirement for the main results.

### Training time estimate on Kaggle T4
- Real training set: **2,808 images** (the `Training` folder), batch size 8 (grad-accum 2 → effective batch 16), images resized to 384×384 (standard TrOCR processor).
- Estimated time per epoch: **10–20 minutes** on a single T4 (interpolated from 2 reference data points in the feasibility study).
- Recommended number of epochs: **15–30**, with early stopping based on val CER (patience of 5 epochs) → a total of **~3–6 hours per full training run**.
- Realistic budget for the entire fine-tuning phase (including rank ablation, A/B augmentation, debugging): **~10–15 GPU hours** — within the free Kaggle limits (30h/week, 9h/session, 2×T4 in parallel).

---

## 3. FINE-TUNING EXPERIMENTAL DESIGN

### 3.1. Data split — use the original 3 splits as-is, do NOT re-split from the training set
To answer your question directly: **use the existing `Validation` split as-is** (≈936 images, already stratified across 78 classes by the original authors of the IEEE iCACCESS 2024 paper), without carving out additional validation data from the 2,808 training images. Reasons:
- Keep the maximum of 2,808 training images intact for fine-tuning (the data is already small, and carving out more validation data from it would shrink the training set and further increase overfitting risk).
- The original validation split is already correctly stratified across 78 classes — re-splitting it risks class imbalance if not done carefully.
- Preserves the ability to compare directly with the original IEEE paper (using their exact 3 splits).

**The test set must be FROZEN as an immutable anchor point**: this is the most important methodological point — the model has already run zero-shot in Phase 1/2 on the exact Testing image set (the 780 images that resolved), so once fine-tuning is done, **it must be re-evaluated on exactly those same 780 images** (not the theoretical 936, and not a fresh random sample) so that the before/after comparison is a valid *paired* comparison (the Wilcoxon signed-rank test requires exactly matched image pairs). Concrete action: save the list of `image_path` values used in Phase 1/2 into an immutable JSON/CSV file (e.g. `results/frozen_test_manifest_kaggle_rx.csv`, `results/frozen_test_manifest_iam.csv`) as soon as the fine-tuning phase begins, and reuse this exact file for every subsequent evaluation.

### 3.2. Augmentation (to combat overfitting on 2,808 images)
> **⚠️ CITATION CORRECTION (17/09/2026):** re-verified via a research agent — Ali et al. (arXiv:2412.18199)
> **is NOT a paper about augmentation methods** (it is a paper about extracting medicine names using Mask R-CNN +
> TrOCR-Base on ~1,000 handwritten Pakistani prescriptions; see the full citation in Section 3.2 below/`docs/04-...`) — it
> should only be used as evidence that "fine-tuning TrOCR at a similar data scale still yields good CER," and should NOT be used to
> justify the augmentation choices (elastic/affine/...) specifically. The augmentation techniques below are standard, common
> practice in HTR generally, and are not tied specifically to any one paper. The corresponding docstring in `src/augmentation.py`
> has been fixed accordingly.

Use **Albumentations**, combining standard augmentation techniques used in HTR:
- Elastic distortion (mild-to-moderate) — TrOCR-**large** is more likely to benefit from elastic distortion (unlike the small variant).
- Random rotation ±3–5°, slight shear.
- Gaussian noise/blur.
- Brightness/contrast jitter.
- Erosion/dilation (simulating varying pen-stroke thickness).
- Random crop-padding.

**Mandatory A/B test**: run 2 configurations — with augmentation vs. without — since the literature warns that elastic distortion is not always beneficial for every TrOCR variant; select based on the lower val CER, and report both numbers in the paper (increasing transparency for Q2 reviewers).

### 3.3. Proposed hyperparameters
| Parameter | Value | Notes |
|---|---|---|
| LoRA rank / alpha | 16 / 32 (ablation 8, 32) | see Section 2 |
| Learning rate | 2e-4 (AdamW), 5-10% warmup steps, cosine decay | LoRA typically uses a higher LR than full fine-tuning (5e-5) |
| Batch size | 8 physical × grad-accum 2 = effective 16 | fp16/bf16 |
| Epochs | 15-30, early-stop patience 5 based on val CER | |
| Max target length | 32 tokens (matching `max_new_tokens=32` used in Phase 2 zero-shot) | |
| Seed | 42 (matching the seed used throughout the current code) | |

### 3.4. Before/after comparison metrics — reuse the exact `src/metrics.py` pipeline
- CER, WER, Exact-match, Top-1 classification accuracy (78 classes) — **identical to the Phase 1/2 measurement approach** so that direct comparison requires no formula changes.
- Bootstrap 95% CI (1,000 resamples, seed=0) for each metric, both before and after fine-tuning, on both datasets (Kaggle-Rx test + IAM).
- **Wilcoxon signed-rank test** (paired, on per-image CER) — 2 main comparisons:
  1. TrOCR zero-shot vs. TrOCR fine-tuned, on the Kaggle-Rx test set (measuring in-domain improvement).
  2. TrOCR zero-shot vs. TrOCR fine-tuned, on IAM (measuring catastrophic forgetting/loss of generalization).
- With n≈780 (the real Kaggle-Rx test set, not the theoretical 936 — see Section 6), the standard error of CER is still small enough (~1%, CI95%≈±2%) to detect differences >4-5% — still sufficiently powered for a paired test.

---

## 4. RESTRUCTURING THE PHASES (from Phase 3 onward)

| Phase | Content | Change from the original plan |
|---|---|---|
| **1** (DONE) | Tesseract + EasyOCR, kaggle_rx (780) + IAM (400) | unchanged |
| **2** (RUNNING) | TrOCR-large-handwritten + Donut-base zero-shot | unchanged |
| **3** | GOT-OCR2.0 zero-shot (kaggle_rx + IAM) | **kept as-is** — still needed for the original benchmark; this model is not fine-tuned (ms-swift is a separate tool, outside the scope of the core contribution) |
| **4** | PaddleOCR-VL zero-shot (kaggle_rx + IAM) | kept as-is |
| **5** | Qwen2.5-VL/Qwen3-VL zero-shot (+ optional closed API) | kept as-is, **still marked optional/bonus** — not included in the fine-tuning scope (high overfitting risk for a 7B VLM on 2,808 images; should only be a supplementary experiment if time allows) |
| **6 (NEW)** | Fine-tuning preparation: freeze the test manifest, build + unit-test the augmentation pipeline, fix the path-resolution bug (see Section 6) to confirm the real n for test/validation | new |
| **7 (NEW)** | LoRA fine-tuning of TrOCR-large-handwritten on the training set (2,808 images): rank ablation (8/16/32), A/B augmentation, select the best configuration based on val CER | new — the core methodological contribution |
| **8 (NEW)** | Re-evaluate the fine-tuned model on **BOTH** datasets (frozen Kaggle-Rx test + frozen IAM) — measuring in-domain improvement AND catastrophic forgetting | new — the most important analysis for Q2 reviewers |
| **9** | Combined quantitative analysis: all zero-shot models × before/after fine-tuning, domain-shift gap, bootstrap CI, Wilcoxon | expanded from the original Day 10 |
| **10** | Qualitative error analysis (including errors before/after fine-tuning — whether fine-tuning fixes any error types: confusing similar-looking names, out-of-vocabulary hallucination, degenerate output) | expanded from the original Day 11 |
| **11 (optional)** | RxHandBD supplementary check (keeping the same supplementary role as the original plan) | unchanged |
| **12 (NEW)** | Write the paper following a JOURNAL structure (longer than a workshop paper): Introduction, expanded Related Work (adding PEFT/LoRA-for-OCR: DLoRA-TrOCR, devanagari-trocr-lora, Ali et al., Aradillas et al., Ancient Greek Qwen3-VL), Method (including the full fine-tuning design), Results, Discussion, **a thorough Threats to Validity/Limitations section** | replaces the original Day 12-14 |
| **13 (NEW)** | Internal review, formatting to the journal template, preparing code+data+adapter weights for public release, submitting to arXiv then to the journal | replaces the original Day 14 |

---

## 5. DETAILED DAY-BY-DAY SCHEDULE (starting 15/09/2026, ~4 weeks total → 12/10/2026; can be shortened to 3 weeks by cutting the items marked *[can be cut]*)

### Week 1 (remaining) — completing the original zero-shot benchmark
| Day | Task |
|---|---|
| 1-2 (15-16/09) | **DONE** — Phase 1 (Tesseract, EasyOCR) |
| 3 (17/09, Thu) | Finish the currently-running Phase 2 (TrOCR-large-handwritten + Donut-base); cross-check TrOCR's CER on IAM against the published figure (2.89%, Li et al. 2023) as a sanity check |
| 4 (18/09, Fri) | Phase 3: set up the environment + run GOT-OCR2.0 zero-shot (kaggle_rx + IAM) |
| 5-6 (19-20/09, weekend, buffer) | Phase 4: set up + run PaddleOCR-VL zero-shot |
| 7 (21/09, Mon) | Phase 5: Qwen2.5-VL/Qwen3-VL zero-shot *[the closed-API bonus part can be cut if time needs saving]*; merge everything into `results_master.csv` — **milestone: completion of the original 2-week benchmark scope** |

### Week 2 — preparing + running fine-tuning
| Day | Task |
|---|---|
| 8 (22/09, Tue) | Investigate + fix the path-resolution bug (only 780/936 test images resolve — see Section 6); freeze `frozen_test_manifest_kaggle_rx.csv` and `frozen_test_manifest_iam.csv`; build + unit-test the Albumentations pipeline |
| 9 (23/09, Wed) | Write the fine-tuning script (`peft` + `LoraConfig`), print `print_trainable_parameters()` to confirm the real number; smoke-test on 20 images/1 epoch to make sure the loop runs correctly and the loss decreases |
| 10 (24/09, Thu) | Run 1: rank 16, no augmentation, 15 epochs, monitor val CER |
| 11 (25/09, Fri) | Run 2: rank 16 + augmentation (A/B); *[if GPU budget allows]* a quick rank 8/32 ablation over a few epochs |
| 12-13 (26-27/09, weekend, buffer) | Select the best configuration based on val CER, train to full convergence (early-stop); save the adapter weights (push to HuggingFace Hub or Kaggle Dataset output so they aren't lost when the session expires) |

### Week 3 — re-evaluation + statistical analysis
| Day | Task |
|---|---|
| 14 (28/09, Mon) | Re-evaluate the fine-tuned model on the **frozen Kaggle-Rx test set** AND the **frozen IAM subsample** (using the same `src/metrics.py` pipeline) |
| 15 (29/09, Tue) | Bootstrap CI (1,000 resamples) + Wilcoxon signed-rank (2 comparisons: in-domain improvement, out-of-domain forgetting) |
| 16 (30/09, Wed) | Interpret the forgetting results — this is the central analysis for Q2; draft it while the numbers are still "fresh" |
| 17 (01/10, Thu) | Compile the combined tables/figures (all models × before/after); qualitative before/after fine-tuning error analysis |
| 18-19 (02-03/10, weekend, buffer) | Clear the remaining technical backlog; start writing Methods + Results (the parts not dependent on final polish) |

### Week 4 — writing + submission
| Day | Task |
|---|---|
| 20 (04/10, Mon) | Introduction + expanded Related Work (adding the PEFT/LoRA-for-OCR branch) |
| 21 (05/10, Tue) | Discussion + a thorough **Limitations/Threats to Validity** section (see the list in Section 6) — *[decision point: if 3 weeks have already passed and an early stop is needed, this is a reasonable cutoff to shorten the paper]* |
| 22 (06/10, Wed) | Abstract + Conclusion; format to the PeerJ CS template; self-review consistency between tables and text |
| 23 (07/10, Thu) | Polish figures/tables; prepare the public code+data+adapter weights repo (README, HuggingFace model card) |
| 24 (08/10, Fri) | Submit to arXiv (cs.CV) first; prepare the journal submission package (cover letter, data availability statement) |
| 25-26 (09-10/10, weekend, buffer) | Final review, handle submission portal glitches |
| 27 (11/10) | Official submission to **PeerJ Computer Science** |
| 28 (12/10) | Buffer / update `docs/` and the project memory |

---

## 6. RISKS SPECIFIC TO THIS DIRECTION AND FALLBACK PLANS

| # | Risk | Fallback plan |
|---|---|---|
| 1 | **Fine-tuning does not improve results meaningfully** (or improves them without statistical significance) | This is still an honest scientific result that can be reported — reframe the paper from "fine-tuning improves X%" to "lightweight LoRA fine-tuning is/is not sufficient to close the domain-shift gap in an extremely small-data regime (2,808 images)" — still a valid contribution for Q2, especially when paired with a qualitative error analysis explaining why |
| 2 | **Exceeding the Kaggle GPU quota** (30h/week, 9h/session) | Switch to Colab Pro (~$10/month) for the fine-tuning portion; or reduce the ablation to a single configuration (rank 16, with augmentation) instead of running all 3 ranks + A/B |
| 3 | **Severe catastrophic forgetting** (IAM CER noticeably worsens after fine-tuning) | This could actually be the paper's most interesting finding — turn it into an analysis of "the cost of specialization" (the trade-off between in-domain gain and out-of-domain loss), exactly the kind of analysis Q2 reviewers expect; mitigate with early stopping based on a metric combining both datasets if a further fix is wanted |
| 4 | **The Kaggle-Rx test set only resolves 780/936 images** (confirmed in `phase1_summary.md` — due to a path-matching bug in `build_manifest_kaggle_rx()`, not because the dataset is missing images) | Before entering the fine-tuning phase (Day 8), investigate the specific cause (possibilities: filenames in the CSV don't fully match the real filenames, or the image folder is nested one level deeper) and try to recover all 936 images to increase statistical power; if it can't be fixed in time, accept n=780, report it honestly in Limitations, and make sure to **use exactly those same 780 images** for every before/after comparison (consistency matters more than the absolute count) |
| 5 | **APC cost exceeds the personal budget** (this is independent research with no CCI funding, per the original stance) | Prioritize submitting to PeerJ CS (cheapest, $1,395) first; if it's still an issue, switch entirely to Health Information Science and Systems (free subscription route) |
| 6 | **Being flagged by a reviewer for overclaiming** (78 closed-vocabulary classes; the fine-tuning data is so small that the model could easily overfit to exactly those 78 medicine names instead of learning general "handwriting reading") | Address it up front in Limitations: clearly state the boundary between "improving handwriting recognition in general" and "improving memorization of the specific 78-class closed vocabulary" — this is the main reason the additional IAM evaluation (Sections 3, 4) is mandatory, not optional |
| 7 | **Overall schedule slippage** (technical issues piling up across multiple phases) | Cutting order if the timeline needs to shrink to 3 weeks: (a) drop the Qwen-VL bonus API branch, (b) drop the rank 8/32 ablation (keep only rank 16), (c) drop the RxHandBD supplementary check, (d) shorten Related Work — the post-fine-tuning re-evaluation on IAM **must not be cut** (this is the core new contribution relative to the original plan) |
| 8 | **DLoRA/DoRA escalation doesn't fit in the timeline** | Already marked from the start as a supplementary experiment (Section 2) — dropping it entirely does not affect the main contribution (basic LoRA rank 16 is already enough to answer the research question) |
| 9 **(new, 17/09/2026)** | **Novelty-overlap risk**: "RxScribe Bench" (arXiv:2609.13280, 09/2026) benchmarks general-purpose VLMs on handwritten prescriptions — the same problem framing | Carefully checked (see `docs/06-verified-references.md` Section 4): it uses a **private** (non-public) dataset and has no LoRA fine-tuning/catastrophic-forgetting component — a difference clear enough that novelty is preserved, but it **must be cited and clearly distinguished** in the Related Work of Phase 12, otherwise a reviewer could easily flag "missing the closest related work" |

---

### Final note
This plan retains the previously locked-in "independent research, public dataset, no IRB" framing — the entire extension (LoRA fine-tuning) uses only Kaggle-Rx data (which already has public CSV/Excel ground truth, with no patient identification) and IAM (general handwriting, free registration) — it does not create any need for hospital/CCI data or any new ethics procedures.
---

## Appendix A: Q2 Journal Verification (SCImago, cross-checked across multiple sources)

## QUARTILE VERIFICATION RESULTS (SCIMAGO) FOR THE CANDIDATE JOURNALS

**Methodological note:** scimagojr.com blocks bots (a Cloudflare "security verification" check), so this page could not be fetched directly (this is also a prohibited action if one tried to bypass CAPTCHA/bot detection, so no attempt was made to get around it). The data below was cross-verified across several aggregator sources that pull original data from Scopus/SJR: researcher.life, resurchify.com, editage.com, askbisht.com, scienceaijournal.com, journalsearches.com — cross-checking at least 2 independent sources for each journal. Recommendation: before submitting, verify directly on scimagojr.com yourself (quartiles are updated ~April-May each year and can change).

### VERIFICATION TABLE FOR THE 14 JOURNALS IN THE ORIGINAL LIST

| # | Journal | Quartile in the closest field | Other quartile | SJR / H-index | Q2 conclusion? |
|---|---------|------|------|------|------|
| 1 | **IJDAR** (Springer) | Computer Vision & Pattern Recognition: **Q1** (2024, best quartile) | Computer Science Applications: Q2; Software: Q2 | SJR ≈0.83, H-index 61, CiteScore 6.7 | ❌ No — Q1 in the most precise field |
| 2 | Pattern Recognition Letters (Elsevier) | Computer Vision & Pattern Recognition: **Q1** (2024, best quartile) | — | CiteScore 8.6 | ❌ No |
| 3 | Journal of Imaging Informatics in Medicine (Springer/SIIM) | **Q1** (best quartile) | — | SJR 0.892, H-index 24, IF 3.1 | ❌ No — contrary to the initial assumption |
| 4 | Multimedia Tools and Applications (Springer) | Media Technology: **Q1**; no separate CV&PR category | Computer Networks and Comm.: Q2; Hardware/Architecture: Q2; Software: Q2 | SJR 0.777, H-index 93 | ❌ No — best quartile is Q1 |
| 5 | Neural Computing and Applications (Springer) | Artificial Intelligence: **Q1**; Software: **Q1** | — | SJR 1.102, H-index 146 | ❌ No |
| 6 | **Journal of Imaging (MDPI)** | Computer Vision & Pattern Recognition: **Q2** ✅ | Radiology/Nuclear Medicine/Imaging: Q2 (SJR)/Q1 (CiteScore); Electrical & Electronic Eng.: Q2; Computer Graphics & CAD: Q2 | SJR 0.66–0.73, H-index 34-53 (sources don't fully agree), CiteScore 4.8–7.3 | ✅ **YES** |
| 7 | **SN Computer Science (Springer)** | Artificial Intelligence: **Q2** ✅ (best quartile Q2 2025) | — | SJR 0.565, H-index 25-65 (sources disagree) | ✅ **YES** |
| 8 | Computers in Biology and Medicine (Elsevier) | Computer Science Applications: **Q1**; Health Informatics: **Q1** | — | SJR 1.375, H-index 113 | ❌ No |
| 9 | Journal of Biomedical Informatics (Elsevier) | **Q1** | — | — | ❌ No |
| 10 | Expert Systems with Applications (Elsevier) | **Q1** (best quartile, SJR 1.854-1.939) | (the companion "X" edition was Q2 in 2020; current figures could not be verified) | Very high H-index | ❌ No |
| 11 | Healthcare Analytics (Elsevier) | **Q1** | — | SJR ~1.1-1.6 | ❌ No |
| 12 | Artificial Intelligence in Medicine (Elsevier) | **Q1** | — | H-index 100, CiteScore 10.4 | ❌ No |
| 13 | Image and Vision Computing (Elsevier) | **Q1** | — | SJR 0.791 | ❌ No |
| 14a | **PeerJ Computer Science** | Computer Science (miscellaneous): **Q2** ✅ (majority of sources agree; 1 source — researcher.life — reports Q1 under the aggregated "Computer Science (all)" label, which appears to be an aggregation error rather than Scimago's real category) | — | SJR 0.618, H-index 84 | ✅ **YES** (with 1 minor conflicting note) |
| 14b | **Health Information Science and Systems (Springer)** | Best quartile **Q2** (2024) ✅ | — | SJR 0.86 | ✅ **YES** |
| 14c | Applied Sciences (MDPI) | Computer Science Applications: **Q2** ✅; but Engineering Multidisciplinary (the main/best category): **Q1** | Several other categories Q2-Q3 | SJR 0.555 | ⚠️ Yes, but it's a multidisciplinary mega-journal |
| 14d | IEEE Access | **Q1** | — | H-index 338 | ❌ No |
| 14e | BMC Medical Informatics and Decision Making | Conflicting: 1 source says Q2, another (broken down by category) says Computer Science Applications/Health Informatics/Health Policy are all **Q1** | — | SJR 1.224 | ❌ No (most likely Q1) |

### RANKING OF THE TOP 5 VERIFIED, GENUINELY Q2 JOURNALS — best fit

**#1 — Journal of Imaging (MDPI)** — [scimagojr](https://www.scimagojr.com/journalsearch.php?q=21100900151&tip=sid) | [MDPI](https://www.mdpi.com/journal/jimaging)
- Confirmed Q2 in *all 4* directly related categories (Computer Vision & Pattern Recognition, Radiology/Imaging, Electrical Eng., Computer Graphics) — the closest match for a medical OCR paper.
- Extremely fast review: ~20-22 days to first decision (median over several years).
- APC: CHF 1800-2400 (sources differ slightly, so check mdpi.com/journal/jimaging/apc) — 100% open access, no free option.
- No specific precedent found yet for a prescription-OCR paper within this journal itself, but its scope (Aims & Scope) matches perfectly with "comparing + fine-tuning vision/medical OCR models."

**#2 — PeerJ Computer Science** — [scimagojr](https://www.scimagojr.com/journalsearch.php?q=21100830173&tip=sid) | [peerj.com](https://peerj.com/computer-science/)
- Q2 (majority of sources), lowest APC in the group: $1,395.
- Fast review per the journal's own published figures (~35 days), although SciRev (an author survey) reports a longer real figure (~14 weeks) — so be mentally prepared for it to take longer than the official number.
- **Strongest precedent**: has published "Adapting multilingual vision language transformers for low-resource Urdu OCR" (2024, using TrOCR) — nearly the exact type of paper (benchmark + adapting an OCR model) that this plan targets.

**#3 — Health Information Science and Systems (Springer)** — [scimagojr](https://www.scimagojr.com/journalsearch.php?q=21100853958&tip=sid) | [Springer](https://link.springer.com/journal/13755)
- Confirmed Q2 (best quartile 2024), SJR 0.86.
- Scope explicitly states: "AI in medicine, medical image processing" — matches the handwritten-prescription-OCR direction directly.
- Hybrid: has a free publication option (subscription route, APC not mandatory) — very suitable for an independent researcher paying out of pocket.
- Drawback: slowest review in the group — averaging ~8 weeks to first decision, plus another 3-4 weeks after revision; APC if choosing OA is high (~$4,090).

**#4 — SN Computer Science (Springer)** — [scimagojr](https://www.scimagojr.com/journalsearch.php?q=21101083109&tip=sid) | [Springer](https://link.springer.com/journal/42979)
- Confirmed Q2 in the Artificial Intelligence category.
- Hybrid, with a free APC option (subscription route).
- Drawback: slowest review in the group (median 96 days ~ 3+ months); very broad scope (all fields of CS), so its prestige/selectivity is lower than the 3 journals above.

**#5 — Applied Sciences (MDPI)** — [scimagojr](https://www.scimagojr.com/journalsearch.php?q=21100829268&tip=sid) | [MDPI](https://www.mdpi.com/journal/applsci)
- Confirmed Q2 *specifically in the Computer Science Applications category* (although its main/overall category — Engineering Multidisciplinary — has moved up to Q1).
- Super-fast review (~15 days), but this is a **multidisciplinary mega-journal** (thousands of papers/year), regarded by many researchers as lower in prestige/selectivity even though the Q2 figure is genuine. Ranked last for this reason — use as a safe fallback, not the first choice.

### SPECIAL NOTE: IJDAR (not in the top 5 for quartile reasons, but worth considering)

IJDAR (International Journal on Document Analysis and Recognition) is the best-fitting journal in terms of **topic** — it is precisely a journal specializing in OCR/document recognition — and has just published 2 papers **very similar** to this plan:
- "Benchmarking OCR and vision-language models for Turkish text recognition: a comprehensive evaluation using synthetic data" (2026)
- "Literally reading behind the lines: a benchmark for OCR on cluttered printed documents" (2026)

However, the *exact* quartile in the closest category (Computer Vision and Pattern Recognition) is currently **Q1**, not Q2 (it is only Q2 in Computer Science Applications/Software — secondary, less relevant categories). Since you require Q2 strictly, IJDAR is excluded from the top 5, but if you're willing to target Q1 for a parallel submission (as a fallback if the Q2 options are rejected), this is the option with the strongest publication precedent. Hybrid, with a free option (subscription route) or OA ~$3,290.

### FINAL RECOMMENDATION

Based on the criteria (a) genuinely confirmed Q2, (b) scope matching a medical OCR benchmark+fine-tuning paper, (c) fast review, (d) reasonable cost/a free option:

**Priority for submission: Journal of Imaging (MDPI) or PeerJ Computer Science** — both are the most certainly confirmed Q2, with fast review, and PeerJ CS has the precedent of publishing a very similar paper (Urdu OCR + TrOCR). Health Information Science and Systems is a good choice if you want to emphasize the "medical application/AI-in-medicine" framing and avoid paying the APC (using the subscription route).

Since some figures (H-index, SJR) differ slightly between aggregator sources (not due to guessing, but because mirror sites cache data from different points in time), it is recommended that you check the exact quartile of the 5 chosen journals directly on scimagojr.com yourself before making the final submission decision.
---

## Appendix B: LoRA Fine-Tuning Feasibility Study on Kaggle GPU T4

## RESEARCH FINDINGS: FEASIBILITY OF LORA OCR FINE-TUNING ON KAGGLE GPU T4

### MAIN RECOMMENDATION (clear, one line)

**Fine-tune the `microsoft/trocr-large-handwritten` model (558M parameters) with LoRA/DoRA using HuggingFace's `peft` library**, applying LoRA to the encoder (BEiT/ViT) and decoder (RoBERTa) of the `VisionEncoderDecoderModel` architecture. This is the lowest-risk option, has a public code precedent, matches the model already benchmarked zero-shot in Phase 2 (allowing a direct "before/after fine-tuning" comparison — exactly the methodological-contribution evidence a Q2 journal needs), and runs comfortably on a single 16GB T4 GPU within a few hours.

If a stronger "bonus" experiment is wanted for the paper (comparing specialized HTR vs. a general-purpose VLM after fine-tuning): **Qwen2.5-VL-7B via QLoRA + Unsloth** is also feasible on a T4 (not plain LoRA — see Question 1 for details).

---

### QUESTION 1: Which models can ACTUALLY be fine-tuned with LoRA/PEFT on a single 16GB T4?

| Model | Size | LoRA feasible on a 16GB T4? | Notes / guidance |
|---|---|---|---|
| **TrOCR-large-handwritten** | ~558M | **YES, easily** | `peft.LoraConfig` applies directly to `VisionEncoderDecoderModel` (HF `transformers`) — there is an official GitHub issue discussing LoRA+VRAM for exactly this class (huggingface/peft #1048). Full fine-tuning (without LoRA) already fits comfortably at 558M, so LoRA only saves additional VRAM/time and isn't required just to fit on the hardware — the safest option. |
| **TrOCR-base-handwritten** | ~334M | **YES, very easily** | A fallback option for even faster/safer fine-tuning; shares the same codebase as the large variant. |
| **Donut-base** | ~200M | **YES** | Also a `VisionEncoderDecoderModel` (Swin encoder + BART-like decoder), peft applies similarly; note that Donut is OCR-free and requires redesigning the task prompt for free-text OCR (this caveat is already noted in your original plan). |
| **GOT-OCR2.0** | ~580M | **YES, via a separate tool** | The official repo uses the **ms-swift** framework (ModelScope), which already supports `--sft_type lora`: `swift sft --model_type got-ocr2 --model_id_or_path stepfun-ai/GOT-OCR2_0 --sft_type lora --dataset train.jsonl`. By default it freezes the vision encoder and only LoRA fine-tunes the LLM+projector — lightweight and T4-friendly. Risk: requires installing ms-swift separately, with less community documentation than TrOCR. |
| **PaddleOCR-VL** | ~0.9B | **SHOULD NOT be used right now** | The PaddlePaddle team has publicly confirmed (HF discussions #13, #4): *"A full workflow for fine-tuning with ERNIEKit is coming soon"* — **official fine-tuning code has NOT been released yet**. There is only 1 unofficial community script (`jzhang533/paddleocr-vl-sft`). Too high a technical risk for a 3-4 week project. |
| **Qwen2.5-VL-7B / Qwen3-VL-8B** | 7-8B | **Plain LoRA: NO** (the fp16 model is already ~18GB > 16GB T4). **4-bit QLoRA via Unsloth: YES** | Unsloth loads Qwen2.5-VL in 4-bit at only ~7.1GB VRAM, with a ready-made free Colab T4 notebook (including a handwriting OCR → LaTeX example); one training run measured ~48.7 minutes on a T4. However, this is a general-purpose 7B VLM — higher overfitting risk with only 2,808 images, and it isn't "lightweight LoRA" in the strict sense (requires 4-bit quantization). Suitable as a **supplementary experiment**, not the core contribution. |
| **Florence-2** (a smaller alternative suggestion) | 0.23B–0.77B | **YES, feasible** | Has an official LoRA tutorial (Roboflow, used for detection but applicable to the `<OCR>` task token). Less direct precedent on medical OCR; the empirically tested VRAM figures are mostly from GPUs stronger than a T4 (RTX 3090/4090) — feasible, but with less T4-specific evidence than TrOCR. |
| **CRNN full training from scratch** | small | **NOT recommended** | Technically runs easily on a T4, but empirical evidence (Aradillas et al. 2018, see Question 4) shows that training from scratch on small data overfits heavily (CER ~18-40% on the validation set) compared to transfer learning from a pretrained model (CER 3-9%) — this runs counter to your goal of a "clear improvement." |

**Final ranking:** TrOCR-large-handwritten (primary) > TrOCR-base-handwritten (safe fallback) > GOT-OCR2.0 (extension if time allows) > Qwen2.5-VL-7B QLoRA (bonus comparison) >> PaddleOCR-VL (excluded, not ready).

---

### QUESTION 2: Most popular LoRA libraries in 2025-2026

- **HuggingFace `peft`** — the **standard choice, and the only one with a direct precedent** for classic encoder-decoder architectures (TrOCR, Donut): `peft.LoraConfig` + `get_peft_model()` applied to `VisionEncoderDecoderModel`. There is a published precedent: a public LoRA adapter on HF (`manishw10/devgen-trocr-devanagari-lora`, trained on 95.4k Devanagari script images using `peft` + `transformers`), and the **DLoRA-TrOCR** paper (arXiv 2404.12734) — using DoRA for the encoder + LoRA for the decoder of TrOCR, with **only 0.7% trainable parameters**, achieving CER 4.02% on IAM, outperforming other PEFT methods.
- **ms-swift** (ModelScope) — the library officially used by GOT-OCR2.0 for LoRA fine-tuning (wrapping peft underneath), needed if GOT-OCR2.0 is chosen.
- **Unsloth** — does **NOT** directly support classic encoder-decoder architectures (TrOCR/Donut/GOT-OCR2.0); in 2026, Unsloth focuses its optimization on **large decoder-only VLMs/LLMs**: Qwen3-VL, Qwen3.5, DeepSeek-OCR/DeepSeek-OCR-2 (a new 3B model released 27/01/2026, with a free fine-tuning notebook, 1.4x faster, using 40% less VRAM). Only use Unsloth if you choose the Qwen-VL/DeepSeek-OCR branch as a supplementary experiment.

**Conclusion for Question 2:** `peft` (HuggingFace) for the core contribution (TrOCR); `ms-swift` if extending to GOT-OCR2.0; `Unsloth` only if adding the VLM branch (Qwen2.5-VL/Qwen3-VL) as a supplementary experiment.

---

### QUESTION 3: Data augmentation for a small dataset (2,808 images) to combat overfitting

Specific evidence from 2 sources:

1. **Ali et al. 2024** (exactly the handwritten prescription problem, closest to your domain): used **brightness adjustment, contrast normalization, translation, minor shearing, elastic transformation, Gaussian noise, cropping with padding** to expand 1,000 → 9,920 images (~×10).
2. **Aradillas et al. 2018** (the classic small-data HTR reference, see Question 4): used **affine transform (rotation, shearing, translation, scaling) + morphological distortion (erosion, dilation)**, reducing CER from 8.2%→6.4% (test) and 5.1%→4.4% (valid) on IAM; on the extremely small Washington-150-lines set, CER dropped 9.4%→8.9%.
3. An important nuance (from a survey of TrOCR): **elastic distortion is not always beneficial** — the small TrOCR variant sometimes gets better CER WITHOUT elastic deformation, while the base/large variants benefit from it. Since you're using TrOCR-**large**, elastic transform is more likely to help — but you should still A/B test with/without it during tuning.

**Specific recommendation for the 2,808-image set:** combine elastic distortion (mild-to-moderate) + small random rotation (±3-5°) + slight shear + Gaussian noise/blur + brightness/contrast jitter + erosion/dilation (simulating varying pen-stroke thickness) + random crop-padding. Use the **Albumentations** library (common in OCR pipelines) or `torchvision.transforms` combined with a custom elastic transform function (following the Simard et al. formula that both papers above reference).

---

### QUESTION 4: Evidence — works that HAVE fine-tuned OCR/HTR on similarly small data, and by how much they improved

Here are 4 citation sources to justify "fine-tuning will improve results substantially" in the paper:

1. **Ali et al., "Leveraging Deep Learning with Multi-Head Attention for Accurate Extraction of Medicine from Handwritten Prescriptions"** (arXiv 2412.18199) — **EXACTLY the same problem**: Mask R-CNN + TrOCR-Base-Handwritten fine-tuned on ~1,000 handwritten Pakistani prescriptions (augmented ×10 → 9,920 images). Results: CER 1.4-15.4% depending on the test scenario (valid 1.4%, changed pattern 3.9%, entirely new images 13.5%) — headline "CER 1.4% on the standard benchmark." This is the closest-domain evidence, at a similar data scale, from the same TrOCR model family.

2. **Aradillas, Murillo-Fuentes & Olmos, "Boosting Handwriting Text Recognition in Small Databases with Transfer Learning"** (arXiv 1804.01527) — the most classic evidence that "transfer learning rescues small datasets": with only **350 training lines**, training from scratch gives CER = **18.2%**, while using transfer learning (pretraining on IAM's 13k lines, then fine-tuning) gives CER = **3.3%** — an absolute improvement of ~15 CER points, equivalent to a ~82% reduction. With 150 lines, CER still stays at 5.8-9.4% thanks to transfer learning (versus complete overfitting if trained from scratch).

3. **DLoRA-TrOCR** (Chang & Li, arXiv 2404.12734) — evidence that **LoRA specifically** (not just full fine-tuning) is strong enough: only 0.7% trainable parameters, achieving CER 4.02% on IAM, F1 94.29% on SROIE, outperforming other PEFT methods — demonstrating that LoRA does not trade away much performance compared to full fine-tuning. **⚠️ Citation update (17/09/2026):** this paper has since been peer-reviewed and published at **ICONIP 2024** (Springer LNCS vol. 15294, DOI 10.1007/978-981-96-6599-0_2) under a **new title**: *"Mixed Text Recognition with Efficient Parameter Fine-Tuning and Transformer"* — the peer-reviewed Springer/ICONIP version should be cited in the paper instead of just an "arXiv preprint."

4. **"Structure-Aware Text Recognition for Ancient Greek Critical Editions"** (arXiv 2603.02803) — more recent evidence (2026) with a larger model: Qwen3-VL-8B fine-tuned on narrow, specialized handwriting data, with CER dropping from **5.2% (zero-shot) → 2.1% (fine-tuned on real data) → 1.0% (combined synthetic+real)** — a different domain (Ancient Greek, not medical) but the same argument that "general-purpose VLMs improve substantially after fine-tuning on a narrow script/domain," which can be cited as supplementary evidence if you try the Qwen-VL branch.

**Summary of Question 4:** All 4 sources show very large CER improvements (a 50-80%+ reduction compared to zero-shot/from-scratch) when fine-tuning on datasets of a similar scale (a few hundred to a few thousand images) to yours — strong enough to justify the research hypothesis.

---

### QUESTION 5: Estimating realistic training time on a T4

There is no exact published figure for *LoRA fine-tuning TrOCR-large-handwritten on exactly 2,808 images*, but interpolating from the empirical data points found:

- Reference data point 1: fine-tuning TrOCR (a smaller variant), effective batch size 8 (physical batch 2 + gradient accumulation 4), **10 epochs ≈ 2 hours 40 minutes** on a GPU **weaker than a T4** (RTX 5060 Laptop, 8GB).
- Reference data point 2: another study fine-tuning TrOCR's full encoder+decoder on a **T4** itself, batch size 8, **40 epochs** (total hours not specified, but this is a typical epoch count chosen for stable convergence).
- Your own estimate from the original plan: TrOCR inference (forward pass only) ~0.1-0.3 seconds/image on a mid-range GPU for the 936-image test set (3-8 minutes).

**Extrapolating for your case** (a training set estimated at ~2,246 images after an 80/20 split of the 2,808, TrOCR-**large** being larger than base and therefore ~1.3-1.5x slower, though LoRA reduces optimizer memory load and allows a larger batch size that partially offsets this):

- **Per epoch: estimated ~10-20 minutes** (batch size 8-16, fp16/bf16, images resized to the standard TrOCR 384×384).
- **Recommended epoch count: 15-30 epochs** (LoRA fine-tuning on an already-pretrained model converges much faster than training from scratch — no need for hundreds of epochs like the from-scratch CNN-BLSTM-CTC training in Aradillas et al.).
- **Total time for one full training run: estimated at 3-6 GPU hours**, plus time to try 2-3 hyperparameter configurations (LoRA rank, learning rate, with/without elastic augmentation) → **the realistic total budget to plan for is ~10-15 GPU hours** for the entire fine-tuning phase (including experimentation/debugging), **entirely within the ~30 GPU hours/week and ~9 hours/session** limits of Kaggle's free dual-T4 tier — feasible within the additional 1-2 weeks in your plan.

---

### Main verified reference sources

- [DLoRA-TrOCR (arXiv 2404.12734)](https://arxiv.org/html/2404.12734v1)
- [manishw10/devgen-trocr-devanagari-lora (HuggingFace)](https://huggingface.co/manishw10/devgen-trocr-devanagari-lora)
- [huggingface/peft Issue #1048 — LoRA + VisionEncoderDecoderModel](https://github.com/huggingface/peft/issues/1048)
- [Ali et al., Leveraging Deep Learning with Multi-Head Attention... (arXiv 2412.18199)](https://arxiv.org/abs/2412.18199)
- [Aradillas et al., Boosting HTR in Small Databases with Transfer Learning (arXiv 1804.01527)](https://arxiv.org/pdf/1804.01527)
- [Structure-Aware Text Recognition for Ancient Greek Critical Editions (arXiv 2603.02803)](https://arxiv.org/pdf/2603.02803)
- [stepfun-ai/GOT-OCR2.0 GitHub + ms-swift issue #2122](https://github.com/modelscope/ms-swift/issues/2122)
- [PaddleOCR-VL fine-tuning discussion (HF #13)](https://huggingface.co/PaddlePaddle/PaddleOCR-VL/discussions/13)
- [Unsloth Qwen3-VL / Qwen2.5-VL fine-tuning docs](https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune/qwen3-vl-how-to-run-and-fine-tune)
- [NielsRogge/Transformers-Tutorials — TrOCR fine-tune notebooks](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Fine_tune_TrOCR_on_IAM_Handwriting_Database_using_Seq2SeqTrainer.ipynb)

*(Note: 2 full PDF files that were downloaded and read during the research process are saved in this session's tool-results folder, in case the detailed data tables need to be referenced again.)*
