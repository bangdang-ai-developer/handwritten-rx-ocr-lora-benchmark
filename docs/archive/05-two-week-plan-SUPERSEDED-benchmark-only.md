> **⚠️ SUPERSEDED — 16/09/2026.** After the real Phase 1 results came in (`results/phase1_summary.md`), the user decided to raise the target to a **Q2** journal instead of just arXiv/a workshop — requiring an additional methodological contribution (LoRA fine-tuning) and extending the timeline to ~3-4 weeks. See the current plan at [`05-q2-research-plan.md`](../05-q2-research-plan.md). This file is kept only for reference/comparison — the Phase 1-2 description in it is still the foundation and remains unchanged.

---

# 5. Two-Week Research Plan (Independent Research, Public Dataset)

*Starting 15/09/2026. Goal: complete the experiments + a paper draft (in the form of a short benchmark/technical report) and submit to arXiv within 14 days — NOT including the peer review time (workshop/conference review always takes an additional several weeks to months, outside the control of this plan).*

---

## 5.1. Analysis of options and selection

## ANALYSIS AND PROPOSED PLAN FOR MEDICAL OCR RESEARCH — 2 WEEKS

*(Based on the search results from the 3 sources already provided, especially the "deep verification" section — the most trustworthy source since it accessed the original pages directly)*

---

### QUICK SUMMARY OF THE CONCLUSION

There is **ONE option ready to start today, with no need to wait for credentialing, no IRB required** (matching your context requirements exactly), the lowest technical risk, and novelty clearly confirmed through the literature search: **Option A — Benchmark pretrained OCR/VLM models (2024-2026) on the Kaggle "Doctor's Handwritten Prescription BD dataset," using IAM as a domain-shift control.** None of the 3 options below require PhysioNet, so there is **no situation where a fallback option must be chosen due to a waiting-time constraint** — all 3 use data that can be downloaded immediately.

---

### OPTION A (primary recommendation)

**Dataset:** the Kaggle "Doctor's Handwritten Prescription BD dataset" (mamun1113) — https://www.kaggle.com/datasets/mamun1113/doctors-handwritten-prescription-bd-dataset — using the **IAM Handwriting Database** (https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) as a domain-shift control.

**Reasons for credibility:**
- The Kaggle dataset has a **genuine peer-reviewed paper**: Mia et al., IEEE iCACCESS 2024 (document 10499631).
- Widely used and stable within the community for 2 years: 34,700 views, 7,078 downloads, 57 upvotes, 29 public notebooks, with a "used in a publication" badge.
- 4,680 images, 78 medicine-name classes, complete CSV/Excel ground truth, an "Open Database" license for research.
- IAM is the classic gold-standard HTR dataset (Marti & Bunke, IJDAR 2002, ~569+ citations), free registration, no IRB.

**Research question:** How do the new generation of pretrained OCR-VLM models (GOT-OCR2.0, PaddleOCR-VL, DeepSeek-OCR, Qwen2.5-VL/Qwen3-VL) — already thoroughly benchmarked on general handwriting (IAM) or receipts (SROIE/CORD) but **never tested on medical handwriting** — perform when reading a doctor's handwritten medicine names? How large is the performance degradation from domain shift (general → medical), and does it differ across model families (classic OCR vs. specialized HTR transformers vs. general-purpose VLMs)?

**Why this has novelty:** The "gap" table in the literature search confirms: **NO ONE** has yet tested GOT-OCR2.0, PaddleOCR-VL, DeepSeek-OCR, or Qwen-VL on any public prescription/medical HTR dataset. The closest benchmark (a dev.to blog post, not peer-reviewed) only tested 4 older engines (Tesseract/EasyOCR/PP-OCRv5/GLM-OCR) on RxHandBD — not touching the models above. The two closest competing works (RxScribe Bench 09/2026 — a different Indian dataset, only mentioning "frontier VLMs" without naming specific ones; "From Handwriting to Structured Data" 04/2026 — only testing closed LLMs via API on FORM, not testing open-source OCR-VLMs) **neither** overlaps with this dataset + model list → a genuine gap, with a low risk of being "scooped" in the next 2 weeks.

**List of models/baselines (8):**
1. Tesseract (classic baseline)
2. PaddleOCR or EasyOCR (classic deep-learning baseline)
3. TrOCR-large (a pretrained HTR transformer, with a published IAM CER figure already available for comparison)
4. Donut (an OCR-free VLM, not yet tested on handwriting/medical text)
5. GOT-OCR2.0 (already trained on IAM — this caveat must be noted clearly — but not yet tested on medical data)
6. PaddleOCR-VL (not yet tested on either IAM or medical data — the most important new target)
7. Qwen2.5-VL/Qwen3-VL (a general-purpose VLM, via local or API)
8. (Optional/bonus) 1 closed model via API (GPT-4V/Gemini/Claude) on a small sample to compare open-vs-closed

**Realistic workload within 2 weeks:** Highly feasible. Small data (~20MB for Kaggle, a subset for IAM), no training needed — only zero-shot inference, runnable on a mid-range GPU or a free Colab/Kaggle-notebook GPU. The biggest risk is setting up the environment for the newer VLMs (PaddleOCR-VL, GOT-OCR2.0, DeepSeek-OCR — dependencies/GPU/downloading many GB of weights), so a 2-day buffer is needed. IAM requires registering in advance (a quick procedure, no real barrier) — should be done right away on Day 1.

**Suitable paper format:** A **short benchmark/technical report (4-6 pages)**, posted as an arXiv preprint and/or submitted to a workshop (an ICDAR workshop or a clinical-NLP workshop) — since the contribution is a systematic empirical comparison, not a new architecture; this matches the format of the closest related works (RxScribe Bench, and the dev.to blog upgraded into a proper academic report).

---

### OPTION B

**Dataset:** RxHandBD (Zenodo DOI 10.5281/zenodo.18478741 / Mendeley DOI 10.17632/dsb5r6vskg.3) **+** Kaggle mamun1113 — using both as a cross-generalization task.

**Reasons for credibility:** Kaggle still has the same peer-reviewed paper as in Option A. RxHandBD adds a larger vocabulary (1,559 words vs. 78 classes) — but **RxHandBD has no peer-reviewed paper**, was only published in 02-03/2026, and has a conflicting license between Zenodo (MIT) and Mendeley (CC BY 4.0) — an academic credibility risk if used as the core pillar.

**Research question:** Does a zero-shot pretrained OCR-VLM model on one handwritten prescription dataset generalize to a different prescription dataset (different vocabulary, different resolution, different collection source) within the same narrow domain (Bangladeshi prescriptions)? Is there an effect of vocabulary diversity (78 fixed classes vs. 1,559 open words) on zero-shot accuracy?

**Why this has novelty:** No one has yet done cross-dataset generalization within the "Bangladeshi prescription" domain using the newer OCR-VLMs — a supplementary angle to Option A.

**Model list:** same as Option A.

**Workload:** Higher than A because 2 different data formats need to be handled, and there is a **higher risk** that a reviewer could flag "RxHandBD isn't credible enough" (not peer-reviewed, conflicting license) if it's used as a primary rather than a secondary data source.

**Format:** Technical report/short paper — similar to A but needs a "limitations" paragraph clearly explaining that RxHandBD is only supplementary/verification data, not the core pillar.

---

### OPTION C

**Dataset:** TCGA-Reports — the original images (`imgs_for_aws`) + Textract OCR results (`aws_response`), Mendeley DOI 10.17632/hyg5xkznpx.1, along with a clean text corpus on GitHub at jkefeli/tcga-path-reports.

**Reasons for credibility:** The Kefeli & Tatonetti paper, *Patterns* (Cell Press) 2024, a very permissive CC BY 4.0 license, and data that already has ready-made image+OCR pairs for comparison (no need to generate fake images).

**Research question:** Can the newer OCR/VLMs (2024-2026) outperform AWS Textract (the baseline used to create TCGA-Reports in 2024) when re-processing the original pathology report page images? Where does the improvement concentrate (data tables, low-quality scans)?

**Why this has novelty:** No one has yet re-benchmarked GOT-OCR2.0/Qwen-VL/PaddleOCR-VL/DeepSeek-OCR against Textract on this exact dataset.

**Model list:** same as above, plus the existing Textract output as a baseline (no need to re-run it).

**Workload & risk:** **MEDIUM-HIGH risk.** (1) The file structure on the Mendeley page could not be fully verified (loaded via JS, not fully visible during the check) — this needs to be re-checked right on Day 1, with a risk that the structure is more complex than expected. (2) This is **printed/typed** text, not handwriting — a technically "easier" problem, with a less compelling story than A. (3) If the downstream classification part is also attempted (as in the original paper, AU-ROC 0.992) to measure the OCR→NLP impact, it would exceed 2 weeks — this part should be CUT, keeping only the pure OCR-quality comparison.

**Format:** Technical report, narrower in scope than A/B due to the necessary cuts.

---

### RANKING

| Rank | Option | Rationale |
|---|---|---|
| **1** | **A** | Primary dataset has a peer-reviewed paper + wide community use, downloadable immediately with no barrier, clearest novelty (exactly the medical handwriting domain — a current hot topic per the RxScribe Bench/04-2026 paper), lowest technical risk |
| 2 | B | A good idea (cross-dataset) but depending on the not-yet-peer-reviewed RxHandBD reduces credibility if used as the core pillar — should be folded in as an *extension* of A, not used as a standalone option |
| 3 | C | A less compelling technical story (printed text, not handwriting), has an unresolved gap in verifying the Mendeley data structure, and easily exceeds 2 weeks if a downstream task is ambitiously added |

**Choice: OPTION A** as the core pillar, with RxHandBD (Option B) insertable as a "supplementary/robustness check" sub-section at the final stage if time allows, and TCGA-Reports (Option C) is not needed within the 2-week scope.

**Confirmation of readiness:** None of the 3 options above requires waiting for PhysioNet credentialing (already excluded during verification since there is no suitable dataset there) — Option A is entirely **ready to start today** (Kaggle: a free account downloads it immediately; IAM: a few minutes to register online, no IRB/HREC). So there is no need for a fallback option due to a time-barrier issue.

---

### DETAILED 14-DAY PLAN — OPTION A

**Working title:** *"Benchmarking Pretrained OCR and Vision-Language Models on Handwritten Medical Prescriptions: A Domain-Shift Study Against General Handwriting Recognition"*

**Metric:** CER, WER, Exact-Match accuracy (matching the measurement approach in the dev.to blog post for comparability), plus Top-1 classification accuracy on the 78 fixed classes (matching the measurement approach of the original IEEE paper for direct comparison).

| Day | Main task |
|---|---|
| **1** | Register for IAM (do it now to eliminate the waiting risk), download the Kaggle dataset via the Kaggle API, carefully read the IEEE iCACCESS 2024 paper to get the correct train/test split and original accuracy figures, set up the repo + environment (Python, jiwer/editdistance to compute CER/WER) |
| **2** | Normalize the 2 datasets into a common format (image path, label), write + unit-test the CER/WER/Exact-Match pipeline, run Tesseract + PaddleOCR/EasyOCR (classic baseline) on the Kaggle test split |
| **3** | Run TrOCR-large on the Kaggle test split; run it on an IAM subsample (for a fair comparison using the same code/prompt, while also cross-checking against the published TrOCR CER on IAM as a sanity check); start setting up the Donut environment |
| **4** | Run Donut zero-shot on Kaggle (clearly note the limitation if a non-standard OCR-free prompt is needed); start setting up the GOT-OCR2.0 environment |
| **5** | Run GOT-OCR2.0 on Kaggle + an IAM subsample; buffer for dependency/GPU errors (the biggest risk point in the whole plan) |
| **6** | *(buffer)* Finish anything running behind from days 3-5; start setting up the PaddleOCR-VL environment (the newest, with little documentation — needs extra time); if things are on track, start writing the Methods section early (dataset description, metric definitions) |
| **7** | *(buffer)* Run PaddleOCR-VL on Kaggle + an IAM subsample; decide whether to include DeepSeek-OCR (the lit review shows very poor CER on non-Chinese handwriting, so it can be skipped in favor of Qwen-VL if time is tight) |
| **8** | Set up Qwen2.5-VL/Qwen3-VL (choose the API route if there isn't enough local GPU for a 7B+ model) |
| **9** | Run Qwen-VL on Kaggle + an IAM subsample; (optional) run 1 closed model (GPT-4V/Gemini) on a small sample of ~200 images as a bonus comparison point; merge all results into the master table |
| **10** | Quantitative analysis: CER/WER/Exact-Match/Top-1 accuracy by model × dataset, bootstrap confidence intervals, compute the "domain-shift gap" (the IAM vs. Kaggle difference for each model), draw tables/figures |
| **11** | Qualitative error analysis: sample the worst-error cases per model, classify the errors (confusing similar medicine names, illegible text, a VLM "hallucinating" outside the vocabulary, repeating/looping) — content that differentiates this from the shallow dev.to blog post |
| **12** | *(optional/extension)* A quick supplementary check on RxHandBD with the 1-2 best/worst models, clearly noted as supplementary (not a core pillar) since its license/credibility is unclear; write the Introduction + Related Work (positioning the distinction from RxScribe Bench and "From Handwriting to Structured Data") |
| **13** | Finalize Results + Discussion + Limitations (narrow scope — closed-vocabulary 78 classes, 1 country/language, the caveat that GOT-OCR2.0 was already trained on IAM); internal review |
| **14** | Finalize the Abstract, format to the arXiv/workshop template, prepare the code + results for public release on GitHub (increasing credibility), ready the final draft |

**Overall risks and mitigation:**
- **Biggest:** setting up the environment for GOT-OCR2.0/PaddleOCR-VL/DeepSeek-OCR (dependencies, downloading many GB of weights) → a 2-day buffer is already in place (Days 6-7); can run on a Kaggle Notebook (the data is already there, with a free-tier GPU) or Colab.
- **Slow IAM registration:** if delayed, use the TrOCR/GOT-OCR2.0 CER figures already published in the literature as a general reference point instead of re-running all of IAM — doesn't block progress.
- **Closed-model API cost:** kept at an optional/small-sample level, not a mandatory part of the main contribution.
- **Academic risk:** the Kaggle dataset is a 78-class closed vocabulary (not fully open free-text OCR) — this needs to be clearly stated in Limitations so a reviewer doesn't flag overclaiming; not a progress-blocking risk.
- **Data ethics:** both datasets are public, with no patient identification (IAM is general handwriting; Kaggle is cropped medicine-name images, with no patient information) — consistent with the "independent research, no IRB/HREC needed" framing you've already established.
---

## 5.2. Detailed 14-day action plan (Option A)

## 14-DAY EXECUTION PLAN — Benchmarking Pretrained OCR/VLM on Handwritten Prescriptions (Option A)

**Start:** Tuesday 15/09/2026 → **End:** Monday 28/09/2026 (complete draft + arXiv submission)
**Note on expectations:** "Completed within 2 weeks" = having a full draft, public code/results, and a submitted preprint — **NOT** already peer-reviewed/published. Workshop review takes an additional 4–8 weeks, and a main conference takes another 3–6 months, outside the scope of these 14 days.

---

### 1. DAY-BY-DAY PLAN TABLE

| Day | Main task | Expected end-of-day output |
|---|---|---|
| **1** (Tue 15/09) | - Register an FKI account to request IAM access at `https://fki.tic.heia-fr.ch/databases/iam-handwriting-database` (submit the form early in the morning since the approval time isn't fixed, and can take a few hours to a few days)<br>- Create the code repo (`git init`), venv, install the full base dependency set (see Section 2)<br>- Download the Kaggle dataset via `kaggle datasets download -d mamun1113/doctors-handwritten-prescription-bd-dataset` (4,680 images, 78 medicine-name classes)<br>- Carefully read the Mia et al. paper (IEEE iCACCESS 2024, doc 10499631) to obtain the original train/test split and baseline accuracy figures for comparison<br>- Create a `manifest.csv` file (image path, label, split) | Repo initialized, Kaggle data ready, IAM application submitted, environment ~70% installed |
| **2** (Wed 16/09) | - Finalize the test split: use the original IEEE paper's split if published, or randomly split 80/20 with stratification across the 78 classes (~936 test images) — fix the seed, save the split file for reproducibility<br>- Write + unit-test the metrics module: `compute_metrics.py` (CER, WER via `jiwer`, Exact-Match, Top-1 accuracy over the 78 classes)<br>- Run Tesseract (pytesseract) + 1 classic deep-learning baseline (EasyOCR **or** PaddleOCR PP-OCRv5) on all 936 test images<br>- Check IAM: if access has been granted → download the subset (see Section 3 for size); if not yet → move to Risk 3 (Section 6) | CER/WER/Exact-Match results for the 2 classic baselines; the measurement pipeline validated |
| **3** (Thu 17/09) | - Install `transformers`, download `microsoft/trocr-large-handwritten` from HuggingFace<br>- Run TrOCR on the 936 Kaggle test images (estimate: mid-range GPU ~0.1–0.3 s/image → 3–5 minutes; CPU would take hours, **prioritize GPU/Colab/Kaggle-Notebook**)<br>- Run TrOCR on an IAM subsample (300–500 images) to cross-check against the published TrOCR CER figure in the original paper (a pipeline sanity check)<br>- Start setting up the Donut environment (`naver-clova-ix/donut-base`) | TrOCR results on both datasets; how much the IAM CER matches/differs from the literature (logged) |
| **4** (Fri 18/09) | - Run Donut zero-shot on the Kaggle test set (using the `<s_synthdog>` task prompt or an equivalent for "raw text reading," NOT the CORD fine-tuned version since that's a receipt-field-extraction task, not free-text OCR — note this caveat clearly in the Method section)<br>- Start setting up the GOT-OCR2.0 environment (`stepfun-ai/GOT-OCR-2.0-hf`, needs a newer `transformers` or install from git if there are compatibility errors) | Donut results (with limitation notes); the GOT-OCR2.0 environment installed or specific error logs |
| **5** (Sat 19/09) | - Run GOT-OCR2.0 on the Kaggle test set + IAM subsample (estimated ~0.5–1 s/image on GPU → 936 images ≈ 15–20 minutes)<br>- **Buffer for dependency/GPU errors** — this is the biggest technical risk point, so a full half-day is set aside as a reserve<br>- If the local machine doesn't have enough VRAM: switch to running on a Kaggle Notebook (free T4 GPU, with the data already there) or Google Colab | GOT-OCR2.0 results on both datasets (with the caveat clearly noted: this model was already trained on IAM, so the IAM figure is not truly zero-shot) |
| **6–7** (Sun 20/09 – Mon 21/09, **buffer**) | - Finish any tasks running behind from Days 3–5<br>- Set up the PaddleOCR-VL environment: `pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/` then `pip install -U "paddleocr[doc-parser]>=3.6.0"` (use PaddleOCR-VL-1.6, released 05/2026 — the newest, more stable than the original 10/2025 release); if there's no suitable CUDA GPU, try the GGUF version (`PaddlePaddle/PaddleOCR-VL-1.6-GGUF`) running via CPU/llama.cpp<br>- Run PaddleOCR-VL on the Kaggle test set + IAM subsample<br>- If there's spare time in the buffer: start writing an early draft of the Methods section (dataset description, metric definitions — this part doesn't depend on the results) | PaddleOCR-VL results; all of Week 1's technical backlog cleared; a rough Methods draft |
| **8** (Tue 22/09) | - Decide the final model list: set up `Qwen/Qwen3-VL-8B-Instruct` (preferred, if VRAM ≥16GB is available) or `Qwen/Qwen2.5-VL-7B-Instruct` as a lighter option<br>- Consider DeepSeek-OCR (`deepseek-ai/DeepSeek-OCR`, ~6.7GB, needs CUDA 11.8+/torch 2.6.0): per the literature, its CER on non-Chinese handwriting is fairly poor — if time is tight, **it can be skipped in favor of Qwen-VL** as already proposed | The Qwen-VL environment ready; a clear decision on whether to run DeepSeek-OCR |
| **9** (Wed 23/09) | - Run Qwen-VL on the Kaggle test set + IAM subsample (estimated on a T4 GPU: 2–5 s/image → 936 images ≈ 30–75 minutes; if running via API, see the timing in Section 2)<br>- (Optional/bonus) Run 1 closed model (GPT-5/GPT-5-mini vision or Gemini 2.5/3.5 Flash) via API on a small sample of 200–300 images<br>- Merge all model results into the master table `results_master.csv` | All 6–8 models have results on both datasets; the combined table complete |
| **10** (Thu 24/09) | - Quantitative analysis: CER/WER/Exact-Match/Top-1 accuracy by model × dataset<br>- Compute 95% confidence intervals via bootstrap (1,000 resamples) for each model<br>- Compute the "domain-shift gap" = CER(IAM) − CER(Kaggle) for each model<br>- Paired statistical test (Wilcoxon signed-rank on per-image CER) to compare the best model vs. the second-best<br>- Draw the main results table + the domain-shift gap chart | Main results table/figure complete, with clear statistical significance |
| **11** (Fri 25/09) | - Qualitative error analysis: take the 15–20 worst-error cases per model, classify the errors (confusing visually similar medicine names, illegible output, a VLM "hallucinating" a word outside the 78-class vocabulary, repeating/looping tokens)<br>- This is the part that creates a clear distinction from the dev.to blog post (which only reports numbers, with no error analysis)<br>- Log the "non-converging/degenerate output" rate separately for each model (not folded into the average CER, to avoid skewing the statistics) | An error-classification table + 3–5 illustrative examples for the Discussion section |
| **12** (Sat 26/09, **optional/extension**) | - If on schedule: a quick supplementary check on RxHandBD (Zenodo DOI 10.5281/zenodo.18478741) with the 1–2 best/worst models, clearly marked as a *supplementary check*, not a core pillar (due to the conflicting MIT/CC-BY-4.0 license and lack of peer review)<br>- Write the Introduction + Related Work, clearly positioning the distinction from RxScribe Bench (09/2026) and "From Handwriting to Structured Data" (04/2026) | Draft Introduction + Related Work; (optional) 1 supplementary RxHandBD table |
| **13** (Sun 27/09) | - Finalize Results + Discussion + Limitations (narrow scope: closed-vocabulary 78 classes, 1 country/language, the caveat that GOT-OCR2.0 was already trained on IAM)<br>- Internal review of the whole paper, checking that figures match between tables and text<br>- Clean up the code, write a README, package the `results_master.csv` results publicly on GitHub | Draft nearly complete (missing only the final Abstract + formatting) |
| **14** (Mon 28/09) | - Write the final Abstract<br>- Format to the arXiv template (or the workshop template if a submission venue has been chosen)<br>- Make the code + data pointer public on GitHub (increasing credibility, meeting reproducibility standards)<br>- Submit to arXiv (cs.CV or cs.CL) | **Draft complete + submitted to arXiv** |

---

### 2. DAY 1 INSTALLATION LIST + COST/TIME ESTIMATES

#### 2.1. Base environment
```bash
python -m venv venv && source venv/bin/activate     # (Windows: venv\Scripts\activate)
pip install torch torchvision --index-url <matching your machine's CUDA>
pip install jiwer editdistance python-Levenshtein pandas numpy matplotlib seaborn scipy statsmodels
pip install kaggle
kaggle datasets download -d mamun1113/doctors-handwritten-prescription-bd-dataset
```

#### 2.2. Classic OCR
```bash
pip install pytesseract            # + install the OS's Tesseract binary
pip install easyocr
pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
pip install -U "paddleocr[doc-parser]>=3.6.0"      # shared for both classic PP-OCR and PaddleOCR-VL
```

#### 2.3. Pretrained HTR/VLM (HuggingFace)
```bash
pip install -U "huggingface_hub[cli]" transformers accelerate tiktoken verovio

hf download microsoft/trocr-large-handwritten
hf download naver-clova-ix/donut-base
hf download stepfun-ai/GOT-OCR-2.0-hf          # use AutoModelForImageTextToText
hf download PaddlePaddle/PaddleOCR-VL-1.6       # the newest version (05/2026), zero-cost migration from 1.5
hf download deepseek-ai/DeepSeek-OCR            # ~6.7GB, needs CUDA 11.8+/torch 2.6.0 — optional, see Day 8
hf download Qwen/Qwen3-VL-8B-Instruct           # or Qwen/Qwen2.5-VL-7B-Instruct if VRAM is limited
```
*Note:* if the pip version of `transformers` doesn't yet support GOT-OCR2/PaddleOCR-VL, install from source: `pip install git+https://github.com/huggingface/transformers.git`.

#### 2.4. API keys (only for the optional bonus part on Day 9)
- **Google AI Studio** (Gemini API key, has a free tier) — sign up at aistudio.google.com
- **OpenAI Platform** (API key for GPT-5/GPT-5-mini vision)

#### 2.5. Realistic cost & runtime estimates

| Model | How it's run | Estimated time (test set ~936 Kaggle images) | Cost |
|---|---|---|---|
| Tesseract, EasyOCR/PaddleOCR | Local CPU/GPU | A few minutes | $0 |
| TrOCR-large, Donut-base | Mid-range GPU | ~3–8 minutes | $0 |
| GOT-OCR2.0 (~580M) | GPU (T4 or better) | ~15–20 minutes | $0 |
| PaddleOCR-VL (0.9B) | GPU or CPU (GGUF version) | ~15–30 minutes (GPU) / longer on CPU | $0 |
| DeepSeek-OCR (~6.7GB weight) | GPU ≥16GB VRAM | ~30–60 minutes | $0 (self-hosted) |
| Qwen3-VL-8B / Qwen2.5-VL-7B | GPU ≥16GB VRAM or API | ~30–75 minutes (GPU) | $0 (self-hosted) or per API pricing |
| GPT-5(-mini) vision (bonus, ~250 images) | API | With concurrency 5, real batch processing takes ~30–60 minutes (including rate-limit/retry) | Estimated ~$0.001/image × 250 ≈ **$0.25–$0.50** |
| Gemini 2.5/3.5 Flash (bonus, ~250 images) | API | Similar, ~20–40 minutes | Estimated ~$0.0005–0.001/image × 250 ≈ **$0.15–$0.25** |

**Total expected API cost for the entire bonus section: under $1**. API pricing is based on tokens (image + prompt + output), not a fixed price per image — the figures above are estimates based on the publicly listed pricing as of 09/2026 ($0.625/$5 million tokens for GPT-5; $0.30/$2.50 million tokens for Gemini 2.5 Flash); **the official pricing page should be re-checked before actually running this** since prices can change.

---

### 3. EXPERIMENTAL DESIGN

**Metric:**
- **CER** (Character Error Rate, via `jiwer`) — the primary metric, comparable to the published TrOCR/GOT-OCR2.0 CER figures on IAM (a sanity check)
- **WER** (Word Error Rate) — secondary, since most Kaggle labels are 1 word/a short phrase
- **Exact-Match accuracy** (normalized: lowercased, whitespace trimmed) — matching the dev.to blog's measurement approach for cross-comparability
- **Top-1 classification accuracy on the 78 fixed classes** (mapping OCR output to the nearest label in the 78-medicine-name dictionary) — matching the original IEEE paper's measurement approach for direct comparison with the 2024 baseline
- The rate of **"non-converging/degenerate output"** logged separately, not folded into the average CER

**Data split:**
- Kaggle: 4,680 images, 78 classes — reuse the original IEEE paper's split if published, otherwise split 80/20 with class stratification → **test set ≈ 936 images**. With n≈936, the standard error of the average CER (assuming a per-image standard deviation of ~0.3) is ≈1%, CI 95% ≈ ±2%, enough to detect a >4–5% CER difference between models — suitable for a 2-week benchmark study, with no need to run all 4,680 images for every model (saving time, especially for slower VLMs).
- IAM: take a **subsample of 300–500 word/line images** (not running the whole of IAM — unnecessary for the domain-shift question, and it keeps time/cost under control for models run via API).
- Bonus API (GPT-5/Gemini): an even smaller subsample, **200–300 images**, enough to see the open-vs-closed comparison trend without inflating cost/time.

**Manual checks:**
- Manually cross-check ~50 random images in the test set to confirm the folder label = accurate ground truth (ruling out the risk of labeling errors in a crowd-sourced dataset).
- Manually review the 15–20 worst-error cases per model (Day 11) for qualitative error classification — not to "fix" the ground truth, but to feed into the Discussion section.

---

### 4. SHORT PAPER STRUCTURE (estimated ~5–6 pages, ~3,000 words, in the ICDAR/DAS short-paper format)

| Section | Content | Estimate |
|---|---|---|
| **Abstract** | Research question, method, main results (best/worst CER figures, domain-shift gap) | 150–200 words |
| **1. Introduction** | Motivation (handwritten prescriptions are a real source of medical errors), the specific gap (no one has tested GOT-OCR2.0/PaddleOCR-VL/DeepSeek-OCR/Qwen-VL on medical HTR), listing 3–4 contributions | 500–600 words (~0.75–1 page) |
| **2. Related Work** | IAM (Marti & Bunke 2002), the Kaggle dataset (Mia et al. 2024), the closest related works (RxScribe Bench 09/2026, "From Handwriting to Structured Data" 04/2026, the dev.to benchmark blog) — clearly stating why they don't overlap | 300–400 words (~0.5 page) |
| **3. Method/Experimental Setup** | A table describing the 2 datasets, a table of the model list (architecture, parameter count, source), metric definitions, split details | 600–800 words + 2 tables (~1–1.25 pages) |
| **4. Results** | The main results table (model × dataset × metric), the domain-shift gap figure, confidence intervals | 400–500 words + 2–3 tables/figures (~1.25–1.5 pages) |
| **5. Discussion** | Qualitative error analysis, explaining why a model trained on IAM (GOT-OCR2.0) still performs poorly on medical data, comparing model families | 400–500 words (~0.5–0.75 page) |
| **6. Limitations** | Closed-vocabulary 78 classes, 1 language/1 country, the GOT-OCR2.0 caveat, RxHandBD is only supplementary | 150–200 words (~0.25 page) |
| **7. Conclusion** | A 2–3 sentence summary | 100–150 words |
| **References** | ~25–35 citations | ~0.5–1 page |

---

### 5. FAST PUBLICATION VENUES (only SUBMISSION, not a guarantee of ACCEPTANCE within 14 days)

1. **arXiv** (cs.CV or cs.CL) — no peer review needed, published almost immediately (a few hours to 1–2 days of moderation for a new account). This is the **only step that is truly completed within 14 days** and is also a way to establish "priority" for the novelty before risking being published first by someone else.
2. **A short workshop attached to ICDAR/DAS** (Document Analysis Systems) — the "short paper"/"resource paper" track if the CFP is open. **The current CFP calendar needs to be checked directly**, since ICDAR/DAS meets on a 2-year cycle and the specific schedule as of 09/2026 may not align with this 14-day submission window — submit if there's still time, otherwise hold the draft for the next CFP round.
3. **A Clinical-NLP workshop attached to a major NLP conference** (e.g., the "Findings"/"short paper" track of a Clinical NLP workshop at ACL/EMNLP/NAACL, or an ML4Health-style workshop) — suitable since the paper sits between an OCR benchmark and a medical application; "findings" tracks usually have a rolling/shorter deadline than the main track — **check the specific CFP before submitting**.
4. **Papers with Code / Hugging Face Papers** (linking to the arXiv version) — not peer review, but increases the chance of quick community visibility and serves as an anchor point for code+data reproducibility.

---

### 6. MAIN RISKS AND FALLBACK PLANS

| # | Risk | Fallback plan |
|---|---|---|
| 1 | Installing GOT-OCR2.0/PaddleOCR-VL/DeepSeek-OCR runs into dependency/CUDA errors (the biggest risk per the original assessment) | Switch to running on a Kaggle Notebook/Google Colab (free-tier T4 GPU, CUDA already configured); if it still fails, use the GGUF/quantized version (running on CPU via llama.cpp) or drop that model from the list and note "excluded due to environment constraints" in Limitations — this doesn't block overall progress since it's only 1 of 7–8 models |
| 2 | API rate limits or costs exceeding expectations (GPT-5/Gemini/Qwen-VL via API) | Reduce the bonus sample to 100–150 images; add exponential backoff/retry; if it's still stuck, **drop the closed-model bonus part entirely** — it was already marked optional from the start and doesn't affect the main contribution (comparing open-source models) |
| 3 | IAM registration access isn't granted in time before Day 3 | Temporarily use a public mirror (Kaggle "iam_handwriting_word_database" or HuggingFace "xReniar/IAM-Dataset") to keep progress moving, noting the license caveat clearly in Limitations; or, for models that can't be re-run on IAM in time, use the IAM CER figure already published in that model's original paper (TrOCR, GOT-OCR2.0) as the domain-shift reference point instead of measuring it directly — doesn't block progress |
| 4 | A model produces repeating/looping output or hallucinates entirely outside the 78-class dictionary (especially Donut with an incorrect prompt, or DeepSeek-OCR on non-Chinese text) | Limit `max_new_tokens`, add a repetition penalty; log these cases separately as a "non-convergence rate" instead of folding them directly into the average CER (to avoid skewing the statistics) |
| 5 | The results don't have the expected novelty (e.g., all models fail in the same way, or the domain-shift gap is negligible) | Shift the paper's focus from "which model is best" to "characterizing error types" (the Day 11 qualitative analysis still retains independent contribution value); or expand the RxHandBD supplementary study (Day 12) into a larger section to create a second angle within the same paper |
| 6 | Risk of being flagged by a reviewer for overclaiming (presenting the 78-class closed-vocabulary dataset as fully open-ended OCR) | Already separated clearly into 2 metrics (free-text CER vs. 78-class Top-1 classification) in Section 3, and clearly stated in Limitations — handled up front, not a fallback plan |
| 7 | Overall schedule slippage from technical issues piling up | The Day 6–7 buffer is already reserved; if that's still not enough, cut in the priority order already marked "optional" from the start: (a) drop RxHandBD on Day 12, (b) drop the closed-model API bonus on Day 9, (c) drop DeepSeek-OCR — keeping the core 5–6 mandatory models (Tesseract, 1 classic deep-learning OCR, TrOCR, Donut, GOT-OCR2.0, PaddleOCR-VL or Qwen-VL) intact to make sure the arXiv submission still lands on time on Day 14 |

---

#### Reference sources used to flesh out the plan
- [stepfun-ai/GOT-OCR-2.0-hf (Hugging Face)](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf)
- [PaddlePaddle/PaddleOCR-VL-1.6 (Hugging Face)](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6)
- [deepseek-ai/DeepSeek-OCR (Hugging Face)](https://huggingface.co/deepseek-ai/DeepSeek-OCR) / [GitHub deepseek-ai/DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)
- [Qwen/Qwen3-VL-8B-Instruct (Hugging Face)](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- [Doctor's Handwritten Prescription BD dataset (Kaggle)](https://www.kaggle.com/datasets/mamun1113/doctors-handwritten-prescription-bd-dataset)
- [IAM Handwriting Database (FKI)](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)
- [Gemini API pricing 2026](https://ai.google.dev/gemini-api/docs/pricing)
- [OpenAI API pricing 2026 (aggregated)](https://www.finout.io/blog/openai-pricing-in-2026)
