# Verified Reference List (17/09/2026)

Compiled from 3 independent investigations (web search + Google Scholar/Semantic Scholar/arXiv + PubMed/Consensus),
used as the primary citation source for the paper manuscript (Phase 12). Every citation below has been verified
through at least 1 source (DOI/arXiv ID/venue), with notes on reliability and any deviation from the original assumptions.

---

## 1. Core Models/Methods (Used in Benchmarking + Fine-Tuning)

| # | Citation | Venue | DOI/arXiv | Notes |
|---|---|---|---|---|
| 1 | Li, Lv, Chen, Cui, Lu, Florencio, Zhang, Li, Wei. "TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models." | **AAAI 2023**, Vol 37 No 11, pp 13094-13102 | DOI 10.1609/aaai.v37i11.26538 / arXiv:2109.10282 | ~480 citations. The main base model for the entire project. |
| 2 | Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen. "LoRA: Low-Rank Adaptation of Large Language Models." | **ICLR 2022** | arXiv:2106.09685 | ~15,183 citations. The foundation of the fine-tuning method. |
| 3 | Kim, Hong, Yim, Nam, Park, Yim, Hwang, Yun, Han, Park. "OCR-free Document Understanding Transformer" (Donut). | **ECCV 2022**, Springer LNCS 13688 | DOI 10.1007/978-3-031-19815-1_29 / arXiv:2111.15664 | Confirms pretraining on SynthDoG (synthetic documents) — solid evidence explaining the domain mismatch when Donut fails on single-word prescription images (Phase 2). |
| 4 | Wei, Liu, Chen, Wang, Kong, Xu, Ge, Zhao, Sun, Peng, Han, Zhang. "General OCR Theory: Towards OCR-2.0 via a Unified End-to-end Model" (GOT-OCR2.0). | **arXiv preprint** (peer review not confirmed) | arXiv:2409.01704 | ~104 citations. |
| 5 | Bai, Chen, Liu, et al. (Qwen Team). "Qwen2.5-VL Technical Report." | **arXiv preprint** (technical report, standard practice for this type of language model) | arXiv:2502.13923 | |
| 6 | Cui, Sun, Liang, Gao, Zhang, Liu, Wang, Zhou, Liu, Lin, Zhang, Zhang, Zheng, Zhang, Zhang, Liu, Yu, Ma (Baidu). "PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model." | **arXiv preprint** | arXiv:2510.14528 | Confirms the exact version tested (`PaddlePaddle/PaddleOCR-VL`, not the newer v1.5/v1.6). The model was excluded from the benchmark due to a library bug (see `phase4_summary.md`). |
| 7 | Chang, Li. "Mixed Text Recognition with Efficient Parameter Fine-Tuning and Transformer" (formerly called "DLoRA-TrOCR"). | **ICONIP 2024**, Springer LNCS 15294 (**peer-reviewed** — no longer just an arXiv preprint) | DOI 10.1007/978-981-96-6599-0_2 / arXiv:2404.12734 (v4, renamed) | **⚠️ Corrected (17/09):** the old title + venue ("DLoRA-TrOCR", arXiv-only) is outdated — use the Springer/ICONIP version when citing. Evidence that LoRA (0.7% of parameters) achieves a CER of 4.02% on IAM. |

## 2. Datasets

| # | Citation | Venue | DOI | Notes |
|---|---|---|---|---|
| 8 | Marti, Bunke. "The IAM-database: an English sentence database for offline handwriting recognition." | **IJDAR**, Vol 5(1), pp 39-46 (2002) | DOI 10.1007/s100320200071 | Confirmed correct — the canonical IAM source. |
| 9 | Mia, Chowdhury, Mamun, Ruddra, Tanny. "A Deep Neural Network Approach with Pioneering Local Dataset to Recognize Doctor's Handwritten Prescription in Bangladesh." | **iCACCESS 2024** (IEEE) | DOI 10.1109/iCACCESS61735.2024.10499631 | The original paper for the Kaggle-Rx dataset. **Needs independent verification** of the 78-class/4,680-image figures directly in the PDF before citing these numbers (the agent could not fetch the full text due to JS rendering). |

## 3. Target Journal Precedent

| # | Citation | Venue | DOI | Notes |
|---|---|---|---|---|
| 10 | Cheema, Shaiq, Mirza, Kamal, Naeem. "Adapting multilingual vision language transformers for low-resource Urdu optical character recognition (OCR)." | **PeerJ Computer Science**, Vol 10, e1964 (2024) | DOI 10.7717/peerj-cs.1964 | **Confirmed correct** — the PeerJ CS precedent. Note when writing: their main model is ViLanOCR (Swin+mBART-50), **TrOCR is only a comparison baseline in the paper** — should not be overstated as "the original TrOCR paper," but rather as "PeerJ CS has previously published a paper using TrOCR as a baseline for low-resource handwritten OCR." |

## 4. Novelty-Overlap Risks — MUST Cite + Distinguish in Related Work

| # | Citation | Venue | Overlap level | How to distinguish in the paper |
|---|---|---|---|---|
| 11 | "RxScribe Bench: A Multi-Axis Benchmark for Evaluating Vision-Language Models on Indian Outpatient Prescriptions." | arXiv:2609.13280 (09/2026) | **Medium-high** — same problem (benchmarking VLMs on handwritten prescriptions) | Their dataset is **private** (newly collected, not public); no LoRA fine-tuning/catastrophic forgetting analysis section. Our paper: **public dataset + fine-tuning + forgetting analysis** — state these 2 differences clearly at the start of Related Work. |
| 12 | Community benchmark blog post on the "RxHandBD" dataset (Zenodo/Mendeley, 02/2026, Md. Masudul Islam) | Not peer-reviewed (dev.to, 04/2026) | Low-medium — a different public Bangladeshi dataset, not the same model set | Can be briefly mentioned in Related Work as "another public prescription dataset that has recently appeared," not a direct competitor. |
| 13 | "From Handwriting to Structured Data: Benchmarking AI Digitisation of Handwritten Forms" (UBOMI BUHLE project, South Africa, 2026) | Venue unclear | Low — medical forms in general, not prescriptions; private dataset | Mention very briefly if needed, not a major risk. |
| 14 | "A Hybrid Deep Learning-Based OCR Model for Handwritten Medical Prescriptions." | **Frontiers in Medicine** (08/2026), DOI 10.3389/fmed.2026.1856485 | Low — different model set (classical Tesseract+PaddleOCR+GPT-4o), no GOT-OCR2.0/Qwen2.5-VL/PaddleOCR-VL | Can be cited as related work on a nearby topic (medical OCR) but does not overlap in method. |

## 5. General Background/Surveys (Related Work/Introduction)

| # | Citation | Venue | Quartile | Notes |
|---|---|---|---|---|
| 15 | Garrido-Munoz, Rios-Vila, Calvo-Zaragoza. "Handwritten Text Recognition: A Survey." | **IEEE TPAMI**, Vol 48(4), pp 4367-4387 (2026) | **Q1** | DOI 10.1109/TPAMI.2025.3646002. The most authoritative HTR survey found. |
| 16 | AlKendi, Gechter, Heyberger, Guyeux. "Advancements and Challenges in Handwritten Text Recognition: A Comprehensive Survey." | **Journal of Imaging** (MDPI), Vol 10(1), a18 (2024) | **Q2** | DOI 10.3390/jimaging10010018. Useful because Journal of Imaging is our backup option. |
| 17 | Crosilla, Klic, Colavizza. "Benchmarking large language models for handwritten text recognition." | **Journal of Documentation** (Emerald), Vol 81(7), pp 334-354 (2025) | **Q1** | arXiv:2503.15195. Very closely matches the framing of our paper (comparing general-purpose VLMs vs. dedicated OCR models). |

## 6. Citations for the Discussion Section — Catastrophic Forgetting in PEFT/LoRA (Phase 7–9)

| # | Citation | Venue | Notes |
|---|---|---|---|
| 18 | Biderman, Portes, Ortiz, Paul, Greengard, Jennings, King, Havens, Chiley, Frankle, Blakeney, Cunningham. "LoRA Learns Less and Forgets Less." | **TMLR 2024** (Featured Certification) | arXiv:2405.09673. **Note the opposite direction**: this paper argues that LoRA forgets LESS than full fine-tuning — our result (LoRA fine-tuning still causes statistically significant forgetting on IAM) is a **complementary/complicating finding** relative to this conclusion, not a full contradiction (we do not compare against full fine-tuning), but it needs to be framed carefully: "LoRA may reduce forgetting relative to full fine-tuning, but it does NOT eliminate forgetting entirely — our results show that updating even <1% of parameters is sufficient to cause a statistically significant drop in generalization." |
| 19 | Shuttleworth, Andreas, Torralba, Sharma. "LoRA vs Full Fine-tuning: An Illusion of Equivalence." | **NeurIPS 2025** | arXiv:2410.21228. Explains the mechanism ("intruder dimensions") — used to explain WHY LoRA can still forget despite having few parameters. |
| 20 | Chen et al. "Bayesian Parameter-Efficient Fine-Tuning for Overcoming Catastrophic Forgetting." | **IEEE/ACM TASLP** | Directly confirms: "catastrophic forgetting remains an issue with PEFT" — the strongest citation supporting our finding. |
| 21 | Kalajdzievski. "Scaling Laws for Forgetting When Fine-Tuning Large Language Models." | **arXiv preprint** (peer review not confirmed — state this clearly when citing) | arXiv:2401.05605 |

## 7. Citations for the Discussion Section — Medication Safety/Confusable Drug Names (Phase 10)

| # | Citation | Venue | Notes |
|---|---|---|---|
| 22 | Bryan, Aronson, Williams, Jordan. "The problem of look-alike, sound-alike name errors: Drivers and solutions." | **British Journal of Clinical Pharmacology**, 87(2), 386-394 (2020) | PMID 32198938. Background evidence for the LASA (look-alike sound-alike) concept. |
| 23 | Ostini, Roughead, Kirkpatrick, Monteith, Tett. "Quality Use of Medicines - medication safety issues in naming; look-alike, sound-alike medicine names." | **International Journal of Pharmacy Practice**, 20(6), 349-357 (2012) | PMID 23134093. |
| 24 | Karet. "Linguistic Analysis of Generic-Generic Drug Name Pairs Prone to Wrong-Drug Errors for which Tall-Man Lettering is Recommended." | **Therapeutic Innovation & Regulatory Science**, 57(4), 751-758 (2023) | PMID 37171707. **Most useful for methodology**: quantifies the orthographic/phonetic similarity between confusable drug pairs per FDA/ISMP — can be referenced when explaining how `confusable_wrong_drug` is defined (normalized edit distance ≤0.34) in `src/analyze_errors.py`. |
| 25 | Lizano-Díez et al. "Prevention strategies to identify LASA errors: building and sustaining a culture of patient safety." | **BMC Health Services Research**, 20, 63 (2020) | PMID 31996197. |
| 26 | Zhou, Blackley, Kowalski, Doan, Acker, Landman, Kontrient, Mack, Meteer, Bates, Goss. "Analysis of Errors in Dictated Clinical Documents Assisted by Speech Recognition Software and Professional Transcriptionists." | **JAMA Network Open**, 1(3), e180530 (2018) | PMID 30370424. The closest precedent for the argument that "classifying errors by clinical danger level matters more than looking at the raw error rate alone" — but in the ASR/clinical-note domain, NOT OCR/prescriptions. |

**Important conclusion for Section 4 (Phase 10) of the paper:** the research agent confirms it **found NO prior work** directly connecting "the type of OCR error when reading prescriptions" with "medication safety risk" — this is a **genuinely novel angle** of the paper (not just a marketing claim), so it should be emphasized clearly in the Discussion, using wording along these lines: "To date, to our knowledge, no study has analyzed the shift in OCR error TYPE (not just error rate) from a medication-safety perspective; the general principle that classifying errors by clinical danger level matters more than the raw error rate has precedent in the field of ASR-assisted clinical documentation [Zhou et al. 2018] and in clinical LLM safety evaluation, and the LASA literature [Bryan 2020; Ostini 2012; Karet 2023] establishes that errors that resemble a different real drug name are a separately recognized category of dangerous error."

---

## Changes Made in This Session (17/09/2026)

1. **Fixed** the `src/augmentation.py` docstring — no longer uses Ali et al. (arXiv:2412.18199) to specifically justify the augmentation choice (that paper is not about augmentation methodology).
2. **Fixed** `docs/05-q2-research-plan.md` Section 3.2 — added a warning note + removed an incorrect claim.
3. **Updated** the DLoRA-TrOCR citation (Section 6, row "ITEM 4") to the peer-reviewed version (ICONIP 2024/Springer).
4. **Added** risk #9 (RxScribe Bench) to the risk table in Section 6 of the plan.

## Tasks Still Needed Before Submission (Not Done in This Session)

1. Independently verify the 78-class/4,680-image figures directly in the original PDF of Mia et al. (iCACCESS 2024) — the agent could not fetch the full text.
2. Check whether the Qwen2.5-VL Technical Report (arXiv:2502.13923) and GOT-OCR2.0 (arXiv:2409.01704) have been accepted at any peer-reviewed venue by the time of final submission (both were still arXiv-only as of the 17/09/2026 check).
3. Read RxScribe Bench (arXiv:2609.13280) carefully in full before writing the novelty-distinction paragraph in Related Work — the agent only summarized from the abstract/description, without reading the full text.
4. Verify the citation "Structure-Aware Text Recognition for Ancient Greek Critical Editions" (arXiv:2603.02803) in `docs/05-q2-research-plan.md` around line 341 — this citation has **NOT been verified** in the 3 recent research passes (it was outside the scope of the questions asked), and should only be used after independent verification.
