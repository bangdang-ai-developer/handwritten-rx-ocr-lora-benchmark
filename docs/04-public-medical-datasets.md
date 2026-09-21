# 4. Public, Reputable Medical OCR Datasets (No Vietnamese, No Hospital Collaboration Required)

*Updated 15/09/2026. This is the result of research for an INDEPENDENT research direction, using readily available public data — not continuing the Vietnamese OCR direction or the CCI-linked pediatric cancer registry direction (see the reasons in [archive/](archive/)).*

This section consists of 3 parts: (4.1) a broad web survey listing ~22 candidate datasets, (4.2) a literature lookup to determine which dataset is the "standard" and which models have NOT yet been tested on it (the opening for novelty), and (4.3) a deep, direct verification of each of the strongest candidates (download link, license, ground truth, and real-world credibility).

---

## 4.1. Broad survey of candidate datasets

## PUBLIC OCR/DOCUMENT AI DATASETS IN THE MEDICAL/CLINICAL DOMAIN
*(Compiled to support a 2-week benchmarking research plan — not limited to a specific disease, not using Vietnamese)*

---

### GROUP 1 — HANDWRITTEN MEDICAL / PRESCRIPTION TEXT (Handwritten Medical/Prescription)

#### 1. RxHandBD — Handwritten Prescription Word Image Dataset
- **Download link:** [Zenodo](https://zenodo.org/records/18478741) | [Mendeley Data](https://data.mendeley.com/datasets/dsb5r6vskg/3)
- **Scale:** 5,578 images of handwritten prescription words, pre-cropped per word, normalized to 128×128px; a vocabulary of 1,559 unique words (generic drug names, brand names, dosage forms, instructions). Already split into train/test (4,463/1,115) with a labeled CSV file.
- **Ground truth:** Full word-level transcription is available (word-level label CSV) — usable directly for HTR/word recognition without additional labeling.
- **License:** Hosted on Zenodo/Mendeley (default open-dataset CC-BY, but the specific page should be checked before citation; in principle free to use for research).
- **Credibility:** A very new dataset (2025–2026), with a related IEEE paper ("A Deep Neural Network Approach with Pioneering Local Dataset to Recognize Doctor's Handwritten Prescription in Bangladesh," iCACCESS 2024) — still lightly cited, but it is a rare specialized data source for this domain and can be downloaded immediately without permission.

#### 2. Doctor's Handwritten Prescription BD dataset (Kaggle)
- **Link:** https://www.kaggle.com/datasets/mamun1113/doctors-handwritten-prescription-bd-dataset
- **Scale:** 4,680–4,688 images (~20 MB), 78 different medication words/terms, collected from multiple hospitals in Bangladesh.
- **Ground truth:** Word labels are available via filename/CSV — full word-level transcription.
- **License:** Follows the dataset's default Kaggle license (usually CC0/Open; check the "License" tab on the page).
- **Credibility:** The most popular Kaggle dataset in the "handwritten prescription" category, used in many derivative notebooks/papers (Swin Transformer, CNN-SSD-ResNet, etc.); downloadable instantly, requiring only a free Kaggle account.

#### 3. Handwritten Medical Prescriptions Collection (Illegible Medical Prescription Images Dataset)
- **Link:** https://www.kaggle.com/datasets/mehaksingal/illegible-medical-prescription-images-dataset
- **Scale:** The exact number of images was not found in the retrieved metadata — the Kaggle page should be checked directly (last updated 2024).
- **Ground truth:** Focused on images of hard-to-read ("illegible") handwriting, used for OCR + NLP to identify disease/medication names — there is a related Springer paper ("Disease Identification from Illegible Medical Prescriptions Using OCR and NLP Techniques").
- **License:** Follows the dataset's Kaggle terms.
- **Credibility:** New, lightly cited, but suitable as a "worst-case OCR" (illegible handwriting) case study — downloadable immediately.

#### 4. A Curated Bangladesh-Based Dataset of Handwritten and Printed Prescription Images
- **Link:** https://data.mendeley.com/datasets/k62rfd23kz/1
- **Characteristics:** Contains both handwritten AND printed prescriptions — suitable for a handwritten-vs-printed classification task or mixed OCR.
- **License:** Mendeley Data (usually CC BY 4.0).
- **Credibility:** A niche, new dataset, not yet widely cited, but a good complement to RxHandBD.

#### 5. MIRAGE — Multimodal Identification and Recognition of Annotations in Indian General Prescriptions
- **Link:** https://arxiv.org/abs/2410.09729 (paper); the publicly released portion is a **small labeled subset of ~100 records** (the original dataset of 743,118 records is owned by Medyug Technology — NOT fully public).
- **Ground truth:** Multimodal annotation (bounding box + transcription) is available for the public subset.
- **License:** Follows the terms accompanying the arXiv/HuggingFace Papers listing.
- **Credibility:** A new 2024 paper, fine-tuning LLaVA 1.6 and Idefics2 — useful as a methodological reference, but the full dataset cannot be freely downloaded (only a demo subset is available).
- **Note:** Since most of the data is not public, it is recommended to use this only as a methodological reference, NOT as a primary data source for a 2-week benchmark.

---

### GROUP 2 — GENERAL HANDWRITING (for cross-checking/supplementary pretraining)

#### 6. IAM Handwriting Database
- **Link:** https://fki.tic.heia-fr.ch/databases/iam-handwriting-database (convenient mirror: [Teklia/IAM-line on HuggingFace](https://huggingface.co/datasets/Teklia/IAM-line))
- **Scale:** 1,539 handwritten pages from 657 writers, 13,353 lines, 115,320 words. Size ranges from a few hundred MB to ~22GB depending on the version (word/line/form).
- **Ground truth:** Full word-level transcription plus bounding boxes (XML), the "gold standard" in HTR benchmarking.
- **License:** Free for non-commercial research use, requires simple registration, no IRB needed.
- **Credibility:** VERY high — the original paper by Marti & Bunke (2002, IJDAR) is the most classic HTR benchmark, cited in nearly every English handwriting OCR/HTR paper for 20+ years (thousands of citations).

#### 7. CVL Database
- **Link:** https://cvl.tuwien.ac.at/research/cvl-databases/an-off-line-database-for-writer-retrieval-writer-identification-and-word-spotting/
- **Scale:** 7 sample texts (1 German, 6 English), 310 writers, RGB images at 300dpi + word bounding boxes (XML).
- **Ground truth:** Full word-level transcription.
- **License:** CC BY-NC 3.0 (non-commercial research only).
- **Credibility:** A popular supplementary benchmark, commonly used alongside IAM in writer-identification/HTR papers.

#### 8. RIMES-2011
- **Link:** [HuggingFace Teklia/RIMES-2011-line](https://huggingface.co/datasets/Teklia/RIMES-2011-line) | original site a2ialab.com (requires emailing for a user agreement)
- **Scale:** 12,723 pages, ~5,605 handwritten French letters, over 1,300 writers.
- **Ground truth:** Full line/page-level transcription.
- **License:** Requires signing a user agreement (free, by email) or using the HuggingFace/Zenodo mirror without needing permission.
- **Credibility:** A standard benchmark for HTR in Latin-script languages other than English — widely used in the ICDAR 2011 competition and cited broadly.

---

### GROUP 3 — MACHINE-READABLE MEDICAL REPORT TEXT (source for synthetic-scan rendering or OCR)

#### 9. TCGA-Reports
- **Link:** [Mendeley Data (Kefeli et al.)](https://data.mendeley.com/datasets/hyg5xkznpx/1)
- **Scale:** 9,523 pathology reports across 32 tissue/cancer types, already OCR'd via AWS Textract from the original TCGA PDFs.
- **Ground truth:** This IS the post-OCR product itself (cleaned text) — extremely useful: the original scanned PDF/images can be re-OCR'd and compared against this "gold" text to compute CER/WER.
- **License:** Mendeley Data, usually CC BY 4.0 (check the specific page).
- **Credibility:** A 2024 paper in *Patterns* (Cell Press) — new but high quality, specifically designed to "benchmark text-based AI models in medicine"; a direct fit for your OCR research goal.

#### 10. MIMIC-IV-Note / MIMIC-III discharge summaries (PhysioNet)
- **Link:** https://physionet.org/content/mimic-iv-note/2.2/
- **Scale:** 331,794 de-identified discharge summaries from 145,915 patients (MIMIC-IV-Note); MIMIC-III has 52,746 discharge summaries.
- **Ground truth:** Clean text (not scanned images) — usable as "ground truth text" to create synthetic scan-like images (rendered to PDF/image and then degraded) for a controlled OCR benchmark.
- **License:** PhysioNet Credentialed Health Data License — **requires credentialing**: completing CITI training (a few hours) + a PhysioNet staff review that typically takes **24–48 hours** (potentially longer, up to a few days, if the application is incomplete), followed by signing a DUA specific to each dataset (usually immediate once credentialed). Total realistic time: **usually 3–7 days**, which fits within a 2-week plan if started on Day 1.
- **Credibility:** Extremely high in medical AI, thousands of papers use MIMIC.

#### 11. n2c2 (i2b2) Clinical NLP Shared Task Corpora — especially Track 1 "2022 n2c2 Contextualized Medication Event" (CMED)
- **Link:** https://n2c2.dbmi.hms.harvard.edu/data-sets → register via https://portal.dbmi.hms.harvard.edu/
- **Scale (e.g., Track 1/2022):** 500 annotated EHR notes, 9,013 medication mentions, split into 350 train/50 val/100 test notes.
- **Ground truth:** Very detailed NER + context annotation (change/no-change, action, negation, temporality, etc.).
- **License:** A free Data Use Agreement (DUA), but mandatory to sign; **IMPORTANT WARNING:** some recent reports (mid-2026) indicate that the n2c2 portal on the DBMI Portal is showing "Temporarily Unavailable" and registration is temporarily closed — access status should be checked before including it in the plan, as there is a risk of not finishing within 2 weeks.
- **Credibility:** A gold-standard clinical NLP dataset (since 2006), heavily cited — but this is plain text, not scanned images, only useful if you render it into images yourself to test OCR.

---

### GROUP 4 — FORM/RECEIPT DATASETS (technical benchmarks, not medical but useful for pretraining/baselines)

#### 12. SROIE (ICDAR 2019 Robust Reading Challenge — Scanned Receipts OCR)
- **Link:** https://rrc.cvc.uab.es/?ch=13 | GitHub mirror: https://github.com/zzzDavid/ICDAR-2019-SROIE
- **Scale:** 1,000 scanned receipt images (600 train + 400 test), ~542MB.
- **Ground truth:** Text localization + OCR transcription + key-information extraction (4 fields: item name, unit price, total amount, date).
- **License:** CC-BY-4.0 for the redistributed annotations.
- **Credibility:** ~262 citations (IEEE Xplore) — a classic OCR+KIE benchmark, used for pretraining text-detection/OCR pipelines before applying them to the medical domain (semi-structured document layouts similar to prescriptions/hospital invoices).

#### 13. FUNSD (Form Understanding in Noisy Scanned Documents)
- **Link:** https://guillaumejaume.github.io/FUNSD/
- **Scale:** 199 real scanned forms (149 train/50 test), 9,707 semantic entities, 31,485 words.
- **Ground truth:** OCR text + bounding boxes + entity labels + linking (complete, this annotation format is the standard used for LayoutLM and other document-AI models).
- **License:** No clear commercial license — for research use (per the original site).
- **Credibility:** ~402 citations (Semantic Scholar) — very popular, a standard benchmark for testing document-layout understanding (techniques similar to what is needed for semi-structured medical forms/prescriptions).

#### 14. CORD (Consolidated Receipt Dataset)
- **Link:** https://github.com/clovaai/cord | HuggingFace: https://huggingface.co/datasets/naver-clova-ix/cord-v2
- **Scale:** >11,000 Indonesian receipts, 30 detailed semantic labels.
- **Ground truth:** Box-level OCR + multi-level parsing classes — the most complete in the receipt-dataset group.
- **License:** CC BY 4.0.
- **Credibility:** ~294 citations; the main training/test dataset for the **Donut** model (OCR-free Document Understanding) — highly relevant if you benchmark Donut/GOT-OCR2.0/Qwen-VL.

#### 15. DocVQA
- **Link:** https://www.docvqa.org/datasets/docvqa (RRC portal, requires free login)
- **Scale:** 50,000 questions over 12,767 document images; ~8.3GB.
- **Ground truth:** QA pairs + OCR text (OCR tokens are already provided).
- **License:** Per RRC terms (for research use, requires a free account registration on the site, no IRB needed).
- **Credibility:** ~1,086 citations (Semantic Scholar) — VERY popular, the standard benchmark for document-VQA for LMMs such as GPT-4V/Gemini/Qwen-VL that you plan to test.

#### 16. PubTables-1M
- **Link:** https://github.com/microsoft/table-transformer (Microsoft Research Open Data)
- **Scale:** ~947,642 fully annotated tables from scientific papers.
- **Ground truth:** Detailed table structure (header, cell, location), addressing the "oversegmentation" problem of previous table datasets.
- **License:** Microsoft Research Open Data license (for research use).
- **Credibility:** ~145 citations — useful if your paper includes a section on extracting tabular data from lab/clinical results in medical documents.

---

### GROUP 5 — RELATED ICDAR/ICFHR COMPETITIONS

#### 17. ICDAR 2024 Competition on Recognition and VQA on Handwritten Documents (HWD)
- **Link:** https://ilocr.iiit.ac.in/icdar_2024_hwd/
- **Characteristics:** 3 tasks — isolated word recognition, page-level recognition, VQA on handwritten documents. NOT specific to medicine, but its test set/leaderboard can be used to benchmark current HTR SOTA.

#### 18. ICDAR 2023 Competition on Indic Handwriting Text Recognition (IHTR)
- **Link:** paper DOI 10.1007/978-3-031-41679-8_25
- **Characteristics:** 18 teams registered, 8 teams submitted results — a multilingual Indian handwriting benchmark, indirectly related to the Indian prescription domain (MIRAGE above).
- **Conclusion for Group 5:** Based on this search, **there is NO official ICDAR/ICFHR track specifically dedicated to "handwritten medical prescription/clinical documents"** as of 2026 — this is a market gap worth stating as motivation in your paper ("no standard ICDAR benchmark yet exists for the medical domain, unlike receipts/invoices which already have SROIE/CORD").

---

### GROUP 6 — MEDICAL IMAGES WITH ACCOMPANYING TEXT/REPORTS REQUIRING OCR

#### 19. MIMIC-CXR (PhysioNet)
- **Link:** https://physionet.org/content/mimic-cxr/2.1.0/
- **Scale:** 377,110 chest X-ray images, 227,835 studies from 65,379 patients, with semi-structured free-text reports.
- **Ground truth:** The report is original text (not a scanned image) — but you can render the reports into images/PDFs yourself to create a synthetic OCR benchmark, or use the labeled X-ray images to test a multimodal pipeline.
- **License:** PhysioNet Credentialed Health Data License (same credentialing process as MIMIC-IV, ~1 week).
- **Credibility:** ~1,594 citations — extremely reputable, nearly the "gold standard" for image+report chest X-ray research.

#### 20. IU X-Ray / OpenI (Indiana University Chest X-rays)
- **Link:** [Kaggle mirror](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university) | original: https://openi.nlm.nih.gov/
- **Scale:** 7,470 chest X-ray images + 3,955 reports (Comparison/Indication/Findings/Impression).
- **License:** CC BY-NC-ND 4.0 — **downloadable immediately, NO credentialing required** (a major difference from MIMIC), suitable if the deadline is tight.
- **Credibility:** A classic dataset in report-generation (used in hundreds of "chest X-ray report generation" papers).

#### 21. PadChest
- **Link:** http://bimcv.cipf.es/bimcv-projects/padchest/ (requires filling out an access-request form — not an IRB, just a simple research registration form)
- **Scale:** >160,000 X-ray images, 67,000 patients, reports in **Spanish** (note: this may not be suitable if you only want English OCR, but is still useful for testing multilingual capability beyond Vietnamese).
- **License:** Per the original research group's request form (free, non-commercial).
- **Credibility:** One of the largest X-ray datasets with original reports, very heavily cited.

#### 22. ROCOv2 (Radiology Objects in COntext v2)
- **Link:** https://zenodo.org/records/8333645 | HuggingFace: https://huggingface.co/datasets/eltorio/ROCOv2-radiology
- **Scale:** 79,789 X-ray/CT/MRI images with captions + medical concepts, drawn from PubMed Open Access.
- **Ground truth:** Captions/descriptions (not scanned text, but a standard text-image pair for testing image-to-text/report generation).
- **License:** Requires signing a simple user agreement on Zenodo (no IRB needed).
- **Credibility:** Used in ImageCLEFmedical Caption 2023 — a standard international benchmark.

---

### STRATEGIC NOTES FOR THE 2-WEEK PLAN

1. **Priority group to download immediately (no credentialing needed, <1 day):** RxHandBD, Doctor's Handwritten Prescription BD (Kaggle), IAM, CVL, RIMES (HF mirror), TCGA-Reports, SROIE, FUNSD, CORD, DocVQA, PubTables-1M, IU X-Ray/OpenI, ROCOv2 — enough to start benchmarking from Day 1–2.
2. **Group requiring credentialing (start applying immediately on Day 1, wait 3–7 days in parallel):** MIMIC-CXR, MIMIC-IV-Note/MIMIC-III (PhysioNet) — still fits within 2 weeks if the application is submitted first.
3. **Group at risk/uncertain to finish in time:** n2c2/i2b2 (the DBMI portal was reported temporarily closed in mid-2026 — a backup plan is advisable), MIRAGE (the full dataset is not public — only the demo subset can be used, or it should be skipped).
4. **Notable research gap worth stating in the paper:** there is no official ICDAR/ICFHR track for "handwritten clinical/prescription documents" — a good argument for positioning your benchmark contribution.

---

**Main references** (cited during the search process): Zenodo, Mendeley Data, Kaggle, PhysioNet/MIMIC, n2c2 DBMI Portal, arXiv (SROIE 2103.10213, FUNSD 1905.13538, CORD, DocVQA 2007.00398, PubTables-1M 2110.00061, ROCOv2 2405.10004, MIRAGE 2410.09729, PadChest 1901.07441, MIMIC-CXR Scientific Data 2019), GitHub (clovaai/cord, microsoft/table-transformer, zzzDavid/ICDAR-2019-SROIE), Semantic Scholar (citation counts).
---

## 4.2. The "standard" dataset, current SOTA, and the gap for creating novelty

## RESEARCH FINDINGS: Public medical OCR/HTR datasets — gold standard, current SOTA, and the gap for a 2024–2026 benchmark

### 1. Identifying the "gold standard" dataset

#### 1.1. IAM Handwriting Database — the gold standard for HTR in general (NOT a medical dataset)
- **Origin**: Marti & Bunke, "The IAM-database: an English sentence database for offline handwriting recognition," *International Journal on Document Analysis and Recognition (IJDAR)*, vol. 5, pp. 39–46, 2002 (DOI: 10.1007/s100320200071). Distributed via the University of Bern (FKI).
- **Credibility**: ~569 citations on Google Scholar (in existence for >20 years) — this is the oldest and most heavily cited general-purpose benchmark in the HTR field (Marti & Bunke, 2002; confirmed by many 2024–2026 papers that still cite it as a baseline).
- **Important note**: IAM is **not** a medical dataset — it consists of 1,066 pages of general English handwriting (based on the LOB corpus) from ~400 writers, 82,227 words. It is used as a "common playground" to measure baseline HTR capability, and medical OCR papers often cite IAM figures to compare a model's degree of generalization, NOT because it contains medical content. → IAM should be included in the research as a **general control benchmark** (not a medical focus) to compare against a real medical dataset.

#### 1.2. RxHandBD / "Doctor's Handwritten Prescription BD dataset" — the closest existing "hands-on medical" standard
This is actually a **family of datasets** of handwritten English/Bangla prescriptions from Bangladeshi doctors, with 2 related versions:
- **Original version**: "Doctor's Handwritten Prescription BD dataset" (Kaggle, Mia, Mamun, Sajid & Ruddra, 2024) — 4,680 images, 78 medication-name classes. First introduced in: *"A Deep Neural Network Approach with Pioneering Local Dataset to Recognize Doctor's Handwritten Prescription in Bangladesh"*, IEEE, 2024 (ieeexplore.ieee.org/document/10499631) — using VGG16; this is the original paper that "gave birth" to this dataset in Bangladesh (97.1% of Bangladeshi doctors handwrite their prescriptions).
- **Expanded/repackaged version**: **RxHandBD** (Md. Masudul Islam, Zenodo, DOI 10.5281/zenodo.18478741, published 04/02/2026; also on Mendeley Data) — 5,578 word images, 1,559 unique text labels, pre-split train/test 80/20, MIT license. The Zenodo page does not explicitly state a direct link to the 2024 Kaggle version, but based on its description (handwritten Bangladeshi prescriptions) it is essentially the same line of research.
- **Credibility**: there is no peer-reviewed "original paper" specifically for RxHandBD yet (published only 02/2026, with almost no citations so far), but this data lineage has been used in ≥8 papers from 2019–2026 (see section 2.2), making it the most-reused **medically specialized** publicly available dataset, despite its small scale and not yet being as "classic" as IAM.

#### 1.3. TCGA-Reports — a standard for clinical NLP, NOT a standard for measuring OCR accuracy
- **Source**: Kefeli et al., *"TCGA-Reports: A machine-readable pathology report resource for benchmarking text-based AI models"*, *Patterns* (Cell Press), 21/02/2024, DOI: 10.1016/j.patter.2024.100933; PMID 38487800; PMC10935496.
- **Content**: 9,523 pathology reports from TCGA, covering 32 tissue/cancer types, OCR'd via AWS Textract plus post-processing, used to benchmark cancer-type classification via NLP/LLM (achieving an AU-ROC of 0.992).
- **Key point to note for your research**: TCGA-Reports **does not provide image+label pairs with word/character-level ground truth** for computing CER/WER — OCR here is only a preprocessing step to produce text for a downstream NLP task (classification). If the goal is to "benchmark OCR accuracy" (CER/WER), then TCGA-Reports is **not suitable** as the primary dataset; it is better suited to a study on "OCR as preprocessing for clinical NLP," a different direction from RxHandBD/IAM.

---

### 2. Models/methods already tested and their results (current SOTA)

#### 2.1. On IAM (general SOTA, not medicine-specific)
| Model | Type | CER (IAM) | Source |
|---|---|---|---|
| GPT-5 | VLM API | ~1.22% | CodeSOTA benchmark registry, 2026 (codesota.com/benchmark/iam) |
| Claude Opus 4.7 | VLM API | ~1.31% | as above |
| Gemini 3 | VLM API | ~1.44% | as above |
| GPT-4o | VLM API | 1.69% (03/2025) — "the point at which VLMs surpassed specialized HTR models" | as above |
| Azure Document Intelligence v4.0 | Commercial service | ~1.8% | as above |
| DTrOCR | Specialized HTR model (non-VLM) | 2.38% (WACV 2024) | as above |
| **TrOCR-Large** | Pretrained transformer | **2.89%** | Li et al., *"TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models"*, AAAI 2023 / arXiv:2109.10282 |
| CNN-BiLSTM+CTC | CNN-RNN | 3.59% CER / 9.44% WER | arXiv:2307.00664, "CNN-BiLSTM model for English Handwriting Recognition" |
| **GOT-OCR2.0** | VLM 580M | *(specific figures not clearly published in the retrieved abstract)* — **importantly: GOT-OCR2.0 uses IAM directly as TRAINING data** (together with Chinese CASIA-HWDB2 and Norwegian NorHand-v3) | Wei et al., *"General OCR Theory: Towards OCR-2.0 via a Unified End-to-end Model"*, arXiv:2409.01704 |
| Donut | OCR-free VDU | No IAM figures found — Donut is evaluated mainly on CORD and SROIE (not line-level handwriting HTR) | Kim et al., ECCV 2022, arXiv:2111.15664 |

**Important note**: GOT-OCR2.0 has been **trained directly on IAM** — so if you plan to "test GOT-OCR2.0 on IAM" as a novel contribution, it needs to be made clear that this is an evaluation on a held-out test split, not a model that has never seen IAM at all — this affects how novelty should be presented in the paper.

#### 2.2. On RxHandBD / the Bangladeshi prescription dataset lineage (medicine-specific SOTA)
| Study | Model | Result | Source |
|---|---|---|---|
| Original dataset paper (2024) | VGG16/CNN | Accuracy on "Doctor's Handwritten Prescription BD" (4,680 images, 78 classes) | IEEE 10499631, 2024 |
| Basic CRNN | CRNN (13 conv + 3 BiLSTM) | 72% accuracy | ResearchGate 336419684, "Medical Handwritten Prescription Recognition Using CRNN" |
| CRNN (different dataset) | CRNN | 95% accuracy | found via WebSearch, DOI not yet verified |
| BiLSTM + SRP data augmentation | BiLSTM | 93.0% average accuracy (max 94.5%, min 92.1%) | Scientific Reports, *"An online cursive handwritten medical words recognition system..."*, Nature, 2022, PMC8897401 |
| Attention model (F1) | CNN+Attention | 89.0% test accuracy, F1 macro/weighted = 0.88 | arXiv:2412.18199, *"Leveraging Deep Learning with Multi-Head Attention for Accurate Extraction of Medicine from Handwritten Prescriptions"*, 12/2024 |
| Multi-backbone feature-concat | Ensemble CNN | 88.44% ± 0.99% accuracy, F1 = 0.88 ± 0.01 | *Advances in Technology Innovation*, ojs.imeti.org/AITI/16426 |
| CNN Bi-LSTM + Lexicon Search | CNN-BiLSTM + dictionary post-processing | (specific result not yet extracted) | ResearchGate 375414751, 2024 |
| RxVLM | VLM contrastive learning | (a proposed new model, not an off-the-shelf benchmark) | *The Visual Computer* (Springer), 2026, DOI 10.1007/s00371-026-04420-2 |
| **Direct benchmark on RxHandBD (5,578 images)** — **closest to your idea** | Tesseract, EasyOCR, PP-OCRv5, **GLM-OCR** (0.9B VLM) | Tesseract: CER 0.785/Exact Match 2.5%; EasyOCR: CER 0.695/2.6%; PP-OCRv5: CER 0.477/WER 0.789/**21.4%** exact match; **GLM-OCR: CER 0.328 (lowest)/WER 0.801/32.6% exact match (highest)** | Non-peer-reviewed technical blog post: *"How Well Can OCR Read Doctor Handwriting in 2026?"*, DEV Community, 2026, dev.to/kaniel_outis/how-well-can-ocr-read-doctor-handwriting-in-2026-54hn |

**Main conclusion from the benchmark above**: "even the best model (GLM-OCR) only gets exactly one-third of the words correct" → not yet reliable enough for full automation, human review is still needed (per the blog post above).

#### 2.3. Most recent related work (very new, important to know to avoid duplication)
- **RxScribe Bench** (arXiv:2609.13280, submitted 08/09/2026) — a multi-axis benchmark (correctness/hallucination/engagement/robustness) for VLMs on **outpatient Indian prescriptions** (DIFFERENT from the RxHandBD dataset). No specific VLM names are stated in the abstract, only "frontier vision-language models." This is a **very new, same-topic** work (handwritten prescriptions + VLMs) — you will need to position your difference (a different dataset, a clinical-risk-based evaluation framework different from the traditional PP-OCRv5/GLM-OCR/CER-WER approach).
- **"From Handwriting to Structured Data: Benchmarking AI Digitisation of Handwritten Forms"** (arXiv:2604.16504, 14/04/2026) — benchmarks 17 frontier MLLMs (GPT-5.4, Claude Sonnet 4.6, Gemini 3.1, etc.) on **a real medical FORM sample** (not a prescription). Gemini 3.1: WER=0.50/CER=0.31 (best on free-text portions); GPT-5.4 is best for date extraction/lowest "hallucination" (6%). It does **NOT** test specialized open-source OCR-VLMs (GOT-OCR2.0, PaddleOCR-VL, DeepSeek-OCR, Qwen-VL) — it only tests closed LLMs via API.
- **OmniHandwritingOCR** (arXiv:2608.18586, CIKM 2026) — an MLLM diagnostic benchmark for general HTR (77.57K images, including handwritten math formulas), NOT medically specialized.
- **ICDAR 2024-HWD** (English/Hindi/Bangla/Telugu) and **ICDAR 2025 Indic HDR** — Indian handwriting-recognition competitions with no specific medical content (per ilocr.iiit.ac.in/icdar_2024_hwd).

---

### 3. THE GAP for creating novelty in a 2-week paper

This is the most important finding (objective 3):

**No work (peer-reviewed or preprint) has yet tested the new generation (2024–2026) of open-source VLM-OCR models on any public prescription/medical HTR dataset** — specifically:

| Model | Tested on RxHandBD/BD dataset? | Tested on IAM? | Confirming source |
|---|---|---|---|
| GOT-OCR2.0 (arXiv:2409.01704) | **NOT YET** | Used as TRAINING data (not an independent test) | Wei et al. 2024 |
| PaddleOCR-VL / -1.5 / -1.6 (arXiv:2510.14528, 2601.21957, 2606.03264) | **NOT YET** | **NOT YET** (only tested on CASIA-HWDB, GNHK, BRUSH — not IAM) | PaddleOCR-VL papers |
| DeepSeek-OCR / -OCR2 (arXiv:2510.18234, 2601.20552) | **NOT YET** | **NOT YET** (only tested on the Fox benchmark, OmniDocBench, Chinese handwriting — CER 154.81, very poor) | DeepSeek-OCR paper |
| Qwen2.5-VL (arXiv:2502.13923) / Qwen3-VL (arXiv:2511.21631) | **NOT YET** | Not clearly confirmed | Qwen technical reports |
| TrOCR | Tested on IAM (2.89% CER) but **NOT YET tested on RxHandBD/medical data** | Yes | Li et al. 2023 |
| Donut | **NOT tested on either** (only CORD/SROIE — receipts/invoices) | Unclear | Kim et al. 2022 |
| GPT-4V/Gemini/Claude (via API) | Tested on a medical FORM (2604.16504) but **NOT YET tested on RxHandBD prescriptions** | Yes (IAM leaderboard) | Multiple sources |

**→ The clearest and most feasible gap within 2 weeks**: Benchmark a set of pretrained models (Tesseract as baseline, TrOCR, Donut, GOT-OCR2.0, PaddleOCR-VL, DeepSeek-OCR, Qwen2.5-VL/Qwen3-VL, and optionally GPT-4V/Gemini via API) on **RxHandBD** (which already has a ready-made 80/20 train/test split, 5,578 images, MIT license — ready to use immediately, no IRB needed since it is public, de-identified data) and cross-check with **IAM** to measure the degree of domain-shift generalization (general in-domain vs. out-of-domain medical). The most closely related existing work (a non-peer-reviewed dev.to blog testing only 4 older engines) has not touched any of the models listed above — this is a genuine "gap," not yet "claimed" by either RxScribe Bench (a different dataset — India) or "From Handwriting to Structured Data" (a different dataset — forms, not prescriptions, testing only closed LLMs via API).

**Warning about competitive speed**: both RxScribe Bench (08/09/2026) and From Handwriting to Structured Data (04/2026) show that the topic of "VLM benchmarking on medical handwriting" is very "hot" in the second half of 2026 — so publication needs to be fast (matching your stated 2-week timeline) and clearly positioned as different: **the RxHandBD dataset (Bangladesh, real prescriptions) + a catalog of specialized open-source OCR-VLM models (not yet tested by anyone) + domain-shift cross-checking with IAM**, rather than simply repeating RxScribe Bench's multi-axis method or the closed-model set from 2604.16504.

---

### Cited sources (compiled)
1. Marti, U.-V. & Bunke, H. (2002). *The IAM-database*. IJDAR 5, 39–46. DOI: 10.1007/s100320200071
2. Kefeli, J. et al. (2024). *TCGA-Reports*. Patterns, Cell Press. DOI: 10.1016/j.patter.2024.100933; PMID 38487800
3. RxHandBD, Zenodo (2026). DOI: 10.5281/zenodo.18478741
4. "Doctor's Handwritten Prescription BD dataset", Kaggle (2024) — IEEE 10499631 (2024)
5. arXiv:2412.18199 — Multi-Head Attention for Handwritten Prescriptions (2024)
6. PMC8897401 — Scientific Reports/Nature (2022), BiLSTM+SRP
7. dev.to/kaniel_outis — "How Well Can OCR Read Doctor Handwriting in 2026?" (blog, not peer-reviewed)
8. arXiv:2609.13280 — RxScribe Bench (2026)
9. arXiv:2604.16504 — From Handwriting to Structured Data (2026)
10. arXiv:2608.18586 — OmniHandwritingOCR (2026)
11. arXiv:2409.01704 — GOT-OCR2.0 (2024)
12. arXiv:2510.14528, 2601.21957, 2606.03264 — PaddleOCR-VL series
13. arXiv:2510.18234, 2601.20552 — DeepSeek-OCR series
14. arXiv:2109.10282 — TrOCR (AAAI 2023)
15. arXiv:2111.15664 — Donut (ECCV 2022)
16. ilocr.iiit.ac.in — ICDAR 2024-HWD, ICDAR 2025 Indic HDR

*Methodological note*: some figures (particularly the exact citation count for RxHandBD/IEEE 10499631, and the specific CER figure for GOT-OCR2.0 on IAM) could not be fully retrieved via WebSearch/WebFetch within the time frame of this research — they should be re-verified directly on Google Scholar/Semantic Scholar and in the original GOT-OCR2.0 PDF before inclusion in the official paper.
---

## 4.3. Deep verification of each candidate (link, license, ground truth — directly confirmed)

## DEEP VERIFICATION REPORT: PUBLIC MEDICAL OCR DATASETS (updated 15/09/2026)

Directly accessed the Zenodo, Mendeley Data, and Kaggle pages (via browser rendering since static WebFetch could not read Kaggle's SPA), GitHub, the official FKI/Bern site (via the Wayback Machine since the original domain had a blocked TLS/connection issue), PubMed Central, and PhysioNet. Below are the results for each dataset.

---

### 1. RxHandBD — "A Handwritten Prescription Word Image Dataset"

- **Direct download link:** Zenodo DOI `10.5281/zenodo.18478741` (https://zenodo.org/records/18478741) AND Mendeley Data DOI `10.17632/dsb5r6vskg.3` (https://data.mendeley.com/datasets/dsb5r6vskg/3)
- **Number of samples:** 5,578 cropped word images, .jpg, a vocabulary of 1,559 unique words (generic drug names, brand names, dosage forms, clinical instructions). Pre-split into train/test 4,463/1,115 (80/20).
- **Ground truth:** Yes — `train_labels.csv` and `test_labels.csv` accompanying each image (not JSON/XML).
- **License — CONFLICTING between the 2 hosts:**
  - Zenodo v1 (04/02/2026): states **MIT License**, images normalized to 128×128px.
  - Mendeley v3 (20/03/2026, latest version): states **CC BY 4.0**, images normalized to 512×512px, 244 MB in size.
  - These are two different versions (different resolution, different license) of "the same" dataset — it is important to state clearly which version you used.
- **Original paper:** **NO** peer-reviewed academic paper introduces this dataset. Only dataset metadata exists (a single author: Md Masudul Islam, Bangladesh University of Business and Technology). This is a self-published dataset that has not undergone peer review.
- **Novelty/credibility risk:** v1 was created 02/03/2026, v2 on 13/03, v3 on 20/03/2026 — extremely new (only ~6 months before "today"), with very rapid version iteration, a single author, and no community/peer-review verification yet. There is one independent benchmark blog post (not peer-reviewed, dev.to, author self-described as having "no affiliation") testing Tesseract/EasyOCR/PP-OCRv5/GLM-OCR on this dataset — showing the dataset is *technically usable*, but it does **not have genuine "international credibility"** (no dataset-paper DOI citation, no official recognition from ICDAR/PhysioNet/HuggingFace).
- **Conclusion:** Downloadable immediately, with full ground truth, but there is a **high academic-credibility risk** if used as the primary dataset for a paper — it should be used as a secondary/supplementary dataset, not as the sole pillar.

---

### 2. "Doctor's Handwritten Prescription BD dataset" on Kaggle (mamun1113)

**This is a COMPLETELY DIFFERENT dataset from RxHandBD** — not the same one (the number of images, number of classes, and structure all differ):

- **Link:** https://www.kaggle.com/datasets/mamun1113/doctors-handwritten-prescription-bd-dataset
- **Number of samples:** 4,680 cropped word images, **78 medication-name classes** (different from RxHandBD's vocabulary of 1,559). Split 60% train / 20% validation / 20% test, stratified by class.
- **Ground truth:** Yes — Excel (.xlsx) and CSV word-index files, separate for each train/val/test folder.
- **License:** Kaggle states "**Database: Open Database, Contents: © Original Authors**" plus the author's note "free to use for educational and research purposes" (with an implicit non-commercial restriction).
- **Original paper — YES, peer-reviewed:** A. R. Mia, M. A.-A.-S. Chowdhury, A. A. Mamun, A. M. Ruddra, N. T. Tanny, **"A Deep Neural Network Approach with Pioneering Local Dataset to Recognize Doctor's Handwritten Prescription in Bangladesh,"** IEEE iCACCESS 2024, Dhaka. (IEEE Xplore document 10499631)
- **Reliability/credibility:** Very good within the Kaggle community — **34,700 views, 7,078 downloads, 57 upvotes**, last updated ~2 years ago (2024), with 29 public notebooks using this dataset, and a "used in a publication" badge. This is a **stable, well-verified dataset used widely by the community, with a real IEEE conference paper.**
- **Conclusion: This is the STRONGEST candidate on the entire list** — ready to use immediately, with full ground truth, a citable paper, and widespread community use.

---

### 3. IAM Handwriting Database (University of Bern)

- **Official link:** https://fki.tic.heia-fr.ch/databases/iam-handwriting-database (the original page currently has a TLS/connection error and cannot be reached directly — verified via the Wayback Machine, content confirmed authentic)
- **Number of samples (v3.0):** 657 writers, 1,539 scanned pages, 5,685 sentences, 13,353 lines, **115,320 words** — all labeled.
- **Ground truth:** PNG images (300dpi scan, 256-level grayscale) + accompanying XML metadata (segmentation + parameters) for each form/line/word.
- **Terms of use (quoted directly from the official page):** *"This database may be used for non-commercial research purpose only. If you publish material based on this database, we request you to include a reference to paper [4]"* (Marti & Bunke, IJDAR 2002). **Registration** is required before download — free, no IRB/HREC needed, just a form to identify who is using it.
- **Original paper:** U. Marti, H. Bunke, ICDAR 1999 (first introduction), IJDAR 2002 (full v3.0 description) — this is the **most classic gold-standard dataset in the HTR field**, cited thousands of times.
- **IMPORTANT — not medical data:** IAM consists of general English sentences (drawn from the LOB Corpus – newspaper/novel text), **unrelated to medicine/prescriptions**. It can only be used as an **internationally recognized general benchmark/baseline** for comparison (e.g., "model X achieves CER=5% on IAM but only 40% on medical prescriptions" — highlighting the difficulty of the medical domain), and cannot serve as the primary pillar for a medical OCR paper.
- **Conclusion:** Ready to use within 2 weeks (quick registration, no barriers), but should only serve as a **secondary benchmark/international standard baseline**, not medical data.

---

### 4. TCGA-Reports (Patterns 2024, Kefeli & Tatonetti)

- **Paper:** Kefeli J, Tatonetti N. "TCGA-Reports: A machine-readable pathology report resource for benchmarking text-based AI models." *Patterns*, Cell Press, February 2024. (PMC10935496; paper DOI: 10.1016/j.patter.2024.100965)
- **Nature of the data — CLEARLY VERIFIED per the exact question asked:** The main published dataset (`TCGA_Reports.csv.zip`, on GitHub at `jkefeli/tcga-path-reports`) is **MACHINE-READABLE TEXT (plain text), NOT scanned images** — 9,523 pathology reports, 23,909 pages, 842,134 lines of text, across 32 tissue/cancer types.
- **OCR origin:** The original PDFs from the Genomic Data Commons (GDC) → cut into page images → OCR'd with **AWS Textract** → post-processed to remove artifacts, **including removal of handwritten annotations**, which were treated as "noise." → The final version does NOT target handwriting.
- **KEY FINDING (directly answering the question "is there an accompanying original scanned image?"):** The GitHub README confirms that **the original page images used as input to Textract (`imgs_for_aws`) AND the original OCR-response files from Textract (`aws_response`) are stored SEPARATELY on Mendeley Data** (due to their large size), under the same DOI as the text version: `10.17632/hyg5xkznpx.1` (https://data.mendeley.com/datasets/hyg5xkznpx/1). In other words, **a real image-scan + OCR-result pair does exist** — but this is **printed/typed** text, not physicians' handwriting.
- **License:** CC BY 4.0 (clearly stated on both the Mendeley page and the GitHub LICENSE file).
- **File format:** CSV (text reports) on GitHub; images + response JSON/text on Mendeley (the exact file structure could not be fully verified since the Mendeley "Files" section loads via JS that did not fully render during this access attempt).
- **Alternative usage direction if an OCR problem is desired:**
  - (a) Use the existing `imgs_for_aws` + `aws_response` pair on Mendeley as an OCR problem for **printed/scanned medical text** (not handwriting) — saving the effort of creating synthetic images yourself.
  - (b) If a synthetic-data problem is still desired, the clean text can be re-rendered into simulated scanned images (adding noise, rotation, blur) as originally proposed.
- **Conclusion:** Ready to use within 2 weeks for **printed/scanned** medical-text OCR (not handwriting), with a very permissive license (CC BY 4.0), and a highly reputable 2024 *Patterns* paper.

---

### 5. PhysioNet — scanned clinical document / handwriting datasets

- Directly checked `physionet.org/about/database/` (the full listing) and the PhysioNet index page (physionet.org).
- **CONCLUSION: NO** dataset on PhysioNet focuses on scanned documents, OCR, or handwriting/prescriptions. PhysioNet is almost entirely physiological signals (ECG/EEG/PPG), medical images (X-ray, CT, fundus), and structured EHR data (MIMIC, eICU). A few datasets have "text-form radiology reports" (MIMIC-CXR, REFLACX), but that is typed text within an EHR, not OCR/scan/handwriting.
- **Recommendation:** Drop the PhysioNet direction — there is no suitable candidate.

---

### 6. CMATERdb (and related Indian/Bangladeshi datasets)

- **Link:** https://code.google.com/archive/p/cmaterdb (now archived; mirrored on TensorFlow Datasets/GitHub)
- Checked the full contents of CMATERdb (created at the CMATER Lab, Jadavpur University, Kolkata):
  - CMATERdb 3.1.1/3.2.1/3.3.1/3.4.1: **handwritten digits** (Bangla/Devanagari/Telugu/Arabic numerals), 32×32px, ~3,000-6,000 images per set, CC BY 4.0 license.
  - CMATERdb1: 150 pages of general handwritten text (100 Bangla pages, 50 Bangla-English pages), unrelated to medicine.
- **CONCLUSION: NO medical/prescription content exists anywhere in the entire CMATERdb family.** Not suitable for medical OCR research.
- **Additional note:** Found one genuinely related dataset from a Bangladesh/Kyushu University group: the **"Handwritten Medical Term Corpus"** (17,431 samples, 480 medical words — 360 English + 120 Bangla, from 39 healthcare workers), associated with a paper in *Scientific Reports* (Nature, 2022, PMC8897401) and IEEE 2021 (document 9488622). However, **no direct public download link was found** (not on Kaggle/Zenodo/GitHub) — the data appears to be used only internally by the Kyushu University research group, requiring contacting the authors to request access — **it does not meet the requirement of being "ready to use immediately within 2 weeks."**

---

### SUMMARY TABLE — READY TO USE WITHIN 2 WEEKS?

| Dataset | Downloadable now? | Full ground truth? | Clear license for research? | Credibility/paper? | Handwritten medical domain? |
|---|---|---|---|---|---|
| **Kaggle "Doctor's Handwritten Prescription BD dataset" (mamun1113)** | ✅ Yes | ✅ CSV/Excel | ✅ Open Database + research note | ✅ **Has an IEEE iCACCESS 2024 paper**, very popular (7k+ downloads) | ✅ Yes — handwritten prescriptions |
| RxHandBD (Zenodo/Mendeley) | ✅ Yes | ✅ CSV | ⚠️ MIT/CC-BY-4.0 conflict between the 2 versions | ❌ No peer-reviewed paper, too new (2026) | ✅ Yes — handwritten prescriptions |
| IAM Handwriting Database | ✅ Yes (registration required) | ✅ PNG+XML | ✅ Non-commercial, must cite | ✅✅ Gold standard (ICDAR/IJDAR) | ❌ No — general English text, NOT medical |
| TCGA-Reports (+ imgs_for_aws/aws_response) | ✅ Yes (GitHub+Mendeley) | ✅ CSV text; separate image+OCR-response | ✅ CC BY 4.0 | ✅✅ *Patterns* 2024 paper (Cell Press) | ⚠️ Medical but **printed/scanned text**, NOT handwriting (already filtered out) |
| PhysioNet (any) | ❌ Does not exist | — | — | — | — |
| CMATERdb | ✅ Yes but irrelevant | — | ✅ CC BY 4.0 | ✅ Has an IJDAR paper | ❌ No — digits/general text |

#### Recommendations for the 2-week plan
1. **Primary pillar:** Kaggle "Doctor's Handwritten Prescription BD dataset" (mamun1113) — the only dataset that has both full ground truth, a real peer-reviewed paper, and widespread, stable community use.
2. **Supplementary/medical-domain cross-check:** RxHandBD — use it additionally to increase vocabulary diversity (1,559 words vs. the Kaggle dataset's 78), but state clearly in the paper that this is a new/unpublished dataset to avoid reviewers flagging it as an "unreliable source."
3. **General/international standard baseline benchmark (non-medical):** IAM — use it to compare "a model performs well on general text but poorly on medical prescriptions," strengthening the case for the domain-specific challenge.
4. **Do not use:** PhysioNet (does not exist) and CMATERdb (unrelated to medicine) — these should be dropped from the plan.
5. TCGA-Reports is suitable if you want to extend to OCR of **printed/low-quality-scanned** medical text (a different angle from "handwriting"), making use of the ready-made image+OCR-response pair on Mendeley instead of creating synthetic images yourself.
