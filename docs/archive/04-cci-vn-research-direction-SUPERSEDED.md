> **⚠️ SUPERSEDED — 15/09/2026.** This document assumes a pediatric cancer registry direction tied to CCI and/or Vietnamese-language OCR. The user has confirmed that this is **independent research, unrelated to CCI**, **not doing Vietnamese OCR**, and needs an **urgent 2-week plan** — these conditions make every direction here (all of which require hospital data/IRB or Vietnamese language) no longer suitable. See the current direction in [`04-public-medical-datasets.md`](../04-public-medical-datasets.md) and [`05-two-week-plan-SUPERSEDED-benchmark-only.md`](05-two-week-plan-SUPERSEDED-benchmark-only.md). This file is kept only for reference/comparison.

---

# ANALYSIS: CURRENT STATE OF OCR AND OPEN RESEARCH DIRECTIONS (MEDICAL PRIORITY)

---

## PART 1 — SUMMARY OF THE CURRENT STATE AND CURRENT LIMITATIONS

- **Architectures are shifting rapidly toward end-to-end OCR-free/VLM approaches** (GOT-OCR2.0, PaddleOCR-VL, dots.ocr, DeepSeek-OCR, Qwen3-VL...) instead of modular pipelines (detect → recognize → parse layout), reducing cumulative error but still leaving a large gap between "being able to read text" and "deep document understanding": OCRBench v2 shows that **most SOTA LMMs (including GPT-4o, Gemini) score under 50/100** on text-localization and reasoning tasks.
- **Clinical handwriting remains the biggest bottleneck, even for the most modern VLMs.** Nicora et al. (2026) tested 3 open-source VLMs on stroke case report forms and concluded that "handwritten fields remained particularly challenging"; Helstad et al. (2025, a Norwegian medical archive) also clearly noted OCR limitations with historical handwriting.
- **Complex layouts (multi-page tables, multi-column forms, blurry/skewed scans) sharply reduce accuracy**: traditional open-source OCR only reaches 88–94% on complex layouts; end-to-end Table Structure Recognition (TSR) remains modest (TEDS ~0.699, even though individual modules score higher — Li et al. 2024) and lacks a unified annotation/benchmark standard (ACM Computing Surveys 2024).
- **Specialized medical terms/abbreviations** (drug names, ICD codes, "bid", "prn"...) fall outside the training distribution of general-purpose OCR/LLMs → error rates rise significantly compared to ordinary text (Datta et al. 2025 explicitly notes that current methods "struggle with multilingual text").
- **Low-resource languages are being left significantly behind.** GlotOCRBench (2026) shows that across 158 Unicode scripts, the best model only correctly transcribes <7.7% of sentences (CER<5%) for low-resource scripts. Vietnamese falls into the "medium-to-low resource" group in global medical OCR, lacking a large-scale benchmark equivalent to OmniDocBench, despite some initial efforts (ViOCRVQA, ViConsFormer, the VDAR 2025 survey).
- **Patient data security/privacy constraints (HIPAA, the Australian Privacy Act, GDPR)** hinder the direct use of cloud APIs (GPT-4V, Gemini) for PHI, driving a trend toward small on-premise multimodal LLMs (Phi-4, Qwen-VL, InternVL3.5) — but these models typically trade off lower accuracy on noisy data (Neveditsin et al. 2025).
- **A lack of large-scale, publicly labeled data for medical documents**, especially multi-specialty/multi-center data, due to ethical barriers. Most existing research (Hom 2022, Ma 2023, Hsu 2022, You 2025...) has only been validated at **a single institution/a single condition**, lacking external validation — the original authors themselves (Damani/Mayo Clinic 2026) explicitly propose expanding to multiple specialties as "future work".
- **A particularly clear and quantifiable gap**: pediatric cancer, melanoma, and lymphoma are "underrepresented" in NLP/OCR research for cancer registries (Hands & Kavuluru, 2025, a review of 156 papers from 2014–2024); an in-depth Consensus search also **found no OCR/LLM research applied specifically to pediatric cancer registries** — this is a gap with direct negative evidence, not just inference.

---

## PART 2 — PROPOSED OPEN RESEARCH DIRECTIONS

### DIRECTION 1 [MEDICAL — HIGHEST PRIORITY] — Automated OCR+LLM pipeline for extracting pediatric cancer registry data (Pediatric Cancer Registry Automation)

- **Research question:** Can a combined OCR + LLM pipeline automatically and accurately extract standard ICCC registry variables (diagnosis, histopathology, stage, age at diagnosis...) from scanned pediatric cancer pathology reports/medical records, and how does its reliability compare to that of a human?
- **Why this is a gap:** Hands & Kavuluru (2025, *AI Review*) explicitly show that pediatric cancer is "underrepresented" across 156 NLP studies on cancer registries (2014–2024); a Consensus search (3 systematic queries) **found no direct evidence** for pediatric cancer registries, while comparable models for adults have already achieved F1 ≥0.85–0.95 (Wiest 2025; Tay et al. 2026, Singapore), and Yoon et al. (2022, JAMIA Open) demonstrated technical feasibility on 29,206 pediatric cancer reports (micro-F1 0.987), but using an older method (not a modern LLM) and only in the US.
- **Feasible approach:** A 2-stage pipeline — (1) OCR/VLM (PaddleOCR-VL or GOT-OCR2.0, run on-premise) converts scanned reports/PDFs into structured text; (2) an LLM (GPT-4/Gemini via a secure API, or a fine-tuned open model) extracts fields according to the ICCC/COG schema, with a "human-in-the-loop" step for ambiguous cases (following the Tay et al. 2026 model, which accepts ~23% needing review).
- **Data/resources:** Requires a set of real pediatric cancer pathology reports/medical records (feasible if partnering with a registry or a children's hospital — e.g., CCIA/Children's Cancer Institute Australia, or a Vietnamese hospital such as the National Children's Hospital/Oncology Hospital). **Difficulty:** high on the ethics side (IRB/HREC, pediatric PHI is more sensitive than adult PHI), but this can be reduced by using de-identified retrospective data or synthetic reports as a proof-of-concept before requesting real data.
- **Feasibility for a small team:** High — no large GPU is needed if using a commercial LLM API + open-source OCR; the main bottleneck is data access, which can be resolved through a partnership with a pediatric cancer center.
- **Suitable publication venues:** *JAMIA Open*, *npj Digital Medicine*, *Journal of Biomedical Informatics*, the AMIA conference; if emphasizing the OCR technical side: *MICCAI Workshop*, *DAS (Document Analysis Systems)*.
- **Expected contribution/novelty:** The first study (according to the current search) to apply a modern LLM to a **pediatric** cancer registry, filling a gap explicitly noted in the literature; could be extended into a public (de-identified) benchmark for the community.

### DIRECTION 2 [MEDICAL] — Clinical handwriting recognition with compact on-premise VLMs (Privacy-Preserving Clinical Handwriting Recognition)

- **Research question:** Can small open-source VLMs (≤8B, capable of running on-premise) achieve acceptable accuracy in recognizing handwritten fields on case report forms/prescriptions without needing to send PHI data to the cloud?
- **Why this is a gap:** This is the gap that is **most explicitly and repeatedly stated** across the aggregated literature — Nicora et al. (2026) concluded that handwritten fields are "particularly challenging" even for Qwen2.5/Mistral/Granite Vision; Helstad et al. (2025) and the VDAR survey (2025) both note similar limitations for Norwegian/Vietnamese.
- **Feasible approach:** Fine-tune/prompt-engineer an open VLM (Qwen2.5-VL-7B, InternVL3.5, or Granite-Docling) on synthetic + real clinical handwriting data (augmenting with simulated handwriting fonts if real data is scarce), comparing against a classic CRNN+CTC pipeline and against GPT-4V/Gemini as an upper bound.
- **Data/resources:** IAM Handwriting Database (public, not medical-specific) for pretraining; RxHandBD (prescriptions, public) for the medication domain; additional real case-report-form/note data would be needed (difficult, requires IRB) — one could start with public + simulated data before expanding to real hospital data.
- **Feasibility for a small team:** High — LoRA fine-tuning on a 7B VLM is feasible with a single 24GB GPU, no supercomputer needed; the starting data is entirely public.
- **Suitable publication venues:** *ICDAR*, *DAS*, *MICCAI (Document/Clinical NLP workshop)*, *JAMIA*.
- **Contribution/novelty:** The first systematic benchmark comparing small on-premise open VLMs vs. cloud APIs for clinical handwriting, along with practical deployment recommendations for resource-limited healthcare facilities with strict privacy requirements.

### DIRECTION 3 [MEDICAL] — A benchmark and specialized OCR/HTR model for Vietnamese-language medical documents

- **Research question:** Build a medium-scale benchmark (scanned/photographed images of Vietnamese medical records, prescriptions, test results) and evaluate/fine-tune OCR-VLM models for Vietnamese-specific characteristics (tone marks, compound vowels) in a medical context.
- **Why this is a gap:** The VDAR survey (2025, arXiv:2506.05061) confirms that Vietnamese lacks large-scale labeled data; GlotOCRBench (2026) quantifies the performance gap between resource-rich and resource-poor scripts; Dinh et al. (2023, ICIS) demonstrated feasibility on tetanus records (CER 2%, WER 12%), but with a narrow scope (1 hospital, 1 record type) — there is still no general benchmark covering multiple types of Vietnamese medical documents equivalent to OmniDocBench.
- **Feasible approach:** Collect/de-identify a diverse range of Vietnamese medical documents (printed prescriptions, test result forms, notes) from 1–2 partner hospitals; label with CER/WER; benchmark existing models (PaddleOCR-VL — already strong at multiple languages, VietOCR, Qwen-VL, GOT-OCR2.0), then fine-tune the best model with LoRA on the medical domain.
- **Data/resources:** Requires partnering with a Vietnamese hospital for real data (difficulty: medium-to-high due to Vietnamese medical privacy regulations, but more feasible than international data thanks to the research team's language/local-relationship advantage); synthetic data (rendering Vietnamese medical text onto simulated scan backgrounds) could supplement this to reduce dependence on real data in the early stages.
- **Feasibility for a small team:** High — this is a topic well-suited to a PhD/graduate thesis, with low computational cost, and language resources that play to the advantage of a Vietnamese research team.
- **Suitable publication venues:** *ICDAR*, *DAS*, *ACL/EMNLP Findings* (multilingual/low-resource track), domestic conferences (NAFOSTED/VLSP), or *Multimedia Systems* (where ViOCRVQA was published).
- **Contribution/novelty:** The first public benchmark specifically for Vietnamese medical OCR (distinct from ViOCRVQA, which is general-purpose VQA), along with a fine-tuned model serving as a new baseline for the community.

### DIRECTION 4 [MEDICAL] — Evaluating multi-center/multi-specialty generalization of an OCR-LLM pipeline (External Validation Study)

- **Research question:** How much does the performance of a published OCR+LLM pipeline (e.g., achieving high F1 at one hospital/for one condition) degrade when applied to a different institution/specialty, and what factors (form layout, scan quality, local terminology) drive that degradation?
- **Why this is a gap:** This limitation is **explicitly stated by the original authors themselves** — Damani et al. (Mayo Clinic, 2026) write clearly in their Future Work about "evaluating the system across additional specialties and institutions"; Ma et al. (2023) and Hom et al. (2022) are both limited to a single institution; Nitayavardhana et al. (2025), despite being multi-center (3 countries), still calls for "wider uptake... to better understand strengths and limitations".
- **Feasible approach:** Choose 1 published open-source pipeline (e.g., the table-extraction pipeline of Li et al. 2024, or an OCR-LLM pipeline like Damani et al.'s), reproduce it on data from ≥2 different institutions (different countries/languages if possible — e.g., Vietnam + Australia), measure the performance degradation, and analyze the errors by cause (domain shift).
- **Data/resources:** Requires at least 2 independent data sources — feasible if the team has multi-institution partnerships (e.g., CCIA Australia + a Vietnamese hospital); medium difficulty since only medium-scale data (a few hundred documents per institution) is needed for external validation, with no large training dataset required.
- **Feasibility for a small team:** Medium-to-high — the work is mainly experimental/evaluative (no need to train a new model from scratch), suitable for a graduate student with inter-institutional partnerships.
- **Suitable publication venues:** *npj Digital Medicine* (already has precedent for this type of scoping study), *Journal of Biomedical Informatics*, *JAMIA Open*.
- **Contribution/novelty:** The first systematic empirical evidence quantifying the "generalization gap" of a medical OCR-LLM pipeline across multiple institutions — a question raised but not yet directly answered in the existing literature.

### DIRECTION 5 [MEDICAL] — End-to-end Table Structure Recognition for complex lab/pathology reports

- **Research question:** How can the gap be closed between individual module performance (table detection, table recognition) and end-to-end performance (currently only TEDS ~0.699, even though the individual parts reach AP50=0.774 and TEDS=0.815 — Li et al. 2024) on low-quality scanned/faxed medical tables?
- **Why this is a gap:** Li et al. (2024, *J. Biomedical Informatics*) explicitly point out this gap; the ACM Computing Surveys (2024) review confirms the lack of a unified benchmark/annotation standard for TSR in general.
- **Feasible approach:** Design a "glue module" between detection and recognition — for example, using an attention mechanism that shares features between the 2 stages instead of a disconnected pipeline, or experimenting with an end-to-end VLM model (a Donut/Pix2Struct-style table model) trained directly on (table image, HTML) pairs to avoid cumulative error.
- **Data/resources:** Li et al.'s dataset (650 tables, 632 lab reports) — would require contacting the authors for reuse/extension, or building a similar set from public data (general-purpose PubTables-1M + fine-tuning on the medical domain). Difficulty: medium, since public general-purpose table data can be leveraged for pretraining.
- **Feasibility for a small team:** Medium — requires some experience with vision architectures (TableFormer-like), but does not require large computing infrastructure.
- **Suitable publication venues:** *ICDAR*, *DAS*, *Journal of Biomedical Informatics*.
- **Contribution/novelty:** A direct improvement on a limitation already clearly quantified in the literature, with an existing baseline for comparison.

### DIRECTION 6 [MEDICAL] — A de-identification framework integrated into the OCR pipeline (Privacy-by-Design OCR for PHI)

- **Research question:** Can an OCR/VLM pipeline be designed that "de-identifies at the source" (detecting and masking PHI right within the OCR step, before the text leaves the secure environment) without significantly reducing the accuracy of extracting the clinical information that needs to be retained?
- **Why this is a gap:** Several studies (Helstad 2025 — Norway; You 2025 — audiograms) have integrated de-identification but evaluated it separately from OCR performance; no study has yet quantified the **trade-off** between the degree of de-identification and extraction accuracy within a single, unified pipeline.
- **Feasible approach:** Combine an image-region-level PHI detection model (bounding-box level NER on the layout, similar to CRAFT but for "sensitive regions" instead of "text regions") with selective OCR — only extracting/storing non-sensitive fields, masking PHI fields before logging/saving.
- **Data/resources:** Synthetic data (synthetic PHI inserted into already de-identified real-record templates) can be used to avoid ethical issues during algorithm development, needing real data only at the final evaluation step (easier to get approval for, since the purpose is privacy protection).
- **Feasibility for a small team:** High — this direction can be implemented almost entirely with synthetic data + a small real-data set for final validation.
- **Suitable publication venues:** *JAMIA*, the *AMIA Annual Symposium*, healthcare security conferences (if a dedicated track exists), *npj Digital Medicine*.
- **Contribution/novelty:** An integrated "privacy-by-design" framework rather than post-hoc de-identification — a direction distinct from existing de-identification methods (which typically process the text only after it is already complete).

### DIRECTION 7 [MEDICAL — LMIC] — Low-cost OCR/Document AI for handwritten medical records in resource-limited settings

- **Research question:** Can an OCR system running on an ordinary mobile device (no dedicated GPU needed) reliably digitize multilingual handwritten medical records at lower-tier/remote healthcare facilities (Vietnam, Southeast Asia)?
- **Why this is a gap:** Kamanga et al. (2026, Malawi, "ScanForm") is one of very few examples in LMIC settings; the IDPA framework (2025) for bilingual Hindi-English records only achieves 74% accuracy, CER 13% — showing significant room for improvement and a clear research gap in regions with limited digital infrastructure, especially since no similar study exists yet for the Vietnam/Southeast Asia context.
- **Feasible approach:** Use ultra-compact OCR/VLM models (PaddleOCR-VL 0.9B, HunyuanOCR 1B, Moondream2) with INT8 quantization to run on smartphones/edge devices, with field trials at commune/district health stations.
- **Data/resources:** Requires partnering with grassroots healthcare facilities (health stations, district hospitals) to collect real photographed records; high difficulty in logistics (fieldwork) but low in computing/technical barriers, and it has a clear social benefit, making it easier to gain consent/support from public health authorities.
- **Feasibility for a small team:** Medium — requires field partnerships but no large computing infrastructure (precisely the strength of the "on-device" direction).
- **Suitable publication venues:** *DAS*, *ICDAR*, *Digital Health*, global health conferences (Global Health Informatics).
- **Contribution/novelty:** The first field study (according to the current literature) on edge-device medical OCR for the Southeast Asian context, filling an LMIC gap beyond the existing Africa/South Asia coverage.

### DIRECTION 8 [GENERAL — for comparison] — Optical Context Compression for long, multi-page medical records

- **Research question:** Can DeepSeek-OCR's "optical context compression" technique (using a page image as a token-compression medium, reducing tokens 7–20x compared to plain text) be applied to process multi-page medical records (multi-report PDFs) more efficiently, allowing an LLM to "read" an entire long record without exceeding the context limit?
- **Why this is a gap:** DeepSeek-OCR (10/2025) and DeepSeek-OCR 2 (1/2026) are very new techniques, verified so far only on the general-purpose OmniDocBench, and **no study has yet applied them to the medical domain** — while the medical "multi-report PDF" problem (Damani et al. 2026, Mayo Clinic) is exactly the type of long, multi-page document this technique targets.
- **Feasible approach:** Apply/compare DeepSeek-OCR (or a similar technique) against a traditional OCR+LLM pipeline on multi-page medical records, measuring both accuracy and computational cost (token count, time, API cost).
- **Data/resources:** Public general-purpose data (OmniDocBench) can be used first to reproduce the technique, then tested on a small set of real/simulated medical records; low difficulty in the early stage since the model is already open-source.
- **Feasibility for a small team:** High — this is mainly reapplying an already-published technique to a new domain, well-suited as a quick comparative/benchmark study.
- **Suitable publication venues:** *MICCAI Workshop*, *DAS*, *EMNLP Findings* (computational-efficiency/long-context track).
- **Contribution/novelty:** The first study evaluating "optical context compression" for long medical documents — a cost-efficiency contribution with high practical value for large-scale deployment.

### DIRECTION 9 [GENERAL — for comparison] — A systematic comparison of ultra-compact dedicated VLM-OCR models vs. large general-purpose VLMs on medical data

- **Research question:** On the same new medical dataset (e.g., combined with Direction 3), do "ultra-compact dedicated OCR" models (PaddleOCR-VL 0.9B, HunyuanOCR 1B, GOT-OCR2.0 580M) outperform large general-purpose VLMs (Qwen3-VL, Gemini, GPT) in cost/performance ratio, especially under a hospital's limited resource conditions?
- **Why this is a gap:** Observation #2 in the 2024–2026 overview points to a "parallel race" between these two groups of models, but **there is no direct comparison yet in the medical domain** — most existing benchmarks (OmniDocBench, OCRBench v2) are general-purpose and do not reflect the specifics of clinical documents (terminology, medical form layouts, handwriting).
- **Feasible approach:** A controlled empirical benchmark: the same input (prescriptions, lab forms, scanned reports), measuring CER/WER, field-extraction F1, inference time, cost (if using an API) or VRAM (if on-premise).
- **Data/resources:** Data from Direction 3 (Vietnamese) or public English-language medical data (RxHandBD, TCGA-Reports) can be reused; low difficulty since no new model training is needed, just comparative inference.
- **Feasibility for a small team:** Very high — this is a computationally light "benchmark study," well-suited as a first paper for a new graduate student.
- **Suitable publication venues:** *DAS*, *ICDAR (short paper/benchmark track)*, *Journal of Imaging Informatics in Medicine*.
- **Contribution/novelty:** An empirically grounded practical recommendation for healthcare facilities choosing an OCR model to deploy — filling the gap between academic research (general-purpose benchmarks) and real deployment needs (the specific medical domain).

---

## PART 3 — RANKING THE TOP 3 BEST DIRECTIONS

### 🥇 #1 — Direction 1: An OCR+LLM pipeline for pediatric cancer registries
**Rationale:** This is the direction with the **strongest gap evidence** in the entire literature (explicitly stated by Hands & Kavuluru 2025 and confirmed by Consensus's negative search result), and it also has the **clearest social/clinical impact** (improving the speed and accuracy of pediatric cancer registration — a rare disease group with scarce data, where every case is valuable for epidemiological research). In terms of feasibility, the underlying techniques (OCR+LLM extraction) have already been proven on adults with high performance (F1 0.85–0.95+), so technical risk is low — the main challenge is data access, which can be resolved through institutional partnerships.

### 🥈 #2 — Direction 3: A Vietnamese medical OCR/HTR benchmark and model
**Rationale:** This direction combines **two independently confirmed gaps in the literature** (low-resource language, per the GlotOCRBench/VDAR survey, and the lack of a specialized medical benchmark) — creating dual novelty while staying focused rather than spread thin. It has a **natural competitive advantage** for the research team (language, local relationships with Vietnamese hospitals), very high technical feasibility (no need for massive data; can start by fine-tuning an existing open-source model such as PaddleOCR-VL), and is a topic very well suited to a graduate thesis (clear, and can be broken into stages: build data → benchmark → fine-tune → publish).

### 🥉 #3 — Direction 2: Clinical handwriting recognition with small on-premise VLMs
**Rationale:** This is the gap **repeated most often and most consistently** across the literature sources (Nicora 2026, Helstad 2025, VDAR 2025, the aggregated survey) — showing that this is a core, unresolved problem, widely recognized by the international community (making it easy to position the contribution and easy for reviewers to accept its urgency). It simultaneously addresses both the technical problem (handwriting) and a pressing practical problem (medical data security, the on-premise trend) — the two biggest areas of concern in the field right now. Computational feasibility is high (LoRA fine-tuning on a 7B VLM, 1 GPU), and the starting data is already public (IAM, RxHandBD).

**Note on choosing a specific topic:** Direction 1 and Direction 3 can **combine** naturally if the research team has relationships with both the Vietnamese healthcare system and an international pediatric cancer organization (e.g., the Children's Cancer Institute) — forming a strong interdisciplinary topic: "OCR/LLM for pediatric cancer registries in a multilingual/multinational context," simultaneously filling the gap on patient population (pediatric) and the gap on language (Vietnamese) — two independent gaps that complement each other very well for a highly novel paper.
