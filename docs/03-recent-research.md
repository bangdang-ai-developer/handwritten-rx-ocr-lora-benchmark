# 3. Latest Research on OCR (2023–2026)

This section synthesizes four independent sources: (3.1) a general survey via web search, (3.2) a specialized survey on medical applications via web search, (3.3) a lookup of peer-reviewed literature on PubMed, and (3.4) synthesized scientific evidence via Consensus. Each source has its own reference/citation list at the end of its subsection — the original sources (DOI/arXiv) should be cross-checked before citing them formally in the paper.

---

## 3.1. General OCR research trends (not limited to medicine)

## Literature Review: OCR / Document AI — 2024–2026 Update

### 1. Context and major trends

The 2024–2026 period marks a clear shift away from traditional OCR architectures (a decoupled pipeline: text-region detection → character recognition → post-hoc layout analysis) toward end-to-end **vision-language models (VLMs)**, often referred to as the **"OCR-free"** or **"OCR 2.0"** approach. Instead of training each module separately, the new models read document page images directly and generate structured text (Markdown/HTML/JSON), simultaneously handling character recognition, tables, mathematical formulas, layout, and semantics in a single inference pass. Industry analyses call this trend "Vision Models Replace OCR" (Extend.ai, 8/2026), further driven by traceability/compliance requirements (the EU AI Act) in the finance and insurance sectors.

### 2. Scene Text Detection & Recognition (STDR)

For text in natural scene images (scene text), 2024–2026 research continues to focus on recognizing arbitrarily shaped text, using transformer/contour-based architectures instead of rectangular boxes: for example, **PBFormer** (Polynomial Band Transformer, 2023) and contour-transformer models (CT-Net) continue to be extended. A notable trend is the use of vision-language foundation models (CLIP) as text detectors (2023–2024) and end-to-end "spotting" models that integrate reading-order estimation with dynamic sampling (2024). A new comprehensive survey (expected to be published in ScienceDirect in 2026) focuses on scene text in Indic scripts, showing that the research focus is expanding beyond English/Latin to more complex scripts.

### 3. OCR-free / VLM for Document Understanding — the central model group

This is the most active development area in 2024–2026:

- **GOT-OCR2.0** (StepFun/UCAS, 9/2024, arXiv:2409.01704) — proposes "General OCR Theory," a unified 580M-parameter model (high-compression encoder + long-context decoder) that handles every type of "optical character": ordinary text, math/chemistry formulas, tables, charts, and even sheet music; its output can be Markdown/LaTeX/SMILES.
- **Nougat** (Meta, 2023) and **Kosmos-2.5** (Microsoft, arXiv:2309.11419) — two "literate" models that convert scientific PDFs/documents into Markdown while preserving structure and formulas; Kosmos-2.5 (1.3B parameters) outperforms Vary-Base despite being much smaller, achieving the best results on MarkdownEval; the key difference is that Nougat uses LaTeX for tables while Kosmos-2.5 uses Markdown.
- **Qwen2.5-VL** (Alibaba, 2/2025, arXiv:2502.13923) and **Qwen3-VL** (arXiv:2511.21631) — surpass GPT-4o/Claude 3.5 Sonnet on many document/diagram-understanding tasks; the 7B version scores 883 on OCRBench and 94.91 on DocVQA.
- **InternVL3** (4/2025, arXiv:2504.10479) and **InternVL3.5** (8/2025, arXiv:2508.18265) — InternVL3-2B scores 88.3 on DocVQA; InternVL3.5-2B scores 89.4 on DocVQA and 70.8 on InfoVQA.
- **DeepSeek-VL2** (MoE, with Tiny/Small/Base variants of 1.0B–4.5B activated parameters) is the foundation for **DeepSeek-OCR** (10/2025, arXiv:2510.18234) — its highlight is "Contexts Optical Compression": using the image itself as a context-compression layer, cutting the number of tokens needed by 7–20× compared with plain text, achieving an Edit Distance <0.25 on OmniDocBench (near-human level). **DeepSeek-OCR 2** (1/2026, arXiv:2601.20552) — a 3B model that scores 91.09 on OmniDocBench v1.5, a new SOTA for structured document understanding.
- **MinerU / MinerU2.5** (OpenDataLab, arXiv:2604.04771) — an open-source system that splits processing into two "coarse-to-fine" stages: global layout analysis on a thumbnail image, followed by localized high-resolution recognition; strong on academic papers, textbooks, and noisy scanned documents.
- **olmOCR / olmOCR 2** (Allen Institute for AI; olmOCR launched 2/2025, olmOCR 2 released 10/2025, arXiv:2510.19817) — a fully open-source model (data, code, and weights), fine-tuned from Qwen2.5-VL-7B, using RL (GRPO) with "unit-test rewards" to improve handling of difficult formulas/tables; it has its own benchmark, **olmOCR-Bench** (7,000+ test cases, 1,400 documents).
- **PaddleOCR-VL** (Baidu, 10/2025, arXiv:2510.14528) — an ultra-compact 0.9B model (NaViT visual encoder + ERNIE-4.5-0.3B), scoring 96.33% on OmniDocBench v1.6, strong on multilingual text; the **PaddleOCR-VL-1.5/1.6** versions add further improvements for robust in-the-wild parsing.
- **dots.ocr** (rednote-hilab/Xiaohongshu, 8/2025) — a 1.7B VLM that integrates layout detection and content recognition into a single model, strong on low-resource languages; later renamed **dots.mocr** (3/2026).
- **HunyuanOCR** (Tencent, 11/2025, arXiv:2511.19575) — a lightweight 1B VLM that took first place in the ICDAR 2025 DIMT Challenge (Small Model Track) and achieves SOTA OCRBench results among sub-3B models; the **HunyuanOCR-1.5** version further improves speed.
- **GLM-OCR** (arXiv:2603.10910), **MonkeyOCR** (a "structure-recognition-relation triplet" model), **Dolphin** (parsing via "heterogeneous anchor prompting"), and **Granite-Docling** (IBM, successor to **SmolDocling**, using a Granite 3 + SigLIP2 backbone, integrated with the open-source **Docling** pipeline, arXiv:2408.09869) are other notable additions from late 2025–2026.

### 4. New benchmarks / leaderboards

- **OmniDocBench** (CVPR 2025, arXiv:2412.07626) — currently the most comprehensive benchmark for document parsing: 1,651 PDF pages, 10 document types (financial reports, printed newspapers, textbooks, handwritten notes, etc.), 5 layout types, 5 languages, annotations covering up to 28 block types and 4 span types, and support for evaluating both LaTeX/HTML for formulas and tables. It is close to being the "gold standard" for comparing MinerU, PaddleOCR-VL, dots.ocr, DeepSeek-OCR, GOT-OCR2, and others.
- **OCRBench** and **OCRBench v2** (1/2025, arXiv:2501.00321) — OCRBench v2 expands the number of tasks fourfold compared with the original, covering 31 scenarios and 10,000 human-verified question-answer pairs, focusing on known weaknesses: text localization, handwriting, and logical reasoning over text. A notable finding: most SOTA LMMs (including GPT-4o, Gemini, and Qwen) score **below 50/100**, revealing a large gap between "being able to read text" and "deeply understanding a document."
- Other derivative/supplementary benchmarks appeared in 2025–2026: **CC-OCR** and **CC-OCR v2** (ICCV 2025, arXiv:2605.03903, evaluating "literacy" in real-world document processing), **MosaicDoc** (bilingual, arXiv:2511.09919), **GlotOCRBench** (multi-script, see section 6), and **Dr. DocBench** (expert-level difficult documents, arXiv:2606.01393).

### 5. Handling complex layouts: tables, multi-column forms, low-quality scans

Table Structure Recognition remains a hard problem: surveys (ACM Computing Surveys 2024, arXiv:2211.08469; arXiv:2312.04808) point to the lack of a unified annotation/benchmark standard. **TableFormer** (IBM, the foundation for Docling) uses a CNN + transformer encoder-decoder to jointly predict HTML structure and cell bounding boxes, handling borderless tables, empty cells, and merged cells well. **PP-StructureV2** (Baidu) is a strong document-analysis system for layout, tables, and information extraction. The 2025 trend is to use large VLMs (vision-LLMs) themselves to enhance table recognition, along with new benchmarks and a "toolchain reasoner" (IJCAI 2025).

For forms and Key Information Extraction (KIE), **LayoutLMv3** remains a popular open-source baseline in 2025 (combining unified text and image masking), but it has clear limitations with handwriting. For multi-column layouts and low-quality scanned documents, studies (LlamaIndex blog, Extend.ai 2025) show that: traditional open-source OCR only reaches 88–94% accuracy when facing multi-page tables, multi-column layouts, or blurry scans; large VLMs (e.g., GPT-5.5 in some 2026 comparisons) show a better ability to preserve reading order as the number of columns increases, while some other models (Doubao) degrade sharply — indicating that generalization to complex layouts still varies widely between models.

### 6. Multilingual and low-resource-language OCR/HTR (including Vietnamese)

**GlotOCRBench** (arXiv:2604.12978) is the most important new multi-script benchmark: it evaluates 158 Unicode scripts, showing that current models are strong on Latin script but degrade severely on mid-resource scripts and fail almost completely on low-resource scripts (the best model correctly transcribes fewer than 7.7% of sentences at CER<5%). **Nayana OCR** (ACL 2025 workshop) is a framework extended for low-resource Indian languages (Odia, Punjabi, Tamil, Telugu). A dedicated survey on OCR for low-resource languages (George Mason NLP, 2024) and LLM-OCR benchmarking studies for "neglected" scripts (arXiv:2412.16119) both confirm that synthetic (rendered) data spanning tens of millions of lines of text across dozens of languages is key to improving model robustness.

For **Vietnamese**, the most notable finding is the survey **"A Survey on Vietnamese Document Analysis and Recognition: Challenges and Future Directions"** (2025, arXiv:2506.05061) — a comprehensive synthesis of language-specific challenges: complex tonal diacritics, a lack of large-scale labeled data, and a proposed direction of leveraging multimodal LLMs/VLMs. Other specific works include: a **transformer-based OCR model for Vietnamese handwriting** (Springer 2025, WER/SER ~9%/24%), a study using a **vision-based LLM for Vietnamese handwriting recognition** (Springer 2025), a digitization pipeline for **handwritten Vietnamese medical records** (text-region detection → image enhancement → transcription → self-correction), and the **ViOCRVQA** dataset (4/2024, arXiv:2404.18397, Multimedia Systems 2025) — 28,000+ images and 120,000+ question-answer pairs about text in Vietnamese images, together with the proposed VisionReader model (EM 0.4116, F1 0.6990); more recent extensions include **ViInfographicVQA** (arXiv:2512.12424) and **ViConsFormer** for Vietnamese scene-text phrases (arXiv:2410.14132). Overall, Vietnamese remains in the "mid-to-low-resource" language group in the global OCR landscape, lacking a large-scale benchmark comparable to OmniDocBench.

### 7. Historical handwriting recognition (Historical HTR)

**Transkribus** remains the leading commercial/academic platform: over 300 public models, support for 100+ languages, and a flagship "Text Titan I" model trained on more than 30 million words drawn from historical documents spanning multiple centuries. Recognized limitations: accuracy depends heavily on source quality (faded ink, staining, damaged parchment), and significant manual correction is still required. In parallel, the open-source **eScriptorium** ecosystem (built on the **Kraken** engine, using RNN + CTC) continues to be used by the digital humanities community for projects such as HTR experiments on medieval Latin manuscripts (KBLab, 6/2025) and 2025–2026 projects applied to early-modern Italian manuscripts and Devanagari script.

### 8. Efficiency / on-device (edge) OCR

The "small but powerful" trend is clearly prominent in 2025–2026: **GOT-OCR2.0** (580M), **PaddleOCR-VL** (0.9B), **dots.ocr** (1.7B), and **HunyuanOCR** (1B) are all specialized OCR VLMs under 2B parameters that achieve performance competitive with much larger models. **Moondream2** (<2B, ~1GB) is designed specifically for mobile/edge devices, scoring 61.2 on OCRBench, performing well on printed forms/tables but weakly on handwriting. **RolmOCR** is positioned for lightweight OCR deployment when GPU resources are insufficient for 30B+ models. On the compression side, INT8 quantization, weight pruning, and lightweight backbones (MobileNetV3, CRNN) continue to be used to achieve low-latency inference on CPU/ARM. A distinct and highly influential new technical direction is DeepSeek-OCR's "optical context compression" — using the image itself as a token-compression medium, cutting token cost by 7–20× compared with plain-text representation, opening a path to resource savings for large-scale document-processing applications (not just for edge use).

### 9. Overall assessment

The three biggest trends of the 2024–2026 period are: (1) convergence toward end-to-end VLM architectures replacing the traditional modular OCR pipeline; (2) a parallel race between "giant, general-purpose" models (Qwen3-VL, InternVL3.5, Gemini, GPT) and "ultra-compact, OCR-specialized" models (PaddleOCR-VL, HunyuanOCR, dots.ocr, GOT-OCR2) — the latter group often achieving better performance-per-cost for pure document-parsing tasks; and (3) a growing quality gap between resource-rich languages (English, Chinese, Latin scripts) and low-resource languages (including Vietnamese) — quantified concretely by OCRBench v2 and GlotOCRBench — showing that this remains a major research gap, particularly regarding large-scale benchmarks and synthetic training data for Vietnamese.

---

### References

1. General OCR Theory: Towards OCR-2.0 via a Unified End-to-end Model (GOT-OCR2.0), 2024 — https://arxiv.org/abs/2409.01704
2. Kosmos-2.5: A Multimodal Literate Model, 2023 — https://arxiv.org/abs/2309.11419
3. Qwen2.5-VL Technical Report, 2025 — https://arxiv.org/abs/2502.13923
4. Qwen3-VL Technical Report, 2025 — https://arxiv.org/pdf/2511.21631
5. InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models, 2025 — https://arxiv.org/pdf/2504.10479
6. InternVL3.5: Advancing Open-Source Multimodal Models, 2025 — https://arxiv.org/pdf/2508.18265
7. DeepSeek-OCR: Contexts Optical Compression, 2025 — https://arxiv.org/html/2510.18234v1 (GitHub: https://github.com/deepseek-ai/DeepSeek-OCR)
8. DeepSeek-OCR 2: Visual Causal Flow, 2026 — https://arxiv.org/pdf/2601.20552
9. MinerU2.5-Pro: Pushing the Limits of Data-Centric Document Parsing at Scale, 2026 — https://arxiv.org/pdf/2604.04771
10. olmOCR 2: Unit Test Rewards for Document OCR, 2025 — https://arxiv.org/pdf/2510.19817 (blog: https://allenai.org/blog/olmocr-2)
11. PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact VLM, 2025 — https://arxiv.org/pdf/2510.14528
12. dots.ocr: Multilingual Document Layout Parsing in a Single Vision-Language Model, 2025 — https://github.com/rednote-hilab/dots.ocr
13. HunyuanOCR Technical Report, 2025 — https://arxiv.org/abs/2511.19575
14. OmniDocBench: Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations, CVPR 2025 — https://arxiv.org/abs/2412.07626
15. OCRBench v2: An Improved Benchmark for Evaluating Large Multimodal Models on Visual Text Localization and Reasoning, 2025 — https://arxiv.org/pdf/2501.00321
16. Docling Technical Report (IBM), 2024 — https://arxiv.org/pdf/2408.09869
17. Deep Learning for Table Detection and Structure Recognition: A Survey, ACM Computing Surveys, 2024 — https://arxiv.org/pdf/2211.08469
18. GlotOCR Bench: OCR Models Still Struggle Beyond a Handful of Unicode Scripts, 2026 — https://arxiv.org/pdf/2604.12978
19. A Survey on Vietnamese Document Analysis and Recognition: Challenges and Future Directions, 2025 — https://arxiv.org/html/2506.05061
20. ViOCRVQA: Novel Benchmark Dataset and Vision Reader for VQA by Understanding Vietnamese Text in Images, 2024 — https://arxiv.org/abs/2404.18397
21. ViConsFormer: Constituting Meaningful Phrases of Scene Texts in Vietnamese Text-based VQA, 2024 — https://arxiv.org/html/2410.14132
22. Transkribus — AI platform for historical documents — https://www.transkribus.org/
23. eScriptorium — Open Source Platform for Historical Document Analysis — https://escriptorium.eu/about/
24. OCR & AI: Vision Models Replace OCR, Extend.ai, 2026 — https://www.extend.ai/resources/ocr-and-ai-vision-language-models-replacing-text-recognition

**Methodological note:** This document was compiled via WebSearch/WebFetch (without directly accessing the full text of each paper except for the Vietnamese survey), so the benchmark figures should be cross-checked against the original PDFs before being cited in formal work. Some sources on the OCRBench leaderboard as of 9/2026 (very recent model version names) have not been fully verified for reliability and have been excluded from the specific figures in this report.
---

## 3.2. OCR / Document AI applications in medicine

## Literature Review: OCR / Document AI Applications in Medicine (prioritizing 2023–2026)

### 1. Digitizing electronic health records (EHR) from scanned/handwritten documents

The majority of healthcare facilities worldwide still retain a significant proportion of paper records even after adopting EHR systems. The 2024–2025 trend shows that digitization is increasingly based on smartphone photos (rather than dedicated scanners), making OCR text "noisier" — character-recognition errors and line-break misalignment significantly harm the quality of downstream data extraction [Wang et al., 2025]. Modern AI-OCR solutions (combining OCR + NLP + deep learning) are converting handwritten/printed notes into standardized formats such as HL7 FHIR and JSON for direct integration into EHR systems. An emerging research direction is using small multimodal language models (multimodal LLMs) — Phi-4, Qwen-VL, InternVL-3.5 — to replace classical OCR (Tesseract, PaddleOCR) for noisy clinical reports, with the advantage of security (on-premise operation) suited to resource-constrained healthcare environments [Neveditsin et al., 2025]. In market terms, the global OCR market reached USD 13.95 billion in 2024 and is forecast to reach USD 46 billion by 2033, with healthcare being one of the fastest-growing sectors.

### 2. Handwritten Prescription Recognition

This is considered the hardest medical OCR problem due to the huge diversity of physicians' handwriting, uneven letter spacing, non-standard abbreviations, and constantly changing pharmaceutical terminology. A frequently cited statistic: 35.7% of handwritten prescriptions contain errors, compared with only 2.5% for electronic prescriptions — demonstrating the urgent need for automation. In terms of datasets, **RxHandBD** (Zenodo/Mendeley, 2024–2025) is a specialized dataset of 5,578 handwritten words cropped from real prescriptions, split 80/20 into train/test, with images normalized to 128×128px — used to benchmark 4 open-source OCR engines. In terms of model architecture, 2024–2025 studies use CRNN combined with CTC loss (no character segmentation required), and notably, the **Donut** model (NAVER Clova AI) has been fine-tuned specifically for handwritten prescriptions, processing image-to-text end-to-end (Hugging Face: chinmays18/medical-prescription-ocr). In a bilingual/multilingual context, the **ViLanOCR** system adapts a multilingual vision-language model to read handwritten prescriptions containing both English and Urdu (disease names, patient names, age, medication, dosage) [PMC11065407].

### 3. Digitizing pathology reports, imaging reports, and lab results

**Pathology:** Recent research has focused on using LLMs to extract structured information from pathology reports. In 2023, GPT-3.5 was evaluated on extracting 17 features from 340 breast cancer reports; in 2024, GPT-4 was tested on 5 features from colorectal cancer (TCGA) reports and 11 features from glioma reports from University College London Hospitals [arXiv:2502.12183]. The **TCGA-Reports** dataset (published 2024, *Patterns* journal) provides a machine-readable repository of pathology reports, used as a benchmark for AI models based on medical text.

**Radiology:** A systematic review combined with meta-analysis (2025, *Journal of Imaging Informatics in Medicine*, DOI: 10.1007/s10278-025-01728-8) examined 28 studies covering 421,692 entities extracted from 51,187 free-text radiology reports, yielding a pooled sensitivity of 91% (95% CI: 87–93) and specificity of 96% (93–97), with an AUC of 0.98 — showing that NLP has achieved very high performance on this type of document, although there is still a notable difference between single anatomical sites and multiple sites [Yang et al., 2025]. Another scoping review in *npj Digital Medicine* (2024, DOI: 10.1038/s41746-024-01219-0) reviewed 34 studies using LLMs to extract information from radiology reports, noting that CT/MRI and chest reports predominate, while also pointing out a common limitation: the lack of external validation, which reduces performance when applied across institutions. The **RadEx** framework (2024, arXiv:2406.15465) is an LLM-based framework specifically designed to extract structured information from radiology reports.

**Lab results:** The most representative study in this group is by Ma et al. (2023, *BMC Medical Informatics and Decision Making*, DOI: 10.1186/s12911-023-02346-6), who built a two-stage OCR + NLP pipeline for 153 paper-based lab reports from Peking University First Hospital (PKU1). The OCR module achieved an average accuracy of 0.93 across three evaluation levels; the information-extraction module (using CRF-based NER) achieved an F1 of 0.86 when extracting 4 entity types (test name, result, unit, reference range), with an inference time of only 0.78 seconds per report on a single CPU — demonstrating the practical feasibility of deployment with minimal computing resources.

### 4. Digitizing historical/archival medical records for research, including cancer registries

This area relates directly to pediatric/childhood cancer. A notable study by Yoon et al. (2022, *JAMIA Open*, DOI: 10.1093/jamiaopen/ooac049) developed a deep learning model to automatically extract information from childhood cancer pathology reports, trained and validated on 29,206 reports (patients aged 0–19) from 6 US state cancer registries, classified according to the ICCC (International Classification of Childhood Cancer) standard. The model achieved a micro-F1 of 0.987 for direct ICCC classification, "rejecting" (deferring, due to insufficient confidence to assign a code) only 14.8% of ambiguous reports — showing that AI can substantially support cancer registry staff in reading and extracting the majority of reports accurately and reliably.

A recent overview survey (Hands & Kavuluru, 2025, *Artificial Intelligence Review*, DOI: 10.1007/s10462-025-11316-5) reviewed 156 papers (2014–2024) on NLP in cancer registry operations, categorized by method (rule-based n=70, machine learning n=66, traditional deep learning n=70, transformer n=29 — sharply increasing since 2019). A notable point: **pediatric cancer, melanoma, and lymphoma are underrepresented** in existing NLP research — a clear research gap for organizations focused on childhood cancer. A more recent study (McPhaul et al., 2026, medRxiv, DOI: 10.64898/2026.03.20.26348915) compared two automated extraction platforms: **Brim Analytics** (LLM-based) and **DeepPhe** (ontology-based), tested on 330 pancreatic cancer reports and 34 breast cancer reports from Johns Hopkins Hospital. Brim Analytics achieved an average accuracy of 96.7% (pancreatic cancer) across 7 registry variables (T stage 96.4%, histologic grade 97.0%), while DeepPhe showed clear weaknesses on T stage (only 83.6% for pancreatic, 70.6% for breast) — showing that the LLM-based approach currently outperforms the ontology-driven approach for this task. In addition, the childhood cancer registry project in Switzerland (2025, medRxiv) illustrates the maintenance of a continuously updated national registry with evolving data-collection methods.

In the broader area of historical archives, 2024–2025 studies (arXiv:2510.06743, arXiv:2509.13236) evaluate LLMs for OCR of historical documents, with preliminary evidence that LLMs outperform traditional systems on handwritten/historical text, although results on LLM-based post-OCR correction remain mixed across studies.

### 5. Processing health insurance documents, informed consent forms, and medical forms

**Health insurance:** Modern AI-OCR systems now automatically extract information from CMS-1500 and UB-04 forms — patient information, insurance codes, provider details — with accuracy claimed by commercial vendors to exceed 99%, combined with NLP to check coverage eligibility and detect duplication/fraud. Notably, the initial claim-denial rate in 2024 reached 11.81%, highlighting the financial importance of accuracy right from the data-entry stage.

**Informed Consent:** A scoping review (2025, *BMC Health Services Research*) examined 27 studies (2012–2024) on digitizing the consent process, showing that digitization improves participants' understanding of procedures and risks/benefits. On the OCR side, clinical-trial documents (including ICFs and case report forms) typically contain both printed text and handwritten signatures/responses — combining OCR with layout parsing helps speed up trial data entry. The **InformGen** system (2025, arXiv:2504.00934) is an AI copilot that assists in drafting regulation-compliant consent documents, using RAG with AWS Textract to convert protocol PDFs (tens to hundreds of pages) into machine-readable Markdown before extracting relevant content.

### 6. Combining OCR + NLP to extract structured clinical information

This is the common "infrastructure" layer for all the applications above. Major commercial platforms already offer integrated solutions: **Amazon Textract** (advanced OCR, recognizing fields/tables in forms) combined with **Amazon Comprehend Medical** (NLP that detects conditions, medications, and PHI from free-text clinical notes — physician notes, discharge summaries); **Google Cloud Healthcare Natural Language API** uses Document AI (OCR) as a preprocessing step before medical NLP models perform analysis. On the academic research side, modern clinical NER pipelines (2024–2025) now have 151 pretrained clinical NER models (John Snow Labs), with a new direction of combining fine-tuned BERT with retrieval-augmented generation (RAG) using medical dictionaries for terminology normalization, as well as LLM-based zero-shot NER frameworks (2025, ACL) that require no labeled training data for each new entity type.

### 7. Challenges specific to medical OCR

**Security/privacy:** Unlike ordinary OCR, medical documents always contain PHI (protected health information), requiring a pipeline compliant with HIPAA (or an equivalent law such as Australia's Privacy Act) — including tokenization/pseudonymization strategies before storage/processing, full-device encryption for mobile capture devices, and compliant destruction of paper documents after digitization. De-identification NER models (2023, arXiv:2312.08495) play a core role in automatically anonymizing large-scale real-world clinical data.

**Diversity of physicians' handwriting:** Already discussed in section 2 — this is the leading cause of medication errors and the biggest technical barrier.

**Medical terminology/abbreviations:** General-purpose OCR is not trained on specialized medical vocabulary (active-ingredient names, ICD codes, abbreviations such as "bid," "prn"), so the error rate rises significantly compared with ordinary text.

**Multilingual records:** A low-cost, scalable processing framework (IDPA, 2025, IET Conference Proceedings, DOI: 10.1049/icp.2025.3682) combines OCR + a vision-language model to digitize bilingual Hindi-English handwritten records containing numerical data — tested on 150 Indian patient records, with the PaliGemma model achieving 74% accuracy and a CER of 13%. For Vietnamese, a study by Dinh et al. (ICIS 2023) built a Vietnamese handwriting-recognition pipeline specifically for tetanus emergency records (Hospital for Tropical Diseases), addressing challenges specific to Vietnamese (6 tones, vowel diacritics) using a BLSTM to handle the "delayed stroke" problem — achieving a Character Error Rate of 2% and a Word Error Rate of 12% on records from 30 doctors/nurses, under emergency time pressure.

**Quality of aged document images:** Long-archived documents (faded, stained, warped) make classical OCR engines (Tesseract) highly sensitive to noise — this is a key driver behind the shift toward multimodal LLMs (section 1), which tolerate noise better.

### 8. Benchmark datasets in this field

- **RxHandBD** (2024–2025): 5,578 handwritten words from real prescriptions, used for medical HTR.
- **TCGA-Reports** (2024, *Patterns*): a machine-readable repository of pathology reports, used as a benchmark for medical text-based AI models.
- **Named Clinical Entity Recognition Benchmark** (2024, arXiv:2410.05046): a multi-entity clinical NER benchmark.
- **IAM Handwriting Database**: a general-purpose handwriting dataset (13,353 lines of text, 657 writers) — still the most cited HTR benchmark in 2024–2025, although not medically specialized.
- **METATR** (2026, arXiv:2605.26712): a multilingual benchmark under development for automated text recognition, potentially applicable to multilingual medical documents.
- **KITAB-Bench** (2025, arXiv:2502.14949): a multi-domain OCR/Document Understanding benchmark for Arabic, a useful reference for low-resource-language problems similar to Vietnamese.

### Brief conclusion

The 2023–2026 period has seen a clear shift from classical OCR (Tesseract, CRNN+CTC) toward multimodal vision-language/LLM models for most applications — from handwritten prescriptions and pathology/radiology reports to cancer registries — delivering higher accuracy on noisy data but raising new challenges around external validation, computational cost, and privacy compliance. Pediatric cancer/childhood cancer registries remain identified as "underrepresented" in existing NLP research [Hands & Kavuluru, 2025] — this is both a challenge and a clear research opportunity for organizations specializing in childhood cancer.

---

### References

1. RxHandBD: A Handwritten Prescription Word Image Dataset. Zenodo, 2025. https://zenodo.org/records/18478741 ; Mendeley Data, 2024. https://data.mendeley.com/datasets/dsb5r6vskg/3
2. Pather, N., Fouché, J., Mundia, S. et al. *From Handwriting to Structured Data: Benchmarking AI Digitisation of Handwritten Forms*. arXiv:2604.16504, 2026.
3. Dinh, M.N., Le, M.T., Bui, T. et al. *A Vietnamese Handwritten Text Recognition Pipeline for Tetanus Medical Records*. ICIS 2023 Proceedings. https://aisel.aisnet.org/icis2023/ishealthcare/ishealthcare/2/
4. Fine-tuned Donut model for handwritten prescriptions. Hugging Face: chinmays18/medical-prescription-ocr.
5. Neveditsin, N., Lingras, P., Patil, S., Patil, S., Mago, V. *Compact Multimodal Language Models as Robust OCR Alternatives for Noisy Textual Clinical Reports*. arXiv:2511.13523, 2025.
6. Wang, Y., Li, Y., Qin, Y., Qian, H. *Key Coverage Matters: Semi-Structured Extraction of OCR Clinical Reports*. arXiv:2605.09440, 2025.
7. Ma, M.-W., Gao, X.-S., Zhang, Z.-Y. et al. *Extracting laboratory test information from paper-based reports*. BMC Medical Informatics and Decision Making, 2023, 23:251. DOI: 10.1186/s12911-023-02346-6 (per PubMed, PMID 37932733).
8. *TCGA-Reports: A machine-readable pathology report resource for benchmarking text-based AI models*. Patterns, 2024. DOI: 10.1016/j.patter.2024.100933.
9. *Leveraging large language models for structured information extraction from pathology reports*. arXiv:2502.12183, 2025.
10. Reichenpfader, D., Müller, H., Denecke, K. *A scoping review of large language model based approaches for information extraction from radiology reports*. npj Digital Medicine, 2024, 7:222. DOI: 10.1038/s41746-024-01219-0 (per PubMed, PMID 39182008).
11. Yang, Q., Jiang, J., Dong, X. et al. *Performance of Natural Language Processing Model in Extracting Information from Free-Text Radiology Reports: A Systematic Review and Meta-Analysis*. Journal of Imaging Informatics in Medicine, 2025, 39(4):3639–3653. DOI: 10.1007/s10278-025-01728-8 (per PubMed, PMID 41152658).
12. *RadEx: A Framework for Structured Information Extraction from Radiology Reports based on Large Language Models*. arXiv:2406.15465, 2024.
13. Yoon, H.-J., Peluso, A., Durbin, E.B. et al. *Automatic information extraction from childhood cancer pathology reports*. JAMIA Open, 2022, 5(2):ooac049. DOI: 10.1093/jamiaopen/ooac049 (per PubMed, PMID 35721398).
14. Hands, I., Kavuluru, R. *A survey of NLP methods for oncology in the past decade with a focus on cancer registry applications*. Artificial Intelligence Review, 2025, 58(10):314. DOI: 10.1007/s10462-025-11316-5 (per PubMed, PMID 40688631).
15. McPhaul, T., Kreimeyer, K., Baras, A., Botsis, T. *Automated Extraction of Cancer Registry Data from Pathology Reports: Comparing LLM-Based and Ontology-Driven NLP Platforms*. medRxiv, 2026. DOI: 10.64898/2026.03.20.26348915 (per PubMed, PMID 41929331).
16. *The childhood cancer registry in Switzerland: methods and results in 2025*. medRxiv, 2025.10.26.25338836.
17. *Digitalizing informed consent in healthcare: a scoping review*. BMC Health Services Research, 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC12225439/
18. *InformGen: An AI Copilot for Accurate and Compliant Clinical Research Consent Document Generation*. arXiv:2504.00934, 2025.
19. Amazon Comprehend Medical & Amazon Textract — AWS Documentation, 2024–2025. https://docs.aws.amazon.com/comprehend-medical/
20. Google Cloud Blog. *Medical Text Processing with the Healthcare Natural Language API*. https://cloud.google.com/blog/topics/healthcare-life-sciences/medical-text-processing-on-google-cloud
21. *A scalable, low-cost framework for multilingual intelligent document processing for continuity of care*. IET Conference Proceedings, 2025. DOI: 10.1049/icp.2025.3682.
22. *Adapting multilingual vision language transformers for low-resource Urdu optical character recognition (OCR)* (ViLanOCR). PMC11065407, 2024 (per PubMed, PMID 38699211).
23. *Building a HIPAA-Compliant OCR Pipeline: A Technical Guide*. IntuitionLabs, 2024–2025. https://intuitionlabs.ai/articles/hipaa-compliant-ocr-pipeline
24. *Beyond Accuracy: Automated De-Identification of Large Real-World Clinical Text Datasets*. arXiv:2312.08495, 2023.
25. *Named Clinical Entity Recognition Benchmark*. arXiv:2410.05046, 2024.

*Methodological note:* This overview was compiled using WebSearch/WebFetch and PubMed lookups (per PubMed) during September 2026, prioritizing 2023–2026 publications. Some sources are preprints (arXiv, medRxiv) that have not undergone full peer review — the official published version should be checked before academic citation.
---

## 3.3. Peer-reviewed scientific literature (PubMed) on OCR in Healthcare

According to PubMed (NCBI), below is the result of a literature review of peer-reviewed scientific publications on the application of OCR (Optical Character Recognition) in healthcare/clinical settings, focusing on the 2022–2026 period.

**Search methodology note:** Six queries were run on PubMed using the requested keyword combinations. Notably: the query "OCR pathology report" returned 112 results, but upon checking all of them, none were relevant — because in mitochondrial biology, "OCR" stands for "oxygen consumption rate," not optical character recognition; this entire branch was excluded. After manually filtering approximately 60 candidate papers (reading the full title and abstract), 15 papers were selected as the most directly relevant and highest quality, spread across application areas (clinical research, laboratory testing, X-ray/DXA, ophthalmology, audiology, paper medical records, and global health).

---

### LIST OF THE 15 MOST RELEVANT PAPERS

#### 1. Facilitating clinical research through automation: Combining optical character recognition with natural language processing
- **Lead author:** Hom J, et al. | **Year:** 2022 | **Journal:** Clinical Trials (Clin Trials) | **PMID:** 35608136 | DOI: 10.1177/17407745221093621
- **Summary:** A team at City of Hope combined OCR (ABBYY FineReader software) with NLP (Linguamatics i2e) to extract performance-status scores from scanned medical records of 189 diffuse large B-cell lymphoma patients. The pipeline achieved high accuracy (<1% error), cutting data-extraction time by 83% (from 108 minutes down to 18 minutes per patient for documents that did not contain the target information).
- **Research gap:** Future work is not stated explicitly, but the study was conducted on only one disease (lymphoma) at a single center — implying a need to verify generalizability to other diseases/institutions.

#### 2. Development of novel optical character recognition system to reduce recording time for vital signs and prescriptions: A simulation-based study
- **Lead author:** Soeno S, et al. | **Year:** 2024 | **Journal:** PLoS ONE | **PMID:** 38241403 | DOI: 10.1371/journal.pone.0296319
- **Summary:** A simulation trial at 3 hospitals and 2 fire stations in Japan with 38 healthcare workers (paramedics, nurses, physicians) compared OCR with manual typing when recording vital signs and prescriptions from photographs. OCR was significantly faster for recording prescriptions (18s vs. 144s for 6 medications) and had a lower error rate for both types of data.
- **Research gap:** The authors themselves note that **OCR reduces prescription-recording time but does NOT reduce vital-sign-recording time** — showing that OCR's effectiveness depends heavily on the type of data/layout complexity, a gap that needs improvement.

#### 3. Extracting laboratory test information from paper-based reports
- **Lead author:** Ma MW, et al. | **Year:** 2023 | **Journal:** BMC Medical Informatics and Decision Making | **PMID:** 37932733 | DOI: 10.1186/s12911-023-02346-6
- **Summary:** Built a two-module NLP pipeline (OCR + CRF-based information extraction) to digitize 153 paper-based lab reports at Peking University First Hospital. Average OCR accuracy was 0.93; the entity-extraction F1-score (test name, result, unit, reference range) reached 0.86, with an inference time of only 0.78s per report on a single CPU.
- **Research gap:** Evaluated only on data from a single hospital (PKU1) — multi-center testing is needed to confirm generalizability to other report formats.

#### 4. Deep learning-based NLP data pipeline for EHR-scanned document information extraction
- **Lead author:** Hsu E, et al. | **Year:** 2022 | **Journal:** JAMIA Open | **PMID:** 35702624 | DOI: 10.1093/jamiaopen/ooac045
- **Summary:** Evaluated a combination of image preprocessing + OCR (Tesseract) + NLP (including ClinicalBERT) on 955 scanned sleep-study reports to extract AHI and SaO2. The ClinicalBERT model achieved an AUROC of 0.97 (AHI) and 0.95 (SaO2), with a document accuracy of >91%.
- **Research gap:** The authors explicitly state: **"scanned documents will remain part of healthcare for many years to come, so developing NLP systems to extract key information is critical"** — while also acknowledging that resource constraints prevented testing every possible combination of preprocessing/models.

#### 5. Improving tabular data extraction in scanned laboratory reports using deep learning models
- **Lead author:** Li Y, et al. | **Year:** 2024 | **Journal:** Journal of Biomedical Informatics | **PMID:** 39393477 | DOI: 10.1016/j.jbi.2024.104735
- **Summary:** Developed an OCR pipeline specialized for tables (table detection using DETR R18/YOLOv8s, table recognition using PaddleOCR/EDD) on 650 tables from 632 scanned/faxed lab reports. DETR R18 achieved AP50=0.774; EDD achieved a TEDS score of 0.815; the overall pipeline achieved a TEDS of 0.699.
- **Research gap:** The overall TEDS score (0.699) is still modest compared with each individual module — showing a gap in the end-to-end combination of table detection and complex table-structure recognition.

#### 6. Automating the segmentation, date extraction, and classification of multi-report PDFs in outside medical records using optical character recognition and generative artificial intelligence
- **Lead author:** Damani S, et al. (Mayo Clinic) | **Year:** 2026 | **Journal:** JAMIA Open | **PMID:** 41924015 | DOI: 10.1093/jamiaopen/ooag027
- **Summary:** Combined OCR with Gemini 1.5 (LLM) to segment, classify, and extract dates from 1303 outside multi-report medical PDFs from 116 different institutions, focused on breast cancer. Achieved F1 of 0.95 (segmentation), 0.96 (classification), 0.90 (date extraction); reduced record-review time by 40% according to clinician assessment.
- **Research gap:** The authors state **explicitly in the Future Work section**: "Future work will focus on evaluating the system across additional specialties and institutions" — the current study is limited to a single specialty (breast cancer) and a single hospital system.

#### 7. Evaluating Open and Accessible Visual Language Models for Optical Character Recognition in Clinical Case Report Forms
- **Lead author:** Nicora G, et al. | **Year:** 2026 | **Journal:** Studies in Health Technology and Informatics | **PMID:** 42174982 | DOI: 10.3233/SHTI260308
- **Summary:** Compared 3 open-source Visual Language Models (Qwen 2.5, Mistral Small 3.1, Granite 3.2 Vision) for recognizing paper case report forms from a stroke trial in Italy, on 80 smartphone photos. Qwen performed best on header recognition (91%) and handwritten dates (75%); Mistral performed better on handwritten Record ID (53%) and checkboxes (80%).
- **Research gap:** The authors emphasize that **"handwritten fields remained particularly challenging"** and conclude that the study "highlight[s] both the promise and current limitations of open VLMs" — a key research gap for clinical handwriting recognition.

#### 8. Streamlining data recording through optical character recognition: a prospective multi-center study in intensive care units
- **Lead author:** Nitayavardhana P, et al. | **Year:** 2025 | **Journal:** Critical Care | **PMID:** 40102894 | DOI: 10.1186/s13054-025-05347-1
- **Summary:** A prospective multi-center study (3 countries) in ICUs, using OCR to enter data from photographs of medical device displays (ventilators, ECMO). Data completeness was 98.5%, accuracy was 96.9%, and data-entry time was reduced by 43.9% (3.4 minutes vs. 6.0 minutes per patient), with high user satisfaction (4.25/5).
- **Research gap:** The authors explicitly call for: **"Wider uptake of these systems should be encouraged to better understand their strengths and limitations in both clinical and research settings"** — more research is needed on broader deployment to understand real-world limitations.

#### 9. Vendor-Agnostic Multisite Automated Dual-Energy X-Ray Absorptiometry Reporting Using Artificial Intelligence-Based Optical Character Recognition: Impact on Workflow Efficiency and Accuracy
- **Lead author:** Lakhani P, et al. | **Year:** 2026 | **Journal:** Journal of the American College of Radiology (JACR) | **PMID:** 42595274 | DOI: 10.1016/j.jacr.2026.08.007
- **Summary:** Deployed an OCR-AI system to automatically draft DXA bone-density reports at 4 sites (2 academic, 2 community). Report-generation time dropped from 3.48 to 0.87 minutes (academic sites); turnaround time (TAT) decreased by ~4 days at community sites; numerical accuracy was ≥99.5%, and report completeness increased at community sites (100% vs. 45% previously).
- **Research gap:** No limitations stated explicitly in the abstract, but the study design applies to only one type of report (DXA) — expansion to other imaging modalities is needed.

#### 10. Ocular Biometry OCR: a machine learning algorithm leveraging optical character recognition to extract intra ocular lens biometry measurements
- **Lead author:** Salvi A, et al. (Stanford) | **Year:** 2025 | **Journal:** Frontiers in Artificial Intelligence | **PMID:** 39834877 | DOI: 10.3389/frai.2024.1428716
- **Summary:** Compared PaddleOCR and Gemini for extracting ocular biometry measurements (axial length, lens thickness, etc.) from nearly 6700 reports (Lenstar, IOL Master 500/700). Agreement scores reached 0.985–0.999; PaddleOCR-to-Annotator achieved near-perfect accuracy (0.999).
- **Research gap:** The authors themselves acknowledge **"in the absence of ground truth"** for 2 of the 3 devices (only the IOLM 500 had annotator ground truth) — a limitation on independent verifiability, requiring a more fully labeled benchmark dataset in the future.

#### 11. Optical Character Recognition-based Biometry Scanner for Easy and Efficient Biometry Calculations
- **Lead author:** Levy I, et al. | **Year:** 2026 | **Journal:** Journal of Refractive Surgery | **PMID:** 42715024 | DOI: 10.3928/1081597X-20260701-02
- **Summary:** Developed and validated a mobile application using OCR (Google Gemini 2.5) to automatically enter biometry data into an intraocular lens power calculator (ESCRS IOL calculator). A survey of 12–21 ophthalmologists gave high satisfaction scores (4.5–4.67/5); data-entry time was reduced by 48.9% compared with manual entry.
- **Research gap:** The user-evaluation sample size is small (12 physicians completed the survey) — larger multi-center studies are needed to confirm clinical reliability.

#### 12. Digitising health history: The creation, function and implementation of the Norwegian Health Archives Registry
- **Lead author:** Helstad G, et al. | **Year:** 2025 | **Journal:** Health Information Management (Health Inf Manag) | **PMID:** 41204650 | DOI: 10.1177/18333583251389095
- **Summary:** Describes a Norwegian national initiative to digitize 1.7 million paper medical records (dating back to 1875) using an OCR tool customized for Norwegian medical terminology, combined with an AI system that automatically anonymizes personal information.
- **Research gap:** The authors state explicitly: **"challenges persist in processing handwritten and historical PHRs due to OCR limitations and language-specific complexities. Key challenges include improving data quality, enhancing OCR accuracy"** — strong evidence of a gap regarding historical handwriting OCR and language-specific complexities for non-English languages.

#### 13. Developing a surveillance system for HIV pre-exposure prophylaxis (PrEP) use in pregnancy in Malawi
- **Lead author:** Kamanga F, et al. | **Year:** 2026 | **Journal:** BMC Pregnancy and Childbirth | **PMID:** 42098649 | DOI: 10.1186/s12884-026-09190-2
- **Summary:** Built a drug-safety surveillance registry for PrEP use in pregnant women in Malawi, using the "ScanForm" tool — an AI-assisted OCR application to digitize and analyze handwritten data from 30+ health facilities, integrated with WHO pharmacovigilance indicators.
- **Research gap:** No specific OCR technical limitations are mentioned, but this is a prime example of the **gap in OCR applications in low- and middle-income countries (LMICs)** — where digital infrastructure remains limited and heavily reliant on handwritten paper forms.

#### 14. Preserving medical information from doctor's prescription ensuring relation among the terminology
- **Lead author:** Datta A, et al. | **Year:** 2025 | **Journal:** Computers in Biology and Medicine | **PMID:** 39983357 | DOI: 10.1016/j.compbiomed.2025.109812
- **Summary:** Proposes an EHR system integrating YOLO (region-of-interest detection, 99.6% accuracy) and OCR to digitize handwritten/printed prescriptions, combined with a medication-name spell-correction algorithm (96% accuracy), linking medication names with dosage and manufacturer information.
- **Research gap:** The authors point out a limitation of existing methods: **"struggle with multilingual text"** — indicating a gap in multilingual prescription processing, though the proposed solution has also not been fully validated across multiple languages.

#### 15. Digitizing audiograms with deep learning: structured data extraction and pseudonymization for hearing big data
- **Lead author:** You S, et al. | **Year:** 2025 | **Journal:** Hearing Research | **PMID:** 40532492 | DOI: 10.1016/j.heares.2025.109337
- **Summary:** Built a CNN combined with OCR to digitize audiogram charts into structured data while anonymizing patient information. Accuracy was 95–98%, with processing 17.72 times faster than manual digitization (3.57s vs. 63.27s per chart).
- **Research gap:** No explicit future work is stated, but the model was trained/tested only on audiogram symbols from a single system — implying a need to evaluate generalizability across the different audiogram formats used by different devices/countries.

---

### SYNTHESIS: NOTABLE RESEARCH GAPS

From the 15 papers above, several trends and important research gaps can be drawn for shaping future research directions:

1. **Handwriting recognition remains the biggest weakness.** Both paper #7 (VLM for case report forms) and #12 (Norwegian archive) clearly state that handwritten fields remain a persistent challenge, even for the most advanced VLM/AI models. This could be the highest-priority research gap for a team seeking a novel contribution.

2. **Lack of multi-center/multi-specialty generalizability.** Many studies (papers #1, #3, #6, #15) were validated at only a single institution/specialty, and some authors (notably paper #6 - Mayo Clinic) explicitly propose expanding testing to multiple specialties/organizations in "future work."

3. **Lack of standard datasets with complete ground truth.** Paper #10 (Ocular Biometry OCR) clearly illustrates this issue, having to assess inter-model agreement rather than compare against a true gold-standard label, due to a lack of fully annotated ground truth.

4. **Gap in multilingual and low-resource-language support.** Papers #12 (Norwegian) and #14 (multilingual prescriptions) both show that current medical OCR is mainly optimized for English; other languages require dedicated customization and achieve lower accuracy.

5. **Uneven OCR effectiveness depending on data type.** Paper #2 shows an interesting paradox: OCR clearly helps with recording prescriptions (long text) but does NOT significantly improve the recording of vital signs (short numeric data, possibly due to rounding errors or complex monitor-screen layouts) — suggesting a need for deeper research into which data types OCR provides real value for.

6. **A shift from traditional OCR toward combining it with Generative AI/LLMs/VLMs** (papers #6, #7, and partly #10) — this is an emerging research direction (2025–2026) replacing the classical OCR + NLP pipeline, but open-source VLMs still have limitations in structured-output-format compliance, as paper #7 showed with the Granite model.

7. **Applications in low- and middle-income countries (LMICs) remain understudied** — paper #13 (Malawi) is one of few examples, showing a large gap in evaluating the reliability and scalability of medical OCR in resource-constrained digital infrastructure contexts.

---

**Data source:** PubMed/PubMed Central (NCBI), retrieved via the PubMed search tool on 15/09/2026. All DOIs are listed with each paper for easy lookup and citation.
---

## 3.4. Synthesized evidence from Consensus

## Scientific Evidence: OCR/AI in Digitizing Clinical Medical Records

### 1. Accuracy of OCR/AI systems when digitizing medical records

A prospective multi-center study in intensive care units (ICUs) across 3 countries found that an OCR system (trained on 868 images, validated on 469 images) achieved 98.5% data completeness and 96.9% data accuracy, while reducing data-entry time by 43.9% compared with manual entry [1]. A meta-analysis on the application of AI-OCR in the pre-analytical clerical stage of clinical laboratories found: untrained OCR accuracy (including handwriting) was 91.08%, rising to 98.4% after training on a relational/hierarchical database, and reaching 99.9–100% with additional training on form fields; when AI-OCR was combined with a human verification step, overall accuracy reached 99.99% [2]. For handwritten prescriptions, an EHR system integrating YOLO (region-of-interest detection, 99.6% accuracy) combined with OCR and a spell-correction algorithm (96% accuracy) enables comprehensive digitization and structuring of prescription information [3].

### 2. Error rates of OCR on medical records, prescriptions, and pathology reports

A simulation study at a hospital in Japan directly compared OCR with manual typing on vital-sign and prescription data: the OCR character error rate was 0% (0/1056) for vital-sign data compared with 1.32% (14/1056) for manual typing, and 0.62% (30/4814) for prescriptions compared with 1.10% (53/4814) for manual typing (p<0.001 for vital signs) — OCR was not only significantly faster for recording prescriptions (18 seconds vs. 144 seconds for a 6-medication prescription) but also had a lower error rate than manual entry [4]. Research on AI-OCR in laboratories also shows that clerical errors in the pre-analytical stage (including entering paper orders) account for as much as 65.09% of all laboratory testing errors — indicating substantial room for improvement if AI-OCR replaces manual entry [2]. The ICU study mentioned above also recorded OCR data-accuracy ranging from 95.3% to 100% depending on the center, reflecting variation based on the quality of the source photographs/devices [1].

### 3. Applications of OCR/document understanding (AI/LLM) in oncology and cancer registries

In oncology, an open-source software framework (LLM-AIx) uses an LLM to extract structured clinical entities (e.g., TNM stage) from 100 pathology reports in the TCGA dataset, running on internal hospital infrastructure to protect patient data privacy without transferring data externally [5]. Another study built a pipeline combining OCR (converting scanned image/PDF pathology reports into text) with an LLM to extract cancer diagnosis, histology, grade, and stage from 829 pathology reports and 569 progress notes across 40 cancer types (26 solid tumors, 14 hematologic malignancies); the results achieved an F1 score of ≥0.85 for most variables (for example, cancer type from pathology reports: precision 0.86, recall 0.89, F1 0.87) [6]. Most closely related to cancer registries, a study at the National Cancer Centre Singapore deployed an LLM pipeline (an internal GPT-5) to automatically extract ICD-10-AM codes, histology, laterality, and diagnosis date from unstructured clinical documents for cancer registry reporting; across two cohorts (760 patients, 859 diagnoses), overall accuracy reached 84.0–91.5%, with F1 scores for individual variables usually above 0.95, substantially shortening processing time (136–144 minutes per cohort) compared with the manual process, which previously had a delay of up to 6 months — although about 23% of cases still required human review (human-in-the-loop) [7].

**Note on pediatric cancer registries:** Within the scope of the three queries performed, I **did not find** any study applying OCR/LLM specifically to **pediatric/childhood** cancer registries. The three oncology-related studies above [5][6][7] were all conducted on adult data (TCGA, multi-cancer-type EHR, the Singapore national cancer registry).

---

## BRIEF SUMMARY

- **OCR/AI accuracy in medical records is generally quite high**, ranging from 95–99%+ depending on context: about 96.9% for ICU data captured via device photographs [1], reaching 98.4–99.99% when OCR/AI is specifically trained and includes a human verification step in a laboratory setting [2], and 96–99.6% for auxiliary steps (medication spell correction, text-region detection) in prescription digitization systems [3].
- **OCR error rates are typically lower than manual data entry**: in a direct trial, OCR had a character error rate of 0-0.62% compared with 1.1-1.32% for manual typing [4], while also saving significant data-entry time (a reduction of 44–92% depending on the task) [1][4].
- **In oncology**, systems combining OCR + LLM have demonstrated the ability to automatically extract diagnosis, histology, and TNM stage from pathology reports and clinical notes with an F1 accuracy usually ≥0.85, with some studies reaching 91–100% for specific variables [5][6][7], showing great potential to support automated cancer registries, reducing manual workload and data latency.
- **Specifically regarding pediatric cancer registries**, no direct evidence was found in these queries — this may be a notable research gap for organizations considering applying this technology to pediatric cancer registries in Vietnam/the region.
- Overall, all the studies emphasize that OCR/AI performance is highest when the system is **specifically trained for the domain** and includes a **human review/verification step** (human-in-the-loop) for complex or ambiguous cases, rather than being deployed fully automatically.

---

### References

[1] [Streamlining data recording through optical character recognition: a prospective multi-center study in intensive care units](https://consensus.app/papers/details/a65c5bc292395b4a9d2965e9729538e0/?utm_source=claude_code) (P. Nitayavardhana et al., 2025, 10 citations, *Critical Care*)

[2] [A-182 Applications of artificial intelligence optical character recognition in laboratory clerical functions offsetting staffing shortages and error reduction](https://consensus.app/papers/details/a03f77bcc1755c17856038bfdb88e9a4/?utm_source=claude_code) (L. Springer et al., 2024, 0 citations, *Clinical Chemistry*)

[3] [Preserving medical information from doctor's prescription ensuring relation among the terminology](https://consensus.app/papers/details/2d782d05ef945c7e917b7bba51361369/?utm_source=claude_code) (A. Datta et al., 2025, 9 citations, *Computers in Biology and Medicine*)

[4] [Development of novel optical character recognition system to reduce recording time for vital signs and prescriptions: A simulation-based study](https://consensus.app/papers/details/9096b96a0ffc5ce18b3329f0908166c9/?utm_source=claude_code) (S. Soeno et al., 2024, 15 citations, *PLOS ONE*)

[5] [A software pipeline for medical information extraction with large language models, open source and suitable for oncology](https://consensus.app/papers/details/ba10b7953a4155c48f3a42de5043a59b/?utm_source=claude_code) (I. Wiest et al., 2025, 23 citations, *NPJ Precision Oncology*)

[6] [Use of large language models to extract cancer diagnosis, histology, grade, and staging from unstructured electronic health records](https://consensus.app/papers/details/5034d4c85acd593ba198b3f974cb5f8c/?utm_source=claude_code) (Gayathri Namasivayam et al., 2025, 0 citations, *Journal of Clinical Oncology*)

[7] [Automated cancer data extraction using large language models: A scalable workflow for clinical documentation processing](https://consensus.app/papers/details/a3ffc59ecb8e56fcb38b9e1e2b5541e0/?utm_source=claude_code) (See Boon Tay et al., 2026, 0 citations, *Journal of Clinical Oncology*)

Create or connect a free Consensus account to return more than 3 results per search in Claude Code.: https://consensus.app/sign-up/?utm_source=claude_code&auth=claude_code