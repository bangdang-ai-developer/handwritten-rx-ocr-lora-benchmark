> **⚠️ SUPERSEDED — 15/09/2026.** This 12-month plan assumes access to real hospital/CCI data and requires IRB/HREC approval — incompatible with the user's requirement of "independent research, urgent within 2 weeks." See the current plan at [`05-two-week-plan-SUPERSEDED-benchmark-only.md`](05-two-week-plan-SUPERSEDED-benchmark-only.md). This file is kept only for reference/comparison.

---

# RESEARCH PLAN AND PUBLICATION ROADMAP
## Topic: An automated OCR + LLM pipeline for extracting pediatric cancer registry data

**Primary direction (chosen): Direction 1** — An OCR+LLM pipeline for Pediatric Cancer Registry Automation
**Fallback option: Direction 2 or Direction 3** — if real pediatric data cannot be obtained within the first 3-4 months, redirect the graduate student's work toward Direction 2 (clinical handwriting, using public IAM/RxHandBD data, without a heavy IRB requirement) or Direction 3 (a Vietnamese medical OCR benchmark) — these two directions share most of the same technical infrastructure (OCR/VLM, the fine-tuning process, the CER/WER/F1 evaluation design), so redirecting does not waste the effort already invested in the pipeline-building stage.

Context note: the original proposal suggested "CCIA/Children's Cancer Institute Australia" and the Australian Childhood Cancer Registry (ACCR — jointly operated by CCI/Cancer Council NSW/AIHW) as a natural data partner for this direction; if the graduate student has an institutional relationship with CCIA, this is the most feasible data-access path and should be prioritized for outreach starting in Month 1.

---

# PART 1 — THE SPECIFIC RESEARCH PLAN

## 1.1. Research questions and hypotheses

**Main research question (RQ):**
> Can a combined OCR/VLM + LLM pipeline automatically extract standardized registry variables (per the ICCC-3 classification or a national pediatric cancer registry schema) from scanned/PDF pathology reports and medical record summaries for childhood cancer, achieving accuracy sufficient to support (not fully replace) cancer registry staff?

**Sub-research questions (sub-RQ):**
- RQ2: Compared to published models for *adults* (F1 0.85–0.95), how much does performance degrade when applied directly (zero-shot/transfer) to *pediatric* cancer reports — where histopathological terminology (embryonal tumors, genetic mutations, the WHO classification for pediatric brain tumors...) differs significantly?
- RQ3: For the disease groups identified as "underrepresented" by Hands & Kavuluru (2025) (melanoma, lymphoma, pediatric cancer in general), is extraction accuracy statistically significantly lower than for common disease groups (leukemia, brain tumors)?
- RQ4: How much can a "human-in-the-loop" mechanism (flagging low-confidence cases for human review) reduce the manual workload by, while still keeping overall accuracy ≥ the clinically acceptable threshold (e.g., ≥95% for the "primary diagnosis" field)?

**Hypotheses (H):**
- **H1:** The OCR+LLM pipeline achieves F1 ≥ 0.80 at the field level for the main structured variables (ICCC code, age at diagnosis, diagnosis date), lower than the adult baseline but still substantially outperforming traditional rule-based NLP.
- **H2:** Performance on rare disease groups (pediatric melanoma, lymphoma) is statistically significantly lower than on common groups (due to out-of-distribution terminology), quantitatively confirming the gap that Hands & Kavuluru described qualitatively.
- **H3:** Lightweight fine-tuning (LoRA, a few hundred to a few thousand labeled cases) on the pediatric domain significantly improves F1 compared to the zero-shot performance of general-purpose models (GPT-4o/Gemini), narrowing the gap with the adult baseline.

## 1.2. Methodology

### a) Data

**Priority 1 — Real data (the main path):**
- Contact the Australian Childhood Cancer Registry (ACCR) / CCIA, or ANZCHOG (Australian and New Zealand Children's Haematology/Oncology Group), or a children's hospital in Vietnam (the National Children's Hospital, Ho Chi Minh City Oncology Hospital) to request a set of retrospective pediatric cancer pathology reports/medical record summaries, de-identified per the organization's standard procedure.
- Target scale: 300–800 reports (enough for lightweight LoRA fine-tuning + a statistically meaningful test set; similar studies — Tay et al. 2026 — use a scale of a few hundred to a few thousand cases).
- Labeling schema: based on ICCC-3 (International Classification of Childhood Cancer, 3rd edition) + standard COG (Children's Oncology Group) fields: primary diagnosis, morphology code, stage, age at diagnosis, anatomical site, diagnosis date, treating institution.

**Priority 2 — Synthetic data for a proof-of-concept before real data is available (reducing ethical risk/IRB waiting time):**
- Generate synthetic pathology reports using an LLM (GPT-4/Claude) based on real, already-public templates (case reports on PubMed Central, pediatric pathology textbooks), then render them as simulated scanned images (adding noise, skew, blur — augmentation) to mimic real hospital scanning conditions.
- Purpose: technically verify that the pipeline works correctly before feeding in sensitive real data — and this can also be published alongside the paper as a supplementary "synthetic benchmark" (partially addressing the lack of a public benchmark).

**Ethics and privacy:**
- Submit the HREC/IRB application right from Month 1 (this is the step with the longest delay, typically 2-4 months) — requesting a "retrospective, de-identified, waiver of consent" approval type (usually more feasible with de-identified retrospective data).
- All processing of pediatric PHI (more sensitive than adult PHI) must run on-premise (no calling commercial cloud APIs like GPT-4V directly on non-de-identified data) — using open-source VLMs/LLMs (Qwen2.5-VL, InternVL3.5, Llama) run internally, or only calling a cloud API after de-identification (masking names, IDs, exact birth dates) via a separate preprocessing step.

### b) Proposed model architecture / pipeline

A 3-stage pipeline:

1. **OCR/VLM stage (Document → Structured Text):**
   - Primary candidates: PaddleOCR-VL (lightweight, multilingual, runs well on-premise) or GOT-OCR2.0 (strong on complex layouts).
   - Compare against Tesseract (a classic OCR baseline) to quantify the benefit of modern VLMs.

2. **Field extraction stage (Text → Structured fields per the ICCC/COG schema):**
   - Zero-shot/few-shot LLM with a structured prompt (JSON schema-constrained generation) — testing both closed models (GPT-4o, Gemini — only on already-de-identified data) and fine-tuned open models (Llama-3, Qwen2.5 fine-tuned with LoRA on a few hundred labeled cases).
   - Technique: controlled chain-of-thought (extracting each field along with a "source citation" from the original text to increase auditability — important for a clinical application).

3. **Human-in-the-loop stage (Confidence scoring → Routing):**
   - The model self-scores confidence for each field (based on log-probability or self-consistency across multiple generations) → low-confidence fields are flagged for manual review by registry staff (following the Tay et al. 2026 model, which accepts ~23% needing review).

### c) Baselines for comparison

| Baseline | Role |
|---|---|
| Human (double-entry by 2 registry staff, cross-checked) | Gold standard / upper bound |
| Traditional rule-based NLP (regex + a terminology dictionary, similar to the method Yoon et al. 2022 used for 29,206 pediatric cancer reports) | The "old" baseline — measures the improvement offered by modern LLMs |
| Traditional OCR (Tesseract) + a zero-shot general-purpose LLM | The "disjointed OCR" baseline — measures the benefit of an end-to-end VLM |
| A published adult OCR+LLM pipeline (reproduced from the description in Damani et al. 2026 / Tay et al. 2026), applied zero-shot to pediatric data | Measures the "domain gap" from adults → children (directly answering RQ2) |
| The proposed model (LoRA fine-tuned on the pediatric domain) | The main method |

### d) Experimental design

- **Main experiment:** Compare all 5 pipelines on the same held-out test set, reporting results at both the field level and the full-document level (document-level exact match).
- **Stratified analysis:** Break down results by (i) common vs. rare disease groups (melanoma, lymphoma, embryonal tumors) to answer RQ3; (ii) scan quality (clear/blurry/skewed); (iii) presence/absence of interspersed handwritten fields.
- **Robustness experiment:** Add artificial noise (reduced resolution, skew angle, blurring) to measure the pipeline's sensitivity — important since multi-year retrospective data typically has inconsistent scan quality.
- **External validation (if possible — linked to Direction 4):** test the pipeline optimized at 1 institution on data from a 2nd institution (e.g., CCIA Australia ↔ a Vietnamese hospital) to measure the generalization gap — if this can be done, it is a major bonus for the paper's novelty.

### e) Evaluation metrics

- **Field-level:** Precision / Recall / F1 for each individual field (with special emphasis on the "primary diagnosis" and "morphology code" fields, as these are the most important for cancer epidemiology).
- **Document-level:** Exact match accuracy (all fields correct) and "clinically acceptable match" (allowing small deviations in unstructured fields).
- **CER/WER** for the pure OCR portion alone (separating OCR errors from semantic-extraction errors — important for knowing where errors originate).
- **Cohen's kappa** between the pipeline and 2 independent annotators (measuring reliability relative to human-to-human variation — inter-annotator agreement is a reasonable reference threshold).
- **The rate of cases requiring manual review** (% flagged for human review) and **time saved** (compared to a fully manual data-entry process) — an important practical-impact metric for an application-oriented paper.

### f) Planned ablation studies

1. Compare OCR engines (PaddleOCR-VL vs. GOT-OCR2.0 vs. Tesseract) — measuring each OCR choice's contribution to the final F1.
2. Compare LLM extraction (GPT-4o vs. Gemini vs. fine-tuned Llama vs. fine-tuned Qwen2.5-VL) — the cost/security/accuracy trade-off.
3. Zero-shot vs. few-shot vs. LoRA fine-tuning — measuring the marginal benefit at each amount of labeled data (a data-scaling curve: 0, 50, 100, 300, 500+ cases).
4. With/without the "source citation" mechanism (grounded extraction) — measuring the effect on error detection ability and the reliability of the human-in-the-loop step.
5. With/without a de-identification step before extraction — measuring the trade-off between security and accuracy (linked to Direction 6 if extending the study).

## 1.3. Timeline (12 months, divided into 6 phases)

| Phase | Timeframe | Main work |
|---|---|---|
| **1. Literature review + administrative preparation** | Months 1–2 | A small systematic review of OCR/NLP research for cancer registries (extending Hands & Kavuluru 2025); submit the HREC/IRB application; contact data partners (ACCR/CCIA or a Vietnamese children's hospital); design a detailed ICCC/COG labeling schema |
| **2. Building proof-of-concept (synthetic) data** | Months 2–3 | Generate a synthetic report set (in parallel while waiting for IRB); set up the basic technical pipeline (OCR + LLM extraction) on synthetic data; verify the correctness of the code, JSON schema, and evaluation process |
| **3. Receiving real data + gold-standard labeling** | Months 3–5 | Receive the data (if IRB approval is on schedule); 2 independent annotators label the gold-standard set (the first 100–200 cases to measure inter-annotator agreement); adjust the schema if needed |
| **4. Building the baselines + main pipeline** | Months 4–7 (running in parallel with the tail end of Phase 3) | Fully implement all 5 baselines; LoRA fine-tune the main model; run the main experiment on the full test set |
| **5. Ablation, error analysis, external validation (if applicable)** | Months 7–9 | Run all ablation studies; qualitative error analysis by rare/common disease group; if a second partner is available, run external validation |
| **6. Writing the paper + submission + peer review** | Months 9–12 | Write the full draft (starting in parallel from month 8 with preliminary results); internal review; submit to the target conference/journal; prepare responses to reviewers |

*Note:* if IRB approval extends beyond Month 4 (the most common risk — see Section 1.4), Phase 2 (synthetic) can be extended into the main content of a first paper (a proof-of-concept), with the real data reserved for a second/extended paper.

## 1.4. Main risks and mitigation plans

| Risk | Severity | Mitigation plan |
|---|---|---|
| **Unable to obtain real pediatric data** (the highest ethical barrier, given children) | High | (1) Use synthetic reports as the main study, publishing it as a "feasibility study" — still novel since no one has done this for pediatrics; (2) pivot entirely to Direction 3 (general Vietnamese medical OCR, not limited to pediatric cancer) or Direction 2 (handwriting, using public IAM/RxHandBD data); (3) use public *adult* cancer data (TCGA-Reports, if available) to write a methodology paper, clearly stating the limitation and proposing a pediatric extension as future work — exactly what Damani et al. 2026 did |
| **IRB/HREC takes longer than expected** (a timing risk, not a fully blocking risk) | Medium–High | Start submitting the application right in Month 1 (in parallel with writing the literature review); use the waiting time to complete the pipeline on synthetic data; have a fallback plan of submitting "part 1" (proof-of-concept + baseline) first, followed by "part 2" (validation on real data) later as an extension |
| **The real-data sample size is too small** (pediatric cancer is rare, with few cases/year) | Medium | Use cross-validation instead of a fixed train/test split; report confidence intervals clearly instead of just a single score; emphasize that this is a "hard, low-resource setting" as part of the scientific contribution (not hiding the limitation, but turning it into a finding) |
| **Cloud models (GPT-4o/Gemini) are not permitted on pediatric PHI under institutional policy** | Medium | The entire main pipeline runs on-premise (Qwen2.5-VL, fine-tuned Llama); the cloud API is only used on synthetic or fully de-identified data, serving as an "upper-bound reference" rather than the actual deployment solution — this is also consistent with the real-world trend noted in the aggregated literature (Neveditsin 2025) |
| **Lack of reliable ground truth** (high inter-annotator disagreement due to complex medical records) | Medium | Measure and transparently report Cohen's kappa between the 2 annotators; for disagreement cases, use a third person's judgment (adjudication) as the final gold standard; use this kappa as a "ceiling" when interpreting the model's results (not expecting the model to exceed human-to-human agreement) |
| **Results are worse than expected compared to the adult baseline** (H1 is wrong) | Low–Medium | This is still a scientifically valuable finding (quantifying the domain gap = RQ2/H2) — not a failure risk but a contribution; the ablation needs to be designed in enough detail to explain *why* (rare terminology, different layout, handwriting) rather than just reporting the numbers |

---

# PART 2 — THE PAPER-WRITING ROADMAP

## 2.1. Proposed paper structure

| Section | Suggested content (1-2 sentences) |
|---|---|
| **Abstract** | State the gap (pediatric cancer is "underrepresented" in NLP/OCR cancer registry research — no study has yet applied a modern LLM); summarize the OCR+LLM pipeline and main results (F1, compared to the adult/rule-based baseline); conclude on feasibility and limitations (the domain gap for rare disease groups). |
| **Introduction** | Open with the importance of pediatric cancer registries for epidemiology and research (every case matters, since the disease is rare); state the current situation where NLP/OCR for cancer registries is already good for adults but overlooks children (citing Hands & Kavuluru 2025); clearly state the research gap + the paper's specific contributions (listing 3 contributions). |
| **Related Work** | Split into 3 groups: (1) modern OCR/Document AI (end-to-end VLMs, GOT-OCR2.0, PaddleOCR-VL) — the technical foundation; (2) NLP/LLM for adult cancer registries (Yoon 2022, Damani 2026, Tay 2026, Wiest 2025) — the conceptual baseline; (3) the specific pediatric gap (Hands & Kavuluru 2025) — positioning the gap this paper fills. |
| **Method** | Describe the 3-stage pipeline in detail (OCR/VLM → schema-based LLM extraction → human-in-the-loop confidence routing); explain the choice of on-premise models due to pediatric PHI constraints; describe the ICCC/COG schema used for labeling. |
| **Experiments** | Describe the data (real + synthetic), the labeling/ethics process, the baselines, the evaluation metrics, and the ablation setup. |
| **Results** | A table comparing F1/CER/WER across the pipelines; a stratified analysis by rare vs. common disease group (answering RQ3/H2 — the most notable "novel" result); the ablation results. |
| **Discussion** | Interpret the adult→pediatric domain gap; discuss real-world deployment feasibility (time saved, % needing review); honestly state the limitations (sample size, one/two institutions). |
| **Conclusion** | State that this is the first study to apply a modern LLM to a pediatric cancer registry; propose a multi-center/multilingual extension direction (linking Direction 3/4 as future work — creating a natural bridge to a second paper). |

## 2.2. List of important related works to cite

*(Drawn from the aggregated literature already provided — full citations/DOIs need to be added when writing the official draft, and should be re-verified via PubMed/Google Scholar since this is an indirect summary)*

- **Hands & Kavuluru (2025)**, *AI Review* — a review of 156 papers from 2014–2024, clearly noting that pediatric cancer/melanoma/lymphoma are "underrepresented" — **the foundational citation for the research gap**.
- **Yoon et al. (2022)**, *JAMIA Open* — extraction on 29,206 pediatric cancer reports, micro-F1 0.987, but using an older (pre-LLM) method — **the most important historical baseline, must be cited and compared directly**.
- **Damani et al. (2026)**, Mayo Clinic — an OCR-LLM pipeline for multi-report PDFs, explicitly proposing a multi-specialty extension — **used as the "adult" baseline and as an anchor point for Damani's own "future work" framing**, to demonstrate that this research fills exactly the gap they identified.
- **Tay et al. (2026)**, Singapore — a pipeline accepting ~23% needing manual review — **the reference model for the human-in-the-loop design**.
- **Wiest et al. (2025)** — F1 0.85–0.95+ for adult cancer registries — **quantifying the expected "upper bound"**.
- **Nicora et al. (2026)** — stroke case report forms, open VLMs struggling with handwritten fields — **directly relevant if the paper has handwritten fields in pediatric medical records**.
- **Li et al. (2024)**, *J. Biomedical Informatics* — end-to-end TSR TEDS ~0.699 — **cite if the pathology reports contain complex tables**.
- Foundational OCR/VLM models: **GOT-OCR2.0**, **PaddleOCR-VL**, **DeepSeek-OCR**, **Qwen-VL/Qwen3-VL**, **TrOCR**, **LayoutLMv3** — cite in the Method section to justify the architecture choices.
- **OCRBench v2** — evidence for the argument that "even GPT-4o/Gemini score under 50/100 on text-localization/reasoning tasks" — used in the Introduction to argue why a specialized pipeline is needed rather than using a general-purpose VLM directly.

*Recommendation:* before submitting the draft, re-run a systematic search on PubMed/Google Scholar/Consensus with the keywords "pediatric cancer registry NLP LLM OCR" and "childhood cancer information extraction" to confirm that no directly competing work has been published while the paper was in preparation (the gap could be "closed" by another group — check periodically every 2-3 months throughout the 12-month execution).

## 2.3. Suggested target conferences/journals (ranked by fit)

1. **JAMIA Open** (Journal of the American Medical Informatics Association, open-access track) — *Best fit*: already has precedent for publishing similar OCR-LLM cancer registry studies (Yoon 2022, Tay 2026); rolling submission, no hard deadline; average review time 2-4 months; prioritizes real clinical applicability over purely technical novelty — very well suited to a paper with real data + clinical evaluation.
2. **npj Digital Medicine** (Nature portfolio) — Higher prestige, accepts the scope/systematicity of external validation; rolling submission; requires content with broad impact — suitable if a multi-center angle (linking Direction 4) is added to increase the paper's weight.
3. **Journal of Biomedical Informatics** (Elsevier) — A good technical fit (already published Li et al. 2024 on medical TSR); accepts both detailed methodology and deep ablations; rolling submission.
4. **AMIA Annual Symposium** (American Medical Informatics Association) — A prestigious conference in medical informatics; **note**: the abstract/paper deadline usually falls around March-April each year for a conference held in November — the exact date should be checked on amia.org close to the submission time, since it can shift year to year; suitable if a fast poster/short-paper publication is wanted before finishing the full journal paper.
5. **MICCAI Workshop** (e.g., a Clinical NLP/Document Analysis workshop within MICCAI) — the main deadline is usually around March for a conference held in September-October; suitable if emphasizing the computer-vision technical side rather than the pure medical-informatics side; the specific workshop needs to be checked for whether it's still running in the submission year (workshops change year to year).

*Important note:* the deadlines above are for reference only, based on recent years' practice — **the exact date must be re-checked on the conference/journal's homepage right before starting to write the final draft** (around Month 6-7 of the 12-month roadmap), since conference schedules can shift.

## 2.4. Practical advice to increase the chance of acceptance

- **A clear novelty framing right from the first sentence of the Abstract:** state directly that "this is the first study to apply a modern LLM to a pediatric cancer registry" and directly cite the statement in Hands & Kavuluru (2025) or the negative search result (no work found), so reviewers immediately see that the gap is evidence-based, not a subjective inference.
- **The baselines must be strong and diverse enough:** there must definitely be both a rule-based baseline (Yoon-2022-style) *and* a zero-shot general-purpose LLM baseline — medical-informatics reviewers very often reject papers for lacking a comparison with an "older but already well-performing" method (Yoon achieved F1 0.987, albeit with different data/techniques — it needs to be clearly explained why the numbers aren't directly comparable if the schema differs).
- **Grounded extraction (source citation) increases credibility with clinical reviewers:** for each extracted field, display the original text snippet as evidence — this helps convince reviewers that the system is auditable, an important implicit requirement in medical-informatics journals.
- **Be transparent about limitations rather than avoiding them:** since the pediatric cancer sample size will certainly be much smaller than in adult studies, proactively frame the Discussion as "quantifying the domain gap and proposing next research directions" rather than trying to prove the system has "fully solved" the problem — this approach actually increases scientific credibility.
- **Reproducibility:** publish (on GitHub, after fully removing PHI) the pipeline source code, prompt templates, and labeling schema — even when the real data cannot be made public, publishing the synthetic dataset + code makes the paper easier to accept at journals that increasingly require reproducibility (both JAMIA Open and npj Digital Medicine encourage this).
- **Prepare a "Data Availability Statement" in advance**, clearly explaining why the real data cannot be made public (pediatric PHI) but describing the process for requesting access through the registry partner — medical-informatics reviewers are familiar with this constraint and will not downgrade the paper for that reason if it is explained transparently.
- **If switching to the fallback option (Direction 2/3):** keep the entire methodological framework as-is (the 3-stage pipeline, CER/WER/F1, OCR/LLM ablation) — only the domain/data changes, so the effort already invested isn't wasted, and the direction can be switched quickly within 2-4 weeks if Month 3-4 confirms that real pediatric data cannot be obtained.
