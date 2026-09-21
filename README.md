# OCR Research Project — Overview

This set of documents has been compiled to prepare an **independent** scientific research project on OCR (Optical Character Recognition) for medical applications, targeting publication in a **Q2 journal**, using only **publicly available, internationally reputable data** (no Vietnamese-language data, no hospital/CCI collaboration required). All detailed content is in the [`docs/`](docs/) folder:

1. [Foundational knowledge of OCR](docs/01-ocr-fundamentals.md) — definitions, pipeline, classical and deep learning methods, evaluation metrics, benchmarks.
2. [History of OCR development](docs/02-history.md) — from the 1913 optophone to multimodal models in 2026.
3. [Most recent research (2023–2026)](docs/03-recent-research.md) — compiled from web search (general + medical), PubMed (15 peer-reviewed papers), and Consensus.
4. [Public, reputable medical OCR datasets](docs/04-public-medical-datasets.md) — ~22 candidates surveyed, with deep verification of download links/license/ground truth, identifying the "standard" dataset and gaps not yet exploited by anyone.
5. **[→ Q2-targeted plan, ~3–4 weeks (read this section first)](docs/05-q2-research-plan.md)** — target journal (with verified real SJR), LoRA fine-tuning design, restructuring of all phases, week/day schedule.

> `docs/archive/` contains **older, superseded** plans/research directions (tied to CCI + pediatric oncology + requiring IRB approval; and the 2-week benchmark version before the target was raised to Q2) — kept only for cross-reference; the Phase 1–2 description within it remains an unchanged foundation.

## Actual progress (continuously updated)

**Zero-shot benchmark (Phase 1–5) COMPLETE** — 5 models successfully run for real + 1 excluded with notes:

| Model | CER kaggle_rx | CER iam | Notes |
|---|---|---|---|
| **Qwen2.5-VL-3B-Instruct** | **0.434 (best)** | 1.10* | *mean skewed by an outlier — median=0.0, see [phase5](results/phase5_summary.md) |
| **GOT-OCR2.0** | 0.479 | **0.386 (best)** | required pinning `transformers==4.57.0` to run correctly, see [phase3](results/phase3_summary.md) |
| EasyOCR | 0.552 | 0.736 | |
| TrOCR-large-handwritten | 0.580 | 0.441 | highest top-1 accuracy among dedicated OCR models (79.5%) |
| Tesseract | 0.625 | 0.836 | classical baseline |
| Donut-base | 1.00 (failed) | 1.00 (failed) | a valid finding — see [phase2](results/phase2_summary.md) |
| ~~PaddleOCR-VL~~ | excluded | excluded | an upstream compatibility bug with no fix yet, see [phase4](results/phase4_summary.md) |

**Most notable finding:** on the exact target domain (prescriptions), a small general-purpose VLM (Qwen2.5-VL-3B) outperforms every dedicated OCR model — including GOT-OCR2.0, which was specifically designed for OCR.

Full raw data: [results/results_master_combined.csv](results/results_master_combined.csv) (8,260 rows).

**LoRA fine-tuning of TrOCR-large-handwritten (Phase 7–8, official configuration: r=32 after ablation) COMPLETE** —
real results, 3,120 training images, paired comparison against zero-shot (n matched 100%, Wilcoxon signed-rank):

| Dataset | CER zero-shot → fine-tuned | Exact-match | Wilcoxon p-value | Significance |
|---|---|---|---|---|
| **Kaggle-Rx test (in-domain, 780 images)** | 0.580 → **0.114** (−80.3%) | 8.1% → **68.1%** | p=6.97×10⁻¹¹¹ | Very large improvement, extremely significant |
| **IAM (out-of-domain, 400 images)** | 0.441 → **0.501** (+13.7%, WORSE) | 57.3% → 26.8% | p=3.18×10⁻⁴ | Statistically significant catastrophic forgetting (milder than r=16) |

Top-1 accuracy (mapped to 78 real drug names) after fine-tuning: **92.4%**. Full details (LoRA configuration, training history, side-by-side interpretation of the two findings) are in [results/phase7_summary.md](results/phase7_summary.md).

**Aggregate analysis (Phase 9) COMPLETE** — most important finding: after fine-tuning, TrOCR-large-handwritten
(558M) **outperforms every zero-shot model, including much larger VLMs (Qwen2.5-VL-3B, GOT-OCR2.0)** on the target
domain (CER 0.114 vs. 0.434 vs. 0.479, Wilcoxon p<10⁻⁵⁸ for both comparisons). Full details (a table of 8 models × 2
datasets, the domain-shift gap, notes on long-tail distribution/outlier robustness) are in [results/phase9_summary.md](results/phase9_summary.md), with the reproduction script at [src/analyze_aggregate.py](src/analyze_aggregate.py).

**Qualitative error analysis (Phase 10) COMPLETE** — fine-tuning substantially shifts the error distribution on the
target domain: "wild guess" (hallucination) errors drop by ~4.1x (18.6%→4.5%), but errors that "resemble a
different real drug name" (more dangerous for clinical safety because they are harder to catch manually) increase
by ~1.7x (1.5%→2.6%). On IAM, catastrophic forgetting is mainly a shift from "exact correct" to "moderate error,"
not a total collapse; a separate check shows **no systematic evidence of drug-name "leakage"** when the model reads
general handwriting (0.5% on IAM). Full details and concrete examples are in [results/phase10_summary.md](results/phase10_summary.md), with the reproduction script at [src/analyze_errors.py](src/analyze_errors.py).

**Ablation (Phase 7b) COMPLETE, conclusions applied** — ran full training for r=8/16/32 (elastic=True) +
r=16 non-elastic. **Key finding: r=16 (the original "main" configuration) is NOT the best rank** — both
r=8 and r=32 beat r=16 with strong statistical significance on kaggle_rx (CER 0.113/0.114 vs. 0.149, p<10⁻⁶), and
**r=32 is best on both criteria** (kaggle_rx AND retaining the best IAM generalization, with no trade-off) →
**r=32 was selected as the paper's official result**, and Phase 7/9/10 have been rewritten around r=32. Elastic
augmentation: no effect in-domain, with a trend (not reaching 0.05 significance) toward slightly reduced
forgetting → kept elastic=True. Full details are in [results/phase7b_ablation_summary.md](results/phase7b_ablation_summary.md).

**Next:** Phase 11 (optional, RxHandBD) → Phase 12–13 (writing + submission). Full details in [docs/05-q2-research-plan.md](docs/05-q2-research-plan.md). The entire pipeline runs through direct control of the Kaggle API (`kaggle kernels push/status/output`, no browser needed) — see [notebooks/kaggle_benchmark.py](notebooks/kaggle_benchmark.py).

## Key points (quick summary)

- **Primary dataset:** Kaggle "Doctor's Handwritten Prescription BD dataset" (`mamun1113/doctors-handwritten-prescription-bd-dataset`, handwritten prescriptions, with a peer-reviewed IEEE iCACCESS 2024 paper, open license) — used as a domain-shift control via the **IAM Handwriting Database** (`nibinv23/iam-handwriting-word-database`).
- **Benchmark novelty:** GOT-OCR2.0, PaddleOCR-VL, DeepSeek-OCR, and Qwen-VL have never been tested on any public prescription/medical-HTR dataset — but the topic is currently "hot" (2 very closely related works appeared in 04/2026 and 09/2026), so rapid publication is needed.
- **Novelty for Q2 (added 16/09/2026):** not just benchmarking existing models — will **LoRA fine-tune TrOCR-large-handwritten** on 2,808 training images, measuring both in-domain improvement (Kaggle-Rx) AND the risk of "forgetting" generalization (catastrophic forgetting on IAM) — this is a genuine methodological contribution, with evidence from several similar papers showing that fine-tuning can reduce CER by 50–80%.
- **Target journal:** **PeerJ Computer Science** (Q2 confirmed via SCImago, APC $2,155 — verified live 2026-09-21; a cheaper "Lifetime Membership" route exists at $755 one-time for authors without institutional funding — with a precedent for publishing a very similar paper — TrOCR for Urdu OCR). Backup options: *Journal of Imaging* (MDPI, most certainly Q2, review time ~20 days) and *Health Information Science and Systems* (Springer, Q2, has a free route with no APC required).
- **Realistic timeline expectation:** ~3–4 weeks for experiments + manuscript + submission — **not** acceptance/publication (journal peer review always takes additional weeks to months, beyond our control).

## Note on source reliability

Sections 1–2 (foundational knowledge, history) are widely established knowledge. Sections 3–5 rely heavily on web/PubMed/Consensus/SCImago lookups — some sources are preprints (arXiv) that have not gone through full peer review, SJR quartiles can change annually (updated around April–May), and scimagojr.com blocks bots, so the quartile figures in doc 5 were cross-verified through multiple secondary sources — **verify directly on scimagojr.com yourself before deciding to submit the final paper**.

Because the topic of "benchmarking + fine-tuning VLMs on medical handwriting" is attracting strong community interest in late 2026, it is preferable to submit to arXiv early (right after the benchmark is finished, without waiting for fine-tuning to complete) to secure priority for this contribution, then update with an extended version once the fine-tuning results are available.
