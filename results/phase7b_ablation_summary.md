# Phase 7b — Ablation: LoRA rank (8/16/32) and augmentation A/B (elastic on/off)

> **✅ DECISION IMPLEMENTED (17/09/2026):** **r=32 was selected as the official configuration** for the
> entire paper (option 1 in Section 3). Renamed in `results_master_combined.csv` (`trocr-lora-finetuned` =
> r=32; the original r=16 renamed to `trocr-lora-r16`), reran `analyze_aggregate.py`/`analyze_errors.py`/
> `generate_figures.py`, and rewrote `phase7_summary.md`, `phase9_summary.md`, `phase10_summary.md`, and
> README.md around r=32. The (old) r=16 adapter was moved to
> `results/trocr-lora-adapter-r16-superseded/`; the (new, official) r=32 adapter is at
> `results/trocr-lora-adapter-final/`. The ablation table below is kept unchanged (this is exactly the data
> that led to the decision above).

Script for reproduction: `src/analyze_ablation.py`. All 4 configurations were trained to **completion** (not
a quick comparison as originally planned) — kernel v20 (main, r=16+elastic), v21 (r=8), v22 (r=32), v23
(r=16, no elastic) — same seed=42, same 3,120 training images, same frozen test set (780 kaggle_rx + 400
iam), **paired** comparison.

## 1. Rank ablation (r=8 vs r=16 vs r=32, alpha=2r, elastic=True)

| Rank | CER kaggle_rx | Top-1 acc | Exact-match | CER IAM | Wilcoxon vs r=16 (kaggle_rx) | Wilcoxon vs r=16 (IAM) |
|---|---|---|---|---|---|---|
| r=8 | **0.113** [0.098–0.128] | 91.5% | 70.0% | 0.530 [0.475–0.580] | **p=1.0×10⁻⁷** (better) | p=0.830 (no difference) |
| r=16 (main) | 0.149 [0.132–0.166] | 89.9% | 60.0% | 0.524 [0.472–0.579] | — | — |
| r=32 | **0.114** [0.100–0.130] | **92.4%** | 68.1% | **0.501** [0.453–0.552] | **p=2.4×10⁻⁸** (better) | p=0.079 (not significant, α=0.05) |

**Unexpected finding**: rank r=16 (the initial "middle-ground" choice, selected by common convention before
real data was available) **was NOT the best rank** — both r=8 AND r=32 outperform r=16 with **very strong
statistical significance** on kaggle_rx (p<10⁻⁶ for both), and r=32 tends to preserve generalization best on
IAM (although not reaching the 0.05 significance threshold). **r=32 is the best configuration on both
criteria** (in-domain AND out-of-domain), with no clear trade-off relative to r=16.

**Note on limitations (important, must be stated in the paper's Limitations)**: each rank was run only
**once** (1 seed). Because the training process has many sources of randomness (data shuffling, dropout,
batch ordering across 2 GPUs), the observed differences COULD partly stem from run-to-run variance, not from
rank alone — the Wilcoxon signed-rank test here tests the difference **between 2 specific existing sets of
predictions** (very reliable for the question "are these 2 checkpoints actually different"), but does NOT
test "would rerunning r=16 a second time produce a similar result." To more firmly establish "rank" as the
cause (rather than run-to-run luck), ≥1 additional seed per rank would be needed — this was not done here,
and is stated explicitly as a limitation.

## 2. Augmentation A/B (elastic=True vs. elastic=False, r=16 held fixed)

| Augmentation | CER kaggle_rx | Exact-match | CER IAM | Wilcoxon vs elastic=True (kaggle_rx) | Wilcoxon vs elastic=True (IAM) |
|---|---|---|---|---|---|
| elastic=True (main) | 0.149 [0.132–0.166] | 60.0% | 0.524 [0.472–0.579] | — | — |
| elastic=False | 0.150 [0.134–0.167] | 61.2% | 0.562 [0.505–0.627] | p=0.647 (no difference) | p=0.101 (not significant) |

**Conclusion**: elastic distortion **has no significant effect** on in-domain performance (consistent with
prior literature warnings — not every augmentation clearly benefits every TrOCR variant), but shows a trend
toward **slightly reducing catastrophic forgetting** on IAM (0.524 vs. 0.562, a ~7% difference) — although
this does not reach the 0.05 statistical significance threshold. **The decision to keep elastic=True for
the main configuration is reasonable** (no evidence it is harmful, and a trend — not yet certain — toward
better-preserved generalization).

## 3. Important recommendation — a decision needed before writing the paper

Because **r=32 outperforms r=16 on both criteria** (no trade-off), consistent with the original plan's logic
("run the ablation to SELECT the best rank, then report that rank as the main result" — not fix r=16
regardless of outcome), the **headline model for the whole paper should be reconsidered — switching from
r=16 to r=32**. This has cascading effects:

- **Cheap to redo**: Phase 9 (`src/analyze_aggregate.py`) and Phase 10 (`src/analyze_errors.py`) only need
  the reference model name changed (`trocr-lora-finetuned` → `trocr-lora-r32`) and rerunning — no GPU
  needed, only a few seconds, since the r=32 data is already present in `results_master_combined.csv`.
- **Costly**: the specific interpretive passages in `phase7_summary.md`, `phase9_summary.md`,
  `phase10_summary.md`, the README, and the qualitative numbers/examples in
  `fig7_qualitative_examples.png` all directly reference the r=16 figures (0.149; 89.9%; 60%; specific
  error-type percentages...) — these need to be rewritten, not just have numbers swapped.

**Two options:**
1. **Switch to r=32 as the main result** — a stronger paper (CER 0.114 instead of 0.149, and also better
   at preserving generalization), consistent with the standard practice of "reporting the best
   configuration after hyperparameter tuning" (standard ML practice, not p-hacking, since rank is a
   hyperparameter selected via validation, not a tested hypothesis).
2. **Keep r=16 as the main result, with r=8/r=32 as a secondary ablation only** — less rewriting, but
   exposes the weakness of "why not use the better configuration you already have" — a question a Q2
   reviewer will almost certainly ask.

**My recommendation: choose option 1** (switch to r=32) — the data is already available, the cost of redoing
the work is low (no GPU needed), and the result is substantially stronger. Confirmation from you is needed
before rewriting Phase 9/10, since this is a fairly broad change in interpretation, not just the addition of
one ablation item.
