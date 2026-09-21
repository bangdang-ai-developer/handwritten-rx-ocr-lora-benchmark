# Phase 10 — Qualitative error analysis (which error types fine-tuning fixes, which new ones it creates)

> **⚠️ Update (17/09/2026):** Fully rerun with **r=32** (the official configuration after the ablation, see
> `phase7b_ablation_summary.md`) in place of r=16. The overall conclusions are unchanged, but the degree of
> the clinical-safety trade-off in Section 2 is **milder** than in the previous r=16 version.

Script for reproduction: `src/analyze_errors.py`. Error classification based on
`results_master_combined.csv`, comparing **trocr-large-handwritten (zero-shot)** with
**trocr-lora-finetuned (r=32)**, on both datasets.

## 1. Unit of analysis: 4 error types on Kaggle-Rx (closed set, 78 drug names)

| Type | Definition |
|---|---|
| `correct` | Exact string match (`exact_match`) |
| `minor_ocr_noise_still_correct_drug` | String does not match exactly, but maps to the nearest of the 78 drug names **correctly** |
| `confusable_wrong_drug` | Wrong drug, and the output **clearly resembles another real drug name** (normalized edit distance ≤0.34) — **the most clinically dangerous error type** |
| `hallucination_far_off` | Wrong drug, and the output does not resemble any real drug name |

## 2. Results — Kaggle-Rx (in-domain)

| Error type | Zero-shot | Fine-tuned (r=32) | Change | (vs. previous r=16) |
|---|---|---|---|---|
| `correct` | 8.08% | **68.08%** | +60.0 percentage points | (r=16: 60.00%) |
| `minor_ocr_noise_still_correct_drug` | 71.79% | 24.87% | −47.0 percentage points | (r=16: 30.77%) |
| `hallucination_far_off` | 18.59% | **4.49%** | −14.1 percentage points (a ~4.1x reduction) | (r=16: 5.77%, a ~3.2x reduction) |
| `confusable_wrong_drug` | 1.54% | **2.56%** | +1.0 percentage points (a ~1.7x increase) | (r=16: 3.46%, a ~2.2x increase) |

**Main interpretation:** the same trend as r=16 — fine-tuning shifts the error distribution strongly toward
"fully correct" and sharply reduces wild-guess errors, but **slightly increases** the rate of
look-alike-drug errors. However, r=32 has a **MILDER** trade-off than r=16 **on both dimensions**: it
reduces hallucination more (4.1x vs. 3.2x) AND increases confusable errors less (a 1.7x increase vs. a 2.2x
increase) — r=32 both improves in-domain performance more AND has a better safety profile than r=16, with
no trade-off between these two goals.

**Still worth flagging for the Discussion/clinical safety**: although milder than r=16, the rate of
`confusable_wrong_drug` still **increases relative to zero-shot** (1.54%→2.56%) — the underlying principle
(remaining errors tend to look "like a real drug" rather than being "clearly nonsensical" as the model gets
better) **still holds**, only the magnitude differs.

### Specific examples (fine-tuned r=32, kaggle_rx)

| reference | hypothesis | top1_pred (nearest) | CER | Type |
|---|---|---|---|---|
| Amodis | Amodin | Axodin | 0.167 | confusable_wrong_drug |
| Dinafex | Fixal | Fixal | 0.714 | confusable_wrong_drug (output exactly matches a different drug) |
| Etizin | Eitrin | Zithrin | 0.500 | confusable_wrong_drug |
| Alatrol | KKKKK | Beklo | 1.000 | hallucination_far_off |
| Bacmax | Bac | Baclon | 0.500 | hallucination_far_off |
| Aceta | Acata | Aceta | 0.200 | minor_ocr_noise_still_correct_drug |

The "Bac-" drug-name family (Bacaid, Backtone, Baclofen, Baclon, Bacmax) remains the hardest bottleneck,
consistent with the finding in the r=16 version.

## 3. Results — IAM (out-of-domain, general handwriting)

| Error level (by CER) | Zero-shot | Fine-tuned (r=32) | (vs. previous r=16) |
|---|---|---|---|
| `correct` (exact match) | 57.25% | 26.75% | (r=16: 25.75%) |
| `minor_error` (0<CER≤0.3) | 6.75% | 13.00% | (r=16: 13.50%) |
| `moderate_error` (0.3<CER≤0.7) | 12.75% | **32.75%** | (r=16: 32.00%) |
| `severe_error` (CER>0.7) | 23.25% | 27.50% | (r=16: 28.75%) |

**Interpretation unchanged**: catastrophic forgetting manifests mainly as a shift from "fully correct" to
"moderate error," not a total collapse — r=32's distribution is nearly identical to r=16's, only slightly
milder on severe_error (27.50% vs. 28.75%).

## 4. Safety check: does the fine-tuned model "leak" drug names when reading lowercase/general handwriting (IAM)?

**Result: 2/400 (0.5%)** (vs. 1/400 = 0.25% for r=16, and 0/400 at zero-shot) — still at a **very
low/negligible** level, two cases:
- the true label "at" (2 characters) was read as "Az" (matching the drug name "Az" in the vocabulary) —
  most likely a coincidental match on a short word, similar to the pattern already seen with r=16.
- the true label "reason" was read as "Nexum" — CER=0.833, a large discrepancy, possibly a genuine case of
  the model "misremembering" a drug name when faced with a long/difficult word, but with only 1/400
  occurrences this is not enough to conclude it is a systematic phenomenon.

**Conclusion unchanged**: there is no significant evidence for a "drug-name leakage" phenomenon at a scale
that would affect the overall conclusions, although the rate increased slightly (1 → 2 cases) going from
r=16 to r=32 — worth noting as a minor observation in the Limitations, not as a main finding.

## 5. Summary for the paper

1. Fine-tuning (r=32) shifts the error distribution on the target domain even more positively than r=16: a
   larger reduction in hallucination_far_off (4.1x) and a smaller increase in confusable_wrong_drug (1.7x)
   — r=32 has a better safety profile than r=16 on this specific dimension, not just a better mean CER.
2. Catastrophic forgetting on IAM: the picture is nearly unchanged from r=16 (the shift from "fully
   correct" to "moderate error" is the dominant effect, not a total collapse).
3. No evidence of statistically meaningful drug-name "leakage" was found for either r=16 or r=32.

## 6. Remaining work

- Phase 11 (optional): additional testing on RxHandBD.
- Phases 12-13: write the paper following the journal structure (PeerJ Computer Science), submit to arXiv
  first, then submit officially — synthesizing all of Phases 1-10 (including the Phase 7b ablation) into
  the Results + Discussion sections.
