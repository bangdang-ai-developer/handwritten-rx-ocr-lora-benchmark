"""Full statistical reporting for the paper's paired comparisons.

Written to satisfy PeerJ CS's explicit statistical-reporting policy, which requires the
test performed, the reason for choosing it, the test statistic, sample size, degrees of
freedom, the exact p-value, effect sizes, and multiple-comparison correction where
multiple tests are performed. The earlier scripts (analyze_aggregate.py,
analyze_ablation.py) reported p-values only.

Design decisions made explicit here because reviewers will ask:

* Test: Wilcoxon signed-rank on per-image CER. Chosen over a paired t-test because
  per-image CER is bounded below at 0, heavily right-skewed, and (for the general VLMs)
  has extreme outliers above 1.0 -- see phase9_summary.md Sec.4. Rank-based inference is
  robust to exactly that shape.
* Degrees of freedom: not defined for Wilcoxon signed-rank; we report n (pairs retained
  after dropping zero differences, per Wilcoxon's standard handling) in its place.
* Effect size: matched-pairs rank-biserial correlation, r = (W+ - W-) / (W+ + W-),
  computed from the signed ranks directly rather than back-derived from the test
  statistic. Bounded [-1, 1]; sign follows (model A - model B).
* Multiple comparisons: Holm-Bonferroni, applied *within* each pre-specified family
  rather than across all tests, because the two families answer different questions
  (cross-model benchmark vs. within-method ablation) and pooling them would over-correct
  the family a reader actually cares about. Families are fixed below, not chosen after
  seeing p-values.

Run from repo root: python src/analyze_statistics.py
Writes results/phase11_statistics_full.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

CSV = "results/results_master_combined.csv"

# Pre-specified comparison families. Each entry: (model_a, model_b, dataset, label).
FAMILY_BENCHMARK = [
    ("trocr-large-handwritten", "trocr-lora-finetuned", "kaggle_rx", "zero-shot vs fine-tuned (in-domain)"),
    ("trocr-large-handwritten", "trocr-lora-finetuned", "iam", "zero-shot vs fine-tuned (out-of-domain)"),
    ("trocr-lora-finetuned", "qwen2.5-vl-3b", "kaggle_rx", "fine-tuned vs Qwen2.5-VL-3B"),
    ("trocr-lora-finetuned", "got-ocr2.0", "kaggle_rx", "fine-tuned vs GOT-OCR2.0"),
    ("trocr-lora-finetuned", "got-ocr2.0", "iam", "fine-tuned vs GOT-OCR2.0 (out-of-domain)"),
    ("qwen2.5-vl-3b", "got-ocr2.0", "kaggle_rx", "Qwen2.5-VL-3B vs GOT-OCR2.0 (in-domain)"),
    ("qwen2.5-vl-3b", "got-ocr2.0", "iam", "Qwen2.5-VL-3B vs GOT-OCR2.0 (out-of-domain)"),
]

FAMILY_ABLATION = [
    ("trocr-lora-r8", "trocr-lora-r16", "kaggle_rx", "r=8 vs r=16 (in-domain)"),
    ("trocr-lora-r8", "trocr-lora-r16", "iam", "r=8 vs r=16 (out-of-domain)"),
    ("trocr-lora-finetuned", "trocr-lora-r16", "kaggle_rx", "r=32 vs r=16 (in-domain)"),
    ("trocr-lora-finetuned", "trocr-lora-r16", "iam", "r=32 vs r=16 (out-of-domain)"),
    ("trocr-lora-r16-noelastic", "trocr-lora-r16", "kaggle_rx", "no-elastic vs elastic (in-domain)"),
    ("trocr-lora-r16-noelastic", "trocr-lora-r16", "iam", "no-elastic vs elastic (out-of-domain)"),
]


def paired_cer(df: pd.DataFrame, model_a: str, model_b: str, dataset: str):
    a = df[(df["model"] == model_a) & (df["dataset"] == dataset)][["image_path", "cer"]]
    b = df[(df["model"] == model_b) & (df["dataset"] == dataset)][["image_path", "cer"]]
    m = a.merge(b, on="image_path", suffixes=("_a", "_b"))
    if len(m) != len(a) or len(m) != len(b):
        raise ValueError(
            f"pairing failed for {model_a} vs {model_b} on {dataset}: "
            f"{len(a)} / {len(b)} rows -> {len(m)} pairs"
        )
    return m["cer_a"].to_numpy(), m["cer_b"].to_numpy()


def rank_biserial(x: np.ndarray, y: np.ndarray):
    """Matched-pairs rank-biserial correlation, computed from signed ranks of the
    non-zero differences. Returns (r, n_nonzero, w_plus, w_minus)."""
    d = x - y
    d = d[d != 0]
    if len(d) == 0:
        return float("nan"), 0, 0.0, 0.0
    ranks = rankdata(np.abs(d))
    w_plus = ranks[d > 0].sum()
    w_minus = ranks[d < 0].sum()
    total = w_plus + w_minus
    return float((w_plus - w_minus) / total), len(d), float(w_plus), float(w_minus)


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, order preserved, monotonicity enforced."""
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        val = (m - i) * pvals[idx]
        running = max(running, val)
        adjusted[idx] = min(running, 1.0)
    return adjusted.tolist()


def run_family(df: pd.DataFrame, family, family_name: str) -> pd.DataFrame:
    rows = []
    for model_a, model_b, dataset, label in family:
        x, y = paired_cer(df, model_a, model_b, dataset)
        stat, p = wilcoxon(x, y)  # two-sided; scipy drops zero differences by default
        r_rb, n_eff, w_plus, w_minus = rank_biserial(x, y)
        rows.append({
            "family": family_name,
            "comparison": label,
            "model_a": model_a,
            "model_b": model_b,
            "dataset": dataset,
            "n_pairs": len(x),
            "n_nonzero_pairs": n_eff,
            "mean_cer_a": x.mean(),
            "mean_cer_b": y.mean(),
            "median_cer_a": float(np.median(x)),
            "median_cer_b": float(np.median(y)),
            "w_plus": w_plus,
            "w_minus": w_minus,
            "wilcoxon_statistic": float(stat),
            "p_raw": float(p),
            "rank_biserial_r": r_rb,
        })
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_raw"].tolist())
    out["significant_holm_0.05"] = out["p_holm"] < 0.05
    return out


def main():
    df = pd.read_csv(CSV)
    benchmark = run_family(df, FAMILY_BENCHMARK, "benchmark")
    ablation = run_family(df, FAMILY_ABLATION, "ablation")
    full = pd.concat([benchmark, ablation], ignore_index=True)
    full.to_csv("results/phase11_statistics_full.csv", index=False)

    pd.set_option("display.width", 200)
    for name, sub in full.groupby("family", sort=False):
        print(f"\n=== FAMILY: {name} (Holm correction applied within family, m={len(sub)}) ===")
        show = sub[["comparison", "n_pairs", "n_nonzero_pairs", "mean_cer_a", "mean_cer_b",
                    "wilcoxon_statistic", "rank_biserial_r", "p_raw", "p_holm",
                    "significant_holm_0.05"]].copy()
        for col in ["mean_cer_a", "mean_cer_b", "rank_biserial_r"]:
            show[col] = show[col].map(lambda v: f"{v:.3f}")
        for col in ["p_raw", "p_holm"]:
            show[col] = show[col].map(lambda v: f"{v:.3e}")
        print(show.to_string(index=False))

    print("\nWrote results/phase11_statistics_full.csv")


if __name__ == "__main__":
    main()
