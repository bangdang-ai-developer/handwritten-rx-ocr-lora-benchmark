"""Phase 9 - aggregate quantitative analysis across every model (zero-shot + fine-tuned),
both datasets: per-model summary table (bootstrap CI), domain-shift gap, and pairwise
Wilcoxon signed-rank significance tests (valid because every model was scored on the
exact same frozen manifest per dataset - seed=42, verified identical image sets across
independent runs, see results/phase6_summary.md).

Run from repo root: python src/analyze_aggregate.py
Writes: results/phase9_summary_table.csv, results/phase9_domain_gap.csv,
results/phase9_wilcoxon_pairs.csv (all gitignored, regenerable from results_master_combined.csv).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

SEED = 0
N_RESAMPLES = 1000


def bootstrap_ci(values, n_resamples=N_RESAMPLES, ci=0.95, seed=SEED):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        means[i] = rng.choice(values, size=n, replace=True).mean()
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return float(values.mean()), float(lo), float(hi)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, dataset), g in df.groupby(["model", "dataset"]):
        cer_m, cer_lo, cer_hi = bootstrap_ci(g["cer"].values)
        row = {
            "model": model,
            "dataset": dataset,
            "n": len(g),
            "cer_mean": cer_m,
            "cer_ci_lo": cer_lo,
            "cer_ci_hi": cer_hi,
            "wer_mean": g["wer"].mean(),
            "exact_match_pct": g["exact_match"].mean() * 100,
            "degenerate_pct": g["degenerate"].mean() * 100,
        }
        if dataset == "kaggle_rx" and "top1_correct" in g.columns:
            row["top1_acc_pct"] = g["top1_correct"].mean() * 100
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["dataset", "cer_mean"]).reset_index(drop=True)
    return out


def domain_gap_table(summary: pd.DataFrame) -> pd.DataFrame:
    piv = summary.pivot(index="model", columns="dataset", values="cer_mean")
    piv["domain_gap_iam_minus_kagglerx"] = piv["iam"] - piv["kaggle_rx"]
    return piv.sort_values("domain_gap_iam_minus_kagglerx").reset_index()


def paired_wilcoxon(df: pd.DataFrame, model_a: str, model_b: str, dataset: str):
    a = df[(df["model"] == model_a) & (df["dataset"] == dataset)][["image_path", "cer"]]
    b = df[(df["model"] == model_b) & (df["dataset"] == dataset)][["image_path", "cer"]]
    m = a.merge(b, on="image_path", suffixes=("_a", "_b"))
    if len(m) == 0 or (m["cer_a"] - m["cer_b"]).abs().sum() == 0:
        return {"model_a": model_a, "model_b": model_b, "dataset": dataset, "n": len(m),
                "cer_a_mean": m["cer_a"].mean() if len(m) else float("nan"),
                "cer_b_mean": m["cer_b"].mean() if len(m) else float("nan"),
                "stat": float("nan"), "p_value": float("nan"), "note": "no variance or no pairs"}
    stat, p = wilcoxon(m["cer_a"], m["cer_b"])
    return {"model_a": model_a, "model_b": model_b, "dataset": dataset, "n": len(m),
            "cer_a_mean": m["cer_a"].mean(), "cer_b_mean": m["cer_b"].mean(),
            "stat": stat, "p_value": p, "note": ""}


def main():
    df = pd.read_csv("results/results_master_combined.csv")

    summary = summary_table(df)
    summary.to_csv("results/phase9_summary_table.csv", index=False)
    print("=== Bang tong hop moi model x dataset ===")
    print(summary.to_string(index=False))
    print()

    gap = domain_gap_table(summary)
    gap.to_csv("results/phase9_domain_gap.csv", index=False)
    print("=== Domain-shift gap (CER_iam - CER_kaggle_rx), sap xep tang dan ===")
    print(gap.to_string(index=False))
    print()

    pairs_to_test = [
        ("trocr-large-handwritten", "trocr-lora-finetuned", "kaggle_rx"),
        ("trocr-large-handwritten", "trocr-lora-finetuned", "iam"),
        ("qwen2.5-vl-3b", "got-ocr2.0", "kaggle_rx"),
        ("qwen2.5-vl-3b", "got-ocr2.0", "iam"),
        ("trocr-lora-finetuned", "qwen2.5-vl-3b", "kaggle_rx"),
        ("trocr-lora-finetuned", "got-ocr2.0", "kaggle_rx"),
        ("trocr-lora-finetuned", "got-ocr2.0", "iam"),
    ]
    results = [paired_wilcoxon(df, a, b, ds) for a, b, ds in pairs_to_test]
    wdf = pd.DataFrame(results)
    wdf.to_csv("results/phase9_wilcoxon_pairs.csv", index=False)
    print("=== Wilcoxon signed-rank pairwise (paired tren cung anh) ===")
    print(wdf.to_string(index=False))


if __name__ == "__main__":
    main()
