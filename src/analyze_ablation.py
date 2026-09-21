"""Phase 7 ablation analysis - LoRA rank (r=8/16/32) and augmentation A/B (elastic on/off).
Compares against the already-completed "main" config (r=16, elastic=True, model name
"trocr-lora-finetuned") using the same frozen test sets (kaggle_rx 780 + iam 400).

Run from repo root AFTER all ablation kernel runs have been downloaded and merged into
results/results_master_combined.csv (see notebooks/kaggle_benchmark.py ABLATION_CONFIG).
Writes: results/phase7b_ablation_table.csv, results/phase7b_ablation_wilcoxon.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

SEED = 0
N_RESAMPLES = 1000


# NOTE (2026-09-17): after the ablation, r=32 was chosen as the main result (best CER +
# least forgetting - see phase7b_ablation_summary.md Section 3) - renamed in
# results_master_combined.csv: "trocr-lora-finetuned" is now r=32 (main); r=16 (the OLD
# main config) was renamed to "trocr-lora-r16".
RANK_MODELS = {
    "trocr-lora-r8": "r=8",
    "trocr-lora-r16": "r=16",
    "trocr-lora-finetuned": "r=32 (main)",
}
AUGMENTATION_MODELS = {
    "trocr-lora-r16": "r=16, elastic=True",
    "trocr-lora-r16-noelastic": "r=16, elastic=False",
}


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


def summarize(df, models_dict):
    rows = []
    for model, label in models_dict.items():
        for dataset in ["kaggle_rx", "iam"]:
            sub = df[(df["model"] == model) & (df["dataset"] == dataset)]
            if len(sub) == 0:
                print(f"  [WARNING] no data for model={model!r} dataset={dataset!r} - skipping")
                continue
            m, lo, hi = bootstrap_ci(sub["cer"].values)
            rows.append({"model": model, "label": label, "dataset": dataset, "n": len(sub),
                         "cer_mean": m, "cer_ci_lo": lo, "cer_ci_hi": hi,
                         "exact_match_pct": sub["exact_match"].mean() * 100,
                         "top1_acc_pct": sub["top1_correct"].mean() * 100 if dataset == "kaggle_rx" else float("nan")})
    return pd.DataFrame(rows)


def pairwise_wilcoxon(df, model_a, model_b, dataset):
    a = df[(df["model"] == model_a) & (df["dataset"] == dataset)][["image_path", "cer"]]
    b = df[(df["model"] == model_b) & (df["dataset"] == dataset)][["image_path", "cer"]]
    m = a.merge(b, on="image_path", suffixes=("_a", "_b"))
    if len(m) == 0:
        return {"model_a": model_a, "model_b": model_b, "dataset": dataset, "n": 0,
                "cer_a_mean": float("nan"), "cer_b_mean": float("nan"), "p_value": float("nan")}
    if (m["cer_a"] - m["cer_b"]).abs().sum() == 0:
        return {"model_a": model_a, "model_b": model_b, "dataset": dataset, "n": len(m),
                "cer_a_mean": m["cer_a"].mean(), "cer_b_mean": m["cer_b"].mean(), "p_value": 1.0}
    stat, p = wilcoxon(m["cer_a"], m["cer_b"])
    return {"model_a": model_a, "model_b": model_b, "dataset": dataset, "n": len(m),
            "cer_a_mean": m["cer_a"].mean(), "cer_b_mean": m["cer_b"].mean(), "p_value": p}


def main():
    df = pd.read_csv("results/results_master_combined.csv")

    print("=== RANK ABLATION (r=8 vs r=16 vs r=32) ===")
    rank_summary = summarize(df, RANK_MODELS)
    print(rank_summary.to_string(index=False))
    print()

    print("=== AUGMENTATION A/B (elastic vs no-elastic, r=16) ===")
    aug_summary = summarize(df, AUGMENTATION_MODELS)
    print(aug_summary.to_string(index=False))
    print()

    combined = pd.concat([rank_summary, aug_summary], ignore_index=True).drop_duplicates()
    combined.to_csv("results/phase7b_ablation_table.csv", index=False)

    pairs = [
        ("trocr-lora-r8", "trocr-lora-r16", "kaggle_rx"),
        ("trocr-lora-r8", "trocr-lora-r16", "iam"),
        ("trocr-lora-finetuned", "trocr-lora-r16", "kaggle_rx"),  # r32 vs r16
        ("trocr-lora-finetuned", "trocr-lora-r16", "iam"),
        ("trocr-lora-r16-noelastic", "trocr-lora-r16", "kaggle_rx"),
        ("trocr-lora-r16-noelastic", "trocr-lora-r16", "iam"),
    ]
    print("=== Wilcoxon signed-rank (paired, each configuration vs. r=16 elastic=True) ===")
    wdf = pd.DataFrame([pairwise_wilcoxon(df, a, b, ds) for a, b, ds in pairs])
    print(wdf.to_string(index=False))
    wdf.to_csv("results/phase7b_ablation_wilcoxon.csv", index=False)


if __name__ == "__main__":
    main()
