"""Sensitivity analysis for the 0.34 confusable-drug distance threshold (Section 3.5).

The threshold separates, among the fine-tuned model's WRONG (non-exact-match, non-degenerate,
top1-incorrect) Kaggle-Rx predictions, "confusable wrong drug" (raw hypothesis closely resembles
a specific different real drug name) from "hallucination, far off" (resembles nothing on the
vocabulary list). It was originally picked by eye; this sweeps a grid of candidate thresholds and
looks for a natural break in the distance distribution to justify a specific value empirically
rather than by inspection alone.

Run from repo root: python src/analyze_threshold_sensitivity.py
Writes: results/phase11_threshold_sensitivity.csv, results/figures/fig10_threshold_sensitivity.png
(both gitignored/regenerable from results_master_combined.csv, except the figure which is tracked
once generated for the paper).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein


def _norm(s) -> str:
    return " ".join(str(s).strip().lower().split())


def main():
    df = pd.read_csv("results/results_master_combined.csv")
    sub = df[(df["model"] == "trocr-lora-finetuned") & (df["dataset"] == "kaggle_rx")].copy()

    # Restrict to the "wrong, and not exact/degenerate/top1-correct" subset - exactly the
    # population the confusable-vs-hallucination split applies to (matches analyze_errors.py's
    # classify_kaggle_rx logic for rows that reach the distance-threshold branch).
    wrong = sub[(~sub["exact_match"].astype(bool)) & (~sub["degenerate"].astype(bool))
                & (~sub["top1_correct"].astype(bool))].copy()
    wrong["dist"] = wrong.apply(
        lambda r: Levenshtein.normalized_distance(_norm(r["hypothesis"]), _norm(r["top1_pred"])),
        axis=1,
    )
    n = len(wrong)
    print(f"Wrong, non-degenerate, top1-incorrect predictions (the population this threshold "
          f"applies to): n = {n}")

    # 1. Full distribution of distances, to look for a natural gap/knee by eye AND numerically.
    dist_sorted = np.sort(wrong["dist"].values)
    print("\nDistance distribution (percentiles):")
    for p in [5, 10, 25, 40, 50, 60, 75, 90, 95]:
        print(f"  p{p:>2d}: {np.percentile(dist_sorted, p):.3f}")

    # Largest gap between consecutive sorted distances in the plausible cutoff range [0.15, 0.55]
    # is a principled candidate for "the" threshold: it is the point that reclassifies the fewest
    # borderline cases either way, i.e. the most STABLE choice.
    mask = (dist_sorted >= 0.15) & (dist_sorted <= 0.55)
    band = dist_sorted[mask]
    if len(band) > 1:
        gaps = np.diff(band)
        knee_idx = int(np.argmax(gaps))
        knee_value = (band[knee_idx] + band[knee_idx + 1]) / 2
        print(f"\nLargest gap in [0.15, 0.55]: between {band[knee_idx]:.3f} and "
              f"{band[knee_idx + 1]:.3f} (gap = {gaps[knee_idx]:.3f}) -> midpoint {knee_value:.3f}")
    else:
        knee_value = None

    # 2. Sweep a grid of candidate thresholds and report the resulting category split + how many
    # of the n_wrong predictions would flip category relative to neighboring candidate values
    # (a measure of how "sensitive"/arbitrary a given choice is).
    candidates = [0.20, 0.25, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.45, 0.50]
    rows = []
    for d in candidates:
        confusable = int((wrong["dist"] <= d).sum())
        hallucination = n - confusable
        rows.append({
            "threshold": d,
            "n_confusable_wrong_drug": confusable,
            "pct_confusable_of_wrong": round(100 * confusable / n, 2) if n else float("nan"),
            "n_hallucination_far_off": hallucination,
            "pct_hallucination_of_wrong": round(100 * hallucination / n, 2) if n else float("nan"),
            # share of ALL kaggle_rx predictions (matches the paper's Table 4 denominator, n=780)
            "pct_confusable_of_all_780": round(100 * confusable / len(sub), 2),
            "pct_hallucination_of_all_780": round(100 * hallucination / len(sub), 2),
        })
    sweep = pd.DataFrame(rows)
    # Sensitivity: how many predictions flip category between consecutive candidate thresholds.
    sweep["flips_vs_prev_candidate"] = sweep["n_confusable_wrong_drug"].diff().abs()
    sweep.to_csv("results/phase11_threshold_sensitivity.csv", index=False)
    print("\nThreshold sweep:")
    print(sweep.to_string(index=False))

    # 3. Plot: histogram of the distance distribution with candidate thresholds marked, plus a
    # small inset-style annotation of category-share stability, to include as a supplementary
    # figure and to visually justify the choice in the manuscript.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.hist(wrong["dist"], bins=40, color="#4C72B0", alpha=0.75, edgecolor="white", linewidth=0.4)
    ax.axvline(0.34, color="#C44E52", linestyle="-", linewidth=2, label="0.34 (used in the paper)")
    if knee_value is not None:
        ax.axvline(knee_value, color="#55A868", linestyle="--", linewidth=1.5,
                   label=f"largest-gap knee ({knee_value:.2f})")
    for d in candidates:
        if d not in (0.34,):
            ax.axvline(d, color="gray", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.set_xlabel("Normalized edit distance: raw hypothesis vs. nearest-vocabulary prediction")
    ax.set_ylabel("Count (wrong, non-degenerate, top1-incorrect predictions)")
    ax.set_title("Distance distribution used to separate \"confusable wrong drug\"\nfrom "
                 "\"hallucination, far off\" (fine-tuned model, Kaggle-Rx, n=%d)" % n)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("results/figures/fig10_threshold_sensitivity.png", dpi=200)
    print("\nWrote results/figures/fig10_threshold_sensitivity.png")


if __name__ == "__main__":
    main()
