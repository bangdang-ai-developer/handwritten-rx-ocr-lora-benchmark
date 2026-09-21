"""Multi-seed replication summary (addressing advisor feedback: "each config trained only once
is not robust, use 3+ seeds x 3 ranks, report mean +- sd").

Aggregates the 9 seed x rank runs (3 ranks {8,16,32} x 3 seeds {42,123,2024}) from
results_master_combined.csv (each run_name of the form "trocr-lora-r{rank}[-seed{seed}]", where
the un-suffixed name implies seed=42, the value used for the original single-seed ablation) into
mean +- SD CER per rank, on both kaggle_rx (in-domain) and iam (out-of-domain generalization).

Run from repo root: python src/analyze_seed_sweep.py
Writes: results/phase12_seed_sweep_summary.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# run_name -> (rank, seed). The three seed=42 runs reuse the ORIGINAL single-seed ablation's
# run_names (no "-seed42" suffix was used for those, to avoid duplicating already-reported rows).
RUN_NAME_TO_RANK_SEED = {
    "trocr-lora-r8": (8, 42),
    "trocr-lora-r16": (16, 42),
    "trocr-lora-finetuned": (32, 42),  # the "official" r=32 run's name, unchanged from before
    "trocr-lora-r8-seed123": (8, 123),
    "trocr-lora-r16-seed123": (16, 123),
    "trocr-lora-r32-seed123": (32, 123),
    "trocr-lora-r8-seed2024": (8, 2024),
    "trocr-lora-r16-seed2024": (16, 2024),
    "trocr-lora-r32-seed2024": (32, 2024),
}


def main():
    df = pd.read_csv("results/results_master_combined.csv")
    df = df[df["model"].isin(RUN_NAME_TO_RANK_SEED)].copy()
    df["rank"] = df["model"].map(lambda m: RUN_NAME_TO_RANK_SEED[m][0])
    df["seed"] = df["model"].map(lambda m: RUN_NAME_TO_RANK_SEED[m][1])

    present_runs = sorted(df["model"].unique())
    missing_runs = sorted(set(RUN_NAME_TO_RANK_SEED) - set(present_runs))
    print(f"Runs present in results_master_combined.csv ({len(present_runs)}/9): {present_runs}")
    if missing_runs:
        print(f"[NOTE] Still missing ({len(missing_runs)}/9), not yet run on Kaggle: {missing_runs}")

    rows = []
    for dataset in ["kaggle_rx", "iam"]:
        sub = df[df["dataset"] == dataset]
        for rank in [8, 16, 32]:
            rank_sub = sub[sub["rank"] == rank]
            per_seed = rank_sub.groupby("seed")["cer"].mean()
            rows.append({
                "dataset": dataset, "rank": rank,
                "n_seeds_available": len(per_seed),
                "seeds": sorted(per_seed.index.tolist()),
                "cer_mean_across_seeds": per_seed.mean() if len(per_seed) else float("nan"),
                "cer_sd_across_seeds": per_seed.std(ddof=1) if len(per_seed) > 1 else float("nan"),
                "cer_per_seed": {int(s): round(float(v), 4) for s, v in per_seed.items()},
            })
    summary = pd.DataFrame(rows)
    pd.set_option("display.width", 140)
    print("\n" + "=" * 70)
    print("MULTI-SEED REPLICATION SUMMARY (mean +- SD across available seeds)")
    print("=" * 70)
    print(summary.drop(columns=["seeds"]).to_string(index=False))

    summary.to_csv("results/phase12_seed_sweep_summary.csv", index=False)
    print("\nWrote results/phase12_seed_sweep_summary.csv")

    if not missing_runs:
        print("\nAll 9 seed x rank runs present - ready to update the manuscript's Limitations "
              "section and Table 2 (ablation) with mean +- SD instead of single-seed values.")


if __name__ == "__main__":
    main()
