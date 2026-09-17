"""Phase 10 - qualitative error analysis: does LoRA fine-tuning fix specific error TYPES
(not just lower mean CER), and does it introduce NEW error types on out-of-domain data
(IAM)? Compares trocr-large-handwritten (zero-shot) vs trocr-lora-finetuned.

Run from repo root: python src/analyze_errors.py
Writes: results/phase10_error_taxonomy_kaggle_rx.csv, results/phase10_error_taxonomy_iam.csv,
results/phase10_drug_leakage_examples.csv, results/phase10_examples_by_category.csv
(all gitignored, regenerable from results_master_combined.csv).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

VOCAB = sorted(pd.read_csv("results/frozen_test_manifest_kaggle_rx.csv")["label"].unique())


def _norm(s) -> str:
    return " ".join(str(s).strip().lower().split())


def classify_kaggle_rx(row) -> str:
    if row["exact_match"]:
        return "correct"
    if row["degenerate"]:
        return "degenerate"
    if row["top1_correct"]:
        return "minor_ocr_noise_still_correct_drug"
    # wrong drug: how close is the raw hypothesis to the (wrong) nearest vocab entry?
    dist = Levenshtein.normalized_distance(_norm(row["hypothesis"]), _norm(row["top1_pred"]))
    return "confusable_wrong_drug" if dist <= 0.34 else "hallucination_far_off"


def classify_open_vocab(row) -> str:
    if row["exact_match"]:
        return "correct"
    if row["degenerate"]:
        return "degenerate"
    if row["cer"] <= 0.3:
        return "minor_error"
    if row["cer"] <= 0.7:
        return "moderate_error"
    return "severe_error"


def drug_leakage_on_iam(ft_iam: pd.DataFrame) -> pd.DataFrame:
    """Rows where the fine-tuned model's IAM hypothesis exactly equals one of the 78
    kaggle_rx drug names, while the true IAM reference is clearly NOT that word
    (high CER to it) - i.e. the model appears to have 'leaked' a memorized drug
    name instead of transcribing the actual handwritten English word it was shown."""
    vocab_norm = {_norm(v) for v in VOCAB}
    rows = []
    for _, row in ft_iam.iterrows():
        hyp_n = _norm(row["hypothesis"])
        if hyp_n in vocab_norm and row["cer"] > 0.3:
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv("results/results_master_combined.csv")

    print("=" * 70)
    print("KAGGLE-RX (in-domain) - error taxonomy: zero-shot vs fine-tuned")
    print("=" * 70)
    rx_rows = []
    for model in ["trocr-large-handwritten", "trocr-lora-finetuned"]:
        sub = df[(df["model"] == model) & (df["dataset"] == "kaggle_rx")].copy()
        sub["error_category"] = sub.apply(classify_kaggle_rx, axis=1)
        counts = sub["error_category"].value_counts(normalize=True) * 100
        counts.name = model
        rx_rows.append(counts)
        sub.to_csv(f"results/phase10_error_taxonomy_kaggle_rx_{model}.csv", index=False)
    rx_table = pd.concat(rx_rows, axis=1).fillna(0.0)
    rx_table.to_csv("results/phase10_error_taxonomy_kaggle_rx.csv")
    print(rx_table.round(2).to_string())
    print()

    print("=" * 70)
    print("IAM (out-of-domain) - error taxonomy: zero-shot vs fine-tuned")
    print("=" * 70)
    iam_rows = []
    iam_frames = {}
    for model in ["trocr-large-handwritten", "trocr-lora-finetuned"]:
        sub = df[(df["model"] == model) & (df["dataset"] == "iam")].copy()
        sub["error_category"] = sub.apply(classify_open_vocab, axis=1)
        counts = sub["error_category"].value_counts(normalize=True) * 100
        counts.name = model
        iam_rows.append(counts)
        iam_frames[model] = sub
    iam_table = pd.concat(iam_rows, axis=1).fillna(0.0)
    iam_table.to_csv("results/phase10_error_taxonomy_iam.csv")
    print(iam_table.round(2).to_string())
    print()

    print("=" * 70)
    print("Drug-name 'leakage' check: does fine-tuned model hallucinate kaggle_rx")
    print("drug names when reading IAM (general English) handwriting?")
    print("=" * 70)
    leakage = drug_leakage_on_iam(iam_frames["trocr-lora-finetuned"])
    leakage.to_csv("results/phase10_drug_leakage_examples.csv", index=False)
    print(f"  {len(leakage)}/400 ({len(leakage)/400*100:.1f}%) fine-tuned IAM predictions exactly "
          f"match a kaggle_rx drug name while being clearly wrong (cer>0.3 vs true IAM label).")
    if len(leakage):
        print(leakage[["image_path", "reference", "hypothesis", "cer"]].head(10).to_string(index=False))
    # sanity check: did zero-shot ever do this too (baseline rate)?
    zs_leak = drug_leakage_on_iam(iam_frames["trocr-large-handwritten"])
    print(f"  (baseline zero-shot rate for comparison: {len(zs_leak)}/400 = {len(zs_leak)/400*100:.1f}%)")
    print()

    print("=" * 70)
    print("Concrete examples per category (kaggle_rx, fine-tuned model)")
    print("=" * 70)
    ft_rx = pd.read_csv("results/phase10_error_taxonomy_kaggle_rx_trocr-lora-finetuned.csv")
    examples = []
    for cat in ["confusable_wrong_drug", "hallucination_far_off", "minor_ocr_noise_still_correct_drug"]:
        sub = ft_rx[ft_rx["error_category"] == cat].head(5)
        examples.append(sub[["reference", "hypothesis", "top1_pred", "cer", "error_category"]])
    ex_df = pd.concat(examples, ignore_index=True) if examples else pd.DataFrame()
    ex_df.to_csv("results/phase10_examples_by_category.csv", index=False)
    print(ex_df.to_string(index=False))


if __name__ == "__main__":
    main()
