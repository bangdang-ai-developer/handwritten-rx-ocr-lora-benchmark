"""Generate all figures for the paper (Phase 12 prep). Reads results already computed by
analyze_aggregate.py (Phase 9) and analyze_errors.py (Phase 10), plus the preserved LoRA
training log (results/phase7_trainer_state_final.json) and local kaggle_rx test images
(data/kaggle_rx/.../Testing/testing_words/) for the qualitative example figure.

Run from repo root: python src/generate_figures.py
Writes PNGs to results/figures/ (tracked in git - these are paper deliverables, not
regeneratable debug output, unlike results/augmentation_preview/).
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = "results/figures"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_ORDER_LABELS = {
    "trocr-lora-finetuned": "TrOCR-LoRA\n(fine-tuned)",
    "qwen2.5-vl-3b": "Qwen2.5-VL-3B",
    "got-ocr2.0": "GOT-OCR2.0",
    "easyocr": "EasyOCR",
    "trocr-large-handwritten": "TrOCR-large\n(zero-shot)",
    "tesseract": "Tesseract",
    "donut-base-synthdog": "Donut",
    "donut-base-synthdog-padded": "Donut (padded)",
}


def fig1_cer_comparison():
    df = pd.read_csv("results/phase9_summary_table.csv")
    models = list(MODEL_ORDER_LABELS.keys())
    rx = df[df["dataset"] == "kaggle_rx"].set_index("model").reindex(models)
    iam = df[df["dataset"] == "iam"].set_index("model").reindex(models)

    y = np.arange(len(models))
    h = 0.35
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(y + h / 2, rx["cer_mean"], height=h, xerr=[rx["cer_mean"] - rx["cer_ci_lo"], rx["cer_ci_hi"] - rx["cer_mean"]],
            label="Kaggle-Rx (in-domain)", color="#2b6cb0", capsize=3)
    ax.barh(y - h / 2, iam["cer_mean"], height=h, xerr=[iam["cer_mean"] - iam["cer_ci_lo"], iam["cer_ci_hi"] - iam["cer_mean"]],
            label="IAM (out-of-domain)", color="#dd6b20", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_ORDER_LABELS[m] for m in models])
    ax.invert_yaxis()
    ax.set_xlabel("Character Error Rate (mean, 95% bootstrap CI)")
    ax.set_title("CER by model and dataset (lower is better)")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig1_cer_comparison.png", dpi=200)
    plt.close(fig)


def fig2_training_curve():
    d = json.load(open("results/phase7_trainer_state_final.json"))
    epochs, cers, losses = [], [], []
    for e in d["log_history"]:
        if "eval_cer" in e:
            epochs.append(e["epoch"])
            cers.append(e["eval_cer"])
            losses.append(e["eval_loss"])
    best_epoch = min(range(len(cers)), key=lambda i: cers[i])

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(epochs, cers, "o-", color="#2b6cb0", label="Validation CER (300-image subsample)")
    ax1.scatter([epochs[best_epoch]], [cers[best_epoch]], color="#c53030", zorder=5, s=100,
                marker="*", label=f"Best checkpoint (epoch {epochs[best_epoch]:.0f}, CER={cers[best_epoch]:.3f})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation CER", color="#2b6cb0")
    ax1.tick_params(axis="y", labelcolor="#2b6cb0")
    ax1.axvline(epochs[best_epoch], color="#c53030", linestyle="--", linewidth=0.8, alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(epochs, losses, "s--", color="#718096", alpha=0.6, label="Validation loss")
    ax2.set_ylabel("Validation loss", color="#718096")
    ax2.tick_params(axis="y", labelcolor="#718096")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    ax1.set_title("LoRA fine-tuning: validation CER/loss per epoch (kernel v20, early-stopped epoch 7)")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig2_training_curve.png", dpi=200)
    plt.close(fig)


def fig3_error_taxonomy_kaggle_rx():
    df = pd.read_csv("results/phase10_error_taxonomy_kaggle_rx.csv", index_col=0)
    cats = ["correct", "minor_ocr_noise_still_correct_drug", "confusable_wrong_drug", "hallucination_far_off"]
    cat_labels = ["Correct\n(exact match)", "Minor OCR noise\n(still correct drug)",
                  "Confusable wrong drug\n(looks like ANOTHER real drug)", "Hallucination\n(far off, no real drug)"]
    colors = ["#2f855a", "#68d391", "#dd6b20", "#c53030"]
    models = ["trocr-large-handwritten", "trocr-lora-finetuned"]
    model_labels = ["Zero-shot", "Fine-tuned (LoRA)"]

    fig, ax = plt.subplots(figsize=(8, 4.7))
    left = np.zeros(len(models))
    for cat, label, color in zip(cats, cat_labels, colors):
        vals = df.loc[cat, models].values.astype(float)
        ax.barh(model_labels, vals, left=left, color=color, label=label)
        for i, v in enumerate(vals):
            if v > 3:
                ax.text(left[i] + v / 2, i, f"{v:.1f}%", ha="center", va="center", fontsize=8)
        left += vals
    ax.set_xlabel("% of test images (n=780)")
    ax.set_title("Kaggle-Rx error taxonomy: zero-shot vs. LoRA fine-tuned")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig3_error_taxonomy_kaggle_rx.png", dpi=200)
    plt.close(fig)


def fig4_error_taxonomy_iam():
    df = pd.read_csv("results/phase10_error_taxonomy_iam.csv", index_col=0)
    cats = ["correct", "minor_error", "moderate_error", "severe_error"]
    cat_labels = ["Correct", "Minor error\n(0<CER<=0.3)", "Moderate error\n(0.3<CER<=0.7)", "Severe error\n(CER>0.7)"]
    colors = ["#2f855a", "#68d391", "#f6ad55", "#c53030"]
    models = ["trocr-large-handwritten", "trocr-lora-finetuned"]
    model_labels = ["Zero-shot", "Fine-tuned (LoRA)"]

    fig, ax = plt.subplots(figsize=(8, 4.7))
    left = np.zeros(len(models))
    for cat, label, color in zip(cats, cat_labels, colors):
        vals = df.loc[cat, models].values.astype(float)
        ax.barh(model_labels, vals, left=left, color=color, label=label)
        for i, v in enumerate(vals):
            if v > 3:
                ax.text(left[i] + v / 2, i, f"{v:.1f}%", ha="center", va="center", fontsize=8)
        left += vals
    ax.set_xlabel("% of test images (n=400)")
    ax.set_title("IAM (out-of-domain) error severity: zero-shot vs. LoRA fine-tuned")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig4_error_taxonomy_iam.png", dpi=200)
    plt.close(fig)


def fig5_domain_gap():
    df = pd.read_csv("results/phase9_domain_gap.csv")
    df = df.sort_values("domain_gap_iam_minus_kagglerx")
    labels = [MODEL_ORDER_LABELS.get(m, m).replace("\n", " ") for m in df["model"]]
    colors = ["#c53030" if g > 0 else "#2b6cb0" for g in df["domain_gap_iam_minus_kagglerx"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels, df["domain_gap_iam_minus_kagglerx"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Domain-shift gap (CER on IAM minus CER on Kaggle-Rx)")
    ax.set_title("Domain-shift gap by model\n(blue = better on IAM; red = better on Kaggle-Rx)")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig5_domain_gap.png", dpi=200)
    plt.close(fig)


def fig6_cer_distribution():
    df = pd.read_csv("results/results_master_combined.csv")
    models = ["trocr-large-handwritten", "trocr-lora-finetuned", "got-ocr2.0", "qwen2.5-vl-3b"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for ax, dataset in zip(axes, ["kaggle_rx", "iam"]):
        data = [df[(df["model"] == m) & (df["dataset"] == dataset)]["cer"].values for m in models]
        bp = ax.boxplot(data, labels=[MODEL_ORDER_LABELS[m].replace("\n", " ") for m in models],
                         showmeans=True, meanprops={"marker": "D", "markerfacecolor": "red", "markeredgecolor": "red"})
        ax.set_title(f"{dataset}")
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylabel("CER" if dataset == "kaggle_rx" else "")
    fig.suptitle("CER distribution (box = median/IQR, red diamond = mean)\n"
                 "Large mean-median gap = heavy-tailed / outlier-driven, see phase9_summary.md Sec.4")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig6_cer_distribution.png", dpi=200)
    plt.close(fig)


def _local_image_path(kaggle_path: str) -> str | None:
    for split_dir in ["testing_words", "training_words", "validation_words"]:
        if split_dir in kaggle_path:
            fname = kaggle_path.split(split_dir + "/")[-1]
            candidates = glob.glob(f"data/kaggle_rx/**/{split_dir}/{fname}", recursive=True)
            if candidates:
                return candidates[0]
    return None


def fig7_qualitative_examples():
    ft = pd.read_csv("results/phase10_error_taxonomy_kaggle_rx_trocr-lora-finetuned.csv")
    zs = pd.read_csv("results/phase10_error_taxonomy_kaggle_rx_trocr-large-handwritten.csv")

    picks = []
    for cat, n in [("correct", 2), ("confusable_wrong_drug", 2), ("hallucination_far_off", 1),
                   ("minor_ocr_noise_still_correct_drug", 1)]:
        sub = ft[ft["error_category"] == cat]
        for _, row in sub.head(n).iterrows():
            local_path = _local_image_path(row["image_path"])
            if local_path:
                picks.append(row)
    if not picks:
        print("  [fig7] No suitable local images found - skipping fig7.")
        return

    n = len(picks)
    cell_w, cell_h, pad = 220, 170, 12
    canvas = Image.new("RGB", (cell_w * n + pad * (n + 1), cell_h + 2 * pad), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    for i, row in enumerate(picks):
        local_path = _local_image_path(row["image_path"])
        img = Image.open(local_path).convert("RGB")
        img.thumbnail((cell_w - 2 * pad, 70))
        x0 = pad + i * (cell_w + pad)
        canvas.paste(img, (x0 + (cell_w - img.width) // 2, pad))
        zs_row = zs[zs["image_path"] == row["image_path"]]
        zs_hyp = zs_row["hypothesis"].values[0] if len(zs_row) else "?"
        text = (f"true: {row['reference']}\n"
                f"zero-shot: {zs_hyp}\n"
                f"fine-tuned: {row['hypothesis']}\n"
                f"[{row['error_category']}]")
        draw.multiline_text((x0 + 5, pad + 75), text, fill="black", font=font, spacing=3)
    canvas.save(f"{OUT_DIR}/fig7_qualitative_examples.png")


def main():
    fig1_cer_comparison()
    print("  fig1_cer_comparison.png done")
    fig2_training_curve()
    print("  fig2_training_curve.png done")
    fig3_error_taxonomy_kaggle_rx()
    print("  fig3_error_taxonomy_kaggle_rx.png done")
    fig4_error_taxonomy_iam()
    print("  fig4_error_taxonomy_iam.png done")
    fig5_domain_gap()
    print("  fig5_domain_gap.png done")
    fig6_cer_distribution()
    print("  fig6_cer_distribution.png done")
    fig7_qualitative_examples()
    print("  fig7_qualitative_examples.png done")
    print(f"\nAll figures written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
