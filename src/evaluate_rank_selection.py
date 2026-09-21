"""Validation-set-based rank selection (fixes the test-set-leakage issue an advisor flagged:
the original ablation picked r=32 by looking at frozen-TEST-set CER across r=8/16/32, which is
improper train/val/test discipline - selection must be done on the VALIDATION split, with the
test set touched only once, after the winner is already chosen).

Loads each already-trained LoRA checkpoint under results/trocr-lora-adapter-*/, runs CPU
inference on the full Kaggle-Rx Validation split (780 images - not just the 300-image
monitoring subsample used for early stopping during training), and reports validation CER per
rank. Whichever rank wins here is the properly-selected "official" model; its already-computed
TEST CER (in results_master_combined.csv) is then reported as the final number, unchanged,
since the test set was never used to make this selection.

Run from repo root: python src/evaluate_rank_selection.py [--limit N] [--checkpoint NAME]
Writes: results/phase11_validation_rank_selection.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, VisionEncoderDecoderConfig
from peft import PeftModel

sys.path.insert(0, os.path.dirname(__file__))
from metrics import cer  # noqa: E402

DATA_ROOT = "data/kaggle_rx"
CHECKPOINTS = {
    # name -> (adapter_dir, lora_r) - lora_r is just for the printed report, read from
    # adapter_config.json too as a cross-check.
    "r16": "results/trocr-lora-adapter-r16-superseded",
    "r32": "results/trocr-lora-adapter-final",
    # "r8" is added once results/trocr-lora-adapter-r8/ exists (Kaggle rerun, 2026-09-21).
}
if os.path.isdir("results/trocr-lora-adapter-r8"):
    CHECKPOINTS["r8"] = "results/trocr-lora-adapter-r8"


def build_validation_manifest(n_sample=None, seed=42):
    root = glob.glob(f"{DATA_ROOT}/*/")[0]  # the single nested "Doctor's Handwritten..." folder
    label_files = glob.glob(f"{root}Validation/*.csv") + glob.glob(f"{root}Validation/*.xlsx")
    if not label_files:
        raise FileNotFoundError(f"No Validation label file found under {root}Validation/")
    path = label_files[0]
    df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
    # Same column-guessing convention as notebooks/kaggle_benchmark.py: image col ends in
    # .jpg/.png, label col is whatever's left (2-column label files, so just pick the other one).
    img_col = next(c for c in df.columns if df[c].astype(str).str.lower().str.endswith((".jpg", ".png", ".jpeg")).mean() > 0.5)
    label_col = next(c for c in df.columns if c != img_col)
    split_dir = os.path.dirname(path)
    img_dir_candidates = [d for d in glob.glob(f"{split_dir}/*/") if os.path.isdir(d)]
    img_dir = img_dir_candidates[0] if img_dir_candidates else split_dir

    def resolve(fname):
        fname = str(fname).strip()
        direct = os.path.join(img_dir, fname)
        if os.path.exists(direct):
            return direct
        hits = glob.glob(f"{img_dir}/**/{fname}", recursive=True)
        return hits[0] if hits else direct

    manifest = pd.DataFrame({"image_path": df[img_col].map(resolve), "label": df[label_col]})
    manifest = manifest[manifest["image_path"].map(os.path.exists)].reset_index(drop=True)
    print(f"Validation manifest: {len(manifest)}/{len(df)} images found on disk (img_dir={img_dir})")
    if n_sample and n_sample < len(manifest):
        manifest = manifest.sample(n=n_sample, random_state=seed).reset_index(drop=True)
    return manifest


def load_checkpoint(adapter_dir):
    if not hasattr(VisionEncoderDecoderConfig, "_vocab_size_patched_for_peft"):
        VisionEncoderDecoderConfig.vocab_size = property(lambda self: self.decoder.vocab_size)
        VisionEncoderDecoderConfig._vocab_size_patched_for_peft = True
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
    base = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
    base.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    base.config.pad_token_id = processor.tokenizer.pad_token_id
    base.config.eos_token_id = processor.tokenizer.sep_token_id
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return processor, model


@torch.no_grad()
def evaluate(processor, model, manifest, tag, log_every=100):
    rows = []
    t0 = time.time()
    for i, r in manifest.iterrows():
        image = Image.open(r["image_path"]).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        ids = model.generate(pixel_values=pixel_values, max_new_tokens=32)
        hyp = processor.batch_decode(ids, skip_special_tokens=True)[0]
        rows.append({"image_path": r["image_path"], "reference": r["label"], "hypothesis": hyp,
                      "cer": cer(r["label"], hyp)})
        if (i + 1) % log_every == 0:
            elapsed = time.time() - t0
            running_cer = np.mean([x["cer"] for x in rows])
            print(f"  [{tag}] {i+1}/{len(manifest)}  elapsed={elapsed:.0f}s  running CER={running_cer:.4f}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Evaluate only the first N validation images (for a quick timing/sanity check before committing to the full 780).")
    ap.add_argument("--checkpoint", default=None, help="Evaluate only this one checkpoint name (r8/r16/r32).")
    args = ap.parse_args()

    checkpoints = CHECKPOINTS if not args.checkpoint else {args.checkpoint: CHECKPOINTS[args.checkpoint]}
    print("Checkpoints to evaluate:", list(checkpoints.keys()))
    if "r8" not in CHECKPOINTS:
        print("[NOTE] results/trocr-lora-adapter-r8/ does not exist yet - r8 will be skipped until "
              "the Kaggle rerun finishes and its adapter is downloaded there.")

    manifest = build_validation_manifest(n_sample=args.limit)

    summary = []
    all_rows = []
    for name, adapter_dir in checkpoints.items():
        if not os.path.isdir(adapter_dir):
            print(f"[SKIP] {name}: {adapter_dir} does not exist")
            continue
        print(f"\n=== Loading checkpoint {name} from {adapter_dir} ===")
        processor, model = load_checkpoint(adapter_dir)
        df = evaluate(processor, model, manifest, tag=name)
        df["rank_config"] = name
        all_rows.append(df)
        mean_cer = df["cer"].mean()
        summary.append({"rank_config": name, "n": len(df), "validation_cer_mean": mean_cer,
                         "validation_cer_median": df["cer"].median()})
        print(f"  {name}: validation CER = {mean_cer:.4f} (n={len(df)})")
        del model, processor

    summary_df = pd.DataFrame(summary).sort_values("validation_cer_mean")
    print("\n" + "=" * 60)
    print("VALIDATION-SET RANK SELECTION (proper train/val/test discipline)")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    if len(summary_df):
        winner = summary_df.iloc[0]["rank_config"]
        print(f"\n-> Winner by VALIDATION CER: {winner}")

    if all_rows:
        out = pd.concat(all_rows, ignore_index=True)
        out_path = "results/phase11_validation_rank_selection.csv"
        out.to_csv(out_path, index=False)
        summary_path = "results/phase11_validation_rank_selection_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nWrote {out_path} and {summary_path}")


if __name__ == "__main__":
    main()
