"""
Kaggle Notebook script — Days 1-9 of the two-week plan (see docs/05-q2-research-plan.md
for the current plan).

HOW TO USE:
1. Go to kaggle.com -> Create -> New Notebook.
2. In the right-hand "Input" panel, click "+ Add Input", search for and add 2 datasets:
     - mamun1113/doctors-handwritten-prescription-bd-dataset   (main focus)
     - nibinv23/iam-handwriting-word-database                   (domain-shift control)
3. Settings (right-hand panel) -> Accelerator -> select "GPU T4 x2".
4. Paste the ENTIRE contents of this file into a single cell and run it (Run All).
   Run one PHASE at a time (set CURRENT_PHASE below) to stay within Kaggle's 9-hour
   GPU/session quota — each phase appends its results to results/results_master.csv
   (append, not overwrite), so you can run it across multiple sessions/days.
5. After each phase, click "Save Version" -> "Save & Run All" to save the output, or
   download results_master.csv directly via the Output panel.

IMPORTANT: PHASE 0 is a DIAGNOSTIC step — run it first, read the output, and confirm
the column names match reality before running PHASE 1+ (this dataset's CSV has
"9 columns" whose exact names we could not determine from outside Kaggle, so the code
here guesses them heuristically and PRINTS them out for you to verify).
"""

import os, glob, json, time, traceback, subprocess, sys

# Bootstrap: install packages that aren't available in the Kaggle base image, BEFORE importing them.
# Runs once per session (Kaggle keeps the pip cache within a session, so later runs are faster).
def _pip_install(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)


def _assert_sane_debug_output(model_name, debug_preds, max_avg_len=60):
    """Shared EMERGENCY BRAKE for every new VLM: if the debug run's first 5 images produce
    output that is too long/nonsensical (a sign of hallucination, a wrong dtype, or a library
    incompatibility — see the GOT-OCR2.0 lesson from kernel v8-v10, where ~90 minutes of GPU
    time were wasted before this was caught), STOP before running the full 1180 images. Call
    this right after the debug loop, before calling run_model_on_manifest."""
    avg_len = sum(len(p) for p in debug_preds) / max(1, len(debug_preds))
    if avg_len > max_avg_len:
        raise RuntimeError(
            f"{model_name}: debug output is too long/nonsensical (average {avg_len:.0f} characters "
            f"for a single short word/label) - suspect a dtype/library/prompt issue, see debug_preds "
            f"above for details. STOP, DO NOT run the full 1180 images."
        )

_pip_install("jiwer", "rapidfuzz", "openpyxl", "pytesseract", "easyocr")
os.system("apt-get install -y tesseract-ocr -q > /tmp/apt.log 2>&1")  # Kaggle kernel runs with root privileges

import pandas as pd
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================
CURRENT_PHASE = 6   # 0=diagnostics, 1=classical OCR, 2=TrOCR+Donut, 3=GOT-OCR2.0,
                     # 4=PaddleOCR-VL, 5=Qwen-VL, 6=(optional) closed API

# Choose which ablation configuration to run when CURRENT_PHASE == 6 (each kernel push runs
# one configuration - change ABLATION_CONFIG and push a new kernel for the next configuration,
# see results/phase7b_ablation_summary.md).
# "main" (r=16, elastic=True) and "r16_noelastic" HAVE ALREADY BEEN RUN - no need to rerun.
# "r8" needs a RERUN (2026-09-21): its adapter checkpoint was not preserved on disk (only its
# metrics survived), and re-evaluating it on the Validation split - not the frozen Test split -
# is needed to fix a test-set-leakage issue in how the reported rank (r=32) was originally chosen
# (see docs/05-q2-research-plan.md and the advisor feedback that prompted this). This rerun at
# seed=42 doubles as one of the 9 seed x rank data points for the multi-seed replication below.
ABLATION_CONFIG = "r8"   # "main" | "r8" | "r32" | "r16_noelastic" | "r8_seed123" | ... (see _SEED_SWEEP below)
_ABLATION_PRESETS = {
    "main":          dict(lora_r=16, lora_alpha=32, use_elastic=True,  run_name="trocr-lora-finetuned"),
    "r8":            dict(lora_r=8,  lora_alpha=16, use_elastic=True,  run_name="trocr-lora-r8"),
    "r32":           dict(lora_r=32, lora_alpha=64, use_elastic=True,  run_name="trocr-lora-r32"),
    "r16_noelastic": dict(lora_r=16, lora_alpha=32, use_elastic=False, run_name="trocr-lora-r16-noelastic"),
}

# Multi-seed replication (2026-09-21, addressing advisor feedback: "each config trained only once
# is not robust, use 3+ seeds x 3 ranks, report mean +- sd"). seed=42 for each rank is already
# covered above (the "r8"/"main"/"r32" runs use SEED=42 by default) - this adds 2 more seeds x 3
# ranks = 6 additional runs, for 3 seeds x 3 ranks = 9 total data points feeding
# results/phase12_seed_sweep_summary.md. Elastic augmentation is held on throughout (already
# decided as the primary setting; this sweep is about rank x seed variance, not augmentation).
_EXTRA_SEEDS = [123, 2024]
for _seed in _EXTRA_SEEDS:
    for _rank, _alpha in [(8, 16), (16, 32), (32, 64)]:
        _ABLATION_PRESETS[f"r{_rank}_seed{_seed}"] = dict(
            lora_r=_rank, lora_alpha=_alpha, use_elastic=True,
            run_name=f"trocr-lora-r{_rank}-seed{_seed}", seed=_seed,
        )
del _seed, _rank, _alpha

# Extra flags for selectively rerunning part of Phase 2 (avoid redoing work that already has good results):
RUN_TROCR = False           # already has good results in results_master_phase2.csv (kernel v6) - no need to rerun
RUN_DONUT_RAW_FULL = False  # already has results (near-100% failure) from kernel v6 - only rerun PADDED this time

RESULTS_DIR = "/kaggle/working/results"
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_CSV = f"{RESULTS_DIR}/results_master.csv"

def _resolve_root(slug, owner):
    """Kaggle's newer container mounts datasets under /kaggle/input/datasets/<owner>/<slug>/
    instead of the classic flat /kaggle/input/<slug>/ - confirmed via an actual run on 2026-09-16
    (os.listdir('/kaggle/input') = ['datasets']). Try both layouts, preferring the newer one first."""
    for cand in [
        f"/kaggle/input/datasets/{owner}/{slug}",
        f"/kaggle/input/{slug}",
    ]:
        if os.path.isdir(cand):
            return cand
    return f"/kaggle/input/{slug}"  # fallback, so phase0_diagnose reports a clear error


RX_ROOT = _resolve_root("doctors-handwritten-prescription-bd-dataset", "mamun1113")
IAM_ROOT = _resolve_root("iam-handwriting-word-database", "nibinv23")

DRUG_VOCAB = [
    "Beklo", "Maxima", "Leptic", "Esoral", "Omastin", "Esonix", "Canazole", "Fixal",
    "Progut", "Diflu", "Montair", "Flexilax", "Maxpro", "Vifas", "Conaz", "Fexofast",
    "Fenadin", "Telfast", "Dinafex", "Ritch", "Renova", "Flugal", "Axodin", "Sergel",
    "Nexum", "Opton", "Nexcap", "Fexo", "Montex", "Exium", "Lumona", "Napa",
    "Azithrocin", "Atrizin", "Monas", "Nidazyl", "Metsina", "Baclon", "Rozith",
    "Bicozin", "Ace", "Amodis", "Alatrol", "Napa Extend", "Rivotril", "Montene",
    "Filmet", "Aceta", "Tamen", "Bacmax", "Disopan", "Rhinil", "Flamyd", "Metro",
    "Zithrin", "Candinil", "Lucan-R", "Backtone", "Bacaid", "Etizin", "Az", "Romycin",
    "Azyth", "Cetisoft", "Dancel", "Tridosil", "Nizoder", "Ketoral", "Ketocon",
    "Ketotab", "Ketozol", "Denixil", "Provair", "Odmon", "Baclofen", "MKast",
    "Trilock", "Flexibac",
]  # 78 classes, matching the dataset's Data Card on Kaggle (confirmed via browser on 2026-09-15)

SEED = 42
np.random.seed(SEED)


def _find_iam_words_txt():
    """words.txt is confirmed to live at IAM_ROOT/iam_words/words.txt (kernel v3 log, 2026-09-16).
    Try the known path FIRST, and only fall back to a recursive glob (slow, scans ~115K images)
    when it's actually needed."""
    direct = f"{IAM_ROOT}/iam_words/words.txt"
    if os.path.exists(direct):
        return [direct]
    return glob.glob(f"{IAM_ROOT}/**/words.txt", recursive=True)

# ============================================================
# PHASE 0 — DIAGNOSE THE DATA STRUCTURE (run first, always run)
# ============================================================
def phase0_diagnose():
    print("=" * 70)
    print("PHASE 0: Checking the structure of /kaggle/input/")
    print("=" * 70)
    print("os.listdir('/kaggle/input') =", os.listdir("/kaggle/input") if os.path.isdir("/kaggle/input") else "NO /kaggle/input DIRECTORY")
    if os.path.isdir("/kaggle/input/datasets"):
        for owner in os.listdir("/kaggle/input/datasets"):
            owner_path = f"/kaggle/input/datasets/{owner}"
            print(f"  /kaggle/input/datasets/{owner}/ ->", os.listdir(owner_path) if os.path.isdir(owner_path) else "?")
    print(f"RX_ROOT resolved to = {RX_ROOT}  (exists: {os.path.isdir(RX_ROOT)})")
    print(f"IAM_ROOT resolved to = {IAM_ROOT}  (exists: {os.path.isdir(IAM_ROOT)})")
    for root in [RX_ROOT, IAM_ROOT]:
        print(f"\n--- {root} ---")
        if not os.path.isdir(root):
            print("  DOES NOT EXIST - double-check that the dataset name was added correctly via Add Input.")
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            depth = dirpath.replace(root, "").count(os.sep)
            if depth >= 2:
                dirnames[:] = []  # stop os.walk from descending further (IAM has ~115K images, very slow without pruning)
            print(f"  {dirpath}/  ({len(filenames)} files, {len(dirnames)} subdirs)")
            for fn in filenames[:5]:
                print(f"      - {fn}")

    print("\n--- Trying to read .csv/.xlsx files in RX_ROOT ---")
    for path in glob.glob(f"{RX_ROOT}/**/*.csv", recursive=True) + glob.glob(f"{RX_ROOT}/**/*.xlsx", recursive=True):
        print(f"\n  File: {path}")
        try:
            df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
            print("  Columns:", list(df.columns))
            print(df.head(3).to_string())
        except Exception as e:
            print("  FILE READ ERROR:", e)

    print("\n--- Trying to read the IAM words.txt (standard format: word_id status graylevel x y w h tag transcription) ---")
    for path in _find_iam_words_txt()[:1]:
        print(f"  File: {path}")
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = [l for l in f.readlines() if not l.startswith("#")]
        print(f"  Line count (excluding comments): {len(lines)}")
        print("  First 3 lines:", lines[:3])


def guess_image_and_label_columns(df: pd.DataFrame, vocab: list[str]):
    """Heuristic: the image column is the column whose strings end in .jpg/.png/.jpeg;
    the label column is the column whose values overlap the most with the 78-drug vocabulary."""
    vocab_lower = {v.lower() for v in vocab}
    img_col, label_col = None, None
    best_overlap = -1
    for col in df.columns:
        sample = df[col].astype(str).head(200)
        if sample.str.lower().str.endswith((".jpg", ".jpeg", ".png")).mean() > 0.5:
            img_col = col
            continue
        overlap = sample.str.strip().str.lower().isin(vocab_lower).mean()
        if overlap > best_overlap:
            best_overlap = overlap
            label_col = col
    print(f"  -> Guessed: image_col={img_col!r}, label_col={label_col!r} (overlap with vocab: {best_overlap:.2%})")
    return img_col, label_col


# ============================================================
# METRICS — a copy of src/metrics.py (kept in sync in both places, see the README here)
# ============================================================
import jiwer
from rapidfuzz.distance import Levenshtein


def _normalize(s):
    return " ".join(str(s).strip().lower().split())


def cer(reference, hypothesis):
    ref, hyp = _normalize(reference), _normalize(hypothesis)
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return Levenshtein.distance(ref, hyp) / len(ref)


def wer(reference, hypothesis):
    ref, hyp = _normalize(reference), _normalize(hypothesis)
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return jiwer.wer(ref, hyp)


def exact_match(reference, hypothesis):
    return _normalize(reference) == _normalize(hypothesis)


def top1_label(hypothesis, vocab):
    hyp = _normalize(hypothesis)
    scores = [Levenshtein.normalized_distance(hyp, _normalize(v)) for v in vocab]
    return vocab[int(np.argmin(scores))]


def is_degenerate(hypothesis, max_len=200):
    hyp = _normalize(hypothesis)
    if len(hyp) == 0 or len(hyp) > max_len:
        return True
    tokens = hyp.split()
    if len(tokens) >= 6 and len(set(tokens)) <= max(1, len(tokens) // 6):
        return True
    return False


# ============================================================
# WRITING RESULTS — APPEND, not overwrite, to be resilient to session timeouts
# ============================================================
def append_results(rows: list[dict]):
    df_new = pd.DataFrame(rows)
    if os.path.exists(RESULTS_CSV):
        df_new.to_csv(RESULTS_CSV, mode="a", header=False, index=False)
    else:
        df_new.to_csv(RESULTS_CSV, mode="w", header=True, index=False)
    print(f"  Wrote {len(rows)} rows to {RESULTS_CSV}")


def run_model_on_manifest(model_name, predict_fn, manifest_df, dataset_name, batch_log_every=50):
    """predict_fn(image_path) -> str (the predicted text). A per-image error does not crash the whole batch."""
    rows = []
    t0 = time.time()
    for i, r in manifest_df.iterrows():
        try:
            hyp = predict_fn(r["image_path"])
        except Exception as e:
            hyp = ""
            print(f"  [ERROR on image {r['image_path']}]: {e}")
        row = {
            "model": model_name,
            "dataset": dataset_name,
            "image_path": r["image_path"],
            "reference": r["label"],
            "hypothesis": hyp,
            "cer": cer(r["label"], hyp),
            "wer": wer(r["label"], hyp),
            "exact_match": exact_match(r["label"], hyp),
            "degenerate": is_degenerate(hyp),
        }
        if dataset_name == "kaggle_rx":
            pred78 = top1_label(hyp, DRUG_VOCAB)
            row["top1_pred"] = pred78
            row["top1_correct"] = _normalize(pred78) == _normalize(r["label"])
        rows.append(row)
        if (i + 1) % batch_log_every == 0:
            elapsed = time.time() - t0
            print(f"  [{model_name}/{dataset_name}] {i+1}/{len(manifest_df)} images, {elapsed:.1f}s, "
                  f"running average CER = {np.mean([x['cer'] for x in rows]):.3f}")
    append_results(rows)
    return pd.DataFrame(rows)


# ============================================================
# BUILDING THE MANIFEST — auto-detects the image/label columns, no need to know the column names in advance
# ============================================================
def build_manifest_kaggle_rx(split_dirname_prefix, n_sample=None):
    """split_dirname_prefix: 'Testing', 'Training', or 'Validation'.
    The dataset actually nests an extra intermediate directory named after the dataset title
    (e.g. '.../Doctor's Handwritten Prescription BD dataset/Testing/...') - confirmed via an
    actual run on 2026-09-16 (kernel v3 log) - so we search recursively instead of assuming
    Testing/Training/Validation sit directly under RX_ROOT."""
    label_files = glob.glob(f"{RX_ROOT}/**/{split_dirname_prefix}/*.csv", recursive=True) + \
                  glob.glob(f"{RX_ROOT}/**/{split_dirname_prefix}/*.xlsx", recursive=True)
    if not label_files:
        raise FileNotFoundError(f"No label file found under {RX_ROOT}/**/{split_dirname_prefix}/")
    path = label_files[0]
    df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
    img_col, label_col = guess_image_and_label_columns(df, DRUG_VOCAB)
    split_dir = os.path.dirname(path)  # the actual Testing/Training/Validation directory
    img_dir_candidates = [d for d in glob.glob(f"{split_dir}/*/") if os.path.isdir(d)]
    img_dir = img_dir_candidates[0] if img_dir_candidates else split_dir
    print(f"  split_dir={split_dir!r}, img_dir={img_dir!r}")

    def resolve_path(fname):
        fname = str(fname).strip()
        direct = os.path.join(img_dir, fname)
        if os.path.exists(direct):
            return direct
        hits = glob.glob(f"{img_dir}/**/{fname}", recursive=True)
        return hits[0] if hits else direct  # may not exist - will raise an error when the image is read

    manifest = pd.DataFrame({
        "image_path": df[img_col].map(resolve_path),
        "label": df[label_col],
    })
    manifest = manifest[manifest["image_path"].map(os.path.exists)].reset_index(drop=True)
    print(f"  build_manifest_kaggle_rx({split_dirname_prefix}): {len(manifest)}/{len(df)} images found on disk")
    if n_sample and n_sample < len(manifest):
        manifest = manifest.sample(n=n_sample, random_state=SEED).reset_index(drop=True)
    return manifest


def build_manifest_iam(n_sample=400):
    """Reads the standard IAM words.txt format: word_id status graylevel x y w h tag transcription.
    word_id looks like a01-000u-00-00 -> image at IAM_ROOT/iam_words/words/a01/a01-000u/a01-000u-00-00.png
    (structure confirmed via kernel v3 log, 2026-09-16).

    IMPORTANT PERFORMANCE NOTE: parse and sample FIRST, only hit disk (os.path.exists/glob)
    AFTER sampling down to n_sample rows — words.txt has 44,565 lines, and globbing for every
    line before sampling would be slow (44K recursive globs) when we only need a few hundred
    images."""
    words_txt = _find_iam_words_txt()
    if not words_txt:
        raise FileNotFoundError(f"words.txt not found under {IAM_ROOT}")
    words_dir = f"{IAM_ROOT}/iam_words/words"  # fallback: search for it if the structure differs

    parsed = []
    with open(words_txt[0], encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split(" ")
            if len(parts) < 9:
                continue  # line doesn't have all 9 fields per the standard IAM format - skip it, don't crash
            word_id, status, transcription = parts[0], parts[1], parts[-1]
            if status == "ok":
                parsed.append((word_id, transcription))
    print(f"  words.txt: {len(parsed)} valid 'ok' lines (before sampling)")

    rng = np.random.default_rng(SEED)
    if n_sample and n_sample < len(parsed):
        idx = rng.choice(len(parsed), size=n_sample, replace=False)
        parsed = [parsed[i] for i in idx]

    rows = []
    for word_id, transcription in parsed:
        a, b = word_id.split("-")[0], "-".join(word_id.split("-")[:2])
        direct = f"{words_dir}/{a}/{b}/{word_id}.png"
        if os.path.exists(direct):
            path = direct
        else:
            hits = glob.glob(f"{IAM_ROOT}/**/{word_id}.png", recursive=True)  # rare, slow fallback
            path = hits[0] if hits else None
        if path:
            rows.append({"image_path": path, "label": transcription})
    manifest = pd.DataFrame(rows)
    print(f"  build_manifest_iam: {len(manifest)}/{len(parsed)} sampled images found on disk")
    return manifest


# ============================================================
# PHASE 1 — Classical OCR: Tesseract (+ EasyOCR if pip-installed)
# The previous cell must have run: !apt-get install -y tesseract-ocr -q && pip install pytesseract easyocr -q
# ============================================================
def phase1_classical_ocr():
    import pytesseract
    from PIL import Image

    def tesseract_predict(path):
        return pytesseract.image_to_string(Image.open(path), config="--psm 7").strip()

    rx_test = build_manifest_kaggle_rx("Testing")
    iam_sub = build_manifest_iam(n_sample=400)

    print("\n--- Tesseract on Kaggle-Rx (Testing, full set) ---")
    run_model_on_manifest("tesseract", tesseract_predict, rx_test, "kaggle_rx")
    print("\n--- Tesseract on IAM (subsample of 400) ---")
    run_model_on_manifest("tesseract", tesseract_predict, iam_sub, "iam")

    try:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=True)

        def easyocr_predict(path):
            res = reader.readtext(path, detail=0)
            return " ".join(res)

        print("\n--- EasyOCR on Kaggle-Rx (Testing, full set) ---")
        run_model_on_manifest("easyocr", easyocr_predict, rx_test, "kaggle_rx")
        print("\n--- EasyOCR on IAM (subsample of 400) ---")
        run_model_on_manifest("easyocr", easyocr_predict, iam_sub, "iam")
    except ImportError:
        print("easyocr is not installed yet — skipping; rerun after `pip install easyocr`.")


# ============================================================
# PHASE 2 — TrOCR-large-handwritten + Donut-base (zero-shot, task <s_synthdog>)
# ============================================================
def phase2_trocr_donut():
    _pip_install("transformers", "accelerate", "sentencepiece", "protobuf")
    import torch
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device = {device}")

    rx_test = build_manifest_kaggle_rx("Testing")
    iam_sub = build_manifest_iam(n_sample=400)

    # --- TrOCR ---
    if not RUN_TROCR:
        print("  RUN_TROCR=False - skipping (already has good results from a previous run, see results_master_phase2.csv)")
    else:
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
            trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten").to(device)
            trocr_model.eval()

            @torch.no_grad()
            def trocr_predict(path):
                image = Image.open(path).convert("RGB")
                pixel_values = trocr_processor(images=image, return_tensors="pt").pixel_values.to(device)
                ids = trocr_model.generate(pixel_values, max_new_tokens=32)
                return trocr_processor.batch_decode(ids, skip_special_tokens=True)[0]

            print("\n--- TrOCR-large-handwritten on Kaggle-Rx (Testing, full set) ---")
            run_model_on_manifest("trocr-large-handwritten", trocr_predict, rx_test, "kaggle_rx")
            print("\n--- TrOCR-large-handwritten on IAM (subsample of 400) ---")
            run_model_on_manifest("trocr-large-handwritten", trocr_predict, iam_sub, "iam")
            del trocr_model
            torch.cuda.empty_cache() if device == "cuda" else None
        except Exception as e:
            print(f"  [TrOCR ERROR, skipping this model]: {e}")
            traceback.print_exc()

    # --- Donut-base (zero-shot document reading via the <s_synthdog> task prompt, NOT the
    #     fine-tuned CORD checkpoint — the CORD checkpoint does receipt field extraction, a
    #     different goal from the free-text OCR here) ---
    #
    # FINDING (kernel v6, 2026-09-16): the "standard" approach (feeding the small crop directly
    # into the processor) produces an EMPTY hypothesis ~100% of the time (degenerate_rate ~1.0 on
    # both datasets). Hypothesis: donut-base (only pretrained on SynthDoG, NOT fine-tuned) was
    # trained on full document page images (page-shaped, wide aspect ratio) — a small, roughly
    # square single-word crop is far enough outside that input distribution that the processor's
    # resize/pad turns it into an almost-blank "page" -> the model immediately predicts EOS. Test
    # BOTH approaches, report both, to make clear this is a "wrong input shape" issue rather than
    # a plain code bug.
    try:
        import re as _re
        from transformers import DonutProcessor, VisionEncoderDecoderModel as DonutVED

        donut_processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
        donut_model = DonutVED.from_pretrained("naver-clova-ix/donut-base").to(device)
        donut_model.eval()
        task_prompt = "<s_synthdog>"
        decoder_input_ids = donut_processor.tokenizer(
            task_prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(device)

        def _pad_to_page_canvas(image, canvas_size=(1280, 960)):
            """Pastes the small crop into the middle of a blank 'page' — mimicking the aspect ratio SynthDoG was trained on."""
            canvas = Image.new("RGB", canvas_size, (255, 255, 255))
            w, h = image.size
            scale = min(canvas_size[0] * 0.6 / w, canvas_size[1] * 0.6 / h, 4.0)
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            resized = image.resize((new_w, new_h))
            canvas.paste(resized, ((canvas_size[0] - new_w) // 2, (canvas_size[1] - new_h) // 2))
            return canvas

        @torch.no_grad()
        def _donut_generate(pixel_values):
            outputs = donut_model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=64,
                pad_token_id=donut_processor.tokenizer.pad_token_id,
                eos_token_id=donut_processor.tokenizer.eos_token_id,
                use_cache=True,
                return_dict_in_generate=True,
            )
            seq = donut_processor.batch_decode(outputs.sequences)[0]
            seq = seq.replace(donut_processor.tokenizer.eos_token, "").replace(donut_processor.tokenizer.pad_token, "")
            seq = _re.sub(r"<.*?>", "", seq, count=1).strip()
            return seq

        def donut_predict_raw(path):
            image = Image.open(path).convert("RGB")
            pixel_values = donut_processor(image, return_tensors="pt").pixel_values.to(device)
            return _donut_generate(pixel_values)

        def donut_predict_padded(path):
            image = _pad_to_page_canvas(Image.open(path).convert("RGB"))
            pixel_values = donut_processor(image, return_tensors="pt").pixel_values.to(device)
            return _donut_generate(pixel_values)

        # Quick debug: print 5 raw outputs (not yet through run_model_on_manifest) for a visual sanity check
        print("\n--- Donut DEBUG: first 5 examples from rx_test, both approaches (raw vs padded) ---")
        for i in range(min(5, len(rx_test))):
            row = rx_test.iloc[i]
            raw = donut_predict_raw(row["image_path"])
            padded = donut_predict_padded(row["image_path"])
            print(f"  ref={row['label']!r}  raw={raw!r}  padded={padded!r}")

        if RUN_DONUT_RAW_FULL:
            print("\n--- Donut-base (<s_synthdog>, RAW - direct crop) on Kaggle-Rx ---")
            run_model_on_manifest("donut-base-synthdog-raw", donut_predict_raw, rx_test, "kaggle_rx")
            print("\n--- Donut-base (<s_synthdog>, RAW) on IAM (subsample of 400) ---")
            run_model_on_manifest("donut-base-synthdog-raw", donut_predict_raw, iam_sub, "iam")
        else:
            print("  RUN_DONUT_RAW_FULL=False - skipping (already have 'raw' data from kernel v6, see results_master_phase2.csv, model='donut-base-synthdog')")

        print("\n--- Donut-base (<s_synthdog>, PADDED - pasted onto a blank 'page') on Kaggle-Rx ---")
        run_model_on_manifest("donut-base-synthdog-padded", donut_predict_padded, rx_test, "kaggle_rx")
        print("\n--- Donut-base (<s_synthdog>, PADDED) on IAM (subsample of 400) ---")
        run_model_on_manifest("donut-base-synthdog-padded", donut_predict_padded, iam_sub, "iam")
    except Exception as e:
        print(f"  [Donut ERROR, skipping this model]: {e}")
        traceback.print_exc()


# ============================================================
# PHASE 3 — GOT-OCR2.0 (stepfun-ai, ~580M, a new-generation OCR-specialized VLM)
# ============================================================
def phase3_got_ocr2():
    # IMPORTANT (kernel v10): the latest transformers from pip is 5.0.0 (a very recent major
    # version, 09/2026). GOT-OCR2.0 was merged into transformers on 2025-01-31 (PR #34721) and
    # was stable across several 4.x releases (~4.57.0), but is not confirmed compatible with the
    # 5.0.0 major bump (there could be an internal breaking change in generate()/cache handling
    # that broke modeling_got_ocr2.py). The prompt/image were confirmed correct (kernel v10:
    # input_ids has the right <img>...OCR:... tokens, pixel_values has the right shape), but the
    # output was still complete nonsense -> pin to 4.57.0 (stable, after this model was merged
    # but before the v5 jump) instead of letting pip pick the latest version.
    _pip_install("transformers==4.57.0", "accelerate")
    import torch
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device = {device}")

    rx_test = build_manifest_kaggle_rx("Testing")
    iam_sub = build_manifest_iam(n_sample=400)

    try:
        import transformers as _tf
        print(f"  transformers.__version__ = {_tf.__version__}, torch.__version__ = {torch.__version__}")

        from transformers import AutoProcessor, AutoModelForImageTextToText

        got_processor = AutoProcessor.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf")
        # IMPORTANT: the T4 (Turing architecture) does NOT have tensor cores with genuine
        # bfloat16 support — kernel v8 used bfloat16 and produced garbled, multi-language,
        # completely wrong output (CER ~55-120, i.e. tens of times longer than the reference) —
        # suspected silent numerical error from forcing bf16 on hardware that doesn't properly
        # support it. Switched to float16 (T4 supports this well, Turing has real fp16 Tensor
        # Cores).
        # KERNEL v9: fp16 still produced similarly garbled output -> not (solely) a dtype issue.
        # Trying again: (a) pass the path/PIL image directly as in the official example (no
        # manual .convert("RGB") beforehand), (b) print the shape/dtype of pixel_values to check
        # whether the image is actually being encoded or the model is "ignoring" the image and
        # generating text from scratch (which would explain the "garbled, multi-language"
        # failure mode seen).
        got_model = AutoModelForImageTextToText.from_pretrained(
            "stepfun-ai/GOT-OCR-2.0-hf", dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        got_model.eval()

        @torch.no_grad()
        def got_predict(path, verbose=False):
            image = Image.open(path).convert("RGB")
            inputs = got_processor(image, return_tensors="pt").to(device)
            if verbose:
                print(f"    inputs.keys()={list(inputs.keys())}")
                for k, v in inputs.items():
                    if hasattr(v, "shape"):
                        print(f"    {k}: shape={tuple(v.shape)} dtype={v.dtype}")
                prompt_text = got_processor.tokenizer.decode(inputs["input_ids"][0])
                print(f"    decoded prompt (input_ids before generate) = {prompt_text!r}")
            generate_ids = got_model.generate(
                **inputs,
                do_sample=False,
                tokenizer=got_processor.tokenizer,
                stop_strings="<|im_end|>",
                max_new_tokens=32,
            )
            text = got_processor.decode(
                generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            return text.strip()

        print("\n--- Debug: first 5 examples of GOT-OCR2.0 on rx_test (dtype=float16) ---")
        debug_preds = []
        for i in range(min(5, len(rx_test))):
            row = rx_test.iloc[i]
            pred = got_predict(row["image_path"], verbose=(i == 0))
            debug_preds.append(pred)
            print(f"  ref={row['label']!r}  pred={pred!r}")

        _assert_sane_debug_output("GOT-OCR2.0", debug_preds)

        print("\n--- GOT-OCR2.0 on Kaggle-Rx (Testing, full set) ---")
        run_model_on_manifest("got-ocr2.0", got_predict, rx_test, "kaggle_rx")
        print("\n--- GOT-OCR2.0 on IAM (subsample of 400) ---")
        run_model_on_manifest("got-ocr2.0", got_predict, iam_sub, "iam")
    except Exception as e:
        print(f"  [GOT-OCR2.0 ERROR, skipping this model]: {e}")
        traceback.print_exc()


# ============================================================
# PHASE 4 — PaddleOCR-VL (0.9B, element-level recognition via transformers)
# ============================================================
def phase4_paddleocr_vl():
    # ============================================================================
    # REMOVED FROM THE BENCHMARK (2026-09-16) - UPSTREAM COMPATIBILITY BUG, NOT A BUG IN THIS CODE.
    # ============================================================================
    # kernel v12: transformers==4.57.0 -> TypeError: create_causal_mask() got an unexpected
    # keyword argument 'inputs_embeds' (in modeling_paddleocr_vl.py from the remote_code of the
    # HF repo PaddlePaddle/PaddleOCR-VL itself, not our code).
    # Looked it up: this is a bug the COMMUNITY HAS ALREADY REPORTED, with NO FIX so far -
    # see the discussion "Newest commit breaks compatibility with transformers==4.57.6,
    # while 5.3.0 is broken as well" at
    # https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5/discussions/22
    # (posted 04/30, no recorded fix; quote: "Anything above transformers==5
    # with AutoModelForImageTextToText has seemingly always been broken"). In other words,
    # PaddleOCR-VL via the transformers/trust_remote_code path currently does not work reliably
    # on ANY transformers version tried (both before and after 5.0), unrelated to our T4/dtype/prompt.
    # DECISION: REMOVE PaddleOCR-VL from the zero-shot benchmark (not worth the cost of debugging
    # an unresolved upstream bug further) - move on to Phase 5 (Qwen-VL, a much more stable
    # ecosystem, with existing precedent of Unsloth running on T4). The code is kept here for
    # reference/retry later if PaddlePaddle releases a fix.
    print("  [SKIPPED] PaddleOCR-VL removed from the benchmark due to an upstream compatibility "
          "bug (transformers/trust_remote_code) with no fix yet - see the code comment and "
          "results/phase4_summary.md for details and references.")
    return

    # --- Original code, kept in case we want to retry if PaddlePaddle fixes this in the future ---
    _pip_install("transformers==4.57.0", "accelerate")
    import torch
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device = {device}")

    rx_test = build_manifest_kaggle_rx("Testing")
    iam_sub = build_manifest_iam(n_sample=400)

    try:
        from transformers import AutoModelForCausalLM, AutoProcessor

        model_path = "PaddlePaddle/PaddleOCR-VL"
        pvl_model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device).eval()
        pvl_processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        @torch.no_grad()
        def pvl_predict(path):
            image = Image.open(path).convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "OCR:"},
            ]}]
            inputs = pvl_processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(device)
            outputs = pvl_model.generate(**inputs, max_new_tokens=32)
            text = pvl_processor.batch_decode(
                outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )[0]
            return text.strip()

        print("\n--- Debug: first 5 examples of PaddleOCR-VL on rx_test ---")
        debug_preds = []
        for i in range(min(5, len(rx_test))):
            row = rx_test.iloc[i]
            pred = pvl_predict(row["image_path"])
            debug_preds.append(pred)
            print(f"  ref={row['label']!r}  pred={pred!r}")

        _assert_sane_debug_output("PaddleOCR-VL", debug_preds)

        print("\n--- PaddleOCR-VL on Kaggle-Rx (Testing, full set) ---")
        run_model_on_manifest("paddleocr-vl", pvl_predict, rx_test, "kaggle_rx")
        print("\n--- PaddleOCR-VL on IAM (subsample of 400) ---")
        run_model_on_manifest("paddleocr-vl", pvl_predict, iam_sub, "iam")
    except Exception as e:
        print(f"  [PaddleOCR-VL ERROR, skipping this model]: {e}")
        traceback.print_exc()


# ============================================================
# PHASE 5 — Qwen2.5-VL-3B-Instruct (a general-purpose VLM, as a control against the specialized OCR models)
# ============================================================
def phase5_qwen_vl():
    # Using the 3B version (not 7B/8B) - fp16 is ~6GB, safe on a 16GB T4 for zero-shot inference
    # (no fine-tuning, so no need to worry about optimizer state). Qwen2.5-VL is a first-class
    # model in transformers (no trust_remote_code needed) - lower risk of API breakage than
    # GOT-OCR2.0/PaddleOCR-VL.
    _pip_install("transformers==4.57.0", "accelerate", "qwen-vl-utils")
    import torch
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device = {device}")

    rx_test = build_manifest_kaggle_rx("Testing")
    iam_sub = build_manifest_iam(n_sample=400)

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
        qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device).eval()
        qwen_processor = AutoProcessor.from_pretrained(model_path)

        QUESTION = "Read the handwritten/printed text in this image. Output ONLY the text itself, nothing else."

        @torch.no_grad()
        def qwen_predict(path):
            image = Image.open(path).convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": QUESTION},
            ]}]
            text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = qwen_processor(text=[text], images=[image], return_tensors="pt").to(device)
            generated_ids = qwen_model.generate(**inputs, max_new_tokens=32)
            trimmed = generated_ids[:, inputs["input_ids"].shape[1]:]
            return qwen_processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

        print("\n--- Debug: first 5 examples of Qwen2.5-VL-3B on rx_test ---")
        debug_preds = []
        for i in range(min(5, len(rx_test))):
            row = rx_test.iloc[i]
            pred = qwen_predict(row["image_path"])
            debug_preds.append(pred)
            print(f"  ref={row['label']!r}  pred={pred!r}")

        _assert_sane_debug_output("Qwen2.5-VL-3B", debug_preds, max_avg_len=100)  # a general-purpose VLM might produce output longer than a single word

        print("\n--- Qwen2.5-VL-3B on Kaggle-Rx (Testing, full set) ---")
        run_model_on_manifest("qwen2.5-vl-3b", qwen_predict, rx_test, "kaggle_rx")
        print("\n--- Qwen2.5-VL-3B on IAM (subsample of 400) ---")
        run_model_on_manifest("qwen2.5-vl-3b", qwen_predict, iam_sub, "iam")
    except Exception as e:
        print(f"  [Qwen2.5-VL ERROR, skipping this model]: {e}")
        traceback.print_exc()


# ============================================================
# PHASE 6 — LoRA fine-tuning of TrOCR-large-handwritten (the main methodological contribution for Q2)
#
# Confirmed (Phase 6 prep, 2026-09-16): Training=3120 images, Validation=780 images, Testing=780
# images (NOT 2808/936/936 as originally assumed from the "60/20/20" description on Kaggle - see
# results/phase6_summary.md). The test set is conceptually "frozen": build_manifest_kaggle_rx
# ("Testing") and build_manifest_iam(n_sample=400, seed=42) have been confirmed to return
# EXACTLY the same 780/400 images across all 7 previous model runs (Phase 1-5) - so no separate
# frozen file is needed; simply calling this function again with the same seed reproduces the
# same test set.
# ============================================================
def phase6_finetune_trocr(lora_r=16, lora_alpha=32, use_elastic=True, run_name="trocr-lora-finetuned",
                           seed=SEED):
    print(f"\n=== phase6_finetune_trocr: lora_r={lora_r}, lora_alpha={lora_alpha}, "
          f"use_elastic={use_elastic}, run_name={run_name!r}, seed={seed} ===")
    output_subdir = run_name.replace("trocr-lora-finetuned", "trocr-lora")  # keeps the old path for the "main" configuration
    # torchao>=0.16.0 is required: the Kaggle base image ships with torchao==0.10.0 (old), and
    # the latest peft refuses to run with the old version (kernel v14: ImportError). Force the
    # torchao upgrade at the same time.
    _pip_install("transformers==4.57.0", "accelerate", "peft", "torchao>=0.16.0",
                 "albumentations", "opencv-python-headless")
    import torch
    from PIL import Image
    import albumentations as A
    import cv2  # noqa: F401 (needed for albumentations to read images/handle border mode)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device = {device}")
    if device != "cuda":
        print("  [WARNING] No GPU available - fine-tuning will be very slow/possibly infeasible, but will still attempt it.")

    # --- A synced copy of src/augmentation.py (reason for keeping 2 copies: see src/metrics.py) ---
    def build_train_augmentation(elastic=True):
        steps = [
            A.Affine(rotate=(-5, 5), shear=(-8, 8), scale=(0.95, 1.05), p=0.7),
            A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.OneOf([
                A.Morphological(scale=(1, 2), operation="erosion", p=1.0),
                A.Morphological(scale=(1, 2), operation="dilation", p=1.0),
            ], p=0.3),
        ]
        if elastic:
            steps.insert(1, A.ElasticTransform(alpha=30, sigma=5, p=0.3))
        return A.Compose(steps)

    def augment_pil(image, augmenter):
        arr = np.array(image.convert("RGB"))
        return Image.fromarray(augmenter(image=arr)["image"])

    # --- Data: Training (train, with augmentation) + Validation subsample (monitored during training) ---
    train_df = build_manifest_kaggle_rx("Training")
    val_df_full = build_manifest_kaggle_rx("Validation")
    val_df = val_df_full.sample(n=min(300, len(val_df_full)), random_state=seed).reset_index(drop=True)
    print(f"  train_df={len(train_df)}, val_df_full={len(val_df_full)}, val_df (monitoring subsample)={len(val_df)}")

    from transformers import (
        TrOCRProcessor, VisionEncoderDecoderModel, VisionEncoderDecoderConfig,
        Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback, TrainerCallback,
        set_seed,
    )
    from peft import LoraConfig, get_peft_model

    # ROOT-CAUSE FIX (kernel v16-v17: 'VisionEncoderDecoderConfig' object has no attribute
    # 'vocab_size', recurring on EVERY checkpoint save, not just the first): peft's
    # get_peft_model_state_dict() checks "does this need embedding resizing" by reading
    # model.config.__class__.from_pretrained(id).vocab_size on a FRESHLY-LOADED CONFIG INSTANCE
    # every time it saves — setting the attribute on the instance (as done below) only fixes the
    # current instance, not this newly-loaded one.
    # Instead, patch a vocab_size property onto the VisionEncoderDecoderConfig CLASS itself
    # (delegating to decoder.vocab_size) so that EVERY instance, including ones freshly loaded
    # inside peft, already has it.
    if not hasattr(VisionEncoderDecoderConfig, "_vocab_size_patched_for_peft"):
        VisionEncoderDecoderConfig.vocab_size = property(lambda self: self.decoder.vocab_size)
        VisionEncoderDecoderConfig._vocab_size_patched_for_peft = True

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
    base_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
    # REQUIRED for fine-tuning VisionEncoderDecoderModel (kernel v15: ValueError if missing) -
    # zero-shot generate() (Phase 2) doesn't need this since it's inferred automatically, but
    # training (shift_tokens_right for teacher-forcing labels) needs the full config. Following
    # the standard TrOCR convention (NielsRogge/Transformers-Tutorials fine-tuning notebook,
    # already cited in the plan).
    base_model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    base_model.config.pad_token_id = processor.tokenizer.pad_token_id
    base_model.config.eos_token_id = processor.tokenizer.sep_token_id
    # No longer assigning base_model.config.vocab_size = ... — vocab_size is now a CLASS-level
    # property (patched above); assigning it directly on the instance would raise AttributeError:
    # can't set attribute.
    MAX_TARGET_LEN = 32

    class RxTorchDataset(torch.utils.data.Dataset):
        def __init__(self, df, augmenter=None):
            self.df = df.reset_index(drop=True)
            self.augmenter = augmenter

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            image = Image.open(row["image_path"]).convert("RGB")
            if self.augmenter is not None:
                image = augment_pil(image, self.augmenter)
            pixel_values = processor(image, return_tensors="pt").pixel_values.squeeze(0)
            label_ids = processor.tokenizer(
                str(row["label"]), padding="max_length", truncation=True, max_length=MAX_TARGET_LEN
            ).input_ids
            label_ids = [l if l != processor.tokenizer.pad_token_id else -100 for l in label_ids]
            return {"pixel_values": pixel_values, "labels": torch.tensor(label_ids)}

    def compute_metrics(pred):
        label_ids = pred.label_ids.copy()
        pred_ids = pred.predictions.copy()
        # When eval batches generate sequences of different lengths, the Trainer concatenates
        # predictions across batches by padding the shorter ones with -100
        # (torch_pad_and_concatenate) — this negative value makes tokenizer.decode overflow
        # (OverflowError: out of range integral type conversion attempted, hit in kernel v19
        # epoch 10). Must filter out -100 before decoding, the same way it's already done for
        # label_ids below.
        pred_ids[pred_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        cers = [cer(l, p) for l, p in zip(label_str, pred_str)]
        return {"cer": float(np.mean(cers))}

    class TimeLimitCallback(TrainerCallback):
        """Safety brake: Kaggle sessions max out at ~9h — stop training early (without crashing)
        if the time limit is exceeded, so there's still time left afterward for the
        frozen-test-set evaluation step instead of losing the page entirely."""
        def __init__(self, max_seconds):
            self.max_seconds = max_seconds
            self.t0 = time.time()

        def on_step_end(self, args, state, control, **kwargs):
            elapsed = time.time() - self.t0
            if elapsed > self.max_seconds:
                print(f"  [TimeLimitCallback] Exceeded {self.max_seconds}s ({elapsed:.0f}s) - STOPPING training early.")
                control.should_training_stop = True
            return control

    # Seed torch/numpy/python RNGs right before LoRA's A/B matrices are randomly initialized, so
    # that a different `seed` here actually produces a different LoRA init (needed for the
    # multi-seed replication in the Limitations section - a single set_seed() call at module
    # import time is NOT enough, since get_peft_model() draws fresh random weights at this exact
    # point, not at import time).
    set_seed(seed)
    lora_config = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.1,
        target_modules=["query", "value", "q_proj", "v_proj"],
        bias="none", task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(base_model, lora_config).to(device)
    # Fix for a peft<->VisionEncoderDecoderConfig compatibility bug (kernel v16: AttributeError
    # during save_pretrained/checkpoint — peft has a "vocab_size resize" check step that reloads
    # AutoConfig.from_pretrained(model_id).vocab_size, but VisionEncoderDecoderConfig (a
    # composite encoder/decoder architecture) has no top-level vocab_size attribute of its own.
    # Setting base_model_name_or_path=None makes peft SKIP this check entirely (safe, since we
    # aren't resizing the embedding).
    for _adapter_name in model.peft_config:
        model.peft_config[_adapter_name].base_model_name_or_path = None
    model.print_trainable_parameters()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_trainable == 0:
        # A common PEFT pitfall: target_modules doesn't match the actual module names -> 0 LoRA
        # parameters get created, and "training" runs without learning anything. Check
        # IMMEDIATELY, don't let it continue.
        names_sample = [n for n, _ in base_model.named_modules()][:80]
        raise RuntimeError(
            f"LoRA has 0 trainable parameters - target_modules={lora_config.target_modules} did "
            f"not match any module name in the model. First 80 module names: {names_sample}"
        )

    # ============================================================
    # SMOKE TEST first (lesson from Phase 3/4): a few dozen steps on a small subset, checking
    # that the loss is finite and decreasing before committing several hours of GPU time to
    # full training.
    # ============================================================
    print("\n--- SMOKE TEST: 20 steps on 64 training images ---")
    smoke_train = RxTorchDataset(train_df.sample(n=min(64, len(train_df)), random_state=seed),
                                  augmenter=build_train_augmentation(elastic=use_elastic))
    smoke_args = Seq2SeqTrainingArguments(
        output_dir=f"/kaggle/working/smoke-{output_subdir}", per_device_train_batch_size=8,
        max_steps=20, logging_steps=5, save_strategy="no", eval_strategy="no",
        fp16=(device == "cuda"), report_to=[], seed=seed,
    )
    smoke_trainer = Seq2SeqTrainer(model=model, args=smoke_args, train_dataset=smoke_train)
    t0 = time.time()
    smoke_result = smoke_trainer.train()
    smoke_elapsed = time.time() - t0
    loss_hist = [h["loss"] for h in smoke_trainer.state.log_history if "loss" in h]
    print(f"  Smoke test: {smoke_elapsed:.1f}s / 20 steps -> ~{smoke_elapsed/20:.2f}s/step. "
          f"Loss history: {loss_hist}")
    if not loss_hist or not all(np.isfinite(loss_hist)):
        raise RuntimeError(f"Smoke test: loss is non-finite/empty ({loss_hist}) - STOPPING, NOT running full training.")
    if len(loss_hist) >= 2 and loss_hist[-1] > loss_hist[0] * 1.5:
        print(f"  [WARNING] Loss increased instead of decreasing ({loss_hist[0]:.3f} -> {loss_hist[-1]:.3f}) - "
              f"learning_rate might be too high, but proceeding with full training anyway (will monitor eval CER).")
    steps_per_epoch = len(train_df) / 16  # effective batch = 8 * grad_accum(2)
    print(f"  Estimate: ~{steps_per_epoch:.0f} steps/epoch, ~{steps_per_epoch*smoke_elapsed/20/60:.1f} minutes/epoch")

    # ============================================================
    # ACTUAL TRAINING (after the smoke test passes)
    # ============================================================
    train_ds = RxTorchDataset(train_df, augmenter=build_train_augmentation(elastic=use_elastic))
    val_ds = RxTorchDataset(val_df, augmenter=None)

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"/kaggle/working/{output_subdir}",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.08,
        num_train_epochs=15,
        fp16=(device == "cuda"),
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LEN,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        report_to=[],
        seed=seed,
    )
    trainer = Seq2SeqTrainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5), TimeLimitCallback(max_seconds=5 * 3600)],
    )
    print(f"\n--- ACTUAL TRAINING: LoRA fine-tune TrOCR-large-handwritten "
          f"(r={lora_r}, alpha={lora_alpha}, elastic={use_elastic}) ---")
    t_train0 = time.time()
    trainer.train()
    print(f"  Training finished after {time.time() - t_train0:.0f}s. Best eval CER: "
          f"{trainer.state.best_metric}")

    # Save the LoRA adapter right away (before doing anything else that could fail) - this is the most important "deliverable".
    adapter_dir = f"/kaggle/working/{output_subdir}-adapter"
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    print(f"  Saved adapter to {adapter_dir}")

    # ============================================================
    # EVALUATE ON BOTH FROZEN TEST SETS (reusing the EXACT SAME build_manifest_* functions as
    # Phase 1-5 — confirmed to return the same 780/400 images every call, see the Phase 6
    # docstring above).
    # ============================================================
    try:
        model.eval()

        @torch.no_grad()
        def trocr_lora_predict(path):
            image = Image.open(path).convert("RGB")
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
            # PeftModelForSeq2SeqLM.generate() ONLY accepts keyword args, not positional ones
            # (unlike the original VisionEncoderDecoderModel.generate() used in Phase 2 - kernel
            # v18: TypeError "takes 1 positional argument but 2 were given" on ALL 1180/1180
            # images, leaving every hypothesis empty).
            ids = model.generate(pixel_values=pixel_values, max_new_tokens=MAX_TARGET_LEN)
            return processor.batch_decode(ids, skip_special_tokens=True)[0]

        rx_test = build_manifest_kaggle_rx("Testing")
        iam_sub = build_manifest_iam(n_sample=400)
        print(f"\n--- {run_name} on Kaggle-Rx (Testing, full set, frozen) ---")
        run_model_on_manifest(run_name, trocr_lora_predict, rx_test, "kaggle_rx")
        print(f"\n--- {run_name} on IAM (subsample of 400, frozen) - checking for catastrophic forgetting ---")
        run_model_on_manifest(run_name, trocr_lora_predict, iam_sub, "iam")
    except Exception as e:
        print(f"  [ERROR while evaluating the frozen test set, but the adapter WAS already saved safely above]: {e}")
        traceback.print_exc()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    phase0_diagnose()  # always run first to log the data structure, regardless of CURRENT_PHASE
    if CURRENT_PHASE == 0:
        print("\n>>> Carefully read the output above. If image_col/label_col was guessed incorrectly "
              "(see the line '-> Guessed: image_col=...'), fix the guess_image_and_label_columns() "
              "function or hardcode the correct values, then set CURRENT_PHASE=1 and rerun.")
    elif CURRENT_PHASE == 1:
        phase1_classical_ocr()
    elif CURRENT_PHASE == 2:
        phase2_trocr_donut()
    elif CURRENT_PHASE == 3:
        phase3_got_ocr2()
    elif CURRENT_PHASE == 4:
        phase4_paddleocr_vl()
    elif CURRENT_PHASE == 5:
        phase5_qwen_vl()
    elif CURRENT_PHASE == 6:
        phase6_finetune_trocr(**_ABLATION_PRESETS[ABLATION_CONFIG])
    elif CURRENT_PHASE >= 7:
        print(f"PHASE {CURRENT_PHASE} has not been added to this script yet. See docs/05-q2-research-plan.md.")
