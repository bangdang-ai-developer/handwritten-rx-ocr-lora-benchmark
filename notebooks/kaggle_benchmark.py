"""
Kaggle Notebook script — Ngay 1-9 cua ke hoach 2 tuan (docs/05-ke-hoach-2-tuan.md).

CACH DUNG:
1. Vao kaggle.com -> Create -> New Notebook.
2. O panel phai "Input", bam "+ Add Input", tim va them 2 dataset:
     - mamun1113/doctors-handwritten-prescription-bd-dataset   (trong tam chinh)
     - nibinv23/iam-handwriting-word-database                   (doi chung domain-shift)
3. Settings (panel phai) -> Accelerator -> chon "GPU T4 x2".
4. Dan TOAN BO noi dung file nay vao 1 cell va chay (Run All).
   Chay theo tung PHASE (dat CURRENT_PHASE ben duoi) de khong vuot quota GPU/session
   9h cua Kaggle - moi phase ghi ket qua vao results/results_master.csv (append,
   khong ghi de) nen co the chay nhieu lan/nhieu ngay khac nhau.
5. Sau moi phase, bam "Save Version" -> "Save & Run All" de luu output, hoac
   download truc tiep results_master.csv qua panel Output.

QUAN TRONG: PHASE 0 la buoc CHAN DOAN - chay truoc, doc output, xac nhan
ten cot dung voi thuc te truoc khi chay PHASE 1+ (bo du lieu nay co CSV
"9 columns" ma ta chua xac dinh duoc ten cot chinh xac tu ben ngoai Kaggle,
nen code o day tu-do-doan qua heuristic va IN RA de ban kiem tra).
"""

import os, glob, json, time, traceback, subprocess, sys

# Bootstrap: cai cac goi khong co san trong Kaggle base image, TRUOC khi import.
# Chay 1 lan/session (Kaggle giu pip cache trong session nen lan sau nhanh hon).
def _pip_install(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)


def _assert_sane_debug_output(model_name, debug_preds, max_avg_len=60):
    """PHANH KHAN CAP dung chung cho moi VLM moi: neu debug 5 anh dau ra output qua dai/vo
    nghia (dau hieu hallucination/dtype-sai/thu vien khong tuong thich - xem bai hoc GOT-OCR2.0
    kernel v8-v10, ~90 phut GPU bi lang phi truoc khi phat hien), DUNG LAI truoc khi chay full
    1180 anh. Goi ngay sau vong debug, truoc khi vao run_model_on_manifest."""
    avg_len = sum(len(p) for p in debug_preds) / max(1, len(debug_preds))
    if avg_len > max_avg_len:
        raise RuntimeError(
            f"{model_name}: debug output qua dai/vo nghia (trung binh {avg_len:.0f} ky tu cho "
            f"1 tu/nhan ngan) - nghi loi dtype/thu vien/prompt, xem chi tiet debug_preds o tren. "
            f"DUNG LAI, KHONG chay full 1180 anh."
        )

_pip_install("jiwer", "rapidfuzz", "openpyxl", "pytesseract", "easyocr")
os.system("apt-get install -y tesseract-ocr -q > /tmp/apt.log 2>&1")  # Kaggle kernel chay quyen root

import pandas as pd
import numpy as np

# ============================================================
# CAU HINH
# ============================================================
CURRENT_PHASE = 6   # 0=chan doan, 1=OCR co dien, 2=TrOCR+Donut, 3=GOT-OCR2.0,
                     # 4=PaddleOCR-VL, 5=Qwen-VL, 6=(tuy chon) API dong

# Cac co phu de tai-chay mot phan Phase 2 (tranh lam lai viec da co ket qua tot):
RUN_TROCR = False           # da co ket qua tot o results_master_phase2.csv (kernel v6) - khong can chay lai
RUN_DONUT_RAW_FULL = False  # da co ket qua (that bai gan 100%) o kernel v6 - chi chay lai PADDED lan nay

RESULTS_DIR = "/kaggle/working/results"
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_CSV = f"{RESULTS_DIR}/results_master.csv"

def _resolve_root(slug, owner):
    """Kaggle's newer container mounts datasets under /kaggle/input/datasets/<owner>/<slug>/
    instead of the classic flat /kaggle/input/<slug>/ - xac nhan qua chay thuc te 16/09/2026
    (os.listdir('/kaggle/input') = ['datasets']). Thu ca 2 kieu, uu tien kieu moi truoc."""
    for cand in [
        f"/kaggle/input/datasets/{owner}/{slug}",
        f"/kaggle/input/{slug}",
    ]:
        if os.path.isdir(cand):
            return cand
    return f"/kaggle/input/{slug}"  # fallback de bao loi ro rang o phase0_diagnose


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
]  # 78 classes, ung theo Data Card cua dataset tren Kaggle (da xac nhan qua browser 15/09/2026)

SEED = 42
np.random.seed(SEED)


def _find_iam_words_txt():
    """words.txt xac nhan nam tai IAM_ROOT/iam_words/words.txt (kernel v3 log, 16/09/2026).
    Thu duong dan biet truoc TRUOC, chi recursive-glob (cham, quet ~115K anh) khi that su can."""
    direct = f"{IAM_ROOT}/iam_words/words.txt"
    if os.path.exists(direct):
        return [direct]
    return glob.glob(f"{IAM_ROOT}/**/words.txt", recursive=True)

# ============================================================
# PHASE 0 — CHAN DOAN CAU TRUC DU LIEU (chay truoc tien, luon chay)
# ============================================================
def phase0_diagnose():
    print("=" * 70)
    print("PHASE 0: Kiem tra cau truc /kaggle/input/")
    print("=" * 70)
    print("os.listdir('/kaggle/input') =", os.listdir("/kaggle/input") if os.path.isdir("/kaggle/input") else "KHONG CO THU MUC /kaggle/input")
    if os.path.isdir("/kaggle/input/datasets"):
        for owner in os.listdir("/kaggle/input/datasets"):
            owner_path = f"/kaggle/input/datasets/{owner}"
            print(f"  /kaggle/input/datasets/{owner}/ ->", os.listdir(owner_path) if os.path.isdir(owner_path) else "?")
    print(f"RX_ROOT da resolve = {RX_ROOT}  (ton tai: {os.path.isdir(RX_ROOT)})")
    print(f"IAM_ROOT da resolve = {IAM_ROOT}  (ton tai: {os.path.isdir(IAM_ROOT)})")
    for root in [RX_ROOT, IAM_ROOT]:
        print(f"\n--- {root} ---")
        if not os.path.isdir(root):
            print("  KHONG TON TAI - kiem tra lai ten dataset da Add Input dung chua.")
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            depth = dirpath.replace(root, "").count(os.sep)
            if depth >= 2:
                dirnames[:] = []  # dung khong de os.walk lan sau vao (IAM co ~115K anh, rat cham neu khong prune)
            print(f"  {dirpath}/  ({len(filenames)} files, {len(dirnames)} subdirs)")
            for fn in filenames[:5]:
                print(f"      - {fn}")

    print("\n--- Doc thu cac file .csv/.xlsx trong RX_ROOT ---")
    for path in glob.glob(f"{RX_ROOT}/**/*.csv", recursive=True) + glob.glob(f"{RX_ROOT}/**/*.xlsx", recursive=True):
        print(f"\n  File: {path}")
        try:
            df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
            print("  Columns:", list(df.columns))
            print(df.head(3).to_string())
        except Exception as e:
            print("  LOI DOC FILE:", e)

    print("\n--- Doc thu words.txt cua IAM (dinh dang chuan: word_id status graylevel x y w h tag transcription) ---")
    for path in _find_iam_words_txt()[:1]:
        print(f"  File: {path}")
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = [l for l in f.readlines() if not l.startswith("#")]
        print(f"  So dong (khong tinh comment): {len(lines)}")
        print("  3 dong dau:", lines[:3])


def guess_image_and_label_columns(df: pd.DataFrame, vocab: list[str]):
    """Heuristic: cot anh la cot co string ket thuc .jpg/.png/.jpeg;
    cot nhan la cot co gia tri trung nhieu nhat voi vocab 78 thuoc."""
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
    print(f"  -> Doan: image_col={img_col!r}, label_col={label_col!r} (overlap voi vocab: {best_overlap:.2%})")
    return img_col, label_col


# ============================================================
# METRICS — ban sao cua src/metrics.py (giu dong bo 2 noi, xem README o day)
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
# GHI KET QUA — APPEND, khong ghi de, de chiu duoc session timeout
# ============================================================
def append_results(rows: list[dict]):
    df_new = pd.DataFrame(rows)
    if os.path.exists(RESULTS_CSV):
        df_new.to_csv(RESULTS_CSV, mode="a", header=False, index=False)
    else:
        df_new.to_csv(RESULTS_CSV, mode="w", header=True, index=False)
    print(f"  Da ghi {len(rows)} dong vao {RESULTS_CSV}")


def run_model_on_manifest(model_name, predict_fn, manifest_df, dataset_name, batch_log_every=50):
    """predict_fn(image_path) -> str (van ban du doan). Loi tung anh khong lam sap ca batch."""
    rows = []
    t0 = time.time()
    for i, r in manifest_df.iterrows():
        try:
            hyp = predict_fn(r["image_path"])
        except Exception as e:
            hyp = ""
            print(f"  [LOI anh {r['image_path']}]: {e}")
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
            print(f"  [{model_name}/{dataset_name}] {i+1}/{len(manifest_df)} anh, {elapsed:.1f}s, "
                  f"CER trung binh tam thoi = {np.mean([x['cer'] for x in rows]):.3f}")
    append_results(rows)
    return pd.DataFrame(rows)


# ============================================================
# BUILD MANIFEST — tu-nhan-dien cot anh/nhan, khong can biet truoc ten cot
# ============================================================
def build_manifest_kaggle_rx(split_dirname_prefix, n_sample=None):
    """split_dirname_prefix: 'Testing', 'Training', hoac 'Validation'.
    Dataset that long them 1 cap thu muc trung gian ten theo tieu de dataset
    (vi du '.../Doctor's Handwritten Prescription BD dataset/Testing/...') -
    xac nhan qua chay thuc te 16/09/2026 (kernel v3 log) - nen tim de quy thay vi
    gia dinh Testing/Training/Validation nam ngay duoi RX_ROOT."""
    label_files = glob.glob(f"{RX_ROOT}/**/{split_dirname_prefix}/*.csv", recursive=True) + \
                  glob.glob(f"{RX_ROOT}/**/{split_dirname_prefix}/*.xlsx", recursive=True)
    if not label_files:
        raise FileNotFoundError(f"Khong tim thay label file duoi {RX_ROOT}/**/{split_dirname_prefix}/")
    path = label_files[0]
    df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
    img_col, label_col = guess_image_and_label_columns(df, DRUG_VOCAB)
    split_dir = os.path.dirname(path)  # thu muc Testing/Training/Validation thuc te
    img_dir_candidates = [d for d in glob.glob(f"{split_dir}/*/") if os.path.isdir(d)]
    img_dir = img_dir_candidates[0] if img_dir_candidates else split_dir
    print(f"  split_dir={split_dir!r}, img_dir={img_dir!r}")

    def resolve_path(fname):
        fname = str(fname).strip()
        direct = os.path.join(img_dir, fname)
        if os.path.exists(direct):
            return direct
        hits = glob.glob(f"{img_dir}/**/{fname}", recursive=True)
        return hits[0] if hits else direct  # co the khong ton tai - se bao loi khi doc anh

    manifest = pd.DataFrame({
        "image_path": df[img_col].map(resolve_path),
        "label": df[label_col],
    })
    manifest = manifest[manifest["image_path"].map(os.path.exists)].reset_index(drop=True)
    print(f"  build_manifest_kaggle_rx({split_dirname_prefix}): {len(manifest)}/{len(df)} anh tim duoc tren dia")
    if n_sample and n_sample < len(manifest):
        manifest = manifest.sample(n=n_sample, random_state=SEED).reset_index(drop=True)
    return manifest


def build_manifest_iam(n_sample=400):
    """Doc words.txt chuan IAM: word_id status graylevel x y w h tag transcription.
    word_id dang a01-000u-00-00 -> anh tai IAM_ROOT/iam_words/words/a01/a01-000u/a01-000u-00-00.png
    (cau truc xac nhan qua kernel v3 log, 16/09/2026).

    QUAN TRONG VE HIEU NANG: parse+sample TRUOC, chi dung cho o dia (os.path.exists/glob)
    SAU KHI da sample xuong n_sample dong — words.txt co 44.565 dong, neu glob cho tung
    dong roi moi sample thi se cham (44K recursive glob) trong khi ta chi can vai tram anh."""
    words_txt = _find_iam_words_txt()
    if not words_txt:
        raise FileNotFoundError(f"Khong tim thay words.txt trong {IAM_ROOT}")
    words_dir = f"{IAM_ROOT}/iam_words/words"  # fallback: do tim neu cau truc khac

    parsed = []
    with open(words_txt[0], encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split(" ")
            if len(parts) < 9:
                continue  # dong khong du 9 truong theo dinh dang chuan IAM - bo qua, khong crash
            word_id, status, transcription = parts[0], parts[1], parts[-1]
            if status == "ok":
                parsed.append((word_id, transcription))
    print(f"  words.txt: {len(parsed)} dong 'ok' hop le (truoc khi sample)")

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
            hits = glob.glob(f"{IAM_ROOT}/**/{word_id}.png", recursive=True)  # fallback hiem gap, cham
            path = hits[0] if hits else None
        if path:
            rows.append({"image_path": path, "label": transcription})
    manifest = pd.DataFrame(rows)
    print(f"  build_manifest_iam: {len(manifest)}/{len(parsed)} anh da sample tim duoc tren dia")
    return manifest


# ============================================================
# PHASE 1 — OCR co dien: Tesseract (+ EasyOCR neu da pip install)
# Cell truoc do phai chay: !apt-get install -y tesseract-ocr -q && pip install pytesseract easyocr -q
# ============================================================
def phase1_classical_ocr():
    import pytesseract
    from PIL import Image

    def tesseract_predict(path):
        return pytesseract.image_to_string(Image.open(path), config="--psm 7").strip()

    rx_test = build_manifest_kaggle_rx("Testing")
    iam_sub = build_manifest_iam(n_sample=400)

    print("\n--- Tesseract tren Kaggle-Rx (Testing, toan bo) ---")
    run_model_on_manifest("tesseract", tesseract_predict, rx_test, "kaggle_rx")
    print("\n--- Tesseract tren IAM (subsample 400) ---")
    run_model_on_manifest("tesseract", tesseract_predict, iam_sub, "iam")

    try:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=True)

        def easyocr_predict(path):
            res = reader.readtext(path, detail=0)
            return " ".join(res)

        print("\n--- EasyOCR tren Kaggle-Rx (Testing, toan bo) ---")
        run_model_on_manifest("easyocr", easyocr_predict, rx_test, "kaggle_rx")
        print("\n--- EasyOCR tren IAM (subsample 400) ---")
        run_model_on_manifest("easyocr", easyocr_predict, iam_sub, "iam")
    except ImportError:
        print("easyocr chua duoc cai — bo qua, chay lai sau khi `pip install easyocr`.")


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
        print("  RUN_TROCR=False - bo qua (da co ket qua tot tu lan chay truoc, xem results_master_phase2.csv)")
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

            print("\n--- TrOCR-large-handwritten tren Kaggle-Rx (Testing, toan bo) ---")
            run_model_on_manifest("trocr-large-handwritten", trocr_predict, rx_test, "kaggle_rx")
            print("\n--- TrOCR-large-handwritten tren IAM (subsample 400) ---")
            run_model_on_manifest("trocr-large-handwritten", trocr_predict, iam_sub, "iam")
            del trocr_model
            torch.cuda.empty_cache() if device == "cuda" else None
        except Exception as e:
            print(f"  [LOI TrOCR, bo qua model nay]: {e}")
            traceback.print_exc()

    # --- Donut-base (zero-shot doc-reading qua task prompt <s_synthdog>, KHONG dung ban fine-tune CORD -
    #     ban CORD la trich xuat truong hoa don, khac muc tieu free-text OCR o day) ---
    #
    # PHAT HIEN (kernel v6, 16/09/2026): cach dung "chuan" (crop nho dua thang vao processor)
    # cho ra hypothesis RONG ~100% (degenerate_rate ~1.0 ca 2 dataset). Gia thuyet: donut-base
    # (chi pretrain SynthDoG, CHUA fine-tune) duoc train tren anh trang tai lieu day du (page-shaped,
    # ty le rong-cao lon) - anh crop 1 tu nho (gan vuong) qua khac phan phoi input khien processor
    # resize/pad thanh mot "trang" gan nhu trong -> model doan ngay EOS. Test CA 2 cach, bao cao ca 2
    # de lam ro day la van de "ap dung sai kieu du lieu" chu khong phai loi code don thuan.
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
            """Dan anh crop nho vao giua 1 'trang' trang - mo phong ty le anh SynthDoG duoc train."""
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

        # Debug nhanh: in 5 output tho (chua qua run_model_on_manifest) de xac nhan truc quan
        print("\n--- Donut DEBUG: 5 vi du dau cua rx_test, ca 2 cach (raw vs padded) ---")
        for i in range(min(5, len(rx_test))):
            row = rx_test.iloc[i]
            raw = donut_predict_raw(row["image_path"])
            padded = donut_predict_padded(row["image_path"])
            print(f"  ref={row['label']!r}  raw={raw!r}  padded={padded!r}")

        if RUN_DONUT_RAW_FULL:
            print("\n--- Donut-base (<s_synthdog>, RAW - crop truc tiep) tren Kaggle-Rx ---")
            run_model_on_manifest("donut-base-synthdog-raw", donut_predict_raw, rx_test, "kaggle_rx")
            print("\n--- Donut-base (<s_synthdog>, RAW) tren IAM (subsample 400) ---")
            run_model_on_manifest("donut-base-synthdog-raw", donut_predict_raw, iam_sub, "iam")
        else:
            print("  RUN_DONUT_RAW_FULL=False - bo qua (da co du lieu 'raw' tu kernel v6, xem results_master_phase2.csv, model='donut-base-synthdog')")

        print("\n--- Donut-base (<s_synthdog>, PADDED - dan vao 'trang' trang) tren Kaggle-Rx ---")
        run_model_on_manifest("donut-base-synthdog-padded", donut_predict_padded, rx_test, "kaggle_rx")
        print("\n--- Donut-base (<s_synthdog>, PADDED) tren IAM (subsample 400) ---")
        run_model_on_manifest("donut-base-synthdog-padded", donut_predict_padded, iam_sub, "iam")
    except Exception as e:
        print(f"  [LOI Donut, bo qua model nay]: {e}")
        traceback.print_exc()


# ============================================================
# PHASE 3 — GOT-OCR2.0 (stepfun-ai, ~580M, VLM chuyen OCR the he moi)
# ============================================================
def phase3_got_ocr2():
    # QUAN TRONG (kernel v10): transformers moi nhat tu pip la 5.0.0 (major version rat moi,
    # 09/2026). GOT-OCR2.0 duoc merge vao transformers ngay 2025-01-31 (PR #34721), on dinh
    # qua nhieu ban 4.x (~4.57.0) nhung CHUA chac tuong thich voi buoc nhay major 5.0.0 (co the
    # co breaking change noi bo ve generate()/cache lam hong modeling_got_ocr2.py). Prompt/anh
    # da xac nhan dung (kernel v10: input_ids co dung token <img>...OCR:..., pixel_values dung
    # shape) nhung output van hoan toan vo nghia -> ghim lai ban 4.57.0 (on dinh, sau khi model
    # nay duoc merge, truoc buoc nhay v5) thay vi de pip tu chon ban moi nhat.
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
        # QUAN TRONG: T4 (kien truc Turing) KHONG co tensor core ho tro bfloat16 that su -
        # kernel v8 dung bfloat16 cho ra output hon loan da ngon ngu, hoan toan sai (CER~55-120,
        # tuc dai gap hang chuc lan tham chieu) - nghi do loi so hoc ngam khi ep bf16 tren phan cung
        # khong ho tro dung. Doi sang float16 (T4 ho tro tot, Turing co Tensor Core fp16 that).
        # KERNEL v9: fp16 van cho output hon loan tuong tu -> khong phai (chi) do dtype.
        # Thu lai: (a) truyen THANG duong dan/PIL nhu vi du chinh thuc (khong .convert("RGB")
        # thu cong truoc), (b) in kich thuoc/dtype cua pixel_values de kiem tra anh co thuc su
        # duoc encode khong hay model dang "phot lo" anh va tu do sinh van ban (giai thich duoc
        # kieu loi "chat lung tung, da ngon ngu" da thay).
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
                print(f"    decoded prompt (input_ids truoc generate) = {prompt_text!r}")
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

        print("\n--- Debug: 5 vi du dau GOT-OCR2.0 tren rx_test (dtype=float16) ---")
        debug_preds = []
        for i in range(min(5, len(rx_test))):
            row = rx_test.iloc[i]
            pred = got_predict(row["image_path"], verbose=(i == 0))
            debug_preds.append(pred)
            print(f"  ref={row['label']!r}  pred={pred!r}")

        _assert_sane_debug_output("GOT-OCR2.0", debug_preds)

        print("\n--- GOT-OCR2.0 tren Kaggle-Rx (Testing, toan bo) ---")
        run_model_on_manifest("got-ocr2.0", got_predict, rx_test, "kaggle_rx")
        print("\n--- GOT-OCR2.0 tren IAM (subsample 400) ---")
        run_model_on_manifest("got-ocr2.0", got_predict, iam_sub, "iam")
    except Exception as e:
        print(f"  [LOI GOT-OCR2.0, bo qua model nay]: {e}")
        traceback.print_exc()


# ============================================================
# PHASE 4 — PaddleOCR-VL (0.9B, element-level recognition qua transformers)
# ============================================================
def phase4_paddleocr_vl():
    # ============================================================================
    # DA LOAI KHOI BENCHMARK (16/09/2026) - LOI TUONG THICH THUONG NGUON, KHONG PHAI LOI O DAY.
    # ============================================================================
    # kernel v12: transformers==4.57.0 -> TypeError: create_causal_mask() got an unexpected
    # keyword argument 'inputs_embeds' (trong modeling_paddleocr_vl.py tai remote_code cua
    # chinh HF repo PaddlePaddle/PaddleOCR-VL, khong phai code cua ta).
    # Da tra cuu: day la loi CONG DONG DA BAO CAO, CHUA CO GIAI PHAP tinh den nay -
    # xem thao luan "Newest commit breaks compatibility with transformers==4.57.6,
    # while 5.3.0 is broken as well" tai
    # https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5/discussions/22
    # (dong 30/04, khong co ban ghi giai phap; nguyen van: "Anything above transformers==5
    # with AutoModelForImageTextToText has seemingly always been broken"). Tuc la PaddleOCR-VL
    # qua duong transformers/trust_remote_code hien khong hoat dong on dinh o BAT KY version
    # transformers nao da thu (ca truoc va sau 5.0), khong lien quan gi den T4/dtype/prompt cua ta.
    # QUYET DINH: LOAI PaddleOCR-VL khoi benchmark zero-shot (khong dang chi phi debug them mot
    # loi thuong nguon chua co fix) - chuyen sang Phase 5 (Qwen-VL, he sinh thai on dinh hon
    # nhieu, da co tien le Unsloth chay tren T4). Ghi lai code o day de tham khao/thu lai sau
    # neu PaddlePaddle phat hanh ban fix.
    print("  [BO QUA] PaddleOCR-VL loai khoi benchmark do loi tuong thich thuong nguon "
          "(transformers/trust_remote_code) chua co fix - xem comment code va "
          "results/phase4_summary.md de biet chi tiet + nguon tham khao.")
    return

    # --- Code goc, giu lai de thu lai neu PaddlePaddle fix trong tuong lai ---
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

        print("\n--- Debug: 5 vi du dau PaddleOCR-VL tren rx_test ---")
        debug_preds = []
        for i in range(min(5, len(rx_test))):
            row = rx_test.iloc[i]
            pred = pvl_predict(row["image_path"])
            debug_preds.append(pred)
            print(f"  ref={row['label']!r}  pred={pred!r}")

        _assert_sane_debug_output("PaddleOCR-VL", debug_preds)

        print("\n--- PaddleOCR-VL tren Kaggle-Rx (Testing, toan bo) ---")
        run_model_on_manifest("paddleocr-vl", pvl_predict, rx_test, "kaggle_rx")
        print("\n--- PaddleOCR-VL tren IAM (subsample 400) ---")
        run_model_on_manifest("paddleocr-vl", pvl_predict, iam_sub, "iam")
    except Exception as e:
        print(f"  [LOI PaddleOCR-VL, bo qua model nay]: {e}")
        traceback.print_exc()


# ============================================================
# PHASE 5 — Qwen2.5-VL-3B-Instruct (VLM tong quat, doi chung voi cac model OCR chuyen biet)
# ============================================================
def phase5_qwen_vl():
    # Dung ban 3B (khong phai 7B/8B) - fp16 ~6GB, an toan tren T4 16GB cho zero-shot inference
    # (khong fine-tune nen khong can lo optimizer state). Qwen2.5-VL la first-class model trong
    # transformers (khong can trust_remote_code) - it rui ro dut gay API hon GOT-OCR2.0/PaddleOCR-VL.
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

        print("\n--- Debug: 5 vi du dau Qwen2.5-VL-3B tren rx_test ---")
        debug_preds = []
        for i in range(min(5, len(rx_test))):
            row = rx_test.iloc[i]
            pred = qwen_predict(row["image_path"])
            debug_preds.append(pred)
            print(f"  ref={row['label']!r}  pred={pred!r}")

        _assert_sane_debug_output("Qwen2.5-VL-3B", debug_preds, max_avg_len=100)  # VLM tong quat co the dai dong hon 1 tu

        print("\n--- Qwen2.5-VL-3B tren Kaggle-Rx (Testing, toan bo) ---")
        run_model_on_manifest("qwen2.5-vl-3b", qwen_predict, rx_test, "kaggle_rx")
        print("\n--- Qwen2.5-VL-3B tren IAM (subsample 400) ---")
        run_model_on_manifest("qwen2.5-vl-3b", qwen_predict, iam_sub, "iam")
    except Exception as e:
        print(f"  [LOI Qwen2.5-VL, bo qua model nay]: {e}")
        traceback.print_exc()


# ============================================================
# PHASE 6 — LoRA fine-tune TrOCR-large-handwritten (dong gop phuong phap chinh cho Q2)
#
# Da xac nhan (Phase 6 chuan bi, 16/09/2026): Training=3120 anh, Validation=780 anh, Testing=780
# anh (KHONG phai 2808/936/936 nhu gia dinh ban dau tu mo ta "60/20/20" tren Kaggle - xem
# results/phase6_summary.md). Test set da "dong bang" ve mat khai niem: build_manifest_kaggle_rx
# ("Testing") va build_manifest_iam(n_sample=400, seed=42) da xac nhan tra ve CHINH XAC cung
# 780/400 anh o ca 7 lan chay model truoc (Phase 1-5) - nen KHONG can file frozen rieng, chi can
# goi lai dung ham nay voi cung seed la tai tao dung tap test cu.
# ============================================================
def phase6_finetune_trocr():
    # torchao>=0.16.0 bat buoc: Kaggle base image co san torchao==0.10.0 (cu), peft moi nhat
    # tu choi chay voi ban cu (kernel v14: ImportError). Ep nang cap torchao cung luc.
    _pip_install("transformers==4.57.0", "accelerate", "peft", "torchao>=0.16.0",
                 "albumentations", "opencv-python-headless")
    import torch
    from PIL import Image
    import albumentations as A
    import cv2  # noqa: F401 (can cho albumentations doc anh/border mode)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device = {device}")
    if device != "cuda":
        print("  [CANH BAO] Khong co GPU - fine-tune se rat cham/khong kha thi, nhung van thu.")

    # --- Ban sao dong bo cua src/augmentation.py (ly do giu 2 ban: xem src/metrics.py) ---
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

    # --- Du lieu: Training (train, augmentation) + Validation subsample (theo doi trong luc train) ---
    train_df = build_manifest_kaggle_rx("Training")
    val_df_full = build_manifest_kaggle_rx("Validation")
    val_df = val_df_full.sample(n=min(300, len(val_df_full)), random_state=SEED).reset_index(drop=True)
    print(f"  train_df={len(train_df)}, val_df_full={len(val_df_full)}, val_df (subsample theo doi)={len(val_df)}")

    from transformers import (
        TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments,
        EarlyStoppingCallback, TrainerCallback,
    )
    from peft import LoraConfig, get_peft_model

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
    base_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
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
        pred_ids = pred.predictions
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        cers = [cer(l, p) for l, p in zip(label_str, pred_str)]
        return {"cer": float(np.mean(cers))}

    class TimeLimitCallback(TrainerCallback):
        """Phanh an toan: Kaggle session toi da ~9h - dung huan luyen som (khong crash) neu vuot
        nguong, de con thoi gian cho buoc danh gia frozen-test-set phia sau khong bi mat trang."""
        def __init__(self, max_seconds):
            self.max_seconds = max_seconds
            self.t0 = time.time()

        def on_step_end(self, args, state, control, **kwargs):
            elapsed = time.time() - self.t0
            if elapsed > self.max_seconds:
                print(f"  [TimeLimitCallback] Vuot {self.max_seconds}s ({elapsed:.0f}s) - DUNG huan luyen som.")
                control.should_training_stop = True
            return control

    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.1,
        target_modules=["query", "value", "q_proj", "v_proj"],
        bias="none", task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(base_model, lora_config).to(device)
    model.print_trainable_parameters()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_trainable == 0:
        # Loi PEFT hay gap: target_modules khong khop ten module thuc te -> 0 tham so LoRA nao
        # duoc tao, "training" se chay ma khong hoc gi ca. Kiem tra NGAY, KHONG cho chay tiep.
        names_sample = [n for n, _ in base_model.named_modules()][:80]
        raise RuntimeError(
            f"LoRA co 0 tham so trainable - target_modules={lora_config.target_modules} khong "
            f"khop ten module nao trong model. 80 ten module dau: {names_sample}"
        )

    # ============================================================
    # SMOKE TEST truoc (bai hoc Phase 3/4): vai chuc step tren subset nho, kiem tra loss huu han
    # va giam dan truoc khi cam ket vai gio GPU cho training day du.
    # ============================================================
    print("\n--- SMOKE TEST: 20 step tren 64 anh train ---")
    smoke_train = RxTorchDataset(train_df.sample(n=min(64, len(train_df)), random_state=SEED),
                                  augmenter=build_train_augmentation(elastic=True))
    smoke_args = Seq2SeqTrainingArguments(
        output_dir="/kaggle/working/smoke", per_device_train_batch_size=8,
        max_steps=20, logging_steps=5, save_strategy="no", eval_strategy="no",
        fp16=(device == "cuda"), report_to=[],
    )
    smoke_trainer = Seq2SeqTrainer(model=model, args=smoke_args, train_dataset=smoke_train)
    t0 = time.time()
    smoke_result = smoke_trainer.train()
    smoke_elapsed = time.time() - t0
    loss_hist = [h["loss"] for h in smoke_trainer.state.log_history if "loss" in h]
    print(f"  Smoke test: {smoke_elapsed:.1f}s / 20 step -> ~{smoke_elapsed/20:.2f}s/step. "
          f"Loss history: {loss_hist}")
    if not loss_hist or not all(np.isfinite(loss_hist)):
        raise RuntimeError(f"Smoke test: loss khong huu han/rong ({loss_hist}) - DUNG, KHONG chay full training.")
    if len(loss_hist) >= 2 and loss_hist[-1] > loss_hist[0] * 1.5:
        print(f"  [CANH BAO] Loss tang thay vi giam ({loss_hist[0]:.3f} -> {loss_hist[-1]:.3f}) - "
              f"co the learning_rate qua cao, nhung van tiep tuc full training (se theo doi eval CER).")
    steps_per_epoch = len(train_df) / 16  # effective batch = 8 * grad_accum(2)
    print(f"  Uoc luong: ~{steps_per_epoch:.0f} step/epoch, ~{steps_per_epoch*smoke_elapsed/20/60:.1f} phut/epoch")

    # ============================================================
    # TRAINING THAT (sau khi smoke test qua)
    # ============================================================
    train_ds = RxTorchDataset(train_df, augmenter=build_train_augmentation(elastic=True))
    val_ds = RxTorchDataset(val_df, augmenter=None)

    training_args = Seq2SeqTrainingArguments(
        output_dir="/kaggle/working/trocr-lora",
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
    )
    trainer = Seq2SeqTrainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5), TimeLimitCallback(max_seconds=5 * 3600)],
    )
    print("\n--- TRAINING THAT: LoRA fine-tune TrOCR-large-handwritten (r=16, alpha=32, elastic=True) ---")
    t_train0 = time.time()
    trainer.train()
    print(f"  Training xong sau {time.time() - t_train0:.0f}s. Best eval CER: "
          f"{trainer.state.best_metric}")

    # Luu adapter LoRA ngay (truoc khi lam gi khac co the loi) - day la "san pham" quan trong nhat.
    model.save_pretrained("/kaggle/working/trocr-lora-adapter")
    processor.save_pretrained("/kaggle/working/trocr-lora-adapter")
    print("  Da luu adapter vao /kaggle/working/trocr-lora-adapter")

    # ============================================================
    # DANH GIA TREN CA 2 FROZEN TEST SET (dung LAI CHINH XAC ham build_manifest_* nhu Phase 1-5
    # - da xac nhan cho cung 780/400 anh moi lan goi, xem docstring Phase 6 o tren).
    # ============================================================
    try:
        model.eval()

        @torch.no_grad()
        def trocr_lora_predict(path):
            image = Image.open(path).convert("RGB")
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
            ids = model.generate(pixel_values, max_new_tokens=MAX_TARGET_LEN)
            return processor.batch_decode(ids, skip_special_tokens=True)[0]

        rx_test = build_manifest_kaggle_rx("Testing")
        iam_sub = build_manifest_iam(n_sample=400)
        print("\n--- TrOCR-LoRA-finetuned tren Kaggle-Rx (Testing, toan bo, frozen) ---")
        run_model_on_manifest("trocr-lora-finetuned", trocr_lora_predict, rx_test, "kaggle_rx")
        print("\n--- TrOCR-LoRA-finetuned tren IAM (subsample 400, frozen) - kiem tra catastrophic forgetting ---")
        run_model_on_manifest("trocr-lora-finetuned", trocr_lora_predict, iam_sub, "iam")
    except Exception as e:
        print(f"  [LOI khi danh gia frozen test set, nhung adapter DA duoc luu an toan o tren]: {e}")
        traceback.print_exc()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    phase0_diagnose()  # luon chay truoc de co log cau truc du lieu, du CURRENT_PHASE la gi
    if CURRENT_PHASE == 0:
        print("\n>>> Doc ky output tren. Neu image_col/label_col doan sai (xem dong "
              "'-> Doan: image_col=...'), sua ham guess_image_and_label_columns() "
              "hoac gan cung truc tiep, roi chuyen CURRENT_PHASE=1 va chay lai.")
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
        phase6_finetune_trocr()
    elif CURRENT_PHASE >= 7:
        print(f"PHASE {CURRENT_PHASE} chua duoc them vao script nay. Xem docs/05-ke-hoach-Q2.md.")
