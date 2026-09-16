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

_pip_install("jiwer", "rapidfuzz", "openpyxl", "pytesseract", "easyocr")
os.system("apt-get install -y tesseract-ocr -q > /tmp/apt.log 2>&1")  # Kaggle kernel chay quyen root

import pandas as pd
import numpy as np

# ============================================================
# CAU HINH
# ============================================================
CURRENT_PHASE = 1   # 0=chan doan, 1=OCR co dien, 2=TrOCR+Donut, 3=GOT-OCR2.0,
                     # 4=PaddleOCR-VL, 5=Qwen-VL, 6=(tuy chon) API dong

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
    elif CURRENT_PHASE >= 2:
        print(f"PHASE {CURRENT_PHASE} (TrOCR/Donut/GOT-OCR2.0/PaddleOCR-VL/Qwen-VL) chua "
              "duoc them vao script nay — se bo sung ngay sau khi xac nhan PHASE 0-1 "
              "chay dung tren du lieu thuc te. Xem docs/05-ke-hoach-2-tuan.md muc 2.3 "
              "cho danh sach lenh cai dat cua tung model.")
