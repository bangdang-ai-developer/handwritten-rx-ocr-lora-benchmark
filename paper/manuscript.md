# Zero-Shot Vision-Language Models versus LoRA-Fine-Tuned TrOCR for Handwritten Medical Prescription Recognition: A Benchmark and Clinical Safety-Aware Error Analysis

**[PLACEHOLDER — confirm before submission]**
**Author:** Bang Dang¹
**Affiliation:** ¹ Independent Researcher
**Corresponding author email:** bangdang007112@gmail.com
**Target venue:** *PeerJ Computer Science* (primary); *Journal of Imaging* and *Health Information Science and Systems* (alternates)

> **Note to self before submission:** confirm real name/affiliation string above, add ORCID if available,
> and re-verify the two items flagged in `docs/06-tai-lieu-tham-khao-xac-minh.md` §"Việc CẦN làm" (exact
> Kaggle-Rx dataset statistics from the primary source paper, and whether GOT-OCR2.0/Qwen2.5-VL have since
> been accepted at a peer-reviewed venue) before final submission.

---

## Abstract

Handwritten medical prescriptions remain a major source of medication error, and automatic reading of
them is a natural target for modern optical character recognition (OCR). Recent general-purpose
vision-language models (VLMs) report strong OCR performance, but none have been systematically benchmarked
on public handwritten-prescription data, nor compared against lightweight, parameter-efficient fine-tuning
of a small OCR-specialist model. We benchmark seven OCR/VLM systems — Tesseract, EasyOCR,
TrOCR-large-handwritten, Donut, GOT-OCR2.0, Qwen2.5-VL-3B-Instruct, and PaddleOCR-VL (excluded after an
unresolved upstream library incompatibility) — zero-shot on a public Bangladeshi handwritten-prescription
dataset (78 drug classes, 780 held-out test images) and on IAM (400 images) as an out-of-domain control.
The best zero-shot system on the prescription domain is a 3B-parameter general VLM (Qwen2.5-VL-3B,
character error rate, CER = 0.434), ahead of every OCR-specialist model. We then apply Low-Rank Adaptation
(LoRA) to TrOCR-large-handwritten (558M parameters) on 3,120 training images, running a rank ablation
(r = 8/16/32) and an augmentation ablation (elastic distortion on/off) before selecting the best configuration
(r = 32). The fine-tuned model reaches CER = 0.114 on the prescription test set — an 80.3% relative
reduction from zero-shot and a statistically significant improvement over every zero-shot system, including
Qwen2.5-VL-3B (Wilcoxon signed-rank, p = 4.4 × 10⁻⁵⁹) and GOT-OCR2.0 (p = 8.0 × 10⁻⁹³) — despite updating
under 2% of parameters on a single free-tier GPU. On IAM, however, the fine-tuned model's CER rises from
0.441 to 0.501 (p = 3.2 × 10⁻⁴), a statistically significant but partial loss of general handwriting
recognition ability. A per-image error-category analysis shows this trade-off is not uniform: fine-tuning
cuts outright-nonsensical predictions by 4.1×, but the residual errors shift toward outputs that resemble a
*different real* drug name — a category that is both statistically more frequent after fine-tuning and,
we argue, clinically more dangerous than obviously wrong output. We find no evidence of a related but
distinct risk (the model inserting memorized drug names into unrelated text). We release the full pipeline,
frozen evaluation manifests, and trained LoRA adapter to support reproduction on free cloud GPU resources.

**Keywords:** optical character recognition, handwritten text recognition, vision-language models, LoRA,
parameter-efficient fine-tuning, catastrophic forgetting, medication safety, clinical NLP

---

## 1. Introduction

Handwritten prescriptions are still routine in much of clinical practice, and illegible or misread
handwriting is a well-documented, recurring contributor to medication error [1,2].
Automatic transcription of prescriptions is therefore an attractive application of optical character
recognition (OCR) — not only to digitize records, but potentially to flag ambiguous or high-risk entries
before they reach a pharmacy. Handwritten text recognition (HTR), the sub-problem this work addresses, has
moved through several technological generations: classical engines built on binarization and template
matching; CNN/RNN sequence models trained with connectionist temporal classification (CTC) loss; transformer
encoder-decoder models such as TrOCR [3] that treat recognition as a translation problem;
"OCR-free" document-understanding transformers such as Donut [4] that skip explicit character
modeling; and, most recently, general-purpose vision-language models (VLMs) — GOT-OCR2.0 [5],
Qwen2.5-VL [6], PaddleOCR-VL [7] — for which OCR is one of many emergent
capabilities rather than the sole training objective.

This rapid diversification raises a practical question for anyone building a real prescription-reading
system today: which of these approaches actually works on handwritten medical text, and is it worth the
engineering cost of fine-tuning a smaller specialist model rather than simply prompting a large general
VLM zero-shot? To our knowledge, no published study has benchmarked this specific combination — classical
OCR, specialist HTR transformers, OCR-free document models, and 2024-2025-generation general VLMs — on a
*public* handwritten-prescription dataset. The closest related work, RxScribe Bench [9],
benchmarks frontier VLMs on handwritten prescriptions but uses a newly collected *private* dataset and does
not evaluate fine-tuning; our benchmark is, to our knowledge, the first to combine a fully public dataset,
this specific model set, and a lightweight fine-tuning intervention with an explicit test of generalization
loss.

We make five contributions:

1. A zero-shot benchmark of seven OCR/VLM systems on a public handwritten-prescription dataset (Kaggle
   "Doctor's Handwritten Prescription BD dataset" [10]) and on IAM [8] as an
   out-of-domain control, with bootstrap confidence intervals and, where the same frozen test images allow
   it, paired Wilcoxon signed-rank tests between models.
2. A demonstration that Low-Rank Adaptation (LoRA [17]) of a 558M-parameter specialist model
   (TrOCR-large-handwritten), updating under 2% of parameters on a single free-tier T4 GPU, statistically
   significantly surpasses every zero-shot system we tested on the target domain — including a 3B-parameter
   general VLM and a purpose-built "OCR-2.0" model with roughly comparable parameter count.
3. A full rank (r = 8/16/32) and augmentation ablation for the fine-tuning step, showing that a commonly
   used default rank (16) was in fact dominated on both axes (in-domain accuracy and out-of-domain
   retention) by a higher rank (32) — a caution against treating LoRA rank as a "safe default" without
   checking it against data.
4. A statistically grounded characterization of catastrophic forgetting under LoRA: fine-tuning
   significantly *degrades* performance on out-of-domain handwriting (IAM), even though under 2% of
   parameters were updated — a finding that complicates a simple reading of prior work suggesting LoRA
   is largely protective against forgetting [19].
5. What is, to our knowledge, the first analysis connecting OCR error *type* (rather than raw error rate)
   to medication-name-confusion safety risk: after fine-tuning, outright-nonsensical predictions fall
   sharply, but the residual errors shift toward outputs that resemble a *different real* drug name — the
   category most likely to be missed on manual review — a nuance invisible to standard character error
   rate (CER) reporting.

The remainder of this paper reviews related work (§2), describes the datasets, models, and evaluation and
fine-tuning protocol (§3), reports the zero-shot benchmark, fine-tuning, ablation, and error-taxonomy
results (§4), and discusses their implications, positions this work relative to the closest related study,
and states limitations (§5–6).

---

## 2. Related Work

### 2.1 Handwritten text recognition: from sequence models to transformers

Classical HTR pipelines combine explicit segmentation with CNN/RNN recognizers trained under CTC loss;
Tesseract's LSTM-based engine [25] remains a widely deployed baseline of this family. TrOCR
[3] reframed recognition as sequence-to-sequence translation, pairing a vision transformer
encoder with an autoregressive text decoder and pretraining on large synthetic corpora before fine-tuning on
handwriting benchmarks such as IAM. Donut [4] removed the explicit recognition step
entirely, training an image-to-text transformer end-to-end on synthetic document images (SynthDoG); as we
show in §4.1, this pretraining choice makes Donut largely unusable, without further fine-tuning, on isolated
handwritten word images, since its training distribution assumes full-page document layouts. Recent surveys
[14,15] catalog this progression in more depth than space allows here.

### 2.2 General-purpose vision-language models as OCR systems

A distinct and more recent line of work treats OCR as one capability among many for large VLMs. GOT-OCR2.0
[5] is purpose-built for a broad "OCR-2.0" task definition (text, formulas, charts, tables) but
is trained and released largely outside the standard `transformers` release cycle, which — as we document in
§4 and §5.6 — makes it unusually sensitive to library version drift. Qwen2.5-VL [6] and
PaddleOCR-VL [7] are general-purpose (Qwen2.5-VL) or document-parsing-focused
(PaddleOCR-VL) VLMs for which text recognition is an emergent rather than sole capability. Crosilla et al.
[16] recently benchmarked several such general-purpose multimodal LLMs against task-specific
HTR models, finding a broadly similar pattern to ours: general models are competitive zero-shot but not
uniformly superior, and evaluation is complicated by non-OCR behavior (e.g., refusals, hedging) that standard
metrics do not anticipate — we encounter an instance of exactly this failure mode with Qwen2.5-VL on IAM
(§4.1).

### 2.3 Parameter-efficient fine-tuning and forgetting

Low-Rank Adaptation (LoRA [17]) fine-tunes a frozen pretrained model by learning a pair of
low-rank update matrices per targeted weight, updating a small fraction of parameters. Chang and Li
[18] apply a related decomposition (DoRA on the encoder, LoRA on the decoder) to TrOCR,
reporting a 4.02% CER on IAM while updating roughly 0.7% of parameters — the closest prior demonstration
that LoRA-family methods are viable for TrOCR specifically, though without a fine-tuning-domain versus
generalization-domain comparison. Whether PEFT methods such as LoRA reduce catastrophic forgetting relative
to full fine-tuning is an active question: Biderman et al. [19] argue LoRA "forgets less"
than full fine-tuning under a matched-performance comparison; Shuttleworth et al. [20]
show LoRA-tuned weights develop structurally distinct "intruder dimensions" that provide a mechanistic
account of *why* LoRA solutions can behave differently from full fine-tuning even at similar in-domain
performance; Chen et al. [21] explicitly note that "catastrophic forgetting remains
an issue with PEFT" and propose a Bayesian regularization scheme to mitigate it; and Kalajdzievski [26]
reports that forgetting under fine-tuning follows a predictable scaling law in the number of fine-tuned
parameters and steps, and is not reliably avoided by early stopping alone — relevant to our own use of
early stopping in §3.4. Our results (§4.4) provide
a concrete, statistically tested data point on the same question in a new domain (handwriting recognition):
LoRA at both r = 16 and r = 32 produces a statistically significant, though partial, degradation on
out-of-domain data, which is more consistent with Chen et al.'s framing than with an unqualified reading of
"LoRA forgets less."

### 2.4 Prior work on prescription and medical handwriting OCR

The dataset we use, the Kaggle "Doctor's Handwritten Prescription BD dataset," originates from Mia et al.
[10], who benchmarked a VGG16-based classifier on it. Ali et al. [11] fine-tune
a Mask R-CNN segmentation stage together with TrOCR-Base-Handwritten on roughly 1,000 handwritten Pakistani
prescriptions (augmented ×10), reporting CER between 1.4% and 15.4% depending on test scenario — the closest
prior demonstration that TrOCR-family fine-tuning is viable at prescription scale, though on a different,
private dataset and without a fine-tuning-versus-forgetting analysis. RxScribe Bench
[9], published within weeks of this work, benchmarks frontier VLMs on a newly collected,
*private* set of Indian outpatient prescriptions along four clinical-risk axes, but does not evaluate
fine-tuning or use a public dataset. A hybrid Tesseract/PaddleOCR/GPT-4o pipeline for prescription OCR has
also been reported [12], using a different model set and web-scraped rather than
benchmark-standard data. Cheema et al. [13] published a closely related methodological
precedent — adapting multilingual VLMs, with TrOCR as one comparison point, for low-resource Urdu OCR — in
*PeerJ Computer Science*, the primary venue we target for this work.

### 2.5 Medication-name confusion as a patient-safety problem

Independent of OCR, the clinical-pharmacy literature has long recognized that structurally or phonetically
similar drug names — "look-alike, sound-alike" (LASA) pairs — are a distinct and well-studied source of
medication error [1,2,23]. Karet [22] quantifies orthographic
and phonetic similarity across FDA/ISMP-flagged LASA pairs, providing a methodological precedent for treating
"resembles a different real drug" as a distinct, measurable error category rather than folding it into an
undifferentiated error rate — the approach we adopt in §3.5 and §4.5. The closest analog for connecting
*error type* (rather than raw rate) to clinical risk in an automated-transcription setting is Zhou et al.
[24], who classify errors in ASR-assisted clinical dictation by clinical significance; to our
knowledge, no prior work makes this connection specifically for OCR-based prescription reading, which is the
gap our error-taxonomy analysis (§4.5) addresses.

---

## 3. Materials and Methods

All code, frozen evaluation manifests, per-image predictions, and the final LoRA adapter are released
alongside this paper (see Data Availability) to support exact reproduction.

### 3.1 Datasets

**Kaggle-Rx.** The "Doctor's Handwritten Prescription BD dataset" [10] (Kaggle,
`mamun1113/doctors-handwritten-prescription-bd-dataset`) contains cropped, single-word handwritten images of
78 drug names, pre-split by the dataset authors into `Training`, `Validation`, and `Testing` folders with
accompanying label files. We verified the actual split sizes by direct enumeration rather than relying on the
dataset page's summary description (which states an approximate 60/20/20 split that does not match the file
counts we observe): **Training = 3,120, Validation = 780, Testing = 780** images (4,680 total), all of which
resolve correctly against their label files. We hold out the entire `Testing` split (780 images) as a frozen
evaluation set, used identically — down to the image-filename level — for every zero-shot model and for the
fine-tuned model's final evaluation, which we verified empirically returns the identical 780-image set across
seven independent runs (fixed seed = 42).

**IAM.** We use IAM Handwriting Word Database [8] as an out-of-domain control representing
general (non-medical) English handwriting, drawing a fixed random subsample of 400 word images (seed = 42,
verified identical across all runs) via the Kaggle mirror `nibinv23/iam-handwriting-word-database`. IAM
serves two purposes: (i) a domain-shift baseline against which to interpret Kaggle-Rx performance, and (ii)
a generalization probe for the fine-tuned model, since it is never seen during fine-tuning.

### 3.2 Models evaluated zero-shot

We group seven systems into three families:

- **Classical:** Tesseract [25] (LSTM-based engine) and EasyOCR (CRAFT detector + CRNN
  recognizer, used here in single-word recognition mode).
- **Specialist encoder-decoder / document-understanding transformers:** `microsoft/trocr-large-handwritten`
  [3] (558M parameters) and `naver-clova-ix/donut-base` [4], tested both on
  raw word crops and on crops padded onto a blank page-sized canvas to better match Donut's SynthDoG
  pretraining aspect ratio.
- **General-purpose vision-language models:** `stepfun-ai/GOT-OCR-2.0-hf` [5] (~580M
  parameters), `Qwen/Qwen2.5-VL-3B-Instruct` [6] (3B parameters, prompted with a fixed
  instruction to output only the transcribed text), and `PaddlePaddle/PaddleOCR-VL` [7]
  (0.9B parameters). PaddleOCR-VL is included in our pipeline but excluded from reported results: it raises
  a `TypeError` inside its own `trust_remote_code` modeling file (`create_causal_mask() got an unexpected
  keyword argument 'inputs_embeds'`) with every `transformers` version we tried, matching an open,
  unresolved community report of the same incompatibility. We treat this as a reproducibility finding in its
  own right (§5.6) rather than a gap to be silently omitted.

All zero-shot models were run with a debug pass on five images (checked for non-degenerate output) before
committing to the full 1,180-image (780 + 400) evaluation, and with `transformers==4.57.0` pinned after an
initial run with an unpinned, newly released major version produced incoherent output for GOT-OCR2.0 (see
§5.6) — a library-version sensitivity we flag as a general reproducibility caution for this class of models.

### 3.3 Evaluation protocol

For every (model, image) pair we compute:

- **Character Error Rate (CER)** and **Word Error Rate (WER)**, via Levenshtein edit distance
  (case-insensitive, whitespace-normalized), using `rapidfuzz` and `jiwer` respectively.
- **Exact match** (post-normalization string equality).
- **Closed-vocabulary top-1 accuracy** (Kaggle-Rx only): the raw prediction is mapped to its nearest of the
  78 drug names by minimum normalized edit distance, and this top-1 label is compared to ground truth. This
  captures cases where the free-text OCR output is "close enough" to be unambiguously resolved against a
  known drug list, which is arguably closer to how such a system would actually be deployed (against a
  pharmacy formulary) than raw string matching.
- **Degenerate-output flag**: empty output, output exceeding 200 characters, or output that is a short
  token repeated so that fewer than one-sixth of tokens are unique — used to separate "the model refused/broke"
  from "the model produced a fluent but wrong answer," which we found necessary after observing near-100%
  such outputs from Donut (§4.1).

We report the **mean and 95% bootstrap confidence interval** (1,000 resamples, fixed seed = 0) for CER, and,
where the same frozen image set was scored by two models, a **paired Wilcoxon signed-rank test** on per-image
CER — valid here specifically because every model's evaluation manifest is generated by the same seeded
procedure and was verified to return an identical image set across independent runs. Where a model's CER
distribution is heavily right-skewed (§4.1), we additionally report the median.

### 3.4 LoRA fine-tuning of TrOCR-large-handwritten

We fine-tune `microsoft/trocr-large-handwritten` with LoRA (`peft` library), targeting the attention
projection modules of both the encoder (`query`, `value`) and decoder (`q_proj`, `v_proj`), with dropout 0.1
and no bias adaptation. Training uses the full 3,120-image `Training` split with an Albumentations
augmentation pipeline (affine rotation/shear/scale, elastic distortion, Gaussian noise and blur,
brightness/contrast jitter, morphological erosion/dilation), a 300-image random subsample of `Validation`
for epoch-level tracking, AdamW with a cosine schedule (peak learning rate 2 × 10⁻⁴, 8%–10% warmup), an
effective batch size of 16 (8 physical × 2 gradient-accumulation steps) under `fp16` mixed precision (`bf16`
is not effectively supported on the Turing-architecture T4 GPUs used here), up to 15 epochs with early
stopping (patience 5 epochs on validation CER), and a 5-hour wall-clock safety cutoff per run. Training runs
on a single Kaggle notebook session (2× T4 GPUs via automatic `DataParallel`).

**Rank and augmentation ablation.** Rather than fixing the LoRA rank a priori, we ran full training-to-early-stopping
for three ranks (r ∈ {8, 16, 32}, α = 2r in each case, elastic distortion enabled) and, holding rank fixed at
16, for augmentation on versus off (elastic distortion enabled versus disabled, all other augmentations
unchanged), evaluating each of the four resulting checkpoints on the same frozen Kaggle-Rx and IAM test sets
described in §3.1. We selected the final reported configuration by frozen-test-set CER on both datasets
jointly, rather than fixing rank in advance, since the ablation directly informs which configuration to
report as primary (§4.2). We flag as a limitation that each configuration was trained with a single random
seed (§6), so part of the observed spread between ranks may reflect run-to-run training variance in addition
to a genuine rank effect; the paired Wilcoxon tests in §4.2 establish that the specific checkpoints compared
are statistically distinguishable, which is a different (weaker) claim than that repeated training runs at
a given rank would reliably reproduce the same ranking.

### 3.5 Qualitative error taxonomy

For Kaggle-Rx (closed 78-class vocabulary), we classify each fine-tuned-model prediction into one of four
categories: **correct** (exact match); **minor OCR noise, still correct drug** (not an exact match, but the
nearest-vocabulary mapping is correct); **confusable wrong drug** (the nearest-vocabulary mapping is
incorrect, and the raw prediction's normalized edit distance to that incorrect nearest match is ≤ 0.34 —
i.e., the output closely resembles a specific, different real drug name); and **hallucination, far off**
(incorrect and not close to any real drug name in the vocabulary). The 0.34 threshold was chosen as a
round cutoff separating "clearly reads as a specific other word" from "not close to any listed word" on
inspection of the distance distribution; we report it as a design choice rather than a validated clinical
threshold (§6). For IAM (open vocabulary), we instead bin predictions into correct / minor (0 < CER ≤ 0.3) /
moderate (0.3 < CER ≤ 0.7) / severe (CER > 0.7) error, since no closed-vocabulary mapping is available. We
additionally test, for the fine-tuned model on IAM only, whether any prediction exactly matches one of the
78 Kaggle-Rx drug names while being clearly wrong relative to the true (non-drug) IAM label (CER > 0.3
against the true label) — a specific check for whether fine-tuning causes the model to intrude memorized
drug names into unrelated text.

---

## 4. Results

### 4.1 Zero-shot benchmark

**Table 1. Zero-shot CER by model and dataset (mean, 95% bootstrap CI; n = 780 Kaggle-Rx, n = 400 IAM).**

| Model | CER, Kaggle-Rx | CER, IAM | Top-1 acc., Kaggle-Rx (78 classes) |
|---|---|---|---|
| Qwen2.5-VL-3B-Instruct | **0.434** [0.372, 0.501] | 1.100 [0.471, 1.960]* | 81.5% |
| GOT-OCR2.0 | 0.479 [0.446, 0.513] | **0.386** [0.275, 0.581] | 67.2% |
| EasyOCR | 0.552 [0.529, 0.574] | 0.736 [0.703, 0.766] | 54.9% |
| TrOCR-large-handwritten | 0.580 [0.555, 0.606] | 0.441 [0.364, 0.515] | 79.5% |
| Tesseract | 0.625 [0.601, 0.648] | 0.836 [0.787, 0.886] | 49.2% |
| Donut (raw crop) | 1.000 | 1.000 | 1.3% |
| Donut (page-padded crop) | 4.201 [3.310, 5.171] | 1.058 [0.993, 1.184] | 3.6% |

\* Median CER for Qwen2.5-VL-3B on IAM is 0.0 (see below); the mean is driven by a small number of
instruction-following failures.

The best zero-shot system on the target (prescription) domain is **Qwen2.5-VL-3B-Instruct**, a
general-purpose 3B-parameter VLM, ahead of every OCR-specialist model including the purpose-built GOT-OCR2.0
(~580M parameters). This is, on its face, a striking result: a model with no OCR-specific training objective
outperforms models built specifically for text recognition on this domain, plausibly because world knowledge
of plausible drug names helps it resolve ambiguous handwriting. On IAM, GOT-OCR2.0 and TrOCR-large-handwritten
(which is directly pretrained toward IAM-like handwriting) are both strong, while Qwen2.5-VL-3B's *mean* CER
(1.100) is the worst of all seven systems — but its *median* CER is 0.0, meaning more than half of its IAM
predictions are exactly correct. Manual inspection shows the mean is inflated by a small subset (roughly 3%)
of cases where the model produces a conversational refusal or hedge rather than following the "output only
the text" instruction (e.g., predicting *"The text is not visible in the image provided. Please upload an
image..."* against a one-character reference), which is not IAM-tolerant text: such refusals are near-total
mismatches against very short ground truths, producing enormous per-image CER outliers. We treat this as
both a substantive finding (mean CER can be a misleading summary for instruction-tuned VLMs; a paired
Wilcoxon test comparing Qwen2.5-VL-3B against GOT-OCR2.0 on IAM is *not* statistically significant,
p = 0.330, despite the large difference in means) and a caution for anyone evaluating general VLMs as OCR
systems using mean-error metrics alone.

Donut-base fails almost completely on both presentations we tried: near-100% degenerate (empty) output on
raw word crops, and, when padded onto a page-sized canvas to better match its SynthDoG pretraining
distribution, a reduced-but-still-high degenerate rate (77.8% Kaggle-Rx / 96.5% IAM) accompanied by severe
hallucination (mean CER 4.20 and 1.06 respectively) on the outputs it does produce. We interpret this as
evidence that Donut's "OCR-free," full-document pretraining does not transfer, without further fine-tuning,
to isolated single-word recognition — a useful negative result given Donut's architecture is often loosely
described as a general-purpose OCR-free document reader.

### 4.2 LoRA fine-tuning and ablation

**Table 2. Rank and augmentation ablation on the frozen Kaggle-Rx test set (n = 780) and IAM (n = 400).**
All configurations trained to early stopping on identical data/seed; each configuration was trained once.

| Configuration | CER, Kaggle-Rx | CER, IAM | Wilcoxon vs. r = 16 elastic (Kaggle-Rx / IAM) |
|---|---|---|---|
| r = 8, elastic | 0.113 [0.098, 0.128] | 0.530 [0.475, 0.580] | p = 1.0 × 10⁻⁷ / p = 0.830 |
| r = 16, elastic | 0.149 [0.132, 0.166] | 0.524 [0.472, 0.579] | — |
| **r = 32, elastic (selected)** | **0.114** [0.100, 0.130] | **0.501** [0.453, 0.552] | p = 2.4 × 10⁻⁸ / p = 0.079 |
| r = 16, no elastic | 0.150 [0.134, 0.167] | 0.562 [0.505, 0.627] | p = 0.647 / p = 0.101 |

The a priori "middle" choice, r = 16, is dominated on Kaggle-Rx by both r = 8 and r = 32 (both
p < 10⁻⁶), and r = 32 additionally shows the lowest (best) point estimate on IAM, though this specific
r = 32-versus-r = 16 IAM comparison does not reach the conventional 0.05 significance threshold
(p = 0.079). Since r = 32 is at least as good as every other tested configuration on both axes with no
apparent trade-off, we select **r = 32** as the configuration reported as "the fine-tuned model" throughout
the remainder of this paper. Disabling elastic distortion (holding r = 16 fixed) leaves in-domain CER
essentially unchanged (0.149 versus 0.150, p = 0.647) but produces a higher (worse) point estimate on IAM
(0.562 versus 0.524), a difference that again does not reach p < 0.05 (p = 0.101); we retain elastic
distortion in the final pipeline since it shows no evidence of harm and a non-significant trend toward
better retention.

Training with r = 32 converges to its best validation CER (0.102 on a 300-image tracking subsample) at
epoch 2 of 15, with early stopping triggered at epoch 7 after five epochs without improvement (Figure 2);
validation loss continues to decrease past epoch 2 even as validation CER plateaus and worsens, consistent
with the model beginning to overfit the small training set on the loss objective before its effect on the
CER metric becomes visible.

### 4.3 Fine-tuned model versus every zero-shot system

**Table 3. Selected model comparison, Kaggle-Rx test set (n = 780).**

| Model | CER | Exact match | Top-1 acc. (78 classes) |
|---|---|---|---|
| **TrOCR + LoRA (r = 32, fine-tuned)** | **0.114** [0.100, 0.130] | **68.1%** | **92.4%** |
| Qwen2.5-VL-3B-Instruct (zero-shot) | 0.434 [0.372, 0.501] | 28.2% | 81.5% |
| GOT-OCR2.0 (zero-shot) | 0.479 [0.446, 0.513] | 15.8% | 67.2% |
| TrOCR-large-handwritten (zero-shot) | 0.580 [0.555, 0.606] | 8.1% | 79.5% |

Fine-tuning TrOCR-large-handwritten with LoRA (r = 32) on 3,120 images reduces CER from 0.580 (zero-shot) to
0.114 — an 80.3% relative reduction — and raises exact match from 8.1% to 68.1% and closed-vocabulary top-1
accuracy from 79.5% to 92.4%. This fine-tuned, 558M-parameter model statistically significantly outperforms
every zero-shot system we tested on this domain, including systems with far more parameters or explicit
OCR-specialization: against Qwen2.5-VL-3B-Instruct (3B parameters), p = 4.4 × 10⁻⁵⁹; against GOT-OCR2.0
(~580M parameters, purpose-built for OCR), p = 8.0 × 10⁻⁹³ (paired Wilcoxon signed-rank, n = 780 for both).
This is, to our knowledge, the first demonstration that a small, resource-efficient specialist model,
adapted with under 2% of its parameters on a single free-tier GPU, can decisively outperform much larger
general-purpose VLMs on a target handwriting domain.

### 4.4 Domain-shift and generalization (catastrophic forgetting)

On IAM, the fine-tuned model's CER rises from 0.441 (zero-shot TrOCR) to 0.501 — a 13.7% relative
*increase* in error, statistically significant (p = 3.2 × 10⁻⁴, paired Wilcoxon, n = 400). The median CER on
IAM also rises, from 0.0 (zero-shot) to 0.400 (fine-tuned), indicating this degradation is a broad,
systematic shift rather than a small number of outlier failures. Figure 5 shows the resulting reversal in
domain-shift gap (CER on IAM minus CER on Kaggle-Rx) across all models: zero-shot OCR-specialist models
(TrOCR, GOT-OCR2.0) score slightly *better* on IAM than on Kaggle-Rx (negative gap, consistent with IAM being
closer to their original pretraining distribution than prescription handwriting), while the fine-tuned model
shows a substantial positive gap — a direct, quantitative signature of the specialization it has undergone.
We note, and clarify explicitly to avoid misreading Table 2 against this gap figure, that this domain-shift
*gap* being similar in magnitude between r = 16 (+0.376) and r = 32 (+0.387) does not contradict r = 32 being
better on *both* datasets in absolute terms (§4.2) — r = 32 improves Kaggle-Rx performance by more than it
improves IAM performance relative to r = 16, which widens the gap even as both individual numbers improve.

This finding is consistent with, but adds a more cautionary note than, prior claims that LoRA-family
fine-tuning is broadly protective against catastrophic forgetting [19]. Updating fewer than
2% of parameters was not sufficient to avoid a statistically significant loss of out-of-domain handwriting
recognition ability in our setting, which — together with the mechanistic account of LoRA-specific
"intruder dimensions" offered by Shuttleworth et al. [20] — argues against treating a
small trainable-parameter fraction alone as a guarantee against generalization loss.

### 4.5 Qualitative error taxonomy and clinical safety implications

**Table 4. Error-category composition, Kaggle-Rx test set, zero-shot vs. fine-tuned (r = 32).**

| Category | Zero-shot | Fine-tuned |
|---|---|---|
| Correct (exact match) | 8.1% | 68.1% |
| Minor OCR noise, still correct drug | 71.8% | 24.9% |
| Confusable wrong drug (resembles a *different* real drug) | 1.5% | 2.6% |
| Hallucination, far off (resembles no real drug) | 18.6% | 4.5% |

Fine-tuning reduces the outright-nonsensical "hallucination, far off" category by 4.1× (18.6% → 4.5%) — the
model becomes markedly less prone to guessing wildly. But the residual, smaller category of errors that
resemble a *different, real* drug name rises from 1.5% to 2.6% (a 1.7× relative increase). We consider this
the paper's most clinically relevant finding: a prediction that reads as *some other real medicine* (e.g.,
ground truth "Amodis" misread as "Amodin," which is itself closer to the unrelated real drug "Axodin" than
to the correct answer) is plausible on its face and, per the look-alike/sound-alike medication-error
literature [1,2,22], considerably harder to catch on routine manual review
than output that is obviously not a real drug name. Aggregate CER improves substantially under fine-tuning,
but the *composition* of the remaining errors shifts toward the category that is arguably most dangerous in
a deployed system — a distinction invisible to CER alone. We verified this checking specifically for one
additional risk: whether the fine-tuned model, having been narrowly specialized on 78 drug names, inserts a
memorized drug name when reading unrelated general handwriting. Of 400 IAM predictions, only 2 (0.5%, versus
0/400 for the zero-shot model) exactly matched a Kaggle-Rx drug name while being clearly wrong relative to
the true IAM label; both are plausibly explained by chance collision on a short or difficult word rather than
systematic drug-name intrusion, and we find no evidence this is a meaningful effect at our sample size.

On IAM, the error-severity distribution shifts from 57.3% exact-match / 12.8% moderate error / 23.3% severe
error (zero-shot) to 26.8% / 32.8% / 27.5% (fine-tuned) — predominantly a shift from *exactly correct* to
*moderately wrong*, rather than a shift toward outright failure, which nuances the mean/median-CER framing
of §4.4: fine-tuning on a narrow domain degrades a broad swath of previously-perfect out-of-domain
predictions to "partially wrong," more than it produces new catastrophic failures.

---

## 5. Discussion

**A small, cheaply adapted specialist model can beat much larger general VLMs on a target domain.** Our
headline result — that LoRA-adapting a 558M-parameter model on a single free-tier GPU, updating under 2% of
parameters, yields a statistically decisive improvement over a 3B-parameter general VLM and a purpose-built
580M-parameter OCR model — has a practical implication beyond the specific numbers: for teams or researchers
without large compute budgets (a category that includes most low- and middle-income-country health
informatics groups, and the independent, self-funded circumstances of this study), fine-tuning a small
specialist model on a modest, domain-relevant training set is a more accessible and more effective strategy
than relying on ever-larger general-purpose models zero-shot.

**Rank matters, and a "reasonable default" is not a substitute for checking.** LoRA rank 16 is a common
default in the fine-tuning literature and in our own initial plan; our ablation shows it was, in this
setting, dominated by both a smaller (r = 8) and a larger (r = 32) rank on the in-domain metric, with no
compensating advantage on generalization. We surface this primarily as a methodological caution: absent an
ablation, we would have reported a materially weaker headline result (CER 0.149 rather than 0.114) under the
mistaken impression that it was near-optimal.

**LoRA does not eliminate catastrophic forgetting in this setting.** Section 4.4's result — a statistically
significant, systematic (not merely outlier-driven) degradation on out-of-domain handwriting recognition
after updating under 2% of parameters — sits in tension with the framing that PEFT methods are largely
protective against forgetting [19], though it does not contradict it directly, since we did
not compare against a matched full fine-tuning run. It is, however, directly consistent with more
qualified accounts: Shuttleworth et al.'s finding that LoRA solutions develop distinctive "intruder
dimensions" not seen in full fine-tuning [20], and Chen et al.'s explicit statement that
forgetting "remains an issue" for PEFT generally [21]. Practically, this means that
teams fine-tuning a general-purpose HTR model for a narrow domain should budget for a generalization check
against out-of-domain data, rather than assuming a small trainable-parameter fraction is sufficient
protection.

**Error type, not just error rate, determines clinical risk.** Section 4.5's finding — that fine-tuning
improves aggregate accuracy while shifting the residual-error mix toward outputs that resemble a different
real medication — is, to our knowledge, the first result connecting this specific mechanism (fine-tuning-induced
error-type shift) to medication-name-confusion safety risk in an OCR setting. The broader principle that
error *type* matters more than error *rate* for clinical risk has precedent in speech-recognition-assisted
clinical documentation [24] and in emerging clinical-LLM safety evaluation frameworks, and the
LASA literature [1,2,22] independently establishes that confusable-real-drug-name
errors are a recognized, clinically significant error class; our contribution is connecting these to a
concrete, measured shift under a common OCR fine-tuning intervention. A practical implication is that OCR
systems intended for prescription reading should be evaluated and possibly optimized against a
confusable-drug-aware metric, not aggregate CER alone, and that a lexicon- or LASA-list-aware post-processing
correction stage — which we did not implement here — is a natural next step (see Conclusion).

**Position relative to the closest related work.** RxScribe Bench [9], published within
weeks of this study, addresses a closely related question — benchmarking frontier VLMs on handwritten
prescriptions along clinical-risk axes — but on a newly collected, private dataset of Indian outpatient
prescriptions, without a fine-tuning or forgetting analysis. Our work is complementary: a fully public,
independently reproducible dataset and pipeline, a specific and directly comparable model set including
purpose-built OCR-2.0 and general VLM systems, and an explicit test of whether lightweight fine-tuning
changes the qualitative safety profile of the resulting errors, not only their aggregate rate.

**Reproducibility lessons for the OCR/VLM community.** Two of the seven systems we set out to benchmark
required nontrivial debugging before producing valid output, and one (PaddleOCR-VL) could not be run to
completion at all: GOT-OCR2.0 produced fluent-looking but essentially random multilingual output under an
unpinned, newly released major `transformers` version, resolved only by pinning to an earlier release; and
PaddleOCR-VL raises an unresolved `trust_remote_code` incompatibility with every `transformers` version we
tried, matching an open, unresolved community bug report at the time of writing. We record these not as
incidental engineering notes but as a substantive observation about the current state of general-purpose
VLM tooling: models released and updated outside the core library's stable release cadence carry a real,
non-obvious reproducibility cost for downstream benchmarking, distinct from their underlying modeling
quality.

---

## 6. Limitations

- **Single training dataset per domain.** We use one public prescription dataset (78 drug classes, a
  specific handwriting population) and one general-handwriting out-of-domain control (IAM). Results may not
  generalize to prescription handwriting in other languages, health systems, or drug-name vocabularies.
- **Single random seed per fine-tuning configuration.** Each of the four ablation configurations (§3.4,
  §4.2) was trained once. The paired Wilcoxon tests establish that the specific trained checkpoints differ
  significantly on the frozen test sets, but do not establish that repeated training runs at a given rank
  would reproducibly yield the same ranking; part of the observed rank effect could reflect run-to-run
  training variance.
- **PaddleOCR-VL could not be benchmarked** due to an unresolved upstream library incompatibility, leaving
  one of the four general-VLM systems we intended to test excluded from quantitative comparison.
- **Small, closed-vocabulary fine-tuning target.** The 3,120-image training set and 78-class label space
  make the fine-tuned model's strong in-domain performance partly attributable to a genuinely narrow task;
  results on prescriptions naming drugs outside this 78-class vocabulary, or containing dosage/frequency
  text beyond the drug name, are not evaluated here.
- **The confusable-drug distance threshold (0.34, §3.5) is a design choice, not a clinically validated
  cutoff.** We report it transparently and note that a formal LASA-pair-based definition (per
  [22]) is a natural refinement for future work.
- **IAM is one specific operationalization of "general handwriting."** Generalization loss measured against
  IAM may not represent generalization loss against other out-of-domain handwriting distributions (e.g.,
  other languages, other prescription vocabularies, or clinical free text).
- **No clinician or pharmacist evaluation.** Our safety-relevant error-taxonomy claims are based on
  automated distance-based categorization and the LASA literature, not a clinical-expert review of the
  specific errors observed; we consider such a review a valuable next step rather than a substitute for the
  quantitative analysis presented here.

---

## 7. Conclusion

We benchmarked seven OCR and vision-language systems, zero-shot, on a public handwritten medical
prescription dataset and an out-of-domain handwriting control, finding that a general-purpose 3B-parameter
VLM outperforms every OCR-specialist system on the target domain. We then showed that lightweight LoRA
fine-tuning of a much smaller (558M-parameter) specialist model, using under 2% of trainable parameters on
a single free-tier GPU, statistically significantly surpasses every zero-shot system tested, after a rank
and augmentation ablation that itself overturned our own initial default choice of hyperparameter. This
improvement comes with a statistically significant, though partial, loss of general handwriting recognition
ability, and — more subtly — with a shift in the *composition* of residual errors toward outputs that
resemble a different real medication, a category we argue is more clinically dangerous than obviously wrong
output despite not showing up as a problem in aggregate error-rate terms. Future work could evaluate a
lexicon- or LASA-list-aware post-correction stage targeting this specific error category, test whether
mixing a small amount of general-handwriting data into fine-tuning mitigates the observed forgetting without
sacrificing in-domain gains, and extend the benchmark to additional public prescription datasets and
languages as they become available.

---

## Additional Information

**Data availability.** All code (benchmark pipeline, fine-tuning script, analysis scripts, figure
generation), the frozen evaluation manifests, per-image predictions for every model, and the final LoRA
adapter weights are available at **[repository URL — to be added]**. Both underlying datasets are publicly
available: Kaggle-Rx at `kaggle.com/datasets/mamun1113/doctors-handwritten-prescription-bd-dataset` and IAM
via its standard registration process (`fki.tic.heia-fr.ch/databases/iam-handwriting-database`); users should
independently verify the exact license terms on each dataset's page before redistribution.

**Funding.** This research received no external funding; compute was provided by Kaggle's free-tier GPU
quota.

**Competing interests.** The author declares no competing interests.

**Author contributions.** [Author] conceived the study, designed and ran all experiments, performed the
analysis, and wrote the manuscript.

---

## References

*(Numbered in order of first citation. Formatting will need to be converted to the target journal's exact
citation style — e.g., PeerJ Computer Science's house style — before submission; DOIs/arXiv IDs are included
for every entry to make that conversion mechanical.)*

1. Bryan R, Aronson JK, Williams A, Jordan S. The problem of look-alike, sound-alike name errors: drivers
   and solutions. *British Journal of Clinical Pharmacology*. 2021;87(2):386–394.
   doi:10.1111/bcp.14285.
2. Ostini R, Roughead EE, Kirkpatrick CMJ, Monteith GR, Tett SE. Quality Use of Medicines — medication
   safety issues in naming; look-alike, sound-alike medicine names. *International Journal of Pharmacy
   Practice*. 2012;20(6):349–357. doi:10.1111/j.2042-7174.2012.00210.x.
3. Li M, Lv T, Chen J, Cui L, Lu Y, Florencio D, Zhang C, Li Z, Wei F. TrOCR: Transformer-based Optical
   Character Recognition with Pre-trained Models. *Proceedings of the AAAI Conference on Artificial
   Intelligence*. 2023;37(11):13094–13102. doi:10.1609/aaai.v37i11.26538. (arXiv:2109.10282)
4. Kim G, Hong T, Yim M, Nam J, Park J, Yim J, Hwang W, Yun S, Han D, Park S. OCR-free Document
   Understanding Transformer. *Computer Vision — ECCV 2022*. Springer LNCS 13688. 2022.
   doi:10.1007/978-3-031-19815-1_29. (arXiv:2111.15664)
5. Wei H, Liu C, Chen J, Wang J, Kong L, Xu Y, Ge Z, Zhao L, Sun J, Peng Y, Han C, Zhang X. General OCR
   Theory: Towards OCR-2.0 via a Unified End-to-end Model. arXiv:2409.01704. 2024.
6. Bai S, Chen K, Liu X, et al. Qwen2.5-VL Technical Report. arXiv:2502.13923. 2025.
7. Cui C, Sun T, Liang S, Gao T, Zhang Z, Liu J, Wang X, Zhou C, Liu H, Lin M, Zhang Y, Zhang Y, Zheng H,
   Zhang J, Zhang J, Liu Y, Yu D, Ma Y. PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B
   Ultra-Compact Vision-Language Model. arXiv:2510.14528. 2025.
8. Marti U-V, Bunke H. The IAM-database: an English sentence database for offline handwriting recognition.
   *International Journal on Document Analysis and Recognition*. 2002;5(1):39–46.
   doi:10.1007/s100320200071.
9. RxScribe Bench: A Multi-Axis Benchmark for Evaluating Vision-Language Models on Indian Outpatient
   Prescriptions. arXiv:2609.13280. 2026. (author list unconfirmed — verify directly on arXiv before
   submission; read the full paper, not just the abstract, before finalizing the Related Work discussion
   of it, per `docs/06-tai-lieu-tham-khao-xac-minh.md`.)
10. Mia AR, Chowdhury MA, Mamun AA, Ruddra AM, Tanny NT. A Deep Neural Network Approach with Pioneering
    Local Dataset to Recognize Doctor's Handwritten Prescription in Bangladesh. *2024 International
    Conference on Advances in Computing, Communication, Electrical, and Smart Systems (iCACCESS)*.
    2024:1–6. doi:10.1109/iCACCESS61735.2024.10499631.
11. Ali U, Ranmbail S, Nadeem M, Ishfaq H, Ramzan MU, Ali W. Leveraging Deep Learning with Multi-Head
    Attention for Accurate Extraction of Medicine from Handwritten Prescriptions. arXiv:2412.18199. 2024.
12. A Hybrid Deep Learning-Based OCR Model for Handwritten Medical Prescriptions. *Frontiers in Medicine*.
    2026. doi:10.3389/fmed.2026.1856485. (author list unconfirmed — verify directly before submission)
13. Cheema MDA, Shaiq MD, Mirza F, Kamal A, Naeem MA. Adapting multilingual vision language transformers
    for low-resource Urdu optical character recognition (OCR). *PeerJ Computer Science*. 2024;10:e1964.
    doi:10.7717/peerj-cs.1964.
14. Garrido-Munoz C, Rios-Vila A, Calvo-Zaragoza J. Handwritten Text Recognition: A Survey. *IEEE
    Transactions on Pattern Analysis and Machine Intelligence*. 2026;48(4):4367–4387.
    doi:10.1109/TPAMI.2025.3646002. (arXiv:2502.08417)
15. AlKendi W, Gechter F, Heyberger L, Guyeux C. Advancements and Challenges in Handwritten Text
    Recognition: A Comprehensive Survey. *Journal of Imaging*. 2024;10(1):18.
    doi:10.3390/jimaging10010018.
16. Crosilla L, Klic L, Colavizza G. Benchmarking large language models for handwritten text recognition.
    *Journal of Documentation*. 2025;81(7):334–354. (arXiv:2503.15195)
17. Hu EJ, Shen Y, Wallis P, Allen-Zhu Z, Li Y, Wang S, Wang L, Chen W. LoRA: Low-Rank Adaptation of Large
    Language Models. *International Conference on Learning Representations (ICLR)*. 2022.
    (arXiv:2106.09685)
18. Chang D, Li Y. Mixed Text Recognition with Efficient Parameter Fine-Tuning and Transformer.
    *Neural Information Processing (ICONIP 2024)*. Springer LNCS 15294. 2025.
    doi:10.1007/978-981-96-6599-0_2. (arXiv:2404.12734)
19. Biderman D, Portes J, Ortiz JJG, Paul M, Greengard P, Jennings C, King D, Havens S, Chiley V, Frankle
    J, Blakeney C, Cunningham JP. LoRA Learns Less and Forgets Less. *Transactions on Machine Learning
    Research*. 2024. (arXiv:2405.09673)
20. Shuttleworth R, Andreas J, Torralba A, Sharma P. LoRA vs Full Fine-tuning: An Illusion of Equivalence.
    *Advances in Neural Information Processing Systems 38 (NeurIPS 2025)*. 2025. (arXiv:2410.21228)
21. Chen H, et al. Bayesian Parameter-Efficient Fine-Tuning for Overcoming Catastrophic Forgetting.
    *IEEE/ACM Transactions on Audio, Speech, and Language Processing*. 2024. (full author list and
    exact volume/issue/pages unconfirmed — verify directly before submission.)
22. Karet GB. Linguistic Analysis of Generic-Generic Drug Name Pairs Prone to Wrong-Drug Errors for which
    Tall-Man Lettering is Recommended. *Therapeutic Innovation & Regulatory Science*. 2023;57(4):751–758.
    doi:10.1007/s43441-023-00526-0.
23. Lizano-Díez I, et al. Prevention strategies to identify LASA errors: building and sustaining a culture
    of patient safety. *BMC Health Services Research*. 2020;20:63. doi:10.1186/s12913-020-4922-3.
24. Zhou L, Blackley SV, Kowalski L, Doan R, Acker WW, Landman AB, Kontrient E, Mack D, Meteer M, Bates
    DW, Goss FR. Analysis of Errors in Dictated Clinical Documents Assisted by Speech Recognition Software
    and Professional Transcriptionists. *JAMA Network Open*. 2018;1(3):e180530.
    doi:10.1001/jamanetworkopen.2018.0530.
25. Smith R. An Overview of the Tesseract OCR Engine. *Ninth International Conference on Document Analysis
    and Recognition (ICDAR 2007)*. 2007:629–633.
26. Kalajdzievski D. Scaling Laws for Forgetting When Fine-Tuning Large Language Models. arXiv:2401.05605.
    2024.

---

## Figure and Table Captions (for reference — see `results/figures/`)

- **Figure 1** (`fig1_cer_comparison.png`): CER by model and dataset, all eight systems (seven zero-shot
  plus the fine-tuned model), mean ± 95% bootstrap CI.
- **Figure 2** (`fig2_training_curve.png`): LoRA fine-tuning (r = 32) validation CER and loss per epoch,
  best checkpoint marked.
- **Figure 3 / 4** (`fig3_error_taxonomy_kaggle_rx.png`, `fig4_error_taxonomy_iam.png`): error-category
  composition, zero-shot versus fine-tuned, Kaggle-Rx (closed-vocabulary categories) and IAM (severity
  bins).
- **Figure 5** (`fig5_domain_gap.png`): domain-shift gap (CER IAM − CER Kaggle-Rx) by model.
- **Figure 6** (`fig6_cer_distribution.png`): per-model CER distributions (boxplots), illustrating the
  heavy-tailed distributions discussed in §4.1.
- **Figure 7** (`fig7_qualitative_examples.png`): representative cropped Kaggle-Rx word images with
  ground truth, zero-shot, and fine-tuned predictions, one example per error category.
- **Figure 8 / 9** (`fig8_rank_ablation.png`, `fig9_augmentation_ablation.png`): rank ablation (r = 8/16/32)
  and augmentation ablation (elastic on/off) CER comparisons.
