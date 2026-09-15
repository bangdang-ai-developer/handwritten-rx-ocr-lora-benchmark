"""Metric functions for the OCR/VLM handwritten-prescription benchmark.

Used identically both locally (for analysis of results_master.csv) and inside
the Kaggle notebook (for computing per-image scores right after inference) —
keep this file byte-identical in both places so numbers are comparable.
"""
from __future__ import annotations

import numpy as np
import jiwer
from rapidfuzz.distance import Levenshtein


def _normalize(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate = (S+D+I) / N_ref_chars, via Levenshtein edit distance."""
    ref, hyp = _normalize(reference), _normalize(hypothesis)
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return Levenshtein.distance(ref, hyp) / len(ref)


def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate via jiwer (handles substitution/deletion/insertion at word level)."""
    ref, hyp = _normalize(reference), _normalize(hypothesis)
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return jiwer.wer(ref, hyp)


def exact_match(reference: str, hypothesis: str) -> bool:
    return _normalize(reference) == _normalize(hypothesis)


def top1_label(hypothesis: str, vocab: list[str]) -> str:
    """Map a free-text OCR/VLM output to its nearest label in a closed vocabulary
    (e.g. the 78 drug-name classes of the Kaggle prescription dataset) by minimum
    normalized edit distance. Ties broken by first occurrence in `vocab`."""
    hyp = _normalize(hypothesis)
    if not vocab:
        raise ValueError("vocab must be non-empty")
    scores = [Levenshtein.normalized_distance(hyp, _normalize(v)) for v in vocab]
    return vocab[int(np.argmin(scores))]


def is_degenerate(hypothesis: str, max_len: int = 200) -> bool:
    """Flag obviously-broken model output (empty, or runaway repetition/looping)
    so it can be logged separately instead of silently dragging down mean CER."""
    hyp = _normalize(hypothesis)
    if len(hyp) == 0:
        return True
    if len(hyp) > max_len:
        return True
    tokens = hyp.split()
    if len(tokens) >= 6 and len(set(tokens)) <= max(1, len(tokens) // 6):
        return True  # a handful of tokens repeated over and over
    return False


def bootstrap_ci(values, n_resamples: int = 1000, ci: float = 0.95, seed: int = 0):
    """Bootstrap confidence interval for the mean of `values`. `seed` is required
    (not the wall-clock) so results are exactly reproducible across re-runs."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(values, size=n, replace=True)
        means[i] = sample.mean()
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return float(values.mean()), float(lo), float(hi)


def score_row(reference: str, hypothesis: str, vocab: list[str] | None = None) -> dict:
    """Compute the full metric set for one (reference, hypothesis) pair."""
    row = {
        "cer": cer(reference, hypothesis),
        "wer": wer(reference, hypothesis),
        "exact_match": exact_match(reference, hypothesis),
        "degenerate": is_degenerate(hypothesis),
    }
    if vocab is not None:
        pred = top1_label(hypothesis, vocab)
        row["top1_pred"] = pred
        row["top1_correct"] = _normalize(pred) == _normalize(reference)
    return row


if __name__ == "__main__":
    # Self-test with hand-checked examples — run this after any edit to this file.
    assert abs(cer("paracetamol", "paracetmol") - 1 / 11) < 1e-9
    assert cer("", "") == 0.0
    assert cer("abc", "") == 1.0
    assert exact_match("Paracetamol ", "paracetamol") is True
    assert exact_match("Napa", "Napa 500") is False
    assert wer("napa 500 mg", "napa 500mg") > 0.0
    vocab = ["Napa", "Ace", "Seclo", "Fexo"]
    assert top1_label("Nap", vocab) == "Napa"
    assert is_degenerate("") is True
    assert is_degenerate("mg mg mg mg mg mg mg mg") is True
    assert is_degenerate("Napa 500 mg tablet") is False
    m, lo, hi = bootstrap_ci([0.1, 0.2, 0.15, 0.3, 0.05], n_resamples=200, seed=42)
    assert lo <= m <= hi
    print("All metrics.py self-tests passed.")
    print(score_row("Napa 500mg", "Napa 500 mg", vocab=vocab))
