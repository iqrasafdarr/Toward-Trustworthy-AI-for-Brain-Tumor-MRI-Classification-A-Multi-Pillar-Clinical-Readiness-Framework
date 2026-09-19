"""
Pillar 2 — Conformal Triage
===========================
Replaces raw softmax confidence (shown in the paper to be poorly aligned
with correctness — 41.94% of MobileNetV2's errors at >90% confidence)
with split Adaptive Prediction Sets (APS) conformal prediction, which
gives a finite-sample marginal coverage guarantee:

    P(y_true in prediction_set(x)) >= 1 - alpha

This directly operationalizes triage: a singleton prediction set is an
"automate" decision, a multi-label set is a "refer to radiologist"
decision, with a mathematical guarantee behind the split, rather than an
arbitrary confidence threshold.

Also includes a coverage-vs-corruption stress test: does the conformal
guarantee hold up as the input distribution shifts (a direct probe of
whether "trustworthy" claims survive outside the clean test set).
"""

import numpy as np
import cv2


CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


# ---------------------------------------------------------------------------
# 1. Split conformal calibration using the Adaptive Prediction Sets (APS)
#    non-conformity score (Romano, Sesia & Candes, 2020).
# ---------------------------------------------------------------------------

def aps_scores(probs, y_true_idx):
    """
    probs: (N, K) softmax probabilities.
    y_true_idx: (N,) integer true class indices.
    Returns (N,) non-conformity scores: the cumulative probability mass of
    all classes ranked at or above the true class's rank.
    """
    order = np.argsort(-probs, axis=1)           # descending
    ranked_probs = np.take_along_axis(probs, order, axis=1)
    cumsum = np.cumsum(ranked_probs, axis=1)

    true_rank = np.array([
        np.where(order[i] == y_true_idx[i])[0][0] for i in range(len(y_true_idx))
    ])
    # score = cumulative mass up to and including the true class's rank
    scores = cumsum[np.arange(len(y_true_idx)), true_rank]
    return scores


def calibrate_aps(probs_cal, y_cal_idx, alpha=0.10):
    """
    probs_cal: (N_cal, K) softmax probs on a held-out calibration split
    (NOT used in training or in the reported test set -- carve this out
    of the existing validation split, e.g. an extra 10% stratified holdout).
    y_cal_idx: (N_cal,) true labels.
    alpha: miscoverage tolerance (0.10 -> 90% guaranteed coverage).

    Returns q_hat: the conformal quantile threshold.
    """
    n = len(y_cal_idx)
    scores = aps_scores(probs_cal, y_cal_idx)
    # finite-sample corrected quantile level
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    q_hat = np.quantile(scores, level, method="higher")
    return q_hat


def predict_sets(probs_test, q_hat):
    """
    probs_test: (N, K) softmax probs.
    q_hat: conformal threshold from calibrate_aps.
    Returns a list of length N, each entry a list of class indices
    included in that image's prediction set.
    """
    order = np.argsort(-probs_test, axis=1)
    ranked_probs = np.take_along_axis(probs_test, order, axis=1)
    cumsum = np.cumsum(ranked_probs, axis=1)

    sets = []
    for i in range(probs_test.shape[0]):
        n_included = int(np.searchsorted(cumsum[i], q_hat) + 1)
        n_included = min(n_included, probs_test.shape[1])
        sets.append(sorted(order[i, :n_included].tolist()))
    return sets


def triage_decision(pred_sets):
    """
    Maps prediction sets to a triage action:
      - singleton set  -> "automate" (report the single predicted class)
      - multi-label set -> "refer"   (route to radiologist review)
    Returns list of dicts: {"action": ..., "candidates": [...]}
    """
    decisions = []
    for s in pred_sets:
        if len(s) == 1:
            decisions.append({"action": "automate", "candidates": [CLASS_NAMES[s[0]]]})
        else:
            decisions.append({"action": "refer", "candidates": [CLASS_NAMES[c] for c in s]})
    return decisions


def empirical_coverage_and_size(pred_sets, y_true_idx):
    covered = np.mean([y in s for y, s in zip(y_true_idx, pred_sets)])
    mean_size = np.mean([len(s) for s in pred_sets])
    referral_rate = np.mean([len(s) > 1 for s in pred_sets])
    return {"coverage": float(covered), "mean_set_size": float(mean_size),
            "referral_rate": float(referral_rate)}


# ---------------------------------------------------------------------------
# 2. Coverage-vs-corruption stress test.
#    Applies a graded corruption grid to the *test images before
#    inference* and re-measures whether APS's 1-alpha guarantee holds,
#    and how referral rate rises as a built-in safety valve under shift.
# ---------------------------------------------------------------------------

def apply_corruption(image_uint8, corruption, severity):
    """
    image_uint8: HxWx3 uint8.
    corruption: one of "gaussian_noise", "gaussian_blur", "brightness", "contrast", "motion_blur".
    severity: int 1-5.
    """
    img = image_uint8.astype(np.float32)
    s = severity

    if corruption == "gaussian_noise":
        sigma = [2, 4, 8, 12, 18][s - 1]
        img = img + np.random.normal(0, sigma, img.shape)

    elif corruption == "gaussian_blur":
        k = [1, 3, 5, 7, 9][s - 1]
        img = cv2.GaussianBlur(img, (k, k), 0)

    elif corruption == "brightness":
        delta = [10, 20, 35, 50, 70][s - 1]
        sign = 1 if s % 2 == 0 else -1
        img = img + sign * delta

    elif corruption == "contrast":
        factor = [0.9, 0.8, 0.65, 0.5, 0.35][s - 1]
        mean = img.mean()
        img = (img - mean) * factor + mean

    elif corruption == "motion_blur":
        k = [3, 5, 9, 13, 17][s - 1]
        kernel = np.zeros((k, k))
        kernel[k // 2, :] = 1.0 / k
        img = cv2.filter2D(img, -1, kernel)

    else:
        raise ValueError(f"unknown corruption: {corruption}")

    return np.clip(img, 0, 255).astype(np.uint8)


CORRUPTION_GRID = ["gaussian_noise", "gaussian_blur", "brightness", "contrast", "motion_blur"]


def coverage_vs_corruption_report(model_predict_fn, images_uint8, y_true_idx, q_hat,
                                   corruptions=CORRUPTION_GRID, severities=(1, 2, 3, 4, 5)):
    """
    model_predict_fn: callable(images_uint8_batch) -> (N, K) softmax probs,
        wrapping your trained model's preprocess_input + predict call.
    images_uint8: (N, H, W, 3) uint8, the clean test set.
    y_true_idx: (N,) true labels.
    q_hat: conformal threshold calibrated on CLEAN data (deliberately not
        re-calibrated per corruption -- the point is to see whether the
        clean-data guarantee degrades under shift).

    Returns a nested dict: {corruption: {severity: {"coverage":.., "mean_set_size":.., "referral_rate":..}}}
    plus the same metrics for the naive softmax-argmax baseline (no
    conformal wrapper) at threshold 0.9, for direct before/after comparison.
    """
    report = {}
    for corruption in corruptions:
        report[corruption] = {}
        for sev in severities:
            corrupted = np.stack([apply_corruption(im, corruption, sev) for im in images_uint8])
            probs = model_predict_fn(corrupted)

            pred_sets = predict_sets(probs, q_hat)
            conformal_metrics = empirical_coverage_and_size(pred_sets, y_true_idx)

            argmax_preds = probs.argmax(axis=1)
            baseline_correct = np.mean(argmax_preds == y_true_idx)
            baseline_high_conf_error_rate = np.mean(
                (argmax_preds != y_true_idx) & (probs.max(axis=1) > 0.9)
            )

            report[corruption][f"severity_{sev}"] = {
                "conformal": conformal_metrics,
                "softmax_baseline": {
                    "accuracy": float(baseline_correct),
                    "high_confidence_error_rate": float(baseline_high_conf_error_rate),
                },
            }
    return report
