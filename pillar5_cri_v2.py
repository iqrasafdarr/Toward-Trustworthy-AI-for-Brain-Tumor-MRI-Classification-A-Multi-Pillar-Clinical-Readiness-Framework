"""
Pillar 5 — CRI v2: Pareto Frontier + Decision Curve Analysis
=============================================================
The paper's own limitations section flags the CRI weights (0.40/0.25/
0.20/0.15) as "hand-selected to preserve interpretability" -- this is the
single most likely reviewer objection. CRI v2 does not propose new
weights (which would have the same problem); it replaces the need for a
single scalar weighting altogether with two weight-free analyses:

  1. Pareto frontier over the four raw pillars (Acc, 1-ECE, 1-HCE, Gen) --
     shows which architectures are non-dominated (no other architecture
     beats them on every axis simultaneously), independent of any
     weighting scheme.
  2. Decision curve analysis (net benefit) for the collapsed binary
     tumor-vs-no-tumor screening task -- shows clinical utility across
     the full range of plausible decision thresholds, rather than at one
     hand-picked operating point.

Both are then cross-checked against the original CRI ranking and its
Table X sensitivity sweep already in the paper, to show convergence.
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. Pareto frontier over the four CRI pillars.
# ---------------------------------------------------------------------------

def pareto_frontier(metrics_dict):
    """
    metrics_dict: {model_name: {"acc": float, "cal": float, "safety": float, "gen": float}}
    All four sub-metrics must already be oriented so "higher is better"
    (i.e. cal = 1 - ECE, safety = 1 - HCE, matching Table IX's columns).

    Returns (frontier_names, dominance_report):
      frontier_names: list of model names on the non-dominated Pareto frontier.
      dominance_report: {model_name: [names of models that dominate it]} --
        empty list means the model is on the frontier.
    """
    names = list(metrics_dict.keys())
    points = np.array([[metrics_dict[n]["acc"], metrics_dict[n]["cal"],
                         metrics_dict[n]["safety"], metrics_dict[n]["gen"]] for n in names])

    dominance_report = {n: [] for n in names}
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if i == j:
                continue
            # j dominates i if j is >= on all axes and > on at least one
            if np.all(points[j] >= points[i]) and np.any(points[j] > points[i]):
                dominance_report[ni].append(nj)

    frontier_names = [n for n in names if len(dominance_report[n]) == 0]
    return frontier_names, dominance_report


def cri_table_ix_defaults():
    """The paper's own Table IX values, pre-loaded so the Pareto check can
    be reported directly against the existing manuscript numbers."""
    return {
        "MobileNetV2":    {"acc": 0.3768 / 0.40, "cal": 0.2427 / 0.25, "safety": 0.1161 / 0.20, "gen": 0.1364 / 0.15},
        "ResNet50":       {"acc": 0.3565 / 0.40, "cal": 0.2382 / 0.25, "safety": 0.1471 / 0.20, "gen": 0.1543 / 0.15},
        "EfficientNetB0": {"acc": 0.3678 / 0.40, "cal": 0.2453 / 0.25, "safety": 0.1628 / 0.20, "gen": 0.1378 / 0.15},
    }
    # dividing each Table IX weighted contribution by its own weight recovers
    # the raw, unweighted (Acc, 1-ECE, 1-HCE, Gen) values used in Eq. (10).


# ---------------------------------------------------------------------------
# 2. Decision curve analysis (net benefit) for the binary tumor-screening
#    collapse described in Section III-H: P(tumor|x) = 1 - p_NT(x).
# ---------------------------------------------------------------------------

def net_benefit_curve(y_true_binary, y_prob_tumor, thresholds=None):
    """
    y_true_binary: (N,) 1 = tumor (glioma/meningioma/pituitary), 0 = no-tumor.
    y_prob_tumor: (N,) P(tumor | x) from the softmax collapse.
    thresholds: array of threshold probabilities p_t to evaluate at
        (default: 0.01 to 0.99 in 0.01 steps -- the standard DCA range).

    Returns dict with:
      thresholds, net_benefit_model, net_benefit_treat_all, net_benefit_treat_none
    Net benefit at threshold p_t:
      NB = (TP/N) - (FP/N) * (p_t / (1 - p_t))
    "Treat all" (flag everyone as tumor) and "treat none" (flag no one) are
    the two reference strategies DCA is judged against; a useful model must
    beat both across a clinically plausible threshold range.
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)

    n = len(y_true_binary)
    prevalence = np.mean(y_true_binary)

    nb_model = np.zeros_like(thresholds)
    nb_all = np.zeros_like(thresholds)

    for i, pt in enumerate(thresholds):
        predicted_positive = y_prob_tumor >= pt
        tp = np.sum(predicted_positive & (y_true_binary == 1))
        fp = np.sum(predicted_positive & (y_true_binary == 0))
        odds = pt / (1 - pt)
        nb_model[i] = (tp / n) - (fp / n) * odds

        # treat-all: everyone predicted positive
        tp_all = np.sum(y_true_binary == 1)
        fp_all = np.sum(y_true_binary == 0)
        nb_all[i] = (tp_all / n) - (fp_all / n) * odds

    nb_none = np.zeros_like(thresholds)  # treating no one always nets zero

    return {
        "thresholds": thresholds,
        "net_benefit_model": nb_model,
        "net_benefit_treat_all": nb_all,
        "net_benefit_treat_none": nb_none,
        "prevalence": prevalence,
    }


def summarize_decision_curve(dca_result, clinical_range=(0.10, 0.50)):
    """
    Reports the range of thresholds over which the model beats both
    reference strategies, and the mean net-benefit advantage within a
    clinically plausible threshold band (default 10-50%, matching typical
    triage/referral thresholds rather than a diagnostic-certainty threshold).
    """
    t = dca_result["thresholds"]
    model = dca_result["net_benefit_model"]
    best_reference = np.maximum(dca_result["net_benefit_treat_all"], dca_result["net_benefit_treat_none"])
    beats_reference = model > best_reference

    in_band = (t >= clinical_range[0]) & (t <= clinical_range[1])
    mean_advantage_in_band = float(np.mean((model - best_reference)[in_band]))

    beats_ranges = []
    start = None
    for i, ok in enumerate(beats_reference):
        if ok and start is None:
            start = t[i]
        if not ok and start is not None:
            beats_ranges.append((start, t[i - 1]))
            start = None
    if start is not None:
        beats_ranges.append((start, t[-1]))

    return {
        "beats_reference_threshold_ranges": beats_ranges,
        "mean_net_benefit_advantage_in_clinical_band": mean_advantage_in_band,
        "clinical_band": clinical_range,
    }


# ---------------------------------------------------------------------------
# 3. Convenience wrapper: run both analyses for all three architectures at
#    once and produce a compact comparison table for the paper.
# ---------------------------------------------------------------------------

def run_cri_v2(per_model_binary_labels, per_model_tumor_probs, cri_table_ix_metrics=None):
    """
    per_model_binary_labels: {model_name: (N,) array} -- typically identical
        across models since it's the same test set, but kept per-model in
        case of any filtering differences.
    per_model_tumor_probs: {model_name: (N,) array} P(tumor|x) per model.
    cri_table_ix_metrics: optional override of cri_table_ix_defaults().

    Returns a dict with the Pareto frontier result and each model's DCA summary.
    """
    metrics = cri_table_ix_metrics or cri_table_ix_defaults()
    frontier_names, dominance_report = pareto_frontier(metrics)

    dca_summaries = {}
    for name in per_model_binary_labels:
        dca = net_benefit_curve(per_model_binary_labels[name], per_model_tumor_probs[name])
        dca_summaries[name] = summarize_decision_curve(dca)

    return {
        "pareto_frontier": frontier_names,
        "dominance_report": dominance_report,
        "decision_curve_summaries": dca_summaries,
    }
