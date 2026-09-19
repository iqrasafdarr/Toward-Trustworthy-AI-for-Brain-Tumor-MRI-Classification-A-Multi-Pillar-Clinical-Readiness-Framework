"""
FAST PATH — uses your already-saved predictions_<model>_seed42.npz files.
Skips Pillar 1 (that one needs raw images + retraining, save it for later).
Gets you Pillar 2 (conformal triage) and Pillar 5 (CRI v2) results right now.

Run:
    python quick_start_pillar2_pillar5.py
"""

import numpy as np
from pillar2_conformal_triage import calibrate_aps, predict_sets, triage_decision, empirical_coverage_and_size
from pillar5_cri_v2 import pareto_frontier, cri_table_ix_defaults, net_benefit_curve, summarize_decision_curve

MODEL_FILES_SEED42 = {
    "MobileNetV2": "predictions_mobilenetv2_seed42.npz",
    "ResNet50": "predictions_resnet50_seed42.npz",
    "EfficientNetB0": "predictions_efficientnetb0_seed42.npz",
}


def load_npz_flexibly(path):
    """
    Handles whatever key names your npz files actually use. Prints the keys
    it found and what it's using, so you can see immediately if it guessed wrong.
    """
    d = np.load(path)
    print(f"  {path} keys: {d.files}")

    prob_keys = ["probs", "y_prob", "y_probs", "predictions", "softmax", "y_pred_probs"]
    label_keys = ["y_true", "labels", "y_test", "targets", "ground_truth"]

    probs = None
    for k in prob_keys:
        if k in d.files:
            probs = d[k]
            break
    labels = None
    for k in label_keys:
        if k in d.files:
            labels = d[k]
            break

    if probs is None or labels is None:
        raise KeyError(
            f"Could not auto-match keys in {path}. Available keys: {d.files}. "
            f"Tell me these keys and I'll fix the loader."
        )

    # labels may be one-hot (N,4) or already integer indices (N,)
    if labels.ndim == 2:
        labels = np.argmax(labels, axis=1)

    return probs, labels


def main():
    print("=== Loading seed-42 predictions ===")
    all_probs, all_labels = {}, {}
    for name, path in MODEL_FILES_SEED42.items():
        probs, labels = load_npz_flexibly(path)
        all_probs[name] = probs
        all_labels[name] = labels
        print(f"  {name}: probs shape {probs.shape}, labels shape {labels.shape}")

    # ------------------------------------------------------------------
    # PILLAR 5 first (fastest — pure re-analysis, no calibration split needed)
    # ------------------------------------------------------------------
    print("\n=== Pillar 5: Pareto frontier ===")
    metrics = cri_table_ix_defaults()
    frontier, dominance = pareto_frontier(metrics)
    print(f"Non-dominated architecture(s): {frontier}")
    for name, dominators in dominance.items():
        if dominators:
            print(f"  {name} is dominated by: {dominators}")

    print("\n=== Pillar 5: Decision curve analysis (tumor vs no-tumor) ===")
    notumor_idx = 2  # [glioma, meningioma, notumor, pituitary] -- confirm this matches your label encoding
    for name in all_probs:
        y_true_bin = (all_labels[name] != notumor_idx).astype(int)
        p_tumor = 1.0 - all_probs[name][:, notumor_idx]
        dca = net_benefit_curve(y_true_bin, p_tumor)
        summary = summarize_decision_curve(dca)
        print(f"  {name}: beats treat-all/none over {summary['beats_reference_threshold_ranges']}, "
              f"mean advantage in 10-50% band = {summary['mean_net_benefit_advantage_in_clinical_band']:.4f}")

    # ------------------------------------------------------------------
    # PILLAR 2 — needs a calibration split. Using a stratified 10% carve-out
    # of the SAME seed-42 test set as a stand-in calibration set is not
    # methodologically ideal (double-dipping) -- if you have a separate
    # validation-set npz with probs, swap it in here instead. For a fast
    # first look, this carve-out gets you a number today.
    # ------------------------------------------------------------------
    print("\n=== Pillar 2: Conformal triage (quick calibration carve-out) ===")
    rng = np.random.default_rng(42)
    for name in all_probs:
        probs = all_probs[name]
        labels = all_labels[name]
        n = len(labels)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_cal = int(0.10 * n)
        cal_idx, test_idx = idx[:n_cal], idx[n_cal:]

        q_hat = calibrate_aps(probs[cal_idx], labels[cal_idx], alpha=0.10)
        pred_sets = predict_sets(probs[test_idx], q_hat)
        metrics_out = empirical_coverage_and_size(pred_sets, labels[test_idx])
        print(f"  {name}: q_hat={q_hat:.4f}, coverage={metrics_out['coverage']:.3f}, "
              f"mean_set_size={metrics_out['mean_set_size']:.2f}, "
              f"referral_rate={metrics_out['referral_rate']:.3f}")

    print("\nDone. Pillar 1 still needs the raw images + a retraining pass -- "
          "do that once you've got GPU time, using run_all_pillars.py.")


if __name__ == "__main__":
    main()
