"""
RUN THIS TOP TO BOTTOM. It assumes you already have, from your existing
notebook/pipeline:
  - X_train, y_train, X_val, y_val, X_test, y_test
      images: float32 arrays (N, 224, 224, 3), pixel range whatever your
      current preprocess_input already produces
      labels: one-hot (N, 4), class order [glioma, meningioma, notumor, pituitary]
  - three trained Keras models, saved as .h5/.keras from your current run:
      mobilenetv2_seed42.keras, resnet50_seed42.keras, efficientnetb0_seed42.keras

Install once:
    pip install opencv-python-headless tensorflow numpy

Run:
    python run_all_pillars.py
"""

import numpy as np
import tensorflow as tf

from pillar1_anatomical_plausibility_loss import (
    precompute_masks, APLModel, make_dataset_with_masks, compute_edge_bias_reduction
)
from pillar2_conformal_triage import (
    calibrate_aps, predict_sets, triage_decision, empirical_coverage_and_size,
    coverage_vs_corruption_report
)
from pillar5_cri_v2 import (
    pareto_frontier, cri_table_ix_defaults, net_benefit_curve, summarize_decision_curve
)

MODEL_FILES = {
    "MobileNetV2": "mobilenetv2_seed42.keras",
    "ResNet50": "resnet50_seed42.keras",
    "EfficientNetB0": "efficientnetb0_seed42.keras",
}
BASELINE_EDGE_BIAS = {"MobileNetV2": 0.1340, "ResNet50": 0.0999, "EfficientNetB0": 0.1464}  # Table VIII, no-tumor column


# ---------------------------------------------------------------------------
# STEP 1 — Pillar 1: Anatomical Plausibility Loss
# ---------------------------------------------------------------------------

def run_pillar1(X_train, y_train, X_val, y_val, X_test, y_test, lambda_apl=0.15):
    print("=== Pillar 1: computing anatomical masks (one-time, cache to disk) ===")
    train_masks = precompute_masks(X_train)
    val_masks = precompute_masks(X_val)
    test_masks = precompute_masks(X_test)
    np.save("train_masks.npy", train_masks)
    np.save("val_masks.npy", val_masks)
    np.save("test_masks.npy", test_masks)

    apl_results = {}
    for name, path in MODEL_FILES.items():
        print(f"--- Pillar 1: fine-tuning {name} with APL, lambda={lambda_apl} ---")
        base_model = tf.keras.models.load_model(path)
        apl_model = APLModel(base_model, lambda_apl=lambda_apl)
        apl_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                           metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")])

        train_ds = make_dataset_with_masks(X_train, train_masks, y_train, batch_size=32,
                                            shuffle=True, augment=True)
        val_ds = make_dataset_with_masks(X_val, val_masks, y_val, batch_size=32,
                                          shuffle=False, augment=False)

        apl_model.fit(
            train_ds, validation_data=val_ds, epochs=12,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_acc", patience=5,
                                                          restore_best_weights=True)],
        )

        out_path = path.replace(".keras", "_apl.keras")
        apl_model.base_model.save(out_path)
        apl_results[name] = out_path
        print(f"    saved APL-trained model -> {out_path}")

        # edge-bias before/after check on the test set
        probs = apl_model.base_model.predict(X_test, batch_size=32)
        with tf.GradientTape() as tape:
            imgs = tf.convert_to_tensor(X_test)
            tape.watch(imgs)
            out = apl_model.base_model(imgs, training=False)
            pred_idx = tf.argmax(out, axis=-1)
            gathered = tf.gather_nd(out, tf.stack(
                [tf.range(tf.shape(out)[0], dtype=tf.int64), pred_idx], axis=1))
        saliency = tf.reduce_sum(tf.abs(tape.gradient(gathered, imgs)), axis=-1).numpy()

        new_eb, reduction_pct = compute_edge_bias_reduction(saliency, test_masks, BASELINE_EDGE_BIAS[name])
        print(f"    {name}: edge-bias {BASELINE_EDGE_BIAS[name]:.4f} -> {new_eb:.4f} "
              f"({reduction_pct:+.1f}% relative change)")

    return apl_results  # {model_name: path_to_apl_trained_checkpoint}


# ---------------------------------------------------------------------------
# STEP 2 — Pillar 2: Conformal Triage (run AFTER pillar 1, on the APL checkpoints)
# ---------------------------------------------------------------------------

def run_pillar2(apl_model_paths, X_val, y_val, X_test, y_test, alpha=0.10):
    print("=== Pillar 2: conformal calibration + coverage-vs-corruption ===")
    y_val_idx = np.argmax(y_val, axis=1)
    y_test_idx = np.argmax(y_test, axis=1)

    # carve a stratified 10% calibration split out of validation
    rng = np.random.default_rng(42)
    cal_idx, keep_idx = [], []
    for c in np.unique(y_val_idx):
        idx_c = np.where(y_val_idx == c)[0]
        rng.shuffle(idx_c)
        n_cal = max(1, int(0.10 * len(idx_c)))
        cal_idx.extend(idx_c[:n_cal])
        keep_idx.extend(idx_c[n_cal:])
    cal_idx, keep_idx = np.array(cal_idx), np.array(keep_idx)

    results = {}
    for name, path in apl_model_paths.items():
        model = tf.keras.models.load_model(path)

        probs_cal = model.predict(X_val[cal_idx], batch_size=32)
        q_hat = calibrate_aps(probs_cal, y_val_idx[cal_idx], alpha=alpha)

        probs_test = model.predict(X_test, batch_size=32)
        pred_sets = predict_sets(probs_test, q_hat)
        decisions = triage_decision(pred_sets)
        clean_metrics = empirical_coverage_and_size(pred_sets, y_test_idx)

        def predict_fn(imgs_uint8_batch):
            imgs = imgs_uint8_batch.astype(np.float32)  # match your existing preprocess_input here
            return model.predict(imgs, batch_size=32, verbose=0)

        corruption_report = coverage_vs_corruption_report(
            predict_fn, (X_test * 255).astype(np.uint8) if X_test.max() <= 1.0 else X_test.astype(np.uint8),
            y_test_idx, q_hat,
        )

        results[name] = {"q_hat": q_hat, "clean_metrics": clean_metrics,
                          "corruption_report": corruption_report}
        n_refer = sum(1 for d in decisions if d["action"] == "refer")
        print(f"    {name}: q_hat={q_hat:.4f}, coverage={clean_metrics['coverage']:.3f}, "
              f"mean_set_size={clean_metrics['mean_set_size']:.2f}, "
              f"referral_rate={clean_metrics['referral_rate']:.3f} "
              f"({n_refer}/{len(decisions)} test images referred)")

    return results


# ---------------------------------------------------------------------------
# STEP 3 — Pillar 5: CRI v2 (no retraining — reanalysis only)
# ---------------------------------------------------------------------------

def run_pillar5(apl_model_paths, X_test, y_test):
    print("=== Pillar 5: Pareto frontier + decision curve analysis ===")
    metrics = cri_table_ix_defaults()
    frontier, dominance = pareto_frontier(metrics)
    print(f"    Pareto-non-dominated architecture(s): {frontier}")
    for name, dominators in dominance.items():
        if dominators:
            print(f"    {name} is dominated by: {dominators}")

    y_test_idx = np.argmax(y_test, axis=1)
    notumor_class_idx = 2  # [glioma, meningioma, notumor, pituitary]
    y_true_binary = (y_test_idx != notumor_class_idx).astype(int)  # 1 = tumor

    dca_summaries = {}
    for name, path in apl_model_paths.items():
        model = tf.keras.models.load_model(path)
        probs = model.predict(X_test, batch_size=32)
        p_tumor = 1.0 - probs[:, notumor_class_idx]

        dca = net_benefit_curve(y_true_binary, p_tumor)
        summary = summarize_decision_curve(dca)
        dca_summaries[name] = summary
        print(f"    {name}: beats treat-all/treat-none over thresholds {summary['beats_reference_threshold_ranges']}, "
              f"mean advantage in 10-50% band = {summary['mean_net_benefit_advantage_in_clinical_band']:.4f}")

    return {"pareto_frontier": frontier, "dominance_report": dominance, "dca_summaries": dca_summaries}


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Load whatever you already saved from your original training run.
    X_train = np.load("X_train.npy")
    y_train = np.load("y_train.npy")
    X_val = np.load("X_val.npy")
    y_val = np.load("y_val.npy")
    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy")

    apl_paths = run_pillar1(X_train, y_train, X_val, y_val, X_test, y_test, lambda_apl=0.15)
    pillar2_results = run_pillar2(apl_paths, X_val, y_val, X_test, y_test, alpha=0.10)
    pillar5_results = run_pillar5(apl_paths, X_test, y_test)

    print("\nAll three pillars done. Checkpoints, edge-bias numbers, conformal coverage "
          "tables, and decision-curve summaries are printed above — copy the printed "
          "numbers into your results tables.")
