"""
Pillar 1 — Anatomical Plausibility Loss (APL)
==============================================
Turns the paper's post-hoc Grad-CAM/edge-bias finding into an end-to-end
training-time method: a saliency-based regularizer that penalizes model
attention falling outside an anatomically plausible brain region, plus a
watermark-masking augmentation that directly attacks the artifact found
in Section IV-C.

This is designed to slot into the existing MobileNetV2 / ResNet50 /
EfficientNetB0 fine-tuning pipeline described in the paper (Focal Loss,
last-50-layers unfrozen, GAP -> BN -> Dense(128) -> Dropout -> Dense(4)).

Usage sketch (fits your existing training script):

    from pillar1_anatomical_plausibility_loss import (
        build_anatomical_mask, APLModel, watermark_masking_augment
    )

    masks = np.stack([build_anatomical_mask(img) for img in X_train])
    model = APLModel(base_model, lambda_apl=0.15)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4))
    model.fit(train_dataset_with_masks, epochs=50, callbacks=[...])
"""

import numpy as np
import tensorflow as tf
import cv2


# ---------------------------------------------------------------------------
# 1. Anatomical prior mask (cheap alternative to a full nnU-Net segmentation
#    model — Otsu thresholding + morphological cleanup reliably isolates the
#    skull/brain silhouette from a dark MRI background and is standard in
#    the brain-MRI preprocessing literature).
# ---------------------------------------------------------------------------

def build_anatomical_mask(image_uint8_or_float, target_size=(224, 224)):
    """
    image_uint8_or_float: HxWx3 or HxW array, any numeric range.
    Returns a float32 mask in [0, 1] of shape target_size, where 1 = brain
    tissue region (anatomically plausible attention target) and 0 =
    background / border, i.e. the region where attention should be
    penalized.
    """
    img = image_uint8_or_float
    if img.ndim == 3:
        gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY) if img.shape[-1] == 3 else img[..., 0]
    else:
        gray = img
    gray = cv2.resize(gray.astype(np.uint8), target_size)

    # Otsu threshold isolates skull/brain tissue from dark background.
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleanup: fill small holes, remove speckle, keep the
    # single largest connected component (the brain silhouette).
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    n_labels, labels = cv2.connectedComponents(mask)
    if n_labels > 1:
        sizes = [(labels == i).sum() for i in range(1, n_labels)]
        largest = 1 + int(np.argmax(sizes))
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)

    # Slight dilation gives the model a little tolerance at the tissue
    # boundary rather than penalizing the exact Otsu edge.
    mask = cv2.dilate(mask, kernel, iterations=1)
    return (mask.astype(np.float32) / 255.0)


def precompute_masks(images, target_size=(224, 224)):
    """images: array (N, H, W, 3). Returns (N, target_size[0], target_size[1])."""
    return np.stack([build_anatomical_mask(im, target_size) for im in images], axis=0)


# ---------------------------------------------------------------------------
# 2. Watermark-masking augmentation
#    Attacks the artifact directly rather than only regularizing around it:
#    with probability p, zero out (mean-fill) a random border/corner strip,
#    matching where the paper found the source banner, rotated copyright
#    text, and faint overlaid text (top strip, top-right corner,
#    lower-left region).
# ---------------------------------------------------------------------------

def watermark_masking_augment(image, p=0.5, border_frac=0.12, seed=None):
    """
    image: tf.Tensor, shape (H, W, 3), float32, any consistent scale.
    Randomly mean-fills one of the four known artifact-prone regions
    (top strip, top-right corner, lower-left region, bottom strip) so the
    model cannot rely on any single fixed watermark location.
    """
    rng = tf.random.Generator.from_seed(seed) if seed is not None else tf.random.get_global_generator()
    do_it = rng.uniform([], 0, 1) < p
    if not do_it:
        return image

    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    bh = tf.cast(tf.cast(h, tf.float32) * border_frac, tf.int32)
    bw = tf.cast(tf.cast(w, tf.float32) * border_frac, tf.int32)
    mean_val = tf.reduce_mean(image)

    region = rng.uniform([], 0, 4, dtype=tf.int32)
    mask = tf.ones_like(image)

    def zero_region(y0, y1, x0, x1, m):
        idx_y = tf.range(h)[:, None]
        idx_x = tf.range(w)[None, :]
        in_y = tf.logical_and(idx_y >= y0, idx_y < y1)
        in_x = tf.logical_and(idx_x >= x0, idx_x < x1)
        in_region = tf.logical_and(in_y, in_x)
        in_region = tf.cast(in_region, tf.float32)[..., None]
        return m * (1.0 - in_region)

    mask = tf.case([
        (tf.equal(region, 0), lambda: zero_region(0, bh, 0, w, mask)),           # top strip
        (tf.equal(region, 1), lambda: zero_region(0, bh, w - bw, w, mask)),       # top-right corner
        (tf.equal(region, 2), lambda: zero_region(h - bh, h, 0, bw, mask)),       # lower-left region
        (tf.equal(region, 3), lambda: zero_region(h - bh, h, 0, w, mask)),        # bottom strip
    ])
    return image * mask + (1.0 - mask) * mean_val


# ---------------------------------------------------------------------------
# 3. Anatomical Plausibility Loss and custom training step.
#
#    Rather than a full double-backprop Grad-CAM inside the loss (expensive
#    and numerically fragile through GAP layers), APL uses the standard
#    "right-for-the-right-reasons" formulation: the input-gradient saliency
#    of the predicted class w.r.t. the input image. This is smooth,
#    single-order-cheaper than Grad-CAM, requires only one nested
#    GradientTape, and directly penalizes exactly the failure mode the
#    paper's edge-bias audit measured (activation mass outside the
#    anatomical region) -- but now as a training signal, not a post-hoc
#    diagnostic.
# ---------------------------------------------------------------------------

class APLModel(tf.keras.Model):
    """
    Wraps an existing classification model (e.g. the MobileNetV2 /
    ResNet50 / EfficientNetB0 head described in the paper) and adds the
    Anatomical Plausibility Loss term to its training step.

    L_total = FocalLoss(y, y_hat)
              + lambda_apl * mean_over_batch( sum( |dY_c/dX| * (1 - mask) ) )

    where Y_c is the predicted class's logit and mask is the anatomical
    prior (1 = brain tissue, 0 = background/border) from build_anatomical_mask.
    """

    def __init__(self, base_model, lambda_apl=0.15, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.lambda_apl = lambda_apl
        self.gamma = gamma
        self.alpha = alpha

    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)

    def focal_loss(self, y_true, y_pred):
        eps = 1e-7
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        ce = -y_true * tf.math.log(y_pred)
        weight = self.alpha * tf.pow(1.0 - y_pred, self.gamma)
        return tf.reduce_sum(weight * ce, axis=-1)

    def train_step(self, data):
        # data = ((images, masks), labels)
        (images, masks), y_true = data

        with tf.GradientTape() as outer_tape:
            with tf.GradientTape() as saliency_tape:
                saliency_tape.watch(images)
                y_pred = self.base_model(images, training=True)
                pred_class_idx = tf.argmax(y_true, axis=-1)
                batch_idx = tf.range(tf.shape(y_pred)[0], dtype=tf.int64)
                gather_idx = tf.stack([batch_idx, tf.cast(pred_class_idx, tf.int64)], axis=1)
                class_scores = tf.gather_nd(y_pred, gather_idx)

            # Input-gradient saliency: this inner gradient is computed
            # *inside* the outer tape, so its dependence on base_model's
            # weights is preserved for the second backward pass below.
            saliency = saliency_tape.gradient(class_scores, images)
            saliency = tf.reduce_sum(tf.abs(saliency), axis=-1)  # (B, H, W)

            off_target = 1.0 - masks  # background/border region
            apl_penalty = tf.reduce_mean(tf.reduce_sum(saliency * off_target, axis=[1, 2]))

            focal = tf.reduce_mean(self.focal_loss(y_true, y_pred))
            total_loss = focal + self.lambda_apl * apl_penalty

        grads = outer_tape.gradient(total_loss, self.base_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))

        self.compiled_metrics.update_state(y_true, y_pred)
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": total_loss, "focal_loss": focal, "apl_penalty": apl_penalty})
        return results

    def test_step(self, data):
        (images, masks), y_true = data
        y_pred = self.base_model(images, training=False)
        focal = tf.reduce_mean(self.focal_loss(y_true, y_pred))
        self.compiled_metrics.update_state(y_true, y_pred)
        results = {m.name: m.result() for m in self.metrics}
        results["loss"] = focal
        return results


def make_dataset_with_masks(images, masks, labels, batch_size=32, shuffle=True, augment=False):
    """
    images: (N,224,224,3) float32, masks: (N,224,224) float32, labels: (N,4) one-hot.
    Returns a tf.data.Dataset yielding ((images, masks), labels), applying
    watermark_masking_augment when augment=True (training set only).
    """
    ds = tf.data.Dataset.from_tensor_slices((images, masks, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(images), reshuffle_each_iteration=True)

    def _map(img, mask, lbl):
        if augment:
            img = watermark_masking_augment(img)
        return (img, mask), lbl

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# 4. Validation metric — re-run the paper's own edge-bias audit on the
#    APL-trained model to report the *causal* improvement (before/after
#    comparison against Table VIII's baseline edge-bias numbers).
# ---------------------------------------------------------------------------

def compute_edge_bias_reduction(saliency_maps, masks, baseline_edge_bias):
    """
    saliency_maps: (N, H, W) abs input-gradient saliency for the APL-trained model.
    masks: (N, H, W) anatomical masks (1 = brain, 0 = background/border).
    baseline_edge_bias: the pre-APL edge-bias value from Table VIII to report
    the relative reduction against (e.g. 0.1464 for EfficientNetB0/no-tumor).
    Returns (new_edge_bias_mean, relative_reduction_pct).
    """
    off_target_mass = (saliency_maps * (1.0 - masks)).sum(axis=(1, 2))
    total_mass = saliency_maps.sum(axis=(1, 2)) + 1e-8
    edge_bias_per_image = off_target_mass / total_mass
    new_mean = float(edge_bias_per_image.mean())
    reduction_pct = 100.0 * (baseline_edge_bias - new_mean) / baseline_edge_bias
    return new_mean, reduction_pct
