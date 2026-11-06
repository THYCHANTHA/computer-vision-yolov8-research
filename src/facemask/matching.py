"""Operating-point evaluation: match predictions to ground truth at a fixed confidence threshold
(the one the web app uses) and build a class-confusion table. Pure NumPy, no Ultralytics.

mAP summarises all thresholds; a deployed app uses one. This module answers "at conf=0.5, how
many faces of each class are found, mislabelled or missed, and how many false alarms occur?"
"""

from __future__ import annotations

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a (N,4) and b (M,4), boxes as xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def match_image(
    gt_boxes: np.ndarray,
    gt_cls: np.ndarray,
    pred_boxes: np.ndarray,
    pred_cls: np.ndarray,
    pred_conf: np.ndarray,
    n_classes: int,
    iou_thr: float = 0.5,
) -> np.ndarray:
    """Confusion counts for one image, shape (n_classes+1, n_classes+1).

    Rows = ground-truth class, last row = background (false positive).
    Columns = predicted class, last column = missed (false negative).
    Greedy matching by descending confidence; a prediction may match a ground-truth box of any
    class, so a face found with the wrong label counts as a confusion, not as two errors.
    """
    cm = np.zeros((n_classes + 1, n_classes + 1), dtype=int)
    order = np.argsort(-pred_conf, kind="stable")
    ious = iou_matrix(pred_boxes[order], gt_boxes)
    taken = np.zeros(len(gt_boxes), dtype=bool)
    for k, p in enumerate(order):
        if len(gt_boxes):
            cand = np.where(~taken, ious[k], -1.0)
            j = int(np.argmax(cand))
            if cand[j] >= iou_thr:
                taken[j] = True
                cm[int(gt_cls[j]), int(pred_cls[p])] += 1
                continue
        cm[n_classes, int(pred_cls[p])] += 1
    for j in np.nonzero(~taken)[0]:
        cm[int(gt_cls[j]), n_classes] += 1
    return cm


def bootstrap_intervals(
    per_image: np.ndarray, n_boot: int = 2000, seed: int = 0, level: float = 0.95
) -> list[dict]:
    """Percentile intervals for per-class precision and recall, resampling whole images.

    `per_image` has shape (n_images, n_classes+1, n_classes+1). Faces in one image are not
    independent, so images (not boxes) are the resampling unit.
    """
    rng = np.random.default_rng(seed)
    n_img, k1, _ = per_image.shape
    n = k1 - 1
    prec = np.full((n_boot, n), np.nan)
    rec = np.full((n_boot, n), np.nan)
    for i in range(n_boot):
        cm = per_image[rng.integers(0, n_img, n_img)].sum(axis=0)
        diag = np.diag(cm)[:n]
        pred, act = cm[:, :n].sum(axis=0), cm[:n, :].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            prec[i] = np.where(pred > 0, diag / pred, np.nan)
            rec[i] = np.where(act > 0, diag / act, np.nan)
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100

    def interval(values: np.ndarray) -> list[float] | None:
        """None when the statistic is undefined in every resample (e.g. no ground truth)."""
        values = values[~np.isnan(values)]
        if values.size == 0:
            return None
        return [float(np.percentile(values, lo)), float(np.percentile(values, hi))]

    return [
        {"class_id": c, "precision_ci": interval(prec[:, c]), "recall_ci": interval(rec[:, c])}
        for c in range(n)
    ]


def per_class_scores(cm: np.ndarray) -> list[dict]:
    """Precision/recall per class from a confusion table built by match_image."""
    n = cm.shape[0] - 1
    out = []
    for c in range(n):
        tp = int(cm[c, c])
        predicted = int(cm[:, c].sum())
        actual = int(cm[c, :].sum())
        out.append(
            {
                "class_id": c,
                "true_positives": tp,
                "predicted": predicted,
                "ground_truth": actual,
                "precision": tp / predicted if predicted else None,
                "recall": tp / actual if actual else None,
            }
        )
    return out
