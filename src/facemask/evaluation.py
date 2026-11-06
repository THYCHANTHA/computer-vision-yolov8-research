"""Detection metrics and CPU latency for a trained YOLO model. Imports Ultralytics lazily so the
data modules and their tests do not need it."""

from __future__ import annotations

import platform
import statistics
import time
from pathlib import Path

from .config import CLASS_NAMES


def evaluate_split(
    weights: Path,
    data_yaml: Path,
    split: str,
    run_dir: Path,
    imgsz: int = 640,
    batch: int = 8,
) -> dict:
    """Ultralytics validation on one split (conf=0.001, IoU 0.7 NMS: the library's defaults for
    mAP). Precision/recall are reported at the confidence that maximises mean F1."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    m = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        batch=batch,
        device="cpu",
        plots=False,
        verbose=False,
        project=str(run_dir),
        name=split,
        exist_ok=True,
    )
    per_class = {}
    for k, cls in enumerate(m.box.ap_class_index):
        cls = int(cls)
        per_class[CLASS_NAMES[cls]] = {
            "precision": float(m.box.p[k]),
            "recall": float(m.box.r[k]),
            "mAP50": float(m.box.ap50[k]),
            "mAP50_95": float(m.box.ap[k]),
        }
    return {
        "split": split,
        "images": int(m.seen) if hasattr(m, "seen") else None,
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "mAP50": float(m.box.map50),
        "mAP50_95": float(m.box.map),
        "per_class": per_class,
        "ultralytics_speed_ms": {k: float(v) for k, v in m.speed.items()},
    }


def operating_point(
    weights: Path,
    image_dir: Path,
    label_dir: Path,
    conf: float = 0.5,
    nms_iou: float = 0.5,
    match_iou: float = 0.5,
    imgsz: int = 640,
) -> dict:
    """Confusion table and per-class precision/recall at the web app's settings
    (app.py: conf=0.5, iou=0.5)."""
    import numpy as np
    from ultralytics import YOLO

    from .matching import bootstrap_intervals, match_image, per_class_scores

    model = YOLO(str(weights))
    n = len(CLASS_NAMES)
    per_image = []
    for img in sorted(image_dir.glob("*.png")):
        r = model.predict(
            str(img), conf=conf, iou=nms_iou, imgsz=imgsz, device="cpu", verbose=False
        )[0]
        h, w = r.orig_shape
        rows = [
            ln.split()
            for ln in (label_dir / f"{img.stem}.txt").read_text().splitlines()
            if ln.strip()
        ]
        lab = np.array(rows, dtype=float).reshape(-1, 5)
        xc, yc, bw, bh = lab[:, 1] * w, lab[:, 2] * h, lab[:, 3] * w, lab[:, 4] * h
        gt = np.stack([xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2], axis=1)
        per_image.append(
            match_image(
                gt,
                lab[:, 0].astype(int),
                r.boxes.xyxy.numpy(),
                r.boxes.cls.numpy().astype(int),
                r.boxes.conf.numpy(),
                n,
                match_iou,
            )
        )
    stacked = np.stack(per_image)
    cm = stacked.sum(axis=0)
    scores = per_class_scores(cm)
    for s, ci in zip(scores, bootstrap_intervals(stacked), strict=True):
        s["precision_95ci"], s["recall_95ci"] = ci["precision_ci"], ci["recall_ci"]
        s["class"] = CLASS_NAMES[s.pop("class_id")]
    return {
        "n_images": len(per_image),
        "bootstrap": "2000 image-level resamples, percentile 95% interval, seed 0",
        "settings": {"conf": conf, "nms_iou": nms_iou, "match_iou": match_iou, "imgsz": imgsz},
        "labels": [*CLASS_NAMES, "background/missed"],
        "confusion_rows_gt_cols_pred": cm.tolist(),
        "per_class": scores,
    }


def measure_latency(
    weights: Path, images: list[Path], imgsz: int = 640, warmup: int = 5, conf: float = 0.5
) -> dict:
    """End-to-end single-image predict time on CPU (pre-process + inference + NMS), batch 1."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    for p in images[:warmup]:
        model.predict(str(p), imgsz=imgsz, conf=conf, device="cpu", verbose=False)
    times = []
    for p in images:
        t0 = time.perf_counter()
        model.predict(str(p), imgsz=imgsz, conf=conf, device="cpu", verbose=False)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {
        "n_images": len(times),
        "median_ms": statistics.median(times),
        "p90_ms": times[int(0.9 * (len(times) - 1))],
        "mean_ms": statistics.fmean(times),
        "imgsz": imgsz,
        "device": "cpu",
        "processor": platform.processor(),
        "platform": platform.platform(),
    }
