"""Score the recorded weights (weights/best.pt) on the recreated val, test and de-duplicated test
splits, check that the val score reproduces the training log, and time CPU inference.
Writes reports/evaluation.json.

Usage:  FACEMASK_DATA_DIR=... FACEMASK_WORK_DIR=... python scripts/evaluate_model.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from facemask.config import PROJECT_ROOT, REPORTS_DIR, WEIGHTS_PATH, work_dir  # noqa: E402
from facemask.evaluation import evaluate_split, measure_latency, operating_point  # noqa: E402


def recorded_best_epoch(results_csv: Path) -> dict:
    """Row of the training log with the highest Ultralytics fitness (0.1*mAP50 + 0.9*mAP50-95),
    the rule Ultralytics uses to choose best.pt."""
    with open(results_csv, newline="") as f:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]

    def fitness(r: dict) -> float:
        return 0.1 * float(r["metrics/mAP50(B)"]) + 0.9 * float(r["metrics/mAP50-95(B)"])

    best = max(rows, key=fitness)
    return {
        "epoch": int(best["epoch"]),
        "precision": float(best["metrics/precision(B)"]),
        "recall": float(best["metrics/recall(B)"]),
        "mAP50": float(best["metrics/mAP50(B)"]),
        "mAP50_95": float(best["metrics/mAP50-95(B)"]),
    }


def main() -> int:
    yolo = work_dir() / "yolo"
    runs = work_dir() / "runs"
    if not (yolo / "data.yaml").exists():
        print("run scripts/build_yolo_dataset.py first", file=sys.stderr)
        return 2
    sha = hashlib.sha256(WEIGHTS_PATH.read_bytes()).hexdigest()

    results = {
        "weights_sha256": sha,
        "recorded_best_epoch_val": recorded_best_epoch(
            PROJECT_ROOT / "reports" / "original_training" / "results.csv"
        ),
        # Sanity check only: training images were seen in training, so this score should be
        # clearly higher than val/test if the recreated split matches the original one.
        "train_sanity_check": evaluate_split(WEIGHTS_PATH, yolo / "data.yaml", "train", runs),
        "val": evaluate_split(WEIGHTS_PATH, yolo / "data.yaml", "val", runs),
        "test": evaluate_split(WEIGHTS_PATH, yolo / "data.yaml", "test", runs),
        "test_dedup": evaluate_split(
            WEIGHTS_PATH, yolo / "data_test_dedup.yaml", "test", runs / "dedup"
        ),
    }
    rec, val = results["recorded_best_epoch_val"], results["val"]
    results["val_reproduction_abs_diff"] = {
        k: round(abs(rec[k] - val[k]), 4) for k in ("precision", "recall", "mAP50", "mAP50_95")
    }
    results["operating_point_test_dedup"] = operating_point(
        WEIGHTS_PATH, yolo / "images" / "test_dedup", yolo / "labels" / "test_dedup"
    )
    test_images = sorted((yolo / "images" / "test").glob("*.png"))
    results["latency"] = measure_latency(WEIGHTS_PATH, test_images)

    import torch
    import ultralytics

    results["versions"] = {
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "python": sys.version.split()[0],
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "evaluation.json").write_text(json.dumps(results, indent=2))
    for k in ("train_sanity_check", "val", "test", "test_dedup"):
        r = results[k]
        print(
            f"{k:11s} P={r['precision']:.4f} R={r['recall']:.4f} "
            f"mAP50={r['mAP50']:.4f} mAP50-95={r['mAP50_95']:.4f}"
        )
    print("recorded   ", rec)
    print("abs diff   ", results["val_reproduction_abs_diff"])
    print("latency    ", {k: results["latency"][k] for k in ("n_images", "median_ms", "p90_ms")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
