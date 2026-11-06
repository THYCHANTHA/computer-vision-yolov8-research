"""Recreate the notebook's YOLO dataset (same labels, same 80/10/10 split) plus a de-duplicated
test split that drops test images with an exact or near duplicate in train/val.

Needs reports/data_audit.json (run scripts/audit_dataset.py first).
Usage:  FACEMASK_DATA_DIR=... FACEMASK_WORK_DIR=... python scripts/build_yolo_dataset.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from facemask.annotations import load_annotations  # noqa: E402
from facemask.config import REPORTS_DIR, raw_data_dir, work_dir  # noqa: E402
from facemask.splits import check_disjoint, notebook_split  # noqa: E402
from facemask.yolo_dataset import write_data_yaml, write_yolo_dataset  # noqa: E402


def main() -> int:
    raw, out = raw_data_dir(), work_dir() / "yolo"
    annotations, issues = load_annotations(raw / "annotations")
    if issues:
        print("unreadable annotation files:", issues, file=sys.stderr)
        return 1
    splits = notebook_split([a.filename for a in annotations])
    check_disjoint(splits)

    audit = json.loads((REPORTS_DIR / "data_audit.json").read_text())
    if sorted(splits["test"]) != audit["test_images"]:
        print("split differs from reports/data_audit.json; re-run the audit", file=sys.stderr)
        return 1
    leaked = set(audit["test_images_with_near_duplicate_in_train_or_val"])
    dedup = {"test_dedup": [f for f in splits["test"] if f not in leaked]}

    # Labels reproduce the notebook arithmetic (no clipping) so the recorded weights are scored
    # against the labels they were validated on.
    actions = write_yolo_dataset(annotations, splits, raw / "images", out, clip=False)
    actions_d = write_yolo_dataset(annotations, dedup, raw / "images", out, clip=False)
    write_data_yaml(out)  # the dedup call overwrote data.yaml with the default; rewrite both
    write_data_yaml(out, test_folder="test_dedup", filename="data_test_dedup.yaml")
    print(
        json.dumps(
            {
                "out": str(out),
                "sizes": {k: len(v) for k, v in {**splits, **dedup}.items()},
                "file_actions": actions,
                "file_actions_dedup": actions_d,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
