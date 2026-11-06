"""Audit the Kaggle face-mask dataset and the notebook's split. Writes reports/data_audit.json and
reports/near_duplicates_cross_split.csv.

Usage:  FACEMASK_DATA_DIR=<folder with annotations/ and images/>  python scripts/audit_dataset.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from facemask.annotations import Issue, load_annotations, validate_annotation  # noqa: E402
from facemask.config import CLASS_NAMES, REPORTS_DIR, raw_data_dir  # noqa: E402
from facemask.duplicates import (  # noqa: E402
    cross_split_pairs,
    dhash,
    file_sha256,
    near_duplicate_pairs,
)
from facemask.splits import check_disjoint, notebook_split  # noqa: E402

NEAR_DUP_MAX_DISTANCE = 4  # of 64 bits; candidates, confirmed by visual inspection


def main() -> int:
    raw = raw_data_dir()
    ann_dir, img_dir = raw / "annotations", raw / "images"
    if not ann_dir.is_dir() or not img_dir.is_dir():
        print(f"dataset not found under {raw}; set FACEMASK_DATA_DIR", file=sys.stderr)
        return 2

    annotations, issues = load_annotations(ann_dir)
    image_files = sorted(p.name for p in img_dir.iterdir() if p.is_file())
    annotated = {a.filename for a in annotations}

    sizes, sha, dh = {}, {}, {}
    for name in image_files:
        path = img_dir / name
        with Image.open(path) as im:
            sizes[name] = im.size
            dh[name] = dhash(im)
        sha[name] = file_sha256(path)
    for a in annotations:
        issues.extend(validate_annotation(a, sizes.get(a.filename)))
        if a.filename not in sizes:
            issues.append(Issue(a.filename, "missing_image", "no image file"))

    class_counts = Counter(b.label for a in annotations for b in a.boxes)
    boxes_per_image = [len(a.boxes) for a in annotations]
    issue_counts = Counter(i.kind for i in issues)

    order = [a.filename for a in annotations]
    splits = notebook_split(order)
    check_disjoint(splits)
    split_of = {f: s for s, files in splits.items() for f in files}
    split_class_counts = {
        s: dict(Counter(b.label for a in annotations if split_of[a.filename] == s for b in a.boxes))
        for s in splits
    }

    by_sha: dict[str, list[str]] = {}
    for n, h in sha.items():
        by_sha.setdefault(h, []).append(n)
    exact_groups = [sorted(v) for v in by_sha.values() if len(v) > 1]

    pairs = near_duplicate_pairs(dh, NEAR_DUP_MAX_DISTANCE)
    cross = cross_split_pairs(pairs, split_of)
    test_touched = sorted(
        {
            x
            for a, b, _, sa, sb in cross
            for x, s in ((a, sa), (b, sb))
            if s == "test" and {sa, sb} & {"train", "val"}
        }
    )

    REPORTS_DIR.mkdir(exist_ok=True)
    with open(REPORTS_DIR / "near_duplicates_cross_split.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_a", "image_b", "dhash_distance", "split_a", "split_b"])
        w.writerows(cross)

    report = {
        "dataset_dir_name": raw.name,
        "n_xml": len(annotations)
        + issue_counts.get("unreadable_xml", 0)
        + issue_counts.get("empty_xml", 0),
        "n_images": len(image_files),
        "images_without_xml": sorted(set(image_files) - annotated),
        "n_boxes": sum(boxes_per_image),
        "class_counts": {c: class_counts.get(c, 0) for c in CLASS_NAMES},
        "boxes_per_image": {
            "min": min(boxes_per_image),
            "median": sorted(boxes_per_image)[len(boxes_per_image) // 2],
            "max": max(boxes_per_image),
        },
        "issue_counts": dict(issue_counts),
        "issues": [i.__dict__ for i in issues][:200],
        "split_sizes": {s: len(v) for s, v in splits.items()},
        "split_class_counts": split_class_counts,
        "test_images": sorted(splits["test"]),
        "exact_duplicate_groups": exact_groups,
        "near_duplicate_threshold_bits": NEAR_DUP_MAX_DISTANCE,
        "near_duplicate_pairs_total": len(pairs),
        "near_duplicate_pairs_cross_split": len(cross),
        "test_images_with_near_duplicate_in_train_or_val": test_touched,
    }
    (REPORTS_DIR / "data_audit.json").write_text(json.dumps(report, indent=2))
    summary = {k: v for k, v in report.items() if k not in ("issues", "test_images")}
    print(json.dumps(summary, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
