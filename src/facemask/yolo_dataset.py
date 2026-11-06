"""Build the Ultralytics dataset folder (images/<split>, labels/<split>, data.yaml).

Images are hard-linked from the raw folder when both are on the same drive, so the dataset costs
no extra disk space; otherwise they are copied after a free-space check.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from .annotations import ImageAnnotation, to_yolo_lines
from .config import CLASS_NAMES


def _link_or_copy(src: Path, dst: Path) -> str:
    if dst.exists():
        return "exists"
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def write_yolo_dataset(
    annotations: list[ImageAnnotation],
    splits: dict[str, list[str]],
    image_dir: Path,
    out_dir: Path,
    clip: bool = False,
) -> dict[str, int]:
    """Write labels and link images for every split; return counts of each file action."""
    by_name = {a.filename: a for a in annotations}
    needed = sum((image_dir / f).stat().st_size for files in splits.values() for f in files)
    out_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out_dir).free
    same_drive = (
        os.path.splitdrive(image_dir.resolve())[0].upper()
        == os.path.splitdrive(out_dir.resolve())[0].upper()
    )
    if not same_drive and needed * 1.2 > free:
        raise OSError(
            f"copying images needs ~{needed / 1e9:.2f} GB but {out_dir} has "
            f"{free / 1e9:.2f} GB free; put FACEMASK_WORK_DIR on the data drive"
        )
    actions: dict[str, int] = {}
    for split, files in splits.items():
        img_out = out_dir / "images" / split
        lbl_out = out_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for f in files:
            action = _link_or_copy(image_dir / f, img_out / f)
            actions[action] = actions.get(action, 0) + 1
            lines = to_yolo_lines(by_name[f], clip=clip)
            (lbl_out / (Path(f).stem + ".txt")).write_text("\n".join(lines) + "\n")
    write_data_yaml(out_dir)
    return actions


def write_data_yaml(out_dir: Path, test_folder: str = "test", filename: str = "data.yaml") -> Path:
    """Write an Ultralytics data file; `test_folder` lets a second file point `test` at the
    de-duplicated test images."""
    path = out_dir / filename
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    path.write_text(
        f"path: {out_dir.resolve().as_posix()}\n"
        f"train: images/train\nval: images/val\ntest: images/{test_folder}\n"
        f"names:\n{names}\n"
    )
    return path
