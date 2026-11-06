"""Pascal-VOC XML parsing, validation and conversion to YOLO labels.

The original notebook flattened every XML tag into a DataFrame. This module parses the same fields
explicitly and adds checks the notebook did not have (unknown classes, degenerate boxes, boxes
outside the image, XML size that disagrees with the real image).
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from .config import CLASS_TO_ID


@dataclass(frozen=True)
class Box:
    label: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    difficult: int = 0
    truncated: int = 0
    occluded: int = 0

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin


@dataclass(frozen=True)
class ImageAnnotation:
    xml_name: str
    filename: str
    width: int
    height: int
    depth: int
    boxes: tuple[Box, ...]


@dataclass(frozen=True)
class Issue:
    filename: str
    kind: str
    detail: str


def _int(node: ET.Element | None, default: int = 0) -> int:
    if node is None or node.text is None or not node.text.strip():
        return default
    return int(float(node.text))


def parse_voc_xml(path: str | Path) -> ImageAnnotation:
    """Parse one VOC annotation file. Raises ValueError on a missing size or box field."""
    path = Path(path)
    root = ET.parse(path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"{path.name}: no <size> element")
    filename = (root.findtext("filename") or "").strip()
    boxes = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        if bb is None:
            raise ValueError(f"{path.name}: <object> without <bndbox>")
        coords = []
        for tag in ("xmin", "ymin", "xmax", "ymax"):
            text = bb.findtext(tag)
            if text is None:
                raise ValueError(f"{path.name}: <bndbox> missing <{tag}>")
            coords.append(float(text))
        boxes.append(
            Box(
                label=(obj.findtext("name") or "").strip(),
                xmin=coords[0],
                ymin=coords[1],
                xmax=coords[2],
                ymax=coords[3],
                difficult=_int(obj.find("difficult")),
                truncated=_int(obj.find("truncated")),
                occluded=_int(obj.find("occluded")),
            )
        )
    return ImageAnnotation(
        xml_name=path.name,
        filename=filename,
        width=_int(size.find("width")),
        height=_int(size.find("height")),
        depth=_int(size.find("depth")),
        boxes=tuple(boxes),
    )


def notebook_file_order(annotation_dir: str | Path) -> list[Path]:
    """XML files in the order the original notebook visited them.

    The notebook used `os.walk`, which on Windows/NTFS lists names in case-insensitive sorted
    order. Sorting explicitly makes the order identical on every operating system.
    """
    files = []
    for dirname, _, filenames in os.walk(annotation_dir):
        files.extend(Path(dirname) / f for f in filenames)
    return sorted(files, key=lambda p: str(p).upper())


def load_annotations(annotation_dir: str | Path) -> tuple[list[ImageAnnotation], list[Issue]]:
    """Parse every XML file; unreadable files become issues instead of silently disappearing."""
    parsed: list[ImageAnnotation] = []
    issues: list[Issue] = []
    for path in notebook_file_order(annotation_dir):
        if path.stat().st_size == 0:
            issues.append(Issue(path.name, "empty_xml", "file is empty"))
            continue
        try:
            parsed.append(parse_voc_xml(path))
        except (ET.ParseError, ValueError) as exc:
            issues.append(Issue(path.name, "unreadable_xml", str(exc)))
    return parsed, issues


def validate_annotation(
    ann: ImageAnnotation, image_size: tuple[int, int] | None = None
) -> list[Issue]:
    """Return every problem found in one image's annotation. `image_size` is (width, height)."""
    out: list[Issue] = []
    name = ann.filename or ann.xml_name
    if Path(ann.xml_name).stem != Path(ann.filename).stem:
        out.append(Issue(name, "filename_mismatch", f"xml {ann.xml_name} names {ann.filename}"))
    if ann.width <= 0 or ann.height <= 0:
        out.append(Issue(name, "bad_size", f"size {ann.width}x{ann.height}"))
        return out
    if image_size is not None and tuple(image_size) != (ann.width, ann.height):
        out.append(
            Issue(name, "size_mismatch", f"xml {ann.width}x{ann.height}, image {image_size}")
        )
    if not ann.boxes:
        out.append(Issue(name, "no_objects", "image has no annotated faces"))
    for i, b in enumerate(ann.boxes):
        if b.label not in CLASS_TO_ID:
            out.append(Issue(name, "unknown_class", f"box {i}: {b.label!r}"))
        if b.width <= 0 or b.height <= 0:
            out.append(Issue(name, "degenerate_box", f"box {i}: {b.width}x{b.height}"))
        if b.xmin < 0 or b.ymin < 0 or b.xmax > ann.width or b.ymax > ann.height:
            out.append(
                Issue(
                    name,
                    "box_outside_image",
                    f"box {i}: ({b.xmin},{b.ymin},{b.xmax},{b.ymax}) in {ann.width}x{ann.height}",
                )
            )
    return out


def to_yolo_lines(ann: ImageAnnotation, clip: bool = False) -> list[str]:
    """YOLO label lines `class x_center y_center w h`, normalised by the XML image size.

    `clip=False` reproduces the notebook's arithmetic exactly. `clip=True` clamps boxes to the
    image first, which is what a corrected pipeline should do (Ultralytics rejects any label
    file with a normalised coordinate above 1).
    """
    lines = []
    for b in ann.boxes:
        xmin, ymin, xmax, ymax = b.xmin, b.ymin, b.xmax, b.ymax
        if clip:
            xmin, xmax = max(0.0, xmin), min(float(ann.width), xmax)
            ymin, ymax = max(0.0, ymin), min(float(ann.height), ymax)
        w = (xmax - xmin) / ann.width
        h = (ymax - ymin) / ann.height
        x = xmin / ann.width + w / 2
        y = ymin / ann.height + h / 2
        lines.append(f"{CLASS_TO_ID[b.label]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    return lines
