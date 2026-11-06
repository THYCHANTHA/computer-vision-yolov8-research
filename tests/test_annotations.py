from pathlib import Path

import pytest

from facemask.annotations import (
    load_annotations,
    notebook_file_order,
    parse_voc_xml,
    to_yolo_lines,
    validate_annotation,
)

XML = """<annotation><folder>images</folder><filename>{fn}</filename>
<size><width>{w}</width><height>{h}</height><depth>3</depth></size><segmented>0</segmented>
{objects}</annotation>"""
OBJ = """<object><name>{label}</name><pose>Unspecified</pose><truncated>0</truncated>
<occluded>0</occluded><difficult>0</difficult><bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin>
<xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox></object>"""


def write_xml(path: Path, fn="img0.png", w=400, h=200, boxes=(("with_mask", 10, 20, 50, 80),)):
    objects = "".join(OBJ.format(label=b[0], x1=b[1], y1=b[2], x2=b[3], y2=b[4]) for b in boxes)
    path.write_text(XML.format(fn=fn, w=w, h=h, objects=objects))
    return path


def test_parse_reads_size_and_every_object(tmp_path):
    p = write_xml(
        tmp_path / "img0.xml", boxes=[("with_mask", 1, 2, 3, 4), ("without_mask", 5, 6, 7, 8)]
    )
    a = parse_voc_xml(p)
    assert (a.filename, a.width, a.height, a.depth) == ("img0.png", 400, 200, 3)
    assert [b.label for b in a.boxes] == ["with_mask", "without_mask"]
    assert a.boxes[1].xmax == 7


def test_parse_rejects_missing_size(tmp_path):
    p = tmp_path / "bad.xml"
    p.write_text("<annotation><filename>x.png</filename></annotation>")
    with pytest.raises(ValueError, match="size"):
        parse_voc_xml(p)


def test_yolo_conversion_matches_notebook_formula(tmp_path):
    a = parse_voc_xml(write_xml(tmp_path / "img0.xml", boxes=[("with_mask", 10, 20, 50, 80)]))
    cls, x, y, w, h = to_yolo_lines(a)[0].split()
    # notebook: w=(xmax-xmin)/width, x=xmin/width + w/2
    assert cls == "1"
    assert float(w) == pytest.approx(40 / 400)
    assert float(h) == pytest.approx(60 / 200)
    assert float(x) == pytest.approx(10 / 400 + 0.05)
    assert float(y) == pytest.approx(20 / 200 + 0.15)


def test_class_ids_follow_notebook_name_map(tmp_path):
    boxes = [
        ("without_mask", 0, 0, 10, 10),
        ("with_mask", 0, 0, 10, 10),
        ("mask_weared_incorrect", 0, 0, 10, 10),
    ]
    a = parse_voc_xml(write_xml(tmp_path / "img0.xml", boxes=boxes))
    assert [ln.split()[0] for ln in to_yolo_lines(a)] == ["0", "1", "2"]


def test_clip_keeps_boxes_inside_image(tmp_path):
    a = parse_voc_xml(write_xml(tmp_path / "img0.xml", boxes=[("with_mask", 380, 0, 401, 10)]))
    _, x, _, w, _ = to_yolo_lines(a, clip=True)[0].split()
    assert float(x) + float(w) / 2 <= 1.0 + 1e-9
    _, x2, _, w2, _ = to_yolo_lines(a, clip=False)[0].split()
    assert float(x2) + float(w2) / 2 > 1.0


def test_validation_flags_each_problem(tmp_path):
    boxes = [("hat", 0, 0, 10, 10), ("with_mask", 10, 10, 10, 30), ("with_mask", 390, 0, 401, 9)]
    a = parse_voc_xml(write_xml(tmp_path / "other.xml", fn="img0.png", boxes=boxes))
    kinds = {i.kind for i in validate_annotation(a, image_size=(401, 200))}
    assert kinds == {
        "unknown_class",
        "degenerate_box",
        "box_outside_image",
        "filename_mismatch",
        "size_mismatch",
    }


def test_clean_annotation_has_no_issues(tmp_path):
    a = parse_voc_xml(write_xml(tmp_path / "img0.xml"))
    assert validate_annotation(a, image_size=(400, 200)) == []


def test_file_order_is_case_insensitive_sorted(tmp_path):
    for n in ["b10.xml", "B2.xml", "a1.xml", "b1.xml"]:
        write_xml(tmp_path / n)
    assert [p.name for p in notebook_file_order(tmp_path)] == [
        "a1.xml",
        "b1.xml",
        "b10.xml",
        "B2.xml",
    ]


def test_load_reports_empty_and_broken_files(tmp_path):
    write_xml(tmp_path / "good.xml", fn="good.png")
    (tmp_path / "empty.xml").write_text("")
    (tmp_path / "broken.xml").write_text("<annotation>")
    anns, issues = load_annotations(tmp_path)
    assert [a.filename for a in anns] == ["good.png"]
    assert sorted(i.kind for i in issues) == ["empty_xml", "unreadable_xml"]
