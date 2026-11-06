"""Checks against the real Kaggle dataset and the recorded weights. Skipped when absent."""

import hashlib
import json

import pytest

from facemask.annotations import load_annotations
from facemask.config import REPORTS_DIR, WEIGHTS_PATH, raw_data_dir
from facemask.splits import notebook_split

RAW = raw_data_dir()
needs_data = pytest.mark.skipif(not (RAW / "annotations").is_dir(), reason="dataset not present")
WEIGHTS_SHA256 = "378dd980300b5a5ec34f131a98eea74e23719ba6e512b9c60baa1465ef971d35"


@pytest.mark.data
@needs_data
def test_dataset_matches_published_description():
    anns, issues = load_annotations(RAW / "annotations")
    assert issues == []
    assert len(anns) == 853
    counts = {}
    for a in anns:
        for b in a.boxes:
            counts[b.label] = counts.get(b.label, 0) + 1
    assert counts == {"with_mask": 3232, "without_mask": 717, "mask_weared_incorrect": 123}


@pytest.mark.data
@needs_data
def test_split_matches_the_audited_test_set():
    anns, _ = load_annotations(RAW / "annotations")
    split = notebook_split([a.filename for a in anns])
    audit = json.loads((REPORTS_DIR / "data_audit.json").read_text())
    assert sorted(split["test"]) == audit["test_images"]


@pytest.mark.skipif(not WEIGHTS_PATH.exists(), reason="weights not present")
def test_weights_are_the_recorded_run():
    assert hashlib.sha256(WEIGHTS_PATH.read_bytes()).hexdigest() == WEIGHTS_SHA256
