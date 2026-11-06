"""Paths and constants. Data locations come from environment variables so nothing machine-specific
is stored in the repository."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Class ids exactly as in the original notebook (`name_map`), so the recorded weights line up.
CLASS_NAMES: tuple[str, ...] = ("without_mask", "with_mask", "mask_weared_incorrect")
CLASS_TO_ID: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}
DISPLAY_NAMES: dict[int, str] = {0: "No mask", 1: "Mask", 2: "Incorrect"}

# The original notebook split: 80 % train, then the remaining 20 % halved into val and test.
SPLIT_SEED = 1
FIRST_TEST_SIZE = 0.2
SECOND_TEST_SIZE = 0.5

WEIGHTS_PATH = PROJECT_ROOT / "weights" / "best.pt"


def raw_data_dir() -> Path:
    """Folder that holds the Kaggle `annotations/` and `images/` folders."""
    return Path(os.environ.get("FACEMASK_DATA_DIR", PROJECT_ROOT / "data" / "raw"))


def work_dir() -> Path:
    """Folder for the generated YOLO dataset and evaluation runs (regenerable, not committed)."""
    return Path(os.environ.get("FACEMASK_WORK_DIR", PROJECT_ROOT / "data" / "work"))


REPORTS_DIR = PROJECT_ROOT / "reports"
