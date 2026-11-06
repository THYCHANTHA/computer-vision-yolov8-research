"""Train/val/test split exactly as the original notebook made it.

Notebook cell: `train_test_split(images, annotations, test_size=0.2, random_state=1)` then
`train_test_split(val_images, val_annotations, test_size=0.5, random_state=1)`. Splitting two
parallel arrays uses one permutation, so splitting the image list alone gives the same result.
"""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.model_selection import train_test_split

from .config import FIRST_TEST_SIZE, SECOND_TEST_SIZE, SPLIT_SEED


def notebook_split(filenames: Sequence[str], seed: int = SPLIT_SEED) -> dict[str, list[str]]:
    """Random image-level 80/10/10 split. `filenames` must be in notebook order."""
    if len(set(filenames)) != len(filenames):
        raise ValueError("filenames must be unique")
    train, rest = train_test_split(list(filenames), test_size=FIRST_TEST_SIZE, random_state=seed)
    val, test = train_test_split(rest, test_size=SECOND_TEST_SIZE, random_state=seed)
    return {"train": list(train), "val": list(val), "test": list(test)}


def check_disjoint(splits: dict[str, list[str]]) -> None:
    """Raise if any filename appears in more than one split."""
    seen: dict[str, str] = {}
    for name, files in splits.items():
        for f in files:
            if f in seen:
                raise ValueError(f"{f} is in both {seen[f]} and {name}")
            seen[f] = name
