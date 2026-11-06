"""Exact and near-duplicate image detection, used to check for leakage between splits.

A random image-level split is only a fair test if the same scene does not appear in both the
training and the test set. The dataset was collected from the web and may contain re-encoded or
resized copies, so files are compared both by content hash and by a 64-bit difference hash
(dHash), which survives resizing and recompression.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from PIL import Image


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(image: Image.Image, hash_size: int = 8) -> int:
    """Difference hash: compare horizontally adjacent pixels of a (hash_size+1) x hash_size
    greyscale thumbnail. Returns an integer with hash_size**2 bits."""
    small = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    px = np.asarray(small, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def near_duplicate_pairs(hashes: dict[str, int], max_distance: int) -> list[tuple[str, str, int]]:
    """All pairs (a, b, distance) with Hamming distance <= max_distance, a < b."""
    names = sorted(hashes)
    values = np.array([hashes[n] for n in names], dtype=np.uint64)
    pairs = []
    for i in range(len(names) - 1):
        x = np.bitwise_xor(values[i], values[i + 1 :])
        # popcount of uint64 without relying on numpy >= 2 bitwise_count
        d = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
        for j in np.nonzero(d <= max_distance)[0]:
            pairs.append((names[i], names[i + 1 + int(j)], int(d[j])))
    return pairs


def cross_split_pairs(
    pairs: Iterable[tuple[str, str, int]], split_of: dict[str, str]
) -> list[tuple[str, str, int, str, str]]:
    """Keep only pairs whose two images fall in different splits."""
    out = []
    for a, b, d in pairs:
        sa, sb = split_of.get(a), split_of.get(b)
        if sa is not None and sb is not None and sa != sb:
            out.append((a, b, d, sa, sb))
    return out
