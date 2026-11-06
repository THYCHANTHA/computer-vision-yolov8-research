import numpy as np
import pytest
from PIL import Image

from facemask.duplicates import cross_split_pairs, dhash, hamming, near_duplicate_pairs
from facemask.matching import bootstrap_intervals, iou_matrix, match_image, per_class_scores
from facemask.splits import check_disjoint, notebook_split

NAMES = [f"maksssksksss{i}.png" for i in range(853)]


def test_split_sizes_match_notebook_proportions():
    s = notebook_split(NAMES)
    assert {k: len(v) for k, v in s.items()} == {"train": 682, "val": 85, "test": 86}
    check_disjoint(s)
    assert sorted(sum(s.values(), [])) == sorted(NAMES)


def test_split_is_deterministic_and_seed_dependent():
    assert notebook_split(NAMES) == notebook_split(NAMES)
    assert notebook_split(NAMES)["test"] != notebook_split(NAMES, seed=2)["test"]


def test_split_rejects_duplicate_names():
    with pytest.raises(ValueError):
        notebook_split(["a.png", "a.png", "b.png"])


def test_check_disjoint_detects_overlap():
    with pytest.raises(ValueError, match="both"):
        check_disjoint({"train": ["a"], "test": ["a"]})


def _gradient(w=64, h=48, flip=False):
    arr = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    arr[: h // 2] = arr[: h // 2, ::-1]
    return Image.fromarray(arr[:, ::-1] if flip else arr)


def test_dhash_is_stable_under_resize_and_differs_for_other_images():
    img = _gradient()
    assert hamming(dhash(img), dhash(img.resize((128, 96)))) <= 2
    assert hamming(dhash(img), dhash(_gradient(flip=True))) > 20


def test_near_duplicate_pairs_and_cross_split_filter():
    hashes = {"a": 0b1111, "b": 0b1110, "c": 0b1 << 40}
    pairs = near_duplicate_pairs(hashes, max_distance=1)
    assert pairs == [("a", "b", 1)]
    assert cross_split_pairs(pairs, {"a": "train", "b": "train"}) == []
    assert cross_split_pairs(pairs, {"a": "train", "b": "test"}) == [("a", "b", 1, "train", "test")]


def test_iou_matrix_values():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], float)
    assert iou_matrix(a, b) == pytest.approx(np.array([[1.0, 50 / 150, 0.0]]))


def test_match_image_counts_hits_confusions_misses_and_false_alarms():
    gt = np.array([[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10]], float)
    gt_cls = np.array([0, 1, 2])
    pred = np.array([[0, 0, 10, 10], [20, 0, 30, 10], [100, 100, 110, 110]], float)
    pred_cls = np.array([0, 2, 1])  # correct, wrong label, false alarm
    cm = match_image(gt, gt_cls, pred, pred_cls, np.array([0.9, 0.8, 0.7]), n_classes=3)
    assert cm[0, 0] == 1  # hit
    assert cm[1, 2] == 1  # with_mask predicted as incorrect
    assert cm[2, 3] == 1  # incorrect face missed
    assert cm[3, 1] == 1  # false alarm predicted as with_mask
    assert cm.sum() == 4
    scores = per_class_scores(cm)
    assert scores[0]["precision"] == 1.0 and scores[0]["recall"] == 1.0
    assert scores[1]["recall"] == 0.0 and scores[1]["precision"] == 0.0


def test_one_ground_truth_box_matches_at_most_once():
    gt = np.array([[0, 0, 10, 10]], float)
    pred = np.array([[0, 0, 10, 10], [0, 0, 10, 9]], float)
    cm = match_image(gt, np.array([1]), pred, np.array([1, 1]), np.array([0.6, 0.9]), 3)
    assert cm[1, 1] == 1 and cm[3, 1] == 1


def test_bootstrap_intervals_contain_point_estimate_and_are_reproducible():
    rng = np.random.default_rng(1)
    per_image = np.zeros((40, 4, 4), dtype=int)
    for k in range(40):
        per_image[k, 1, 1] = rng.integers(1, 5)  # hits
        per_image[k, 1, 3] = rng.integers(0, 2)  # misses
    cm = per_image.sum(axis=0)
    recall = cm[1, 1] / cm[1].sum()
    a = bootstrap_intervals(per_image, n_boot=300, seed=3)
    assert a == bootstrap_intervals(per_image, n_boot=300, seed=3)
    lo, hi = a[1]["recall_ci"]
    assert lo <= recall <= hi and hi - lo > 0
    assert a[0]["recall_ci"] is None  # class with no ground truth has no interval


def test_no_predictions_means_all_missed():
    gt = np.array([[0, 0, 10, 10]], float)
    cm = match_image(gt, np.array([0]), np.zeros((0, 4)), np.zeros(0, int), np.zeros(0), 3)
    assert cm[0, 3] == 1 and cm.sum() == 1
