# Face-Mask Detection with YOLOv8: Audit and Held-out Re-evaluation

**Portfolio Project 09 (Computer Vision).** Status: **in progress** (audit and held-out re-evaluation
complete; retraining on a leakage-free split not yet done; see [docs/NEXT_STEPS.md](docs/NEXT_STEPS.md)).

This project takes the owner's earlier course project
([THYCHANTHA/Face-mask-detection](https://github.com/THYCHANTHA/Face-mask-detection): a YOLOv8n model
trained in a notebook and served by a Flask app) and asks a narrower research question:

> **How well does the recorded model really perform on images it has never seen, and is the original
> evaluation trustworthy?**

No model was retrained. The recorded weights (`weights/best.pt`, SHA-256 in
[reports/provenance.json](reports/provenance.json)) were re-scored after rebuilding the original data
split from the notebook's code.

## Main findings

1. **The original split was rebuilt exactly.** Scoring the recorded weights on the rebuilt validation
   split reproduces the training log's best epoch (epoch 38) to within 0.008 on precision, recall,
   mAP50 and mAP50-95. The training split scores clearly higher (mAP50 0.944), as expected for images
   the model was trained on.
2. **The original test split was never scored** (the notebook only displayed predictions), and the
   previously quoted "85% accuracy" has no source in the notebook or training log.
3. **Train/test leakage.** The dataset contains duplicate images. 4 of the 86 test images
   (`maksssksksss293/428/545/656`) are exact or near copies of training or validation images (checked
   by file hash and a 64-bit difference hash, then confirmed by eye). Removing them lowers test mAP50
   from 0.833 to 0.781.
4. **The rare class is weak and uncertain.** At the web app's threshold (conf 0.5), the model finds 4
   of 10 "mask worn incorrectly" faces in the de-duplicated test set (recall 0.40, 95% bootstrap
   interval 0.09 to 0.80); 4 of the other 6 are labelled "mask". Ten instances are far too few for a
   reliable estimate.

## Results (recorded weights, CPU, Ultralytics 8.4.203)

Standard detection metrics (`model.val`, conf 0.001, NMS IoU 0.7; precision/recall at the
confidence that maximises F1). Source: [reports/evaluation.json](reports/evaluation.json).

| Split | Images | Faces | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| Train (sanity check only) | 682 | 3,324 | 0.951 | 0.882 | 0.944 | 0.682 |
| Validation (used to choose `best.pt`) | 85 | 320 | 0.890 | 0.679 | 0.779 | 0.565 |
| Training log, best epoch 38 (recorded) | 85 | 320 | 0.887 | 0.678 | 0.785 | 0.572 |
| Test, original | 86 | 428 | 0.873 | 0.752 | 0.833 | 0.583 |
| **Test, de-duplicated (headline)** | **82** | **370** | **0.814** | **0.750** | **0.781** | **0.558** |

Per class, de-duplicated test (mAP50): with mask 0.957, without mask 0.817, mask worn incorrectly
0.568 (10 faces).

**At the web app's settings** (conf 0.5, NMS IoU 0.5, match IoU 0.5), de-duplicated test, 82 images.
Rows are the true class, columns the predicted class:

| True \ Predicted | without mask | with mask | incorrect | missed |
|---|---|---|---|---|
| without mask (52) | 36 | 1 | 0 | 15 |
| with mask (308) | 1 | 288 | 2 | 17 |
| incorrect (10) | 0 | 4 | 4 | 2 |
| false alarm | 3 | 15 | 0 | - |

| Class | Precision (95% CI) | Recall (95% CI) |
|---|---|---|
| without mask | 0.90 (0.79 to 0.98) | 0.69 (0.52 to 0.86) |
| with mask | 0.94 (0.90 to 0.96) | 0.94 (0.89 to 0.97) |
| mask worn incorrectly | 0.67 (0.20 to 1.00) | 0.40 (0.09 to 0.80) |

Intervals: 2,000 bootstrap resamples of whole images (faces in one image are not independent).

**Latency.** End-to-end single-image prediction (batch 1, 640 px, CPU, Intel Core i5-11300H laptop):
median 56 to 73 ms across four runs on the same 86 test images (the machine was not otherwise
controlled). The model has 3.0 M parameters and 8.1 GFLOPs.

## What was built

| Part | Content |
|---|---|
| `src/facemask/annotations.py` | Explicit Pascal-VOC parser (replaces the notebook's generic tag flattening), validation (unknown class, degenerate box, box outside image, size and file-name mismatches), YOLO conversion with the notebook's exact formula, optional clipping |
| `src/facemask/splits.py` | The notebook's 80/10/10 split (`train_test_split`, `random_state=1`) in the notebook's file order |
| `src/facemask/duplicates.py` | SHA-256 and difference-hash duplicate search; cross-split leakage report |
| `src/facemask/yolo_dataset.py` | Ultralytics dataset writer; hard-links images (no extra disk), free-space check before copying |
| `src/facemask/matching.py` | IoU matching at a fixed threshold, class-confusion table, image-level bootstrap |
| `src/facemask/evaluation.py` | Ultralytics validation per split, operating-point evaluation, latency |
| `scripts/` | `audit_dataset.py`, `build_yolo_dataset.py`, `evaluate_model.py` |
| `tests/` | 23 pytest tests (synthetic data; real-data checks skip when the dataset is absent) |

## Data audit ([reports/data_audit.json](reports/data_audit.json))

* 853 images, 853 XML files, 4,072 faces: with mask 3,232 (79%), without mask 717 (18%), mask worn
  incorrectly 123 (3%). 1 to 115 faces per image (median 2).
* All XML files parse; every image has an annotation.
* 12 boxes in 11 images end at x = 401 in 400-px-wide images (a 1-pixel overshoot). In YOLO
  centre/width form all values stay at or below 1, so Ultralytics accepts them; left unchanged so
  the recorded weights are scored on the labels they were trained with.
* 5 pairs of byte-identical images and 17 near-duplicate pairs in total; 6 pairs cross a split
  boundary ([reports/near_duplicates_cross_split.csv](reports/near_duplicates_cross_split.csv)).

## Reproduce

The dataset is not stored in this repository ([docs/DATA_SOURCE.md](docs/DATA_SOURCE.md)). Put the
extracted `annotations/` and `images/` folders anywhere and point the scripts at them:

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt
export FACEMASK_DATA_DIR=/path/to/face-mask-detection     # contains annotations/ and images/
export FACEMASK_WORK_DIR=/path/on/the/same/drive/work      # generated YOLO dataset (hard links)
python scripts/audit_dataset.py       # -> reports/data_audit.json, near_duplicates_cross_split.csv
python scripts/build_yolo_dataset.py  # -> $FACEMASK_WORK_DIR/yolo
python scripts/evaluate_model.py      # -> reports/evaluation.json  (about 3 minutes on a laptop CPU)
python -m pytest
```

Unedited outputs of the recorded run: [reports/pytest_output.txt](reports/pytest_output.txt),
[reports/ruff_output.txt](reports/ruff_output.txt),
[reports/evaluate_model_output.txt](reports/evaluate_model_output.txt).

## Limitations

* **No retraining.** All results describe one recorded model (YOLOv8n, 50 epochs). Its training
  data included the 3 test images that duplicate training images, so the original test result is
  inflated; the de-duplicated result is the fair one but is based on only 82 images.
* **Random image-level split**, as in the original notebook. Duplicates were removed from the test
  set only; train and validation still contain duplicate pairs (validation, which chose `best.pt`,
  shares one scene with test image 293, now excluded).
* **Small test set and rare class**: 10 "incorrect" faces. Treat that class's numbers as indicative.
* **Domain**: web photographs collected around 2020 (crowds, mostly East Asian street scenes). Not
  evaluated on Cambodian images, CCTV, night-time or video.
* The Flask app from the original repository is not part of this project; its threshold (conf 0.5,
  IoU 0.5) is used for the operating-point table.
* Single CPU environment; GPU numbers were not measured. Results depend on Ultralytics/torch
  versions (the val reproduction differs from the 2025 training log by up to 0.008).

## Responsible use

The model detects faces and classifies mask wearing only. It does **not** identify people and must
not be used for identification, biometric matching or enforcement against individuals. Test images
are of real people; they are not redistributed here and no example photographs are shown.

## Licence and credits

Code: MIT ([LICENSE](LICENSE)). Dataset: *Face Mask Detection* by andrewmvd on Kaggle, listed as
CC0: Public Domain (see [docs/DATA_SOURCE.md](docs/DATA_SOURCE.md) for caveats). Model:
[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (AGPL-3.0; this project imports it
and does not modify it).

**Licence note:** the MIT licence covers this repository's own code only. Ultralytics is licensed
under AGPL-3.0, and the weights in `weights/best.pt` were produced with it; anyone who builds a
network service on this code and model must also meet the AGPL-3.0 terms (or obtain an Ultralytics
Enterprise licence).
