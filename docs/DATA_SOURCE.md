# Data source

| Field | Value |
|---|---|
| Dataset | *Face Mask Detection*, uploaded by andrewmvd |
| Page | https://www.kaggle.com/datasets/andrewmvd/face-mask-detection |
| Licence | **CC0: Public Domain**, as returned by the Kaggle API (`licenseName`) on 2026-11-06 |
| Last updated on Kaggle | 2020-05-22 |
| Contents | 853 PNG images, 853 Pascal-VOC XML annotation files, 3 classes |
| Archive | 416,968,083 bytes, SHA-256 `cabae358a60492a1b96da0510c31475414b64100480ae52cadbe00753bbef904` |
| Download | `https://www.kaggle.com/api/v1/datasets/download/andrewmvd/face-mask-detection` (worked without a Kaggle login on 2026-11-06) |
| Stored in this repository | **No.** Images of real people are not redistributed |

## Caveats

* The CC0 label is the uploader's declaration. The photographs appear to be collected from the
  web; the uploader's licence may not cover the original photographers' rights. This project uses
  the images for evaluation only and does not republish them.
* The Kaggle page gives no collection protocol, annotator information or inter-annotator check.
* Class names are kept exactly as in the XML files (`mask_weared_incorrect` included) because the
  recorded model's class ids depend on them.
* The original notebook took its data from the same Kaggle dataset; the local copy it used
  (`D:\I4-AMS\...`) no longer exists, so the dataset was downloaded again. The rebuilt validation
  split reproduces the recorded validation metrics to within 0.008, which indicates the files are the
  same.

## Where the data lived for the recorded run

To protect the small D: drive, the archive was downloaded to `C:\Users\<user>\portfolio-data\p09-face-mask\`
(`download/` zip, `raw/` extracted, `work/` generated YOLO dataset as hard links to `raw/`). The scripts
read these locations from `FACEMASK_DATA_DIR` and `FACEMASK_WORK_DIR`; nothing machine-specific is
in the code.
