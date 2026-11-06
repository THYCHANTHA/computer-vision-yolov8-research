# Next steps (not done)

Ordered by value. None of these has been started; no result below exists yet.

1. **Retrain with a leakage-free split** (needs the owner's approval: the original run took about
   6 hours on CPU). Group exact and near duplicates before splitting, keep the class balance per
   split (the rare class has only 123 faces), and keep the de-duplicated test set fixed so old and
   new models are compared on the same images. Use `to_yolo_lines(clip=True)` for new labels.
2. **Repeat training with several seeds** to show run-to-run variation; one run cannot show whether
   a difference between models is real.
3. **Rare-class handling**: compare class-weighted loss or oversampling of images containing
   "incorrect" faces against the baseline, judged on recall of that class with bootstrap intervals.
4. **Larger model comparison** (YOLOv8s) on the same split, reporting mAP and CPU latency together.
5. **Inference module and app**: a small `predict.py`/Streamlit page using the project's settings,
   replacing the Flask app (which has a Jinja expression inside a static JS file and a missing
   favicon; see the original repository's README).
6. ~~**Licensing decision for publication**~~ Done 2026-11-06: the owner approved MIT for this
   repository's code with an AGPL-3.0 note for Ultralytics and the weights (README, "Licence note").
7. **CI**: GitHub Actions cannot run while the account billing lock remains (see the portfolio
   trackers). The unit tests need no dataset and would run in CI; the data tests skip.
