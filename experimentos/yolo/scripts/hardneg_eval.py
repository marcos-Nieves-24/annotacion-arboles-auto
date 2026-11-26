"""Hard-negative verifier test: gold crowns vs QUARANTINED fragments
(same blob-like shapes the model actually produces). Reuses feature
function by import. Reports CV + separation. CPU local."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, "/home/biohacker24/annotacion-arboles-auto/scripts")
from features_profile import features, rast, load_img, FEAT_NAMES

ROOT = "/home/biohacker24/annotacion-arboles-auto"


def main():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    coco = json.load(open(f"{ROOT}/data/gold_v1/coco.json"))
    byimg = {i["id"]: i["file_name"] for i in coco["images"]}
    # map (farm, frame) -> file for quarantine lookup
    farm_frame_file = {}
    shapes = {}
    for farm in ["villas", "magdala"]:
        meta = json.load(open(f"{ROOT}/data/raw/{farm}/meta.json"))
        names = [f["name"] for f in meta["frames"]]
        ann = json.load(open(f"{ROOT}/data/raw/{farm}/annotations.json"))
        for s in ann.get("shapes", []):
            shapes[(farm, s["id"])] = (s.get("frame"), s.get("points"))
        for idx, n in enumerate(names):
            farm_frame_file[(farm, idx)] = n
    Xg, Xq = [], []
    for a in coco["annotations"]:
        if a["iscrowd"]:
            continue
        img = load_img(byimg[a["image_id"]])
        if img is None:
            continue
        f = features(img, rast(a["segmentation"][0]))
        if f:
            Xg.append(f)
    q = json.load(open(f"{ROOT}/data/gold_v1/quarantine.json"))
    for e in q:
        fr_pts = shapes.get((e["farm"], e["shape_id"]))
        if not fr_pts:
            continue
        fr, pts = fr_pts
        fn = farm_frame_file.get((e["farm"], fr))
        if not fn:
            continue
        img = load_img(fn)
        if img is None:
            continue
        f = features(img, rast(pts))
        if f:
            Xq.append(f)
    print(f"gold={len(Xg)} quar={len(Xq)}", flush=True)
    X = np.array(Xg + Xq, dtype=np.float32)
    y = np.array([1] * len(Xg) + [0] * len(Xq))
    clf = RandomForestClassifier(n_estimators=300, random_state=7, n_jobs=4)
    cv = cross_val_score(clf, X, y, cv=3)
    print("HARD-neg CV accuracy:", cv.round(3).tolist(), flush=True)
    clf.fit(X, y)
    imp = sorted(zip(FEAT_NAMES, clf.feature_importances_),
                 key=lambda t: -t[1])
    print("importances:", [(k, round(float(v), 3)) for k, v in imp],
          flush=True)
    import pickle
    pickle.dump(clf, open(f"{ROOT}/data/profile/verifier_hard.pkl", "wb"))
    # separation per top feature: gold p5-p95 vs quar median
    Xt = X[y == 1]
    X0 = X[y == 0]
    for i, k in enumerate(FEAT_NAMES):
        print(f"{k}: tree p5={np.percentile(Xt[:, i], 5):.3f} "
              f"med={np.median(Xt[:, i]):.3f} | quar med={np.median(X0[:, i]):.3f}")


if __name__ == "__main__":
    main()
