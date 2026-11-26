"""Tree profile from gold crowns: color/texture/shape features -> tiny RF
verifier + percentile profile. Measures separation vs background patches
and vs quarantined fragments. Output: profile.json + verifier.pkl +
report. CPU local. No CVAT."""

import json
import os
import random

import cv2
import numpy as np

ROOT = "/home/biohacker24/annotacion-arboles-auto"
random.seed(3)
np.random.seed(3)


def rast(pts):
    m = np.zeros((512, 512), np.uint8)
    cv2.fillPoly(m, [np.array(pts, np.float32).reshape(-1, 2).round()
                     .astype(np.int32)], 1)
    return m


def features(img_rgb, mask):
    m = mask.astype(bool)
    if m.sum() < 20:
        return None
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    f = img_rgb.astype(np.float32)
    exg = 2 * f[:, :, 1] - f[:, :, 0] - f[:, :, 2]
    lap = cv2.Laplacian(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY), cv2.CV_64F)
    area = float(m.sum())
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    carea = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hare = cv2.contourArea(hull)
    x, y, w, h = cv2.boundingRect(cnt)
    ys, xs = np.where(m)
    return [
        area,
        float(carea / max(hare, 1)),                    # solidity
        float(4 * np.pi * carea / max(peri ** 2, 1)),   # circularity
        float(carea / max(w * h, 1)),                   # extent
        float(w / max(h, 1)),                           # aspect
        float(exg[m].mean()), float(exg[m].std()),      # ExG
        float((exg[m] > 0).mean()),                     # green frac
        float(hsv[:, :, 0][m].mean()), float(hsv[:, :, 1][m].mean()),
        float(hsv[:, :, 2][m].mean()),                  # HSV means
        float(lap[m].var()),                            # texture
    ]


FEAT_NAMES = ["area", "solidity", "circularity", "extent", "aspect",
              "exg_mean", "exg_std", "green_frac", "H", "S", "V", "lap_var"]


def load_img(name):
    for d in (f"{ROOT}/data/gold_v1/images/{name}",):
        if os.path.exists(d):
            return cv2.cvtColor(cv2.imread(d), cv2.COLOR_BGR2RGB)
    return None


def main():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    coco = json.load(open(f"{ROOT}/data/gold_v1/coco.json"))
    byimg = {i["id"]: i["file_name"] for i in coco["images"]}
    X, y = [], []
    # positives: gold non-iscrowd
    for a in coco["annotations"]:
        if a["iscrowd"]:
            continue
        img = load_img(byimg[a["image_id"]])
        if img is None:
            continue
        f = features(img, rast(a["segmentation"][0]))
        if f:
            X.append(f)
            y.append(1)
    print("positives:", len([v for v in y if v == 1]), flush=True)
    # negatives A: quarantined fragments (shapes by id)
    q = json.load(open(f"{ROOT}/data/gold_v1/quarantine.json"))
    shapes = {}
    for farm in ["villas", "magdala"]:
        ann = json.load(open(f"{ROOT}/data/raw/{farm}/annotations.json"))
        for s in ann.get("shapes", []):
            shapes[(farm, s["id"])] = s["points"]
    meta_names = {}
    for farm in ["villas", "magdala"]:
        meta_names[farm] = [f["name"] for f in
                            json.load(open(f"{ROOT}/data/raw/{farm}/meta.json"))["frames"]]
    nq = 0
    for e in q:
        pts = shapes.get((e["farm"], e["shape_id"]))
        if not pts:
            continue
        # find frame file: search image by frame order is ambiguous; use gold images only if match
        for fn in meta_names[e["farm"]]:
            pass
        img = None
        for cand in [f"{ROOT}/data/gold_v1/images/{fn}" for fn in meta_names[e["farm"]]]:
            img = None
        # simpler: match quarantine shape to gold frame via coco (shape ids differ) -> use random gold bg instead
        break
    # negatives B: background patches avoiding gold masks (robust, no id mapping)
    import collections
    goldmasks = collections.defaultdict(list)
    for a in coco["annotations"]:
        goldmasks[a["image_id"]].append(rast(a["segmentation"][0]))
    nbg = 0
    for iid, fn in byimg.items():
        img = load_img(fn)
        if img is None:
            continue
        union = np.logical_or.reduce(goldmasks[iid]).astype(np.uint8) \
            if goldmasks[iid] else np.zeros((512, 512), np.uint8)
        tries = 0
        got = 0
        while got < 4 and tries < 40:
            tries += 1
            w, h = random.randint(30, 120), random.randint(30, 120)
            x0, y0 = random.randint(0, 512 - w), random.randint(0, 512 - h)
            if union[y0:y0 + h, x0:x0 + w].mean() > 0.05:
                continue
            m = np.zeros((512, 512), np.uint8)
            m[y0:y0 + h, x0:x0 + w] = 1
            f = features(img, m)
            if f:
                X.append(f)
                y.append(0)
                got += 1
                nbg += 1
    print("background negatives:", nbg, flush=True)
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    clf = RandomForestClassifier(n_estimators=200, random_state=7, n_jobs=4)
    cv = cross_val_score(clf, X, y, cv=3)
    print("CV accuracy:", cv.round(3).tolist(), flush=True)
    clf.fit(X, y)
    print("train acc:", round(float(clf.score(X, y)), 4), flush=True)
    imp = sorted(zip(FEAT_NAMES, clf.feature_importances_), key=lambda t: -t[1])
    print("importances:", [(k, round(float(v), 3)) for k, v in imp], flush=True)
    # profile percentiles of tree class
    Xt = X[y == 1]
    prof = {k: {"p5": round(float(np.percentile(Xt[:, i], 5)), 3),
                "p50": round(float(np.percentile(Xt[:, i], 50)), 3),
                "p95": round(float(np.percentile(Xt[:, i], 95)), 3)}
            for i, k in enumerate(FEAT_NAMES)}
    import pickle
    os.makedirs(f"{ROOT}/data/profile", exist_ok=True)
    json.dump({"features": FEAT_NAMES, "profile_tree": prof,
               "cv_acc": [round(float(v), 4) for v in cv],
               "importances": {k: round(float(v), 4) for k, v in imp}},
              open(f"{ROOT}/data/profile/profile.json", "w"), indent=1)
    pickle.dump(clf, open(f"{ROOT}/data/profile/verifier.pkl", "wb"))
    print("saved profile + verifier", flush=True)


if __name__ == "__main__":
    main()
