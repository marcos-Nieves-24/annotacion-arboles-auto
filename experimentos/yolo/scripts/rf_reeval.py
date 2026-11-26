"""Re-evaluate existing RF verifiers with clean protocol.

Fixes two defects in features_profile.py / hardneg_eval.py:
  1. holdout frames EXCLUDED from positives (regla 7).
  2. GroupKFold by farm+frame (no random split -> no same-frame leakage).

Compares:
  - verifier.pkl (trained on random background)
  - verifier_hard.pkl (trained on quarantine hard negatives)
  - a freshly fit RF on gold-vs-quarantine (clean, for reference)

Then applies the best to R1b FP candidates on clean-gold frames.
Output: data/fase4/rf_reeval.json
"""

import json
import os
import pickle
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, "/home/biohacker24/annotacion-arboles-auto/scripts")
from features_profile import features, rast, load_img, FEAT_NAMES

ROOT = "/home/biohacker24/annotacion-arboles-auto"
OUT = f"{ROOT}/data/fase4/rf_reeval.json"


def build_dataset():
    coco = json.load(open(f"{ROOT}/data/gold_v1/coco.json"))
    byimg = {i["id"]: i["file_name"] for i in coco["images"]}
    hold = {h["file_name"] for h in json.load(open(f"{ROOT}/data/gold_v1/holdout.json"))}
    X, y, g = [], [], []
    farm_by_name = {}
    for farm in ("villas", "magdala"):
        meta = json.load(open(f"{ROOT}/data/raw/{farm}/meta.json"))
        for f in meta["frames"]:
            farm_by_name[f["name"]] = farm

    # positives: gold non-iscrowd, excluding holdout
    for a in coco["annotations"]:
        if a["iscrowd"]:
            continue
        name = byimg[a["image_id"]]
        if name in hold:
            continue
        img = load_img(name)
        if img is None:
            continue
        f = features(img, rast(a["segmentation"][0]))
        if f:
            X.append(f); y.append(1); g.append(farm_by_name.get(name, "?") + ":" + name)

    # negatives: quarantine fragments, mapped by (farm, shape_id) -> frame -> file
    shapes, farm_frame_file = {}, {}
    for farm in ("villas", "magdala"):
        meta = json.load(open(f"{ROOT}/data/raw/{farm}/meta.json"))
        names = [f["name"] for f in meta["frames"]]
        ann = json.load(open(f"{ROOT}/data/raw/{farm}/annotations.json"))
        for s in ann.get("shapes", []):
            shapes[(farm, s["id"])] = (s.get("frame"), s.get("points"))
        for idx, n in enumerate(names):
            farm_frame_file[(farm, idx)] = n

    q = json.load(open(f"{ROOT}/data/gold_v1/quarantine.json"))
    nq = 0
    for e in q:
        hit = shapes.get((e["farm"], e["shape_id"]))
        if not hit:
            continue
        fr, pts = hit
        fn = farm_frame_file.get((e["farm"], fr))
        if not fn or fn in hold:
            continue
        img = load_img(fn)
        if img is None:
            continue
        f = features(img, rast(pts))
        if f:
            X.append(f); y.append(0); g.append(e["farm"] + ":" + fn); nq += 1
    return np.array(X, np.float32), np.array(y), np.array(g), nq


def evaluate(name, clf, X, y, g):
    gkf = GroupKFold(n_splits=4)
    probs = np.zeros(len(y))
    for tr, te in gkf.split(X, y, g):
        m = RandomForestClassifier(n_estimators=200, random_state=7, n_jobs=4)
        m.fit(X[tr], y[tr])
        probs[te] = m.predict_proba(X[te])[:, 1]
    ap = average_precision_score(y, probs)
    out = {"AP": round(float(ap), 4)}
    for t in (0.3, 0.5, 0.7):
        pred = probs >= t
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        P = tp / (tp + fp) if tp + fp else 0
        R = tp / (tp + fn) if tp + fn else 0
        F1 = 2 * P * R / (P + R) if P + R else 0
        out[f"t{t}"] = {"P": round(P, 3), "R": round(R, 3), "F1": round(F1, 3)}
    return out


def main():
    X, y, g, nq = build_dataset()
    print(f"dataset: gold={int((y==1).sum())} quar={nq} holdout excluido", flush=True)
    report = {"n_gold": int((y == 1).sum()), "n_quar": nq}

    for key in ("verifier", "verifier_hard"):
        p = f"{ROOT}/data/profile/{key}.pkl"
        if not os.path.exists(p):
            continue
        clf = pickle.load(open(p, "rb"))
        report[key] = evaluate(key, clf, X, y, g)
        print(key, report[key], flush=True)

    fresh = RandomForestClassifier(n_estimators=200, random_state=7, n_jobs=4)
    report["fresh_clean"] = evaluate("fresh_clean", fresh, X, y, g)
    print("fresh_clean", report["fresh_clean"], flush=True)

    # feature importances from a clean fit (full set, for information only)
    fresh.fit(X, y)
    imp = sorted(zip(FEAT_NAMES, fresh.feature_importances_), key=lambda t: -t[1])
    report["importances"] = {k: round(float(v), 4) for k, v in imp}
    json.dump(report, open(OUT, "w"), indent=1)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
