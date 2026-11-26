"""postprocess_v1 (frozen): r1 predict -> GSD area filter -> color verifier
(mid-size only) -> count-guided merge -> labels + report.

Usage: postprocess_v1.py <images_dir> <out_dir> <farm>
Farm GSD bands + median crowns baked in. CPU local. No CVAT.
"""

import json
import os
import pickle
import sys

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = "/home/biohacker24/annotacion-arboles-auto"
AREA_BAND = {"villas": (882, 39224), "magdala": (410, 18240),
             "yulima": (767, 34096)}
MEAN_CROWN = {"villas": 9798, "magdala": 6636, "yulima": 4057}
# mean-fraction dirt floor per farm (measured 2026-11-11: villas k=0.10
# +2.9pts, yulima k=0.15 +1.6pts, magdala ~0 -> keep GSD floor only)
MEAN_FRAC = {"villas": 0.10, "magdala": 0.0, "yulima": 0.15}


def area_min(farm):
    gmin, _ = AREA_BAND[farm]
    return max(gmin, MEAN_FRAC[farm] * MEAN_CROWN[farm])
MEDIAN_CROWN = {"villas": 9798, "magdala": 6636, "yulima": 4057}
sys.path.insert(0, f"{ROOT}/scripts")
from features_profile import features

VERIFIER = pickle.load(open(f"{ROOT}/data/profile/verifier_hard.pkl", "rb"))
MODEL = YOLO(f"{ROOT}/data/kaggle-out-ronda1b/runs1b/r1/weights/best.pt")


def correct_image(path, farm, conf=0.5, nms=0.35):
    img_bgr = cv2.imread(path)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    r = MODEL.predict(path, imgsz=640, conf=conf, iou=nms, verbose=False)[0]
    masks = []
    if r.masks is not None and len(r.masks):
        import torch
        import torch.nn.functional as F

        md = r.masks.data.float()
        if md.shape[1] != 512 or md.shape[2] != 512:
            md = F.interpolate(md.unsqueeze(0), size=(512, 512),
                               mode="nearest").squeeze(0)
        masks = [(md[k] > 0.5).cpu().numpy().astype("uint8")
                 for k in range(md.shape[0])]
    amin, amax = area_min(farm), AREA_BAND[farm][1]
    kept = []
    for m in masks:
        a = int(m.sum())
        if not (amin <= a <= amax):
            continue
        if a < MEDIAN_CROWN[farm]:
            f = features(img, m)
            if f is None or VERIFIER.predict([f])[0] == 0:
                continue
        kept.append(m)
    union = np.logical_or.reduce(kept).astype(np.uint8) if kept \
        else np.zeros((512, 512), np.uint8)
    expected = max(1, int(round(union.sum() / MEDIAN_CROWN[farm])))
    cur = [m.astype(bool) for m in kept]
    guard = 0
    while len(cur) > int(expected * 1.1) + 1 and guard < 200:
        guard += 1
        best, bi, bj = 0, -1, -1
        for i in range(len(cur)):
            di = cv2.dilate(cur[i].astype(np.uint8),
                            np.ones((3, 3), np.uint8))
            for j in range(i + 1, len(cur)):
                sh = int(np.logical_and(di, cur[j]).sum())
                if sh > best and (cur[i] | cur[j]).sum() <= amax:
                    best, bi, bj = sh, i, j
        if bi < 0 or best < 8:
            break
        cur = [m for k, m in enumerate(cur) if k not in (bi, bj)] + \
              [(cur[bi] | cur[bj]).astype(np.uint8)]
    return [m.astype(np.uint8) for m in cur], expected, len(masks)


if __name__ == "__main__":
    print("import ok: postprocess_v1 ready")
