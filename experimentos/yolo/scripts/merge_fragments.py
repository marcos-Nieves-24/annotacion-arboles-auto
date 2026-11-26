"""Treetop-constrained merge: every final instance must contain exactly one
treetop (ExG local maxima). Fragments without a treetop are absorbed by the
neighbor they touch most; pieces with their own treetop survive.

Measures P/R/F1 vs gold before/after on Villas sample30 + Yulima40.
CPU local. No CVAT access.
"""

import json
import os

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = "/home/biohacker24/annotacion-arboles-auto"


def rast(pts):
    m = np.zeros((512, 512), np.uint8)
    cv2.fillPoly(m, [np.array(pts, np.float32).reshape(-1, 2).round()
                     .astype(np.int32)], 1)
    return m


def miou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def treetops(img, min_dist=18):
    """ExG peaks as treetop markers."""
    f = img.astype(np.float32)
    exg = 2 * f[:, :, 1] - f[:, :, 0] - f[:, :, 2]
    exg = cv2.GaussianBlur(exg, (0, 0), 3)
    dil = cv2.dilate(exg, np.ones((min_dist, min_dist), np.uint8))
    peaks = (exg == dil) & (exg > np.percentile(exg, 85))
    n, lab = cv2.connectedComponents(peaks.astype(np.uint8))
    return [(int((lab == i).sum() and np.where(lab == i)[1].mean()),
             int(np.where(lab == i)[0].mean())) for i in range(1, n)]


def merge_fragments(pred_masks, peaks):
    """Absorb masks with no treetop into most-touched neighbor."""
    has = []
    for m in pred_masks:
        inside = [p for p in peaks
                  if 0 <= p[1] < 512 and 0 <= p[0] < 512 and m[p[1], p[0]]]
        has.append(len(inside) > 0)
    keep = [m.copy() for m, h in zip(pred_masks, has) if h]
    orphans = [m for m, h in zip(pred_masks, has) if not h]
    for o in orphans:
        d = cv2.dilate(o, np.ones((5, 5), np.uint8))
        best, bj = 0, -1
        for j, k in enumerate(keep):
            v = np.logical_and(d, k).sum()
            if v > best:
                best, bj = v, j
        if bj >= 0 and best > 0:
            keep[bj] = np.logical_or(keep[bj], o).astype(np.uint8)
        # else: background fragment -> dropped
    return keep


def evaluate(files, gt_list, model, conf=0.5):
    res = model.predict(files, imgsz=640, conf=conf, iou=0.35, verbose=False)
    P0 = R0 = P1 = R1 = 0
    n = len(files)
    for r, g, f in zip(res, gt_list, files):
        pm = []
        if r.masks is not None and len(r.masks):
            import torch
            import torch.nn.functional as F

            md = r.masks.data.float()
            if md.shape[1] != 512 or md.shape[2] != 512:
                md = F.interpolate(md.unsqueeze(0), size=(512, 512),
                                   mode="nearest").squeeze(0)
            pm = [(md[k] > 0.5).cpu().numpy().astype("uint8")
                  for k in range(md.shape[0])]
        img = cv2.imread(f)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        merged = merge_fragments(pm, treetops(rgb))

        def score(preds):
            used, tp = set(), 0
            for p in preds:
                best, bj = 0.0, -1
                for j, a in enumerate(g):
                    if j in used:
                        continue
                    v = miou(p, a)
                    if v > best:
                        best, bj = v, j
                if best >= 0.5:
                    tp += 1
                    used.add(bj)
            P = tp / max(len(preds), 1)
            R = tp / max(len(g), 1)
            return P, R

        p0, r0 = score(pm)
        p1, r1 = score(merged)
        P0 += p0
        R0 += r0
        P1 += p1
        R1 += r1
    return {"base": (round(P0 / n, 3), round(R0 / n, 3)),
            "merged": (round(P1 / n, 3), round(R1 / n, 3))}


def main():
    coco = json.load(open(f"{ROOT}/data/gold_v1/coco.json"))
    gv = {}
    for a in coco["annotations"]:
        if not a["iscrowd"]:
            gv.setdefault(a["image_id"], []).append(rast(a["segmentation"][0]))
    byname = {i["file_name"]: i["id"] for i in coco["images"]}
    vs = json.load(open(f"{ROOT}/data/gold_v1/villas_sample30.json"))
    vfiles = [f"{ROOT}/data/gold_v1/images/{s['file']}" for s in vs]
    vgt = [gv.get(byname[s["file"]], []) for s in vs]
    model = YOLO(f"{ROOT}/data/kaggle-out-fase2/runs/splitB/weights/best.pt")
    out = {"villas30": evaluate(vfiles, vgt, model)}
    print(json.dumps(out, indent=1), flush=True)
    json.dump(out, open(f"{ROOT}/data/gold_v1/merge_report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
