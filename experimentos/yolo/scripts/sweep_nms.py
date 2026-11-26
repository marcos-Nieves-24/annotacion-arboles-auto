"""Fase 2b: sweep NMS IoU (0.35-0.6) at conf 0.5 + GSD area filter.
Villas sample30 vs gold (non-iscrowd), Yulima40 vs auto. Mask-level greedy
matching. Picks operating point by mask F1. CPU local."""

import json
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

ROOT = "/home/biohacker24/annotacion-arboles-auto"
AREA_MIN = {"villas": 882, "yulima": 767}


def rast(pts):
    m = np.zeros((512, 512), np.uint8)
    cv2.fillPoly(m, [np.array(pts, np.float32).reshape(-1, 2).round()
                     .astype(np.int32)], 1)
    return m


def miou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def load_gt():
    coco = json.load(open(f"{ROOT}/data/gold_v1/coco.json"))
    gv = {}
    for a in coco["annotations"]:
        if not a["iscrowd"]:
            ys, xs = np.where(rast(a["segmentation"][0]))
            gv.setdefault(a["image_id"], []).append(
                {"mask": rast(a["segmentation"][0]),
                 "bbox": [float(xs.min()), float(ys.min()),
                          float(xs.max() - xs.min() + 1),
                          float(ys.max() - ys.min() + 1)]})
    byname = {i["file_name"]: i["id"] for i in coco["images"]}
    farm_of = {i["file_name"]: i["farm"] for i in coco["images"]}
    ann = json.load(open(f"{ROOT}/data/raw/yulima/annotations.json"))
    ya = {}
    for s in ann.get("shapes", []):
        if s.get("type") == "polygon":
            m = rast(s["points"])
            ys, xs = np.where(m)
            ya.setdefault(s["frame"], []).append(
                {"mask": m,
                 "bbox": [float(xs.min()), float(ys.min()),
                          float(xs.max() - xs.min() + 1),
                          float(ys.max() - ys.min() + 1)]})
    ymeta = json.load(open(f"{ROOT}/data/raw/yulima/meta.json"))
    ynames = [f["name"] for f in ymeta["frames"]]
    return gv, byname, farm_of, ya, ynames


def eval_set(model, files, gt_list, iou_thr, conf=0.5, amin=0):
    res = model.predict(files, imgsz=640, conf=conf, iou=iou_thr,
                        verbose=False)
    P, R, I, CN = [], [], [], []
    for r, g in zip(res, gt_list):
        pm = []
        if r.masks is not None and len(r.masks):
            md = r.masks.data.float()
            if md.shape[1] != 512 or md.shape[2] != 512:
                md = F.interpolate(md.unsqueeze(0), size=(512, 512),
                                   mode="nearest").squeeze(0)
            for k in range(md.shape[0]):
                m = (md[k] > 0.5).cpu().numpy().astype("uint8")
                if m.sum() >= amin:
                    pm.append(m)
        used, ious, tp = set(), [], 0
        for p in pm:
            best, bj = 0.0, -1
            for j, a in enumerate(g):
                if j in used:
                    continue
                v = miou(p, a["mask"])
                if v > best:
                    best, bj = v, j
            if best >= 0.5:
                tp += 1
                used.add(bj)
                ious.append(best)
        P.append(tp / max(len(pm), 1))
        R.append(tp / max(len(g), 1))
        I.extend(ious)
        CN.append(abs(len(pm) - len(g)))
    mp, mr = float(np.mean(P)), float(np.mean(R))
    return {"P": round(mp, 3), "R": round(mr, 3),
            "F1": round(2 * mp * mr / max(mp + mr, 1e-6), 3),
            "med_iou": round(float(np.median(I)), 3) if I else 0.0,
            "MAE": round(float(np.mean(CN)), 2)}


def main():
    gv, byname, farm_of, ya, ynames = load_gt()
    vs = json.load(open(f"{ROOT}/data/gold_v1/villas_sample30.json"))
    ys = json.load(open(f"{ROOT}/data/gold_v1/yulima_qc40.json"))
    model = YOLO(f"{ROOT}/data/kaggle-out-fase2/runs/splitB/weights/best.pt")
    vfiles = [f"{ROOT}/data/gold_v1/images/{s['file']}" for s in vs]
    vgt = [[{"mask": a["mask"], "bbox": a["bbox"]}
            for a in gv.get(byname[s["file"]], [])] for s in vs]
    yfiles = [f"{ROOT}/data/gold_v1/yulima_qc/{s['file']}" for s in ys]
    ymeta_frames = {s["file"]: s["frame"] for s in ys}
    ygt = [ya.get(ymeta_frames[s["file"]], []) for s in ys]
    table = {}
    for iouv in [0.35, 0.45, 0.5, 0.6]:
        rv = eval_set(model, vfiles, vgt, iouv, amin=AREA_MIN["villas"])
        ry = eval_set(model, yfiles, ygt, iouv, amin=AREA_MIN["yulima"])
        table[str(iouv)] = {"villas": rv, "yulima": ry}
        print(iouv, "V:", rv, "Y:", ry, flush=True)
    json.dump(table, open(f"{ROOT}/data/gold_v1/nms_sweep.json", "w"), indent=1)


if __name__ == "__main__":
    main()
