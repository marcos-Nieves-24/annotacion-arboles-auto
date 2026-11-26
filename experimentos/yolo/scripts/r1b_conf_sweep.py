"""Confidence sweep for R1b box prompts on clean gold frames.

For each conf in {0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8}:
  - match predicted boxes to gold polygons with one-to-one IoU >= 0.5
  - report instance P/R/F1, count MAE, small-box count (<882 Villas / <410 Magdala)
Read-only. Output: data/fase4/conf_sweep.json
"""

import json
import os

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = "/home/biohacker24/annotacion-arboles-auto"
OUT = f"{ROOT}/data/fase4/conf_sweep.json"
WEIGHTS = f"{ROOT}/data/kaggle-out-ronda1b/runs1b/r1/weights/best.pt"
CONFS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
IOU_NMS = 0.35
MIN_AREA = {"villas": 882, "magdala": 410}


def src_of(name):
    for d in (f"{ROOT}/data/gold_v1/images/{name}",
              f"{ROOT}/data/r1/images/villas/{name}",
              f"{ROOT}/data/r1/images/magdala/{name}"):
        if os.path.exists(d):
            return d
    return None


def rast(pts):
    m = np.zeros((512, 512), np.uint8)
    cv2.fillPoly(m, [np.array(pts, np.float32).reshape(-1, 2).round().astype(np.int32)], 1)
    return m


def match(gt_masks, pred_boxes, iou_thr=0.5):
    """One-to-one greedy matching by IoU between gold mask and box raster."""
    pm = []
    for x0, y0, x1, y1 in pred_boxes:
        m = np.zeros((512, 512), np.uint8)
        cv2.rectangle(m, (int(x0), int(y0)), (int(x1), int(y1)), 1, -1)
        pm.append(m)
    pairs = []
    for gi, g in enumerate(gt_masks):
        for pi, p in enumerate(pm):
            inter = np.logical_and(g, p).sum()
            union = np.logical_or(g, p).sum()
            if union and inter / union >= iou_thr:
                pairs.append((inter / union, gi, pi))
    pairs.sort(reverse=True)
    ug, up, tp = set(), set(), 0
    for _, gi, pi in pairs:
        if gi not in ug and pi not in up:
            ug.add(gi); up.add(pi); tp += 1
    return tp


def main():
    coco = json.load(open(f"{ROOT}/data/gold_v1/coco.json"))
    byname = {i["file_name"]: i["id"] for i in coco["images"]}
    gold = {}
    for a in coco["annotations"]:
        if not a["iscrowd"]:
            gold.setdefault(a["image_id"], []).append(a["segmentation"][0])

    items = []
    for farm, sfile in (("villas", "villas_sample30.json"),
                        ("magdala", "magdala_cands30.json")):
        for e in json.load(open(f"{ROOT}/data/gold_v1/{sfile}")):
            name = e["file"]
            if byname.get(name) not in gold:
                continue
            p = src_of(name)
            if p:
                items.append((farm, name, gold[byname[name]], p))

    model = YOLO(WEIGHTS)
    # one inference pass per conf over all items
    results = {}
    for conf in CONFS:
        tp = fp = fn = 0
        small = 0
        errs = []
        for farm, name, polys, path in items:
            r = model.predict(path, imgsz=640, conf=conf, iou=IOU_NMS, verbose=False)[0]
            boxes = r.boxes.xyxy.tolist() if r.boxes is not None else []
            confs = r.boxes.conf.tolist() if r.boxes is not None and len(r.boxes) else []
            # small-box count using box area
            for (x0, y0, x1, y1) in boxes:
                if (x1 - x0) * (y1 - y0) < MIN_AREA[farm] * 0.5:
                    small += 1
            gtm = [rast(p) for p in polys]
            t = match(gtm, boxes)
            tp += t; fp += len(boxes) - t; fn += len(polys) - t
            errs.append(abs(len(boxes) - len(polys)))
        P = tp / (tp + fp) if tp + fp else 0
        R = tp / (tp + fn) if tp + fn else 0
        F1 = 2 * P * R / (P + R) if P + R else 0
        results[str(conf)] = {"tp": tp, "fp": fp, "fn": fn,
                              "precision": round(P, 4), "recall": round(R, 4),
                              "f1": round(F1, 4),
                              "count_MAE": round(float(np.mean(errs)), 2),
                              "small_boxes": small, "n_frames": len(items)}
        print(conf, results[str(conf)], flush=True)
    json.dump(results, open(OUT, "w"), indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
