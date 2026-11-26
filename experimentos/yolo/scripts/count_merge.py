"""Count-guided merge: expected count per frame = canopy_union / median crown.
While n_pred > 1.1*expected, merge the adjacent pair with longest shared
border whose combined area stays within the crown band. Fragments below
GSD min-area are dropped first. Evaluates P/R/F1 vs gold (Villas30) and
vs auto (Yulima40). CPU local. No CVAT access."""

import json
import os

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = "/home/biohacker24/annotacion-arboles-auto"
MEDIAN_CROWN = {"villas": 9798, "magdala": 6636, "yulima": 4057}
AREA_BAND = {"villas": (882, 39224), "magdala": (410, 18240),
             "yulima": (767, 34096)}


def rast(pts):
    m = np.zeros((512, 512), np.uint8)
    cv2.fillPoly(m, [np.array(pts, np.float32).reshape(-1, 2).round()
                     .astype(np.int32)], 1)
    return m


def miou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def shared_border(a, b):
    da = cv2.dilate(a, np.ones((3, 3), np.uint8))
    return int(np.logical_and(da, b).sum())


def count_guided(masks, farm):
    masks = [m for m in masks
             if AREA_BAND[farm][0] <= m.sum() <= AREA_BAND[farm][1]]
    union = np.logical_or.reduce(masks).astype(np.uint8) if masks \
        else np.zeros((512, 512), np.uint8)
    expected = max(1, int(round(union.sum() / MEDIAN_CROWN[farm])))
    cur = [m.astype(bool) for m in masks]
    guard = 0
    while len(cur) > int(expected * 1.1) + 1 and guard < 200:
        guard += 1
        best, bi, bj = 0, -1, -1
        for i in range(len(cur)):
            di = cv2.dilate(cur[i].astype(np.uint8),
                            np.ones((3, 3), np.uint8))
            for j in range(i + 1, len(cur)):
                shared = int(np.logical_and(di, cur[j]).sum())
                comb = (cur[i] | cur[j]).sum()
                if shared > best and comb <= AREA_BAND[farm][1]:
                    best, bi, bj = shared, i, j
        if bi < 0 or best < 8:
            break
        merged = (cur[bi] | cur[bj]).astype(np.uint8)
        cur = [m for k, m in enumerate(cur) if k not in (bi, bj)] + [merged]
    return [m.astype(np.uint8) for m in cur], expected


def run_set(model, files, gt_list, farm):
    res = model.predict(files, imgsz=640, conf=0.5, iou=0.35, verbose=False)
    P0 = R0 = P1 = R1 = 0
    n = len(files)
    for r, g in zip(res, gt_list):
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
        merged, exp = count_guided(pm, farm)

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
            return tp / max(len(preds), 1), tp / max(len(g), 1)

        p0, r0 = score(pm)
        p1, r1 = score(merged)
        P0 += p0
        R0 += r0
        P1 += p1
        R1 += r1
    f = lambda p, r: round(2 * p * r / max(p + r, 1e-6), 3)
    return {"base_P": round(P0 / n, 3), "base_R": round(R0 / n, 3),
            "base_F1": f(P0 / n, R0 / n), "merged_P": round(P1 / n, 3),
            "merged_R": round(R1 / n, 3), "merged_F1": f(P1 / n, R1 / n)}


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
    ann = json.load(open(f"{ROOT}/data/raw/yulima/annotations.json"))
    ya = {}
    for s in ann.get("shapes", []):
        if s.get("type") == "polygon":
            ya.setdefault(s["frame"], []).append(rast(s["points"]))
    ys = json.load(open(f"{ROOT}/data/gold_v1/yulima_qc40.json"))
    yfiles = [f"{ROOT}/data/gold_v1/yulima_qc/{s['file']}" for s in ys]
    ygt = [ya.get(s["frame"], []) for s in ys]
    model = YOLO(f"{ROOT}/data/kaggle-out-ronda1b/runs1b/r1/weights/best.pt")
    out = {"villas30": run_set(model, vfiles, vgt, "villas"),
           "yulima40": run_set(model, yfiles, ygt, "yulima")}
    print(json.dumps(out, indent=1), flush=True)
    json.dump(out, open(f"{ROOT}/data/gold_v1/countmerge_report.json", "w"),
              indent=1)


if __name__ == "__main__":
    main()
