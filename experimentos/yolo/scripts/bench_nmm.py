"""Benchmark NMM (supervision) vs count-merge on clean-gold frames.

Branches per frame (r1 weights, conf 0.5):
  A: NMS 0.35 + GSD area + count-merge (ours).
  B: sv.Detections.with_nmm(threshold=T), T in {0.05, 0.1, 0.15, 0.2}.
Metric: mask P/R/F1 vs CLEAN gold only (Villas 0-199, Magdala 0-130).
Rest reported separately as agreement-vs-auto. Winner = best F1 with
recall >= best_recall - 0.02. CPU local. No CVAT access.
"""

import json
import os

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = "/home/biohacker24/annotacion-arboles-auto"
AREA_BAND = {"villas": (882, 39224), "magdala": (410, 18240),
             "yulima": (767, 34096)}
MEDIAN_CROWN = {"villas": 9798, "magdala": 6636, "yulima": 4057}


def rast(pts):
    m = np.zeros((512, 512), np.uint8)
    cv2.fillPoly(m, [np.array(pts, np.float32).reshape(-1, 2).round()
                     .astype(np.int32)], 1)
    return m


def miou(a, b):
    i = np.logical_and(a, b).sum()
    u = np.logical_or(a, b).sum()
    return float(i / u) if u else 0.0


def main():
    import supervision as sv

    print("supervision", sv.__version__)
    coco = json.load(open(f"{ROOT}/data/gold_v1/coco.json"))
    gv = {}
    for a in coco["annotations"]:
        if not a["iscrowd"]:
            gv.setdefault(a["image_id"], []).append(rast(a["segmentation"][0]))
    byname = {i["file_name"]: i["id"] for i in coco["images"]}
    farm_of = {i["file_name"]: i["farm"] for i in coco["images"]}
    # clean-gold frames present locally
    files = []
    for fn, iid in byname.items():
        for d in (f"{ROOT}/data/r1/images/{farm_of[fn]}/{fn}",
                  f"{ROOT}/data/gold_v1/images/{fn}"):
            if os.path.exists(d):
                files.append((fn, d))
                break
    print("clean frames locally:", len(files))
    model = YOLO(f"{ROOT}/data/kaggle-out-ronda1b/runs1b/r1/weights/best.pt")
    agg = {}
    B = 25
    for b in range(0, len(files), B):
        ch = files[b:b + B]
        res = model.predict([p for _, p in ch], imgsz=640, conf=0.5,
                            iou=0.5, verbose=False)
        for (fn, _), r in zip(ch, res):
            farm = farm_of[fn]
            g = gv.get(byname[fn], [])
            nb = len(r.boxes) if r.boxes is not None else 0
            boxes = r.boxes.xyxy.cpu().numpy() if nb else np.zeros((0, 4))
            confs = r.boxes.conf.cpu().numpy() if nb else np.zeros(0)
            masks = None
            if r.masks is not None and len(r.masks):
                import torch
                import torch.nn.functional as F

                md = r.masks.data.float()
                if md.shape[1] != 512 or md.shape[2] != 512:
                    md = F.interpolate(md.unsqueeze(0), size=(512, 512),
                                       mode="nearest").squeeze(0)
                masks = (md > 0.5).cpu().numpy().astype("uint8")
            det = sv.Detections(xyxy=boxes, confidence=confs,
                                class_id=np.zeros(nb, dtype=int), mask=masks)

            def score(pred_masks):
                used, tp = set(), 0
                for p in pred_masks:
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
                P = tp / max(len(pred_masks), 1)
                R = tp / max(len(g), 1)
                return P, R

            # branch A: ours (area filter + count-merge)
            amin, amax = AREA_BAND[farm]
            filt = [masks[k] for k in range(nb)
                    if masks is not None and amin <= masks[k].sum() <= amax]
            union = np.logical_or.reduce(filt).astype(np.uint8) if filt \
                else np.zeros((512, 512), np.uint8)
            exp = max(1, int(round(union.sum() / MEDIAN_CROWN[farm])))
            cur = [m.astype(bool) for m in filt]
            guard = 0
            while len(cur) > int(exp * 1.1) + 1 and guard < 200:
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
            pa, ra = score([m.astype(np.uint8) for m in cur])
            # branch B: supervision NMM sweep
            for T in (0.05, 0.1, 0.15, 0.2):
                try:
                    mrg = det.with_nmm(threshold=T)
                    pm = []
                    if mrg.mask is not None:
                        for k in range(len(mrg)):
                            m = mrg.mask[k].astype(np.uint8)
                            if amin <= m.sum() <= amax:
                                pm.append(m)
                    pb, rb = score(pm)
                except Exception as e:
                    pb, rb = -1.0, -1.0
                    print("nmm fail", T, repr(e)[:120])
                agg.setdefault(f"nmm_{T}", []).append((pb, rb))
            agg.setdefault("ours", []).append((pa, ra))
        print(f"batch {b // B + 1} done", flush=True)
    out = {}
    for k, v in agg.items():
        P = float(np.mean([p for p, _ in v if p >= 0]))
        R = float(np.mean([r for _, r in v if r >= 0]))
        out[k] = {"P": round(P, 3), "R": round(R, 3),
                  "F1": round(2 * P * R / max(P + R, 1e-6), 3), "n": len(v)}
    print(json.dumps(out, indent=1))
    json.dump(out, open(f"{ROOT}/data/mejora/nmm_bench.json", "w"), indent=1)


if __name__ == "__main__":
    main()
