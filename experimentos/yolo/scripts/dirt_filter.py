"""Mean-crown dirt filter: drop predictions with mask area < k * farm MEAN.
Sweep k in {0.10,0.15,0.20,0.25,0.30} with real mask-IoU P/R vs refs on
sample90. One inference pass (masks cached as RLE), offline sweep.
CPU local. No CVAT."""

import json
import os

import cv2
import numpy as np
from pycocotools import mask as cmask
from ultralytics import YOLO

ROOT = "/home/biohacker24/annotacion-arboles-auto"
MEAN_CROWN = {"villas": 11304, "magdala": 11429, "yulima": 6940}


def rast(pts):
    m = np.zeros((512, 512), np.uint8)
    cv2.fillPoly(m, [np.array(pts, np.float32).reshape(-1, 2).round()
                     .astype(np.int32)], 1)
    return m


def miou(a, b):
    i = np.logical_and(a, b).sum()
    u = np.logical_or(a, b).sum()
    return float(i / u) if u else 0.0


def dec(rle):
    c = rle["counts"]
    if isinstance(c, str):
        rle = dict(rle)
        rle["counts"] = c.encode()
    return cmask.decode(rle)


def main():
    rows = json.load(open(f"{ROOT}/data/gold_v1/sample90.json"))
    coco = json.load(open(f"{ROOT}/data/gold_v1/coco.json"))
    gv = {}
    for a in coco["annotations"]:
        if not a["iscrowd"]:
            gv.setdefault(a["image_id"], []).append(rast(a["segmentation"][0]))
    byname = {i["file_name"]: i["id"] for i in coco["images"]}
    model = YOLO(f"{ROOT}/data/kaggle-out-ronda1b/runs1b/r1/weights/best.pt")
    cache, refs = [], []
    B = 30
    for b in range(0, len(rows), B):
        ch = rows[b:b + B]
        paths = []
        for r in ch:
            for d in (f"{ROOT}/data/r1/images/{r['farm']}/{r['f']}",
                      f"{ROOT}/data/gold_v1/images/{r['f']}"):
                if os.path.exists(d):
                    paths.append(d)
                    break
        res = model.predict(paths, imgsz=640, conf=0.5, iou=0.35,
                            verbose=False)
        for row, r in zip(ch, res):
            f, farm = row["f"], row["farm"]
            entr = []
            if r.masks is not None and len(r.masks):
                import torch
                import torch.nn.functional as F

                md = r.masks.data.float()
                if md.shape[1] != 512 or md.shape[2] != 512:
                    md = F.interpolate(md.unsqueeze(0), size=(512, 512),
                                       mode="nearest").squeeze(0)
                for k in range(md.shape[0]):
                    m = (md[k] > 0.5).cpu().numpy().astype("uint8")
                    e = cmask.encode(np.asfortranarray(m))
                    c = e["counts"]
                    e["counts"] = c.decode() if isinstance(c, bytes) else c
                    entr.append({"rle": e, "area": int(m.sum())})
            cache.append(entr)
            if farm == "villas" and f in byname:
                refs.append(gv.get(byname[f], []))
            else:
                ann = json.load(open(f"{ROOT}/data/raw/{farm}/annotations.json"))
                meta = json.load(open(f"{ROOT}/data/raw/{farm}/meta.json"))
                idx = [x["name"] for x in meta["frames"]].index(f)
                refs.append([rast(s["points"]) for s in ann["shapes"]
                             if s.get("frame") == idx and s.get("type") == "polygon"])
        print(f"batch {b // B + 1} done", flush=True)
    print("k P R F1", flush=True)
    out = {}
    for k in (0.0, 0.10, 0.15, 0.20, 0.25, 0.30):
        P = R = 0
        for entr, g, row in zip(cache, refs, rows):
            thr = k * MEAN_CROWN[row["farm"]]
            pm = [dec(e["rle"]) for e in entr if e["area"] >= thr]
            used, tp = set(), 0
            for p in pm:
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
            P += tp / max(len(pm), 1)
            R += tp / max(len(g), 1)
        P /= len(rows)
        R /= len(rows)
        F1 = 2 * P * R / max(P + R, 1e-6)
        out[str(k)] = {"P": round(P, 3), "R": round(R, 3),
                       "F1": round(F1, 3)}
        print(f"{k} P={P:.3f} R={R:.3f} F1={F1:.3f}", flush=True)
    json.dump(out, open(f"{ROOT}/data/mejora/dirt_sweep.json", "w"), indent=1)


if __name__ == "__main__":
    main()
