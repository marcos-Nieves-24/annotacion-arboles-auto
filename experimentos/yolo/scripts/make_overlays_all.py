"""All-frame overlays with r1 (conf 0.5, NMS 0.35): green = gold GT,
blue = CVAT auto, red = model. Villas 0-688, Magdala all, Yulima all.
Images from data/r1/images (unlabeled) + data/gold_v1/images (gold)."""

import json
import os

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = "/home/biohacker24/annotacion-arboles-auto"
OUT = f"{ROOT}/data/gold_v1/overlays_all"
os.makedirs(OUT, exist_ok=True)

coco = json.load(open(f"{ROOT}/data/gold_v1/coco.json"))
gold = {}
for a in coco["annotations"]:
    if not a["iscrowd"]:
        gold.setdefault(a["image_id"], []).append(a["segmentation"][0])
byname = {i["file_name"]: i["id"] for i in coco["images"]}

RAW = {}
for farm, jf in [("villas", f"{ROOT}/data/raw/villas/annotations.json"),
                 ("magdala", f"{ROOT}/data/raw/magdala/annotations.json"),
                 ("yulima", f"{ROOT}/data/raw/yulima/annotations.json")]:
    auto = {}
    for s in json.load(open(jf)).get("shapes", []):
        if s.get("type") == "polygon":
            auto.setdefault(s["frame"], []).append(s["points"])
    meta = json.load(open(f"{ROOT}/data/raw/{farm}/meta.json"))
    RAW[farm] = {"auto": auto,
                 "names": [f["name"] for f in meta["frames"]]}


def src_of(farm, name):
    for d in (f"{ROOT}/data/r1/images/{farm}/{name}",
              f"{ROOT}/data/gold_v1/images/{name}",
              f"{ROOT}/data/gold_v1/yulima_qc/{name}"):
        if os.path.exists(d):
            return d
    return None


def draw(img, polys, color):
    for p in polys:
        pts = np.array(p, np.float32).reshape(-1, 2).astype(int)
        cv2.polylines(img, [pts], True, color, 2)


model = YOLO(f"{ROOT}/data/kaggle-out-ronda1b/runs1b/r1/weights/best.pt")
GOLD_RANGES = {"villas": set(range(0, 200)), "magdala": set(range(0, 131)),
               "yulima": set()}
done = 0
for farm, R in RAW.items():
    idxs = sorted(set(list(R["auto"].keys()) +
                      [i for i, n in enumerate(R["names"])
                       if os.path.exists(f"{ROOT}/data/r1/images/{farm}/{n}")
                       or os.path.exists(f"{ROOT}/data/gold_v1/images/{n}")]))
    B = 25
    for b in range(0, len(idxs), B):
        chunk = idxs[b:b + B]
        paths, valid = [], []
        for i in chunk:
            p = src_of(farm, R["names"][i])
            if p:
                paths.append(p)
                valid.append(i)
        if not paths:
            continue
        res = model.predict(paths, imgsz=640, conf=0.5, iou=0.35,
                            verbose=False)
        for i, r in zip(valid, res):
            img = cv2.imread(src_of(farm, R["names"][i]))
            if img is None:
                continue
            if i in GOLD_RANGES[farm]:
                draw(img, gold.get(byname.get(R["names"][i], -1), []),
                     (0, 255, 0))
            else:
                draw(img, R["auto"].get(i, []), (255, 0, 0))
            if r.masks is not None and len(r.masks):
                import torch.nn.functional as F

                md = r.masks.data.float()
                if md.shape[1] != 512 or md.shape[2] != 512:
                    md = F.interpolate(md.unsqueeze(0), size=(512, 512),
                                       mode="nearest").squeeze(0)
                for k in range(md.shape[0]):
                    m = (md[k] > 0.5).cpu().numpy().astype("uint8")
                    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(img, cnts, -1, (0, 0, 255), 2)
            cv2.imwrite(f"{OUT}/{farm}_{R['names'][i]}", img)
            done += 1
        print(f"{farm} batch {b // B + 1} done={done}", flush=True)
print(f"ALL done: {done}", flush=True)
