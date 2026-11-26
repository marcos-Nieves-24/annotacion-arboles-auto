"""300 Villas overlays (frames 0-299): green = gold GT, blue = CVAT auto
(frames 200+), red = splitB model. Run after frames exist in
data/gold_v1/images (gold) — downloads 200-299 itself? No: expects
download done by caller. Pure local read + inference."""

import json
import os

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = "/home/biohacker24/annotacion-arboles-auto"
META = json.load(open(f"{ROOT}/data/raw/villas/meta.json"))
NAMES = [f["name"] for f in META["frames"]]
IMGDIR = f"{ROOT}/data/gold_v1/images"
OUTDIR = f"{ROOT}/data/gold_v1/overlays_villas"
os.makedirs(OUTDIR, exist_ok=True)

coco = json.load(open(f"{ROOT}/data/gold_v1/coco.json"))
gold = {}
for a in coco["annotations"]:
    if not a["iscrowd"]:
        gold.setdefault(a["image_id"], []).append(a["segmentation"][0])
byname = {i["file_name"]: i["id"] for i in coco["images"]}

ann = json.load(open(f"{ROOT}/data/raw/villas/annotations.json"))
auto = {}
for s in ann.get("shapes", []):
    if s.get("type") == "polygon":
        auto.setdefault(s["frame"], []).append(s["points"])


def draw(img, polys, color):
    for p in polys:
        pts = np.array(p, np.float32).reshape(-1, 2).astype(int)
        cv2.polylines(img, [pts], True, color, 2)


model = YOLO(f"{ROOT}/data/kaggle-out-fase2/runs/splitB/weights/best.pt")
frames = list(range(0, 300))
allres = []
B = 25
slist = [os.path.join(IMGDIR, NAMES[f]) for f in frames]
for b in range(0, len(slist), B):
    allres += model.predict(slist[b:b + B], imgsz=640, conf=0.5, verbose=False)
    print(f"batch {b//B + 1}/{(len(slist) + B - 1)//B}", flush=True)
res = allres
done = 0
for f, r in zip(frames, res):
    img = cv2.imread(os.path.join(IMGDIR, NAMES[f]))
    if img is None:
        print("missing", NAMES[f])
        continue
    if f < 200:
        draw(img, gold.get(byname.get(NAMES[f], -1), []), (0, 255, 0))
    else:
        draw(img, auto.get(f, []), (255, 0, 0))
    n = 0
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
            n += 1
    cv2.imwrite(os.path.join(OUTDIR, NAMES[f]), img)
    done += 1
print(f"overlays done: {done}/300")
