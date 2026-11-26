"""Sweep conf x min-area on Yulima QC 40 vs CVAT auto. One CPU pass at
conf=0.05 saving all candidates, then offline sweep. No CVAT access."""

import json
import os

import numpy as np
from ultralytics import YOLO

ROOT = "/home/biohacker24/annotacion-arboles-auto"
IMGDIR = f"{ROOT}/data/gold_v1/yulima_qc"


def main():
    sample = json.load(open(f"{ROOT}/data/gold_v1/yulima_qc40.json"))
    model = YOLO(f"{ROOT}/data/kaggle-out-fase2/runs/splitB/weights/best.pt")
    paths = [os.path.join(IMGDIR, s["file"]) for s in sample]
    res = model.predict(paths, imgsz=640, conf=0.05, verbose=False)
    allc = []
    for r, s in zip(res, sample):
        fr = []
        nb = len(r.boxes) if r.boxes is not None else 0
        for k in range(nb):
            x0, y0, x1, y1 = r.boxes.xyxy[k].tolist()
            fr.append({"bbox": [x0, y0, x1 - x0, y1 - y0],
                       "score": float(r.boxes.conf[k]),
                       "area": float((x1 - x0) * (y1 - y0))})
        allc.append({"file": s["file"], "frame": s["frame"], "cands": fr})
    json.dump(allc, open(f"{ROOT}/data/gold_v1/yulima_cands.json", "w"))
    print("candidates saved:", sum(len(c["cands"]) for c in allc))


if __name__ == "__main__":
    main()
