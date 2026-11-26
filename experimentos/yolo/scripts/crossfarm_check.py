"""Local CPU cross-farm check: splitA weights -> holdout Magdala only,
splitB weights -> holdout Villas only. No leakage by construction."""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as cmask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

ROOT = "/home/biohacker24/annotacion-arboles-auto"
coco = json.load(open(f"{ROOT}/data/gold_v1/coco.json"))
hold = json.load(open(f"{ROOT}/data/gold_v1/holdout.json"))
IMG = f"{ROOT}/data/gold_v1/images"
W = f"{ROOT}/data/kaggle-out-fase2/runs"


def evaluate(weights, farm):
    model = YOLO(weights)
    subset = [h for h in hold if h["farm"] == farm]
    paths = [os.path.join(IMG, h["file_name"]) for h in subset]
    res = model.predict(paths, imgsz=640, conf=0.25, verbose=False)
    names = {h["file_name"] for h in subset}
    gimgs = [i for i in coco["images"] if i["file_name"] in names]
    remap = {i["id"]: n for n, i in enumerate(gimgs)}
    gt = {"images": [{"id": n, "width": 512, "height": 512}
                     for n in range(len(gimgs))],
          "annotations": [],
          "categories": [{"id": 1, "name": "crop_tree"}]}
    aid = 0
    for a in coco["annotations"]:
        if a["image_id"] in remap and not a["iscrowd"]:
            gt["annotations"].append(
                {"id": aid, "image_id": remap[a["image_id"]],
                 "category_id": 1, "bbox": a["bbox"],
                 "segmentation": a["segmentation"],
                 "area": a["area"], "iscrowd": 0})
            aid += 1
    open("/tmp/gtx.json", "w").write(json.dumps(gt))
    cocoGt = COCO("/tmp/gtx.json")
    inv = {i["file_name"]: remap[i["id"]] for i in gimgs}
    dt_box, dt_seg, errs = [], [], []
    for r, h in zip(res, subset):
        iid = inv[h["file_name"]]
        n = 0
        confs = r.boxes.conf.tolist() if r.boxes is not None and len(r.boxes) else []
        boxes = r.boxes.xyxy.tolist() if r.boxes is not None and len(r.boxes) else []
        for (x0, y0, x1, y1), cf in zip(boxes, confs):
            dt_box.append({"image_id": iid, "category_id": 1,
                           "bbox": [x0, y0, x1 - x0, y1 - y0],
                           "score": float(cf)})
            n += 1
        if r.masks is not None and len(r.masks):
            md = r.masks.data.float()
            if md.shape[1] != 512 or md.shape[2] != 512:
                md = F.interpolate(md.unsqueeze(0), size=(512, 512),
                                   mode="nearest").squeeze(0)
            for k in range(md.shape[0]):
                m = (md[k] > 0.5).cpu().numpy().astype("uint8")
                rle = cmask.encode(np.asfortranarray(m))
                c = rle["counts"]
                rle["counts"] = c.decode() if isinstance(c, bytes) else c
                dt_seg.append({"image_id": iid, "category_id": 1,
                               "segmentation": rle,
                               "score": float(confs[k]) if k < len(confs) else 0.25})
        gn = sum(1 for a in coco["annotations"]
                 if a["image_id"] == next(i["id"] for i in gimgs
                                          if i["file_name"] == h["file_name"])
                 and not a["iscrowd"])
        errs.append(abs(n - gn))
    out = {}
    if dt_box:
        ev = COCOeval(cocoGt, cocoGt.loadRes(dt_box), "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        out["box_map50"] = round(float(ev.stats[1]), 4)
        out["box_map5095"] = round(float(ev.stats[0]), 4)
    if dt_seg:
        ev = COCOeval(cocoGt, cocoGt.loadRes(dt_seg), "segm")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        out["mask_map50"] = round(float(ev.stats[1]), 4)
        out["mask_map5095"] = round(float(ev.stats[0]), 4)
    out["count_MAE"] = round(float(np.mean(errs)), 2)
    out["n_images"] = len(subset)
    return out


report = {
    "splitA_on_magdala_holdout": evaluate(f"{W}/splitA/weights/best.pt", "magdala"),
    "splitB_on_villas_holdout": evaluate(f"{W}/splitB/weights/best.pt", "villas"),
}
print(json.dumps(report, indent=1))
json.dump(report, open(f"{ROOT}/data/gold_v1/crossfarm_report.json", "w"), indent=1)
