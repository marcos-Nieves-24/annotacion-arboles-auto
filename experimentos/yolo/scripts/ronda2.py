"""Ronda 2 pseudo: r1 weights, conf>=0.65, NMS 0.35, GSD area band,
count-guided merge as postprocess. Output data/r2/{images->symlink to
r1 images,labels/{farm}/}. No CVAT access."""

import json
import os

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = "/home/biohacker24/annotacion-arboles-auto"
R1IMG = f"{ROOT}/data/r1/images"
R2 = f"{ROOT}/data/r2"
MEDIAN_CROWN = {"villas": 9798, "magdala": 6636, "yulima": 4057}
AREA_BAND = {"villas": (882, 39224), "magdala": (410, 18240),
             "yulima": (767, 34096)}
CONF, NMS_IOU, MAXN = 0.65, 0.35, 40


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
    return [m.astype(np.uint8) for m in cur]


def main():
    sample = None
    model = YOLO(f"{ROOT}/data/kaggle-out-ronda1b/runs1b/r1/weights/best.pt")
    rep = {}
    for farm in ["villas", "magdala", "yulima"]:
        meta = json.load(open(f"{ROOT}/data/raw/{farm}/meta.json"))
        names = [f["name"] for f in meta["frames"]]
        if farm == "villas":
            idxs = list(range(200, 689))
        elif farm == "magdala":
            idxs = list(range(131, 1302))
        else:
            idxs = [i for i in range(756) if i not in {700, 701, 704, 725}]
        ld = f"{R2}/labels/{farm}"
        os.makedirs(ld, exist_ok=True)
        idir = f"{R2}/images/{farm}"
        os.makedirs(idir, exist_ok=True)
        kept_fr = kept_inst = over = 0
        B = 40
        for b in range(0, len(idxs), B):
            ch = idxs[b:b + B]
            paths = []
            for i in ch:
                src = f"{R1IMG}/{farm}/{names[i]}"
                dst = f"{idir}/{names[i]}"
                if os.path.exists(src) and not os.path.exists(dst):
                    try:
                        os.symlink(src, dst)
                    except Exception:
                        pass
                paths.append(dst)
            res = model.predict(paths, imgsz=640, conf=CONF, iou=NMS_IOU,
                                verbose=False)
            print(f"{farm} batch {b // B + 1}", flush=True)
            for n, r in zip([names[i] for i in ch], res):
                nb = len(r.boxes) if r.boxes is not None else 0
                if nb > MAXN:
                    over += 1
                    continue
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
                pm = count_guided(pm, farm)
                lines = []
                for m in pm:
                    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                    if not cnts:
                        continue
                    c = max(cnts, key=cv2.contourArea)
                    peri = cv2.arcLength(c, True)
                    ap = cv2.approxPolyDP(c, 0.005 * peri, True).ravel()
                    if len(ap) < 6:
                        continue
                    lines.append("0 " + " ".join(
                        f"{ap[j] / 512:.6f} {ap[j + 1] / 512:.6f}"
                        for j in range(0, len(ap), 2)))
                if lines:
                    kept_fr += 1
                    kept_inst += len(lines)
                    with open(f"{ld}/{n.rsplit('.', 1)[0]}.txt", "w") as fh:
                        fh.write("\n".join(lines) + "\n")
        rep[farm] = {"frames_kept": kept_fr, "instances": kept_inst,
                     "over_cap": over}
        print(farm, rep[farm], flush=True)
    json.dump(rep, open(f"{R2}/pseudo_report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
