"""Mejora de output (puntos 2-4): pipeline de postproceso sobre la muestra 90.

r1 (conf 0.5, NMS 0.35) -> filtro area GSD -> count-merge (media por finca).
Escribe: labels corregidos, overlays antes/después, métricas antes/después,
reporte de conteo esperado vs final por frame. Todo local. Sin CVAT."""

import json
import os

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = "/home/biohacker24/annotacion-arboles-auto"
OUT = f"{ROOT}/data/mejora"
os.makedirs(f"{OUT}/labels", exist_ok=True)
os.makedirs(f"{OUT}/overlays", exist_ok=True)

MEDIAN_CROWN = {"villas": 9798, "magdala": 6636, "yulima": 4057}
AREA_BAND = {"villas": (882, 39224), "magdala": (410, 18240),
             "yulima": (767, 34096)}


def rast(pts):
    m = np.zeros((512, 512), np.uint8)
    cv2.fillPoly(m, [np.array(pts, np.float32).reshape(-1, 2).round()
                     .astype(np.int32)], 1)
    return m


def miou(a, b):
    i = np.logical_and(a, b).sum()
    u = np.logical_or(a, b).sum()
    return float(i / u) if u else 0.0


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


def main():
    rows = json.load(open(f"{ROOT}/data/gold_v1/sample90.json"))
    coco = json.load(open(f"{ROOT}/data/gold_v1/coco.json"))
    gv = {}
    for a in coco["annotations"]:
        if not a["iscrowd"]:
            gv.setdefault(a["image_id"], []).append(rast(a["segmentation"][0]))
    byname = {i["file_name"]: i["id"] for i in coco["images"]}
    ann = {}
    meta_names = {}
    for farm in ["villas", "magdala", "yulima"]:
        au = {}
        for s in json.load(open(f"{ROOT}/data/raw/{farm}/annotations.json")).get("shapes", []):
            if s.get("type") == "polygon":
                au.setdefault(s["frame"], []).append(rast(s["points"]))
        ann[farm] = au
        meta_names[farm] = [f["name"] for f in
                            json.load(open(f"{ROOT}/data/raw/{farm}/meta.json"))["frames"]]

    def src_of(farm, name):
        for d in (f"{ROOT}/data/r1/images/{farm}/{name}",
                  f"{ROOT}/data/gold_v1/images/{name}",
                  f"{ROOT}/data/gold_v1/yulima_qc/{name}"):
            if os.path.exists(d):
                return d
        return None

    def ref_of(farm, name):
        if farm == "villas" and name in byname:
            return gv.get(byname[name], [])
        idx = meta_names[farm].index(name)
        return ann[farm].get(idx, [])

    model = YOLO(f"{ROOT}/data/kaggle-out-ronda1b/runs1b/r1/weights/best.pt")
    rep_rows = []
    B = 30
    for b in range(0, len(rows), B):
        ch = rows[b:b + B]
        paths = [src_of(r["farm"], r["f"]) for r in ch]
        res = model.predict(paths, imgsz=640, conf=0.5, iou=0.35,
                            verbose=False)
        for row, r, p in zip(ch, res, paths):
            farm = row["farm"]
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
            fixed, expected = count_guided(pm, farm)
            g = ref_of(farm, row["f"])

            def score(preds):
                used, tp = set(), 0
                for x in preds:
                    best, bj = 0.0, -1
                    for j, a in enumerate(g):
                        if j in used:
                            continue
                        v = miou(x, a)
                        if v > best:
                            best, bj = v, j
                    if best >= 0.5:
                        tp += 1
                        used.add(bj)
                P = tp / max(len(preds), 1)
                R = tp / max(len(g), 1)
                return P, R

            p0, r0 = score(pm)
            p1, r1 = score(fixed)
            # labels + overlay (cyan = corregido)
            img = cv2.imread(p)
            for m in fixed:
                cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cnts, -1, (255, 255, 0), 2)
            cv2.imwrite(f"{OUT}/overlays/{farm}_{row['f']}", img)
            with open(f"{OUT}/labels/{farm}_{row['f'].rsplit('.', 1)[0]}.txt", "w") as fh:
                for m in fixed:
                    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                    if not cnts:
                        continue
                    c = max(cnts, key=cv2.contourArea)
                    ap = cv2.approxPolyDP(c, 0.005 * cv2.arcLength(c, True),
                                          True).ravel()
                    if len(ap) >= 6:
                        fh.write("0 " + " ".join(
                            f"{ap[j] / 512:.6f} {ap[j + 1] / 512:.6f}"
                            for j in range(0, len(ap), 2)) + "\n")
            rep_rows.append({"f": row["f"], "farm": farm,
                             "n_raw": len(pm), "n_fixed": len(fixed),
                             "expected": expected, "n_ref": len(g),
                             "P0": round(p0, 3), "R0": round(r0, 3),
                             "P1": round(p1, 3), "R1": round(r1, 3)})
        print(f"batch {b // B + 1} done", flush=True)
    import statistics

    for farm in ["villas", "magdala", "yulima"]:
        rr = [x for x in rep_rows if x["farm"] == farm]
        if not rr:
            continue
        f0 = statistics.mean(2 * x["P0"] * x["R0"] / max(x["P0"] + x["R0"], 1e-6) for x in rr)
        f1 = statistics.mean(2 * x["P1"] * x["R1"] / max(x["P1"] + x["R1"], 1e-6) for x in rr)
        print(farm, len(rr), "F1 antes=%.3f despues=%.3f" % (f0, f1), flush=True)
    json.dump(rep_rows, open(f"{OUT}/mejora_report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
