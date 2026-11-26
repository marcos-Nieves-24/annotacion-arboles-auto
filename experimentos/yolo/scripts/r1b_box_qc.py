"""R1b box QC: draw actual YOLO boxes/masks on a stratified sample.

Green = gold polygon (only gold frames). Blue = CVAT auto polygon.
Red = R1b box, yellow = R1b mask contour. CPU, no CVAT writes.

Output: data/fase4/box_qc/{overlays,contact_*.jpg,report.json}
"""

import json
import os

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = "/home/biohacker24/annotacion-arboles-auto"
OUT = f"{ROOT}/data/fase4/box_qc"
os.makedirs(f"{OUT}/overlays", exist_ok=True)

WEIGHTS = f"{ROOT}/data/kaggle-out-ronda1b/runs1b/r1/weights/best.pt"
CONF, IOU = 0.5, 0.35


def src_of(name):
    for d in (f"{ROOT}/data/r1/images/villas/{name}",
              f"{ROOT}/data/r1/images/magdala/{name}",
              f"{ROOT}/data/gold_v1/images/{name}",
              f"{ROOT}/data/gold_v1/yulima_qc/{name}"):
        if os.path.exists(d):
            return d
    return None


def rast(pts):
    m = np.zeros((512, 512), np.uint8)
    cv2.fillPoly(m, [np.array(pts, np.float32).reshape(-1, 2).round().astype(np.int32)], 1)
    return m


def main():
    coco = json.load(open(f"{ROOT}/data/gold_v1/coco.json"))
    byname = {i["file_name"]: i["id"] for i in coco["images"]}
    gold = {}
    for a in coco["annotations"]:
        if not a["iscrowd"]:
            gold.setdefault(a["image_id"], []).append(a["segmentation"][0])

    auto = {}
    for farm in ("villas", "magdala"):
        ann = json.load(open(f"{ROOT}/data/raw/{farm}/annotations.json"))
        for s in ann.get("shapes", []):
            if s.get("type") == "polygon":
                auto.setdefault((farm, s["frame"]), []).append(s["points"])

    sample = []
    for farm, sfile in (("villas", "villas_sample30.json"),
                        ("magdala", "magdala_cands30.json")):
        for e in json.load(open(f"{ROOT}/data/gold_v1/{sfile}")):
            sample.append((farm, e["file"], e.get("frame"), True))
    # 20 non-gold production frames, spread over downloaded ranges
    for farm, start, end in (("villas", 200, 689), ("magdala", 131, 1302)):
        meta = json.load(open(f"{ROOT}/data/raw/{farm}/meta.json"))
        names = [f["name"] for f in meta["frames"]]
        step = (end - start) // 10
        for k in range(10):
            i = start + k * step
            sample.append((farm, names[i], i, False))

    model = YOLO(WEIGHTS)
    report = []
    B = 20
    for b in range(0, len(sample), B):
        chunk = sample[b:b + B]
        paths = [src_of(n) for _, n, _, _ in chunk]
        valid = [(f, n, i, g, p) for (f, n, i, g), p in zip(chunk, paths) if p]
        if not valid:
            continue
        res = model.predict([p for *_, p in valid], imgsz=640, conf=CONF,
                            iou=IOU, verbose=False)
        for (farm, name, idx, is_gold, path), r in zip(valid, res):
            img = cv2.imread(path)
            if img is None:
                continue
            # reference polygons
            gp = gold.get(byname.get(name, -1), []) if is_gold else []
            ap = auto.get((farm, idx), [])
            for p in ap:
                pts = np.array(p, np.float32).reshape(-1, 2).astype(int)
                cv2.polylines(img, [pts], True, (255, 128, 0), 1)
            for p in gp:
                pts = np.array(p, np.float32).reshape(-1, 2).astype(int)
                cv2.polylines(img, [pts], True, (0, 220, 0), 2)

            nb = len(r.boxes) if r.boxes is not None else 0
            confs = r.boxes.conf.tolist() if nb else []
            for k in range(nb):
                x0, y0, x1, y1 = [int(v) for v in r.boxes.xyxy[k].tolist()]
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)
                cv2.putText(img, f"{k}:{confs[k]:.2f}", (x0, max(12, y0 - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1, cv2.LINE_AA)
            if r.masks is not None and len(r.masks):
                import torch.nn.functional as F
                md = r.masks.data.float()
                if md.shape[1] != 512 or md.shape[2] != 512:
                    md = F.interpolate(md.unsqueeze(0), size=(512, 512),
                                       mode="nearest").squeeze(0)
                for k in range(md.shape[0]):
                    m = (md[k] > 0.5).cpu().numpy().astype("uint8")
                    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(img, cnts, -1, (0, 255, 255), 1)

            cv2.imwrite(f"{OUT}/overlays/{farm}_{name}", img)
            report.append({"farm": farm, "file": name, "frame": idx, "gold": is_gold,
                           "n_boxes": nb, "n_gold_poly": len(gp), "n_auto_poly": len(ap),
                           "conf_min": round(min(confs), 3) if confs else None,
                           "conf_max": round(max(confs), 3) if confs else None})
        print(f"batch {b // B + 1}/{(len(sample) + B - 1) // B}", flush=True)

    json.dump(report, open(f"{OUT}/report.json", "w"), indent=1)
    # contact sheets, 5 cols x 4 rows, 256px thumbnails
    for farm in ("villas", "magdala"):
        items = [r for r in report if r["farm"] == farm][:20]
        if not items:
            continue
        tw, th = 256, 256
        sheet = np.full((th * 4, tw * 5, 3), 40, np.uint8)
        for k, r in enumerate(items):
            im = cv2.imread(f"{OUT}/overlays/{farm}_{r['file']}")
            if im is None:
                continue
            im = cv2.resize(im, (tw, th))
            cv2.putText(im, f"{r['file'][-9:-4]} g{r['n_gold_poly']} p{r['n_boxes']}",
                        (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            sheet[(k // 5) * th:(k // 5 + 1) * th,
                  (k % 5) * tw:(k % 5 + 1) * tw] = im
        cv2.imwrite(f"{OUT}/contact_{farm}.jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"DONE: {len(report)} overlays", flush=True)


if __name__ == "__main__":
    main()
