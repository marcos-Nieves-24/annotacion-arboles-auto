"""Build gold_v2 from gold_v1 + exported CVAT raw (T0).

Inputs (local or Kaggle-mounted):
  data/gold_v1/            coco.json, holdout.json, splits.json, PROTOCOLO.md
  data/gold_v2_raw/        {villas,magdala}_annotations.json, manifest.json
  data/raw/{farm}/meta.json  frame index -> filename
  data/r1/images/{farm}/   downloaded images for the new ranges

Output: data/gold_v2/
  coco.json, yolo/labels/*.txt, images/*.png (symlink or copy),
  splits.json, dev.json, holdout.json, quarantine.json,
  duplicates_report.json, conversion_report.json

Protocol = gold_v1 (DP eps 1.5, GSD bands, iscrowd >=3 border vertices).
No CVAT writes. No pseudo-label filtering (user confirmed all shapes reviewed).
"""

import json
import math
import os
import shutil
import sys

ROOT = "/home/biohacker24/annotacion-arboles-auto"
sys.path.insert(0, f"{ROOT}/scripts")
from convert_gold import (poly_area, douglas_peucker, perp_dist,
                          W, H, BORDER_TOL, BORDER_MIN_VERTICES, DP_EPSILON)

GOLD1 = f"{ROOT}/data/gold_v1"
RAW = f"{ROOT}/data/gold_v2_raw"
OUT = f"{ROOT}/data/gold_v2"
os.makedirs(f"{OUT}/yolo/labels", exist_ok=True)
os.makedirs(f"{OUT}/images", exist_ok=True)

FARMS = {
    "villas": {"gsd_cm": 3.58, "frames": range(200, 233),
               "area_min": 882, "area_max": 39224, "job": 4307835},
    "magdala": {"gsd_cm": 5.25, "frames": range(200, 241),
                "area_min": 410, "area_max": 18240, "job": 4307818},
}
HOLDOUT_FILES = {h["file_name"] for h in
                 json.load(open(f"{GOLD1}/holdout.json"))}


def border_vertices(pts):
    import numpy as np
    xs, ys = np.array(pts[0::2]), np.array(pts[1::2])
    return int(((xs <= BORDER_TOL) | (ys <= BORDER_TOL) |
                (xs >= W - BORDER_TOL) | (ys >= H - BORDER_TOL)).sum())


def convert(farm, cfg):
    ann = json.load(open(f"{RAW}/{farm}_annotations.json"))
    meta = json.load(open(f"{ROOT}/data/raw/{farm}/meta.json"))
    names = [f["name"] for f in meta["frames"]]
    rng = set(cfg["frames"])
    warns = []
    # group shapes by frame, ALL sources (user reviewed every shape)
    byframe = {}
    for s in ann.get("shapes", []):
        fr = s.get("frame")
        if fr in rng and s.get("type") == "polygon":
            byframe.setdefault(fr, []).append(s)
    # duplicate audit IoU>0.9 between shapes of same frame
    dups = []
    for fr, shapes in byframe.items():
        masks = []
        for s in shapes:
            pts = s["points"]
            pv = list(zip(pts[0::2], pts[1::2]))
            m = set()
            for (x0, y0), (x1, y1) in zip(pv, pv[1:] + pv[:1]):
                steps = int(max(abs(x1-x0), abs(y1-y0))) or 1
                for k in range(steps + 1):
                    t = k / steps
                    m.add((int(x0 + t*(x1-x0)), int(y0 + t*(y1-y0))))
            masks.append(m)
        for i in range(len(masks)):
            for j in range(i+1, len(masks)):
                inter = len(masks[i] & masks[j])
                union = len(masks[i] | masks[j])
                if union and inter/union > 0.9:
                    dups.append({"frame": fr, "ids": [shapes[i]["id"], shapes[j]["id"]],
                                 "iou": round(inter/union, 4)})
    # convert
    records, quarantine = [], []
    for fr, shapes in sorted(byframe.items()):
        name = names[fr] if fr < len(names) else None
        if not name or name in HOLDOUT_FILES:
            continue
        src = f"{ROOT}/data/r1/images/{farm}/{name}"
        if not os.path.exists(src):
            warns.append(f"missing image {farm}/{name}")
            continue
        dst = f"{OUT}/images/{name}"
        if not os.path.exists(dst):
            try: os.symlink(src, dst)
            except Exception: shutil.copyfile(src, dst)
        ylines = []
        for s in shapes:
            pts = s["points"]
            area = poly_area(pts)
            if not (cfg["area_min"] <= area <= cfg["area_max"]):
                quarantine.append({"farm": farm, "frame": fr, "shape_id": s["id"],
                                   "area_px": round(area, 1), "reason": "area_out_of_range"})
                continue
            pv = [(float(pts[2*i]), float(pts[2*i+1])) for i in range(len(pts)//2)]
            simp = douglas_peucker(pv, DP_EPSILON)
            bv = border_vertices([c for p in simp for c in p])
            iscrowd = bv >= BORDER_MIN_VERTICES
            if not iscrowd:
                flat = [round(c/W, 6) for p in simp for c in p]
                if len(flat) >= 6:
                    ylines.append("0 " + " ".join(f"{v:.6f}" for v in flat))
            records.append({"frame": fr, "file_name": name, "farm": farm,
                            "area": round(area,1), "iscrowd": iscrowd,
                            "segmentation": [[c for p in simp for c in p]],
                            "source": s.get("source")})
        if ylines:
            open(f"{OUT}/yolo/labels/{name.rsplit('.',1)[0]}.txt", "w").write(
                "\n".join(ylines) + "\n")
    return records, quarantine, dups, warns


def main():
    report = {"protocol": "gold_v1", "dp_eps": DP_EPSILON,
              "source_filter": "none (user reviewed all shapes)",
              "holdout_excluded": sorted(HOLDOUT_FILES), "farms": {}}
    all_recs, all_q, all_d = [], [], {}
    for farm, cfg in FARMS.items():
        recs, q, d, w = convert(farm, cfg)
        report["farms"][farm] = {"frames": len(set(r["frame"] for r in recs)),
                                  "instances": len(recs),
                                  "iscrowd": sum(1 for r in recs if r["iscrowd"]),
                                  "quarantined": len(q), "duplicates_iou>0.9": len(d),
                                  "warnings": w[:20]}
        all_recs += recs; all_q += q; all_d[farm] = d
    # merge with gold_v1 annotations (already converted) via coco.json
    g1 = json.load(open(f"{GOLD1}/coco.json"))
    byid = {i["id"]: i["file_name"] for i in g1["images"]}
    g1_farms = {"villas": range(0,200), "magdala": range(0,131)}
    base = []
    for a in g1["annotations"]:
        fn = byid[a["image_id"]]
        farm = "villas" if "Villas" in fn else "magdala"
        seg = a["segmentation"][0] if a.get("segmentation") else []
        base.append({"frame": None, "file_name": fn, "farm": farm,
                     "area": round(a["area"],1), "iscrowd": bool(a["iscrowd"]),
                     "segmentation": [list(map(float, seg))], "source": "gold_v1"})
    total = base + all_recs
    # keep ORIGINAL gold_v1 image ids so v1 segmentation stays aligned
    imgs = [{"id": i["id"], "file_name": i["file_name"], "farm": i["farm"]}
            for i in g1["images"]]
    seen = set(i["file_name"] for i in imgs)
    for farm in FARMS:
        meta = json.load(open(f"{ROOT}/data/raw/{farm}/meta.json"))
        for fr in sorted(FARMS[farm]["frames"]):
            nm = meta["frames"][fr]["name"]
            if nm not in seen and nm not in HOLDOUT_FILES:
                imgs.append({"id": max(i["id"] for i in imgs)+1, "file_name": nm, "farm": farm})
                seen.add(nm)
    imap = {i["file_name"]: i["id"] for i in imgs}
    coco = {"info": {"description": "gold_v2 — gold_v1 + reviewed 200-240/200-232"},
            "images": imgs, "categories": [{"id": 1, "name": "crop_tree"}],
            "annotations": []}
    for k, r in enumerate(total):
        fn = r["file_name"]
        coco["annotations"].append({
            "id": k+1, "image_id": imap[fn], "category_id": 1,
            "area": r["area"], "iscrowd": 1 if r["iscrowd"] else 0,
            "bbox": [0,0,0,0],
            "segmentation": [list(map(float, r.get("segmentation", [[]])[0]))]})
    json.dump(coco, open(f"{OUT}/coco.json", "w"))
    # copy gold_v1 YOLO labels too (T0 previously only wrote labels for new frames)
    nv1lab = 0
    for f in os.listdir(f"{GOLD1}/yolo/labels"):
        dst = f"{OUT}/yolo/labels/{f}"
        if not os.path.exists(dst):
            shutil.copyfile(f"{GOLD1}/yolo/labels/{f}", dst)
            nv1lab += 1
    print("copied gold_v1 labels:", nv1lab, flush=True)
    json.dump(all_q, open(f"{OUT}/quarantine.json", "w"), indent=1)
    json.dump(all_d, open(f"{OUT}/duplicates_report.json", "w"), indent=1)
    json.dump({"holdout": sorted(HOLDOUT_FILES), "train_val_excludes": sorted(HOLDOUT_FILES)},
              open(f"{OUT}/holdout.json", "w"), indent=1)
    # farm-wise splits with holdout exclusion; dev 10% per farm
    import random
    random.seed(3)
    byfarm = {}
    for i in imgs:
        if i["file_name"] in HOLDOUT_FILES: continue
        byfarm.setdefault(i["farm"], []).append(i["id"])
    A = {"train": byfarm.get("villas", []), "val": byfarm.get("magdala", [])}
    B = {"train": byfarm.get("magdala", []), "val": byfarm.get("villas", [])}
    devA, devB = [], []
    for farm, ids in byfarm.items():
        k = max(1, len(ids)//10)
        dev = random.Random(7).sample(ids, k)
        devA += dev if farm == "villas" else []
        devB += dev if farm == "magdala" else []
    json.dump({"split_A": A, "split_B": B}, open(f"{OUT}/splits.json", "w"), indent=1)
    json.dump({"A": devA, "B": devB}, open(f"{OUT}/dev.json", "w"), indent=1)
    report["totals"] = {"frames": len(imgs), "instances": len(total),
                        "quarantined": len(all_q),
                        "duplicates_iou>0.9": sum(len(v) for v in all_d.values())}
    json.dump(report, open(f"{OUT}/conversion_report.json", "w"), indent=1)
    print(json.dumps(report["totals"], indent=1))
    print(json.dumps({k: v["duplicates_iou>0.9"] for k,v in report["farms"].items()}, indent=1))


if __name__ == "__main__":
    main()
