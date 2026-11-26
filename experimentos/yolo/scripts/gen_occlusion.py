"""Targeted occlusion augmentation: paste soil/shadow strips over crown
junctions (where >=2 gold polygons touch) + random cutouts on crowns.
Labels unchanged (standard cutout practice). Output data/occ/{images,labels}/
+ manifest. CPU local. No CVAT."""

import json
import os
import random

import cv2
import numpy as np

ROOT = "/home/biohacker24/annotacion-arboles-auto"
OUT = f"{ROOT}/data/occ"
FARM_FILTER = os.environ.get("OCC_FARM")  # "villas" | "magdala" | None=all
os.makedirs(f"{OUT}/images", exist_ok=True)
os.makedirs(f"{OUT}/labels", exist_ok=True)
random.seed(11)
np.random.seed(11)


def rast(pts):
    m = np.zeros((512, 512), np.uint8)
    cv2.fillPoly(m, [np.array(pts, np.float32).reshape(-1, 2).round()
                     .astype(np.int32)], 1)
    return m


def soil_patch(img, gold_union, w=64, h=32):
    """Sample a background-looking strip (low ExG) from the same image."""
    f = img.astype(np.float32)
    exg = 2 * f[:, :, 1] - f[:, :, 0] - f[:, :, 2]
    bg = ((exg < np.percentile(exg, 25)) & (gold_union == 0)).astype(np.uint8)
    ys, xs = np.where(bg)
    if len(xs) < w * h:
        return np.zeros((h, w, 3), np.uint8)
    i = random.randrange(len(xs))
    x, y = xs[i] % 512, ys[i] % 512
    x0, y0 = max(0, x - w // 2), max(0, y - h // 2)
    return img[y0:y0 + h, x0:x0 + w].copy()


def main():
    coco = json.load(open(f"{ROOT}/data/gold_v1/coco.json"))
    byimg = {i["id"]: i["file_name"] for i in coco["images"]}
    farm_of = {i["id"]: i["farm"] for i in coco["images"]}
    polys = {}
    for a in coco["annotations"]:
        if not a["iscrowd"]:
            polys.setdefault(a["image_id"], []).append(a["segmentation"][0])
    manifest = []
    for iid, fn in byimg.items():
        if FARM_FILTER and farm_of.get(iid) != FARM_FILTER:
            continue
        src = f"{ROOT}/data/gold_v1/images/{fn}"
        img = cv2.imread(src)
        if img is None or iid not in polys:
            continue
        union = np.logical_or.reduce(
            [rast(p) for p in polys[iid]]).astype(np.uint8)
        # junction map: dilated overlap of >=2 crowns
        dil = [cv2.dilate(rast(p), np.ones((7, 7), np.uint8))
               for p in polys[iid]]
        overlap = (sum(dil) >= 2).astype(np.uint8) if dil else \
            np.zeros((512, 512), np.uint8)
        ys, xs = np.where(overlap)
        for v in (1, 2):
            aug = img.copy()
            if len(xs) > 0 and v == 1:
                # strip over a random junction point
                i = random.randrange(len(xs))
                cx, cy = int(xs[i]), int(ys[i])
                w, h = random.randint(40, 90), random.randint(16, 36)
                patch = soil_patch(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                   union, w, h)
                patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR) \
                    if patch.shape[2] == 3 else patch
                x0 = max(0, min(512 - w, cx - w // 2))
                y0 = max(0, min(512 - h, cy - h // 2))
                if random.random() < 0.5:
                    aug[y0:y0 + h, x0:x0 + w] = cv2.GaussianBlur(
                        aug[y0:y0 + h, x0:x0 + w], (15, 15), 0)
                else:
                    ph, pw = patch.shape[:2]
                    aug[y0:y0 + ph, x0:x0 + pw] = patch
            else:
                # random cutout on a random crown
                p = random.choice(polys[iid])
                m = rast(p)
                ys2, xs2 = np.where(m)
                if len(xs2) == 0:
                    continue
                i = random.randrange(len(xs2))
                cx, cy = int(xs2[i]), int(ys2[i])
                w, h = random.randint(30, 70), random.randint(30, 70)
                x0 = max(0, min(512 - w, cx - w // 2))
                y0 = max(0, min(512 - h, cy - h // 2))
                aug[y0:y0 + h, x0:x0 + w] = 0
            oname = fn.rsplit(".", 1)[0] + f"_occ{v}.png"
            cv2.imwrite(f"{OUT}/images/{oname}", aug)
            # same labels as source
            lab = f"{ROOT}/data/gold_v1/yolo/labels/{fn.rsplit('.', 1)[0]}.txt"
            import shutil
            shutil.copyfile(lab, f"{OUT}/labels/{oname.rsplit('.', 1)[0]}.txt")
            manifest.append({"src": fn, "out": oname, "variant": v})
    json.dump(manifest, open(f"{OUT}/manifest.json", "w"))
    print(f"occluded copies: {len(manifest)}", flush=True)


if __name__ == "__main__":
    main()
