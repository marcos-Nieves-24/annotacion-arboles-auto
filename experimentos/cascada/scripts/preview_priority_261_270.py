#!/usr/bin/env python3
"""Preview of priority painting on Villas 00771-00780 (no model changes).

For each accepted mask, keep SAM's own predicted score. Sort descending and
paint sequentially: a mask only writes pixels not claimed by a higher-score
mask. Writes overlays and a report under data/centroides/priority_261_270/.
"""
import glob
import json
import math
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from segment_anything import sam_model_registry, SamPredictor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_split_261_270 import load_ns, shape_stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "centroides", "priority_261_270")
CLEAN = os.path.join(ROOT, "data", "centroides", "priority_261_270_clean")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CLEAN, exist_ok=True)


def main():
    ns = load_ns()
    net = ns["CenterNetLite"]()
    net.load_state_dict(torch.load(os.path.join(ROOT, "data", "centroides", "v1",
                                                "model.pt"), map_location="cpu",
                                   weights_only=True))
    net.eval()
    sam = sam_model_registry["vit_b"](checkpoint=os.path.join(
        ROOT, "data", "centroides", "cascade", "sam_vit_b_01ec64.pth"))
    predictor = SamPredictor(sam)

    meta = json.load(open(os.path.join(ROOT, "data", "raw", "villas", "meta.json")))
    frames = [meta["frames"][i]["name"] for i in range(261, 271)]
    idx = {}
    for p in glob.glob(os.path.join(ROOT, "data", "r1", "images", "villas", "*.png")):
        idx.setdefault(os.path.basename(p), p)

    report = {}
    with torch.no_grad():
        for name in frames:
            img = np.asarray(Image.open(idx[name]).convert("RGB"), dtype=np.uint8)
            ph, pr = net(torch.from_numpy(img.astype(np.float32) / 255.).permute(2, 0, 1)[None])
            predictor.set_image(img)
            props = ns["decode"](ph, pr, thr=0.5, dist_factor=0.7, fixed_r=58.5)
            masks, sam_scores = [], []
            for cx, cy, r, _ in props:
                box = np.array([max(0, cx - 1.15 * r), max(0, cy - 1.15 * r),
                                min(512, cx + 1.15 * r), min(512, cy + 1.15 * r)])
                ms, sc, _ = predictor.predict(box=box, multimask_output=True)
                k = int(np.argmax(sc))
                masks.append(ms[k])
                sam_scores.append(float(sc[k]))

            kept, rejected = ns["validate"](props, masks, img, name)
            _, ambiguas, npares = ns["annotate_confidence"](
                [p for p, _ in kept], [m for _, m in kept], img)
            score_by_centre = {(round(p[0], 1), round(p[1], 1)): sam_scores[i]
                               for i, p in enumerate(props)}
            items = []
            for p, m in kept:
                c = (round(p[0], 1), round(p[1], 1))
                items.append({"prop": p, "mask": m,
                              "sam_score": score_by_centre.get(c, float(p[3]))})
            items.sort(key=lambda x: -x["sam_score"])

            claimed = np.zeros((512, 512), bool)
            painted, trimmed = [], 0
            for it in items:
                m = it["mask"]
                pm = m & ~claimed
                trimmed += int(np.count_nonzero(m & claimed))
                claimed |= m
                painted.append((it["prop"], pm))

            ious = []
            for i in range(len(painted)):
                for j in range(i + 1, len(painted)):
                    a, b = painted[i][1], painted[j][1]
                    u = np.count_nonzero(a | b)
                    if u:
                        ious.append(np.count_nonzero(a & b) / u)
            ratios = []
            for p, m in painted:
                area, _, _ = shape_stats(m)
                ratios.append(area / (math.pi * p[2] ** 2))

            vis = img.copy()
            for p, m in painted:
                vis[m] = (0.55 * vis[m] + 0.45 * np.array([70, 220, 110])).astype(np.uint8)
            clean_vis = vis.copy()
            for it in items:
                cx, cy = it["prop"][0], it["prop"][1]
                old = it["mask"]
                lost = old & ~next(m for p, m in painted
                                   if p[0] == it["prop"][0] and p[1] == it["prop"][1])
                if lost.any():
                    vis[lost] = (0.5 * vis[lost] + 0.5 * np.array([255, 60, 200])).astype(np.uint8)
            for vis_out, folder in ((vis, OUT), (clean_vis, CLEAN)):
                im = Image.fromarray(vis_out)
                d = ImageDraw.Draw(im)
                for p, _ in painted:
                    cx, cy = int(p[0]), int(p[1])
                    d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6],
                              outline=(255, 40, 40), width=2)
                im.save(os.path.join(folder, name))

            report[name] = {
                "aceptadas": len(kept), "pares_ambiguos_antes": len(ambiguas),
                "max_iou_despues": round(float(max(ious)), 3) if ious else 0.0,
                "pixeles_recortados": trimmed,
                "area_ratio_med": round(float(np.median(ratios)), 2),
            }
            print(name, report[name], flush=True)

    json.dump(report, open(os.path.join(OUT, "report.json"), "w"), indent=1)

    files = sorted(OUT.glob("Limon_*.png")) if hasattr(OUT, "glob") else sorted(
        f for f in os.listdir(OUT) if f.startswith("Limon_") and f.endswith(".png"))
    cols, tw, th, lab = 2, 512, 512, 26
    sheet = Image.new("RGB", (cols * tw, ((len(files) + 1) // 2) * (th + lab)), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for i, f in enumerate(files):
        im = Image.open(os.path.join(OUT, f)).convert("RGB").resize((tw, th))
        x, y = (i % 2) * tw, (i // 2) * (th + lab)
        sheet.paste(im, (x, y))
        d.text((x + 6, y + th + 4), f.replace(".png", ""), fill=(0, 0, 0))
    sheet.save(os.path.join(OUT, "contact.png"))

    files_clean = sorted(f for f in os.listdir(CLEAN)
                         if f.startswith("Limon_") and f.endswith(".png"))
    sheet2 = Image.new("RGB", (cols * tw, ((len(files_clean) + 1) // 2) * (th + lab)),
                       (255, 255, 255))
    d2 = ImageDraw.Draw(sheet2)
    for i, f in enumerate(files_clean):
        im = Image.open(os.path.join(CLEAN, f)).convert("RGB").resize((tw, th))
        x, y = (i % 2) * tw, (i // 2) * (th + lab)
        sheet2.paste(im, (x, y))
        d2.text((x + 6, y + th + 4), f.replace(".png", ""), fill=(0, 0, 0))
    sheet2.save(os.path.join(CLEAN, "contact.png"))
    print("contacts guardados", sheet.size, sheet2.size)


if __name__ == "__main__":
    main()
