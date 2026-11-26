#!/usr/bin/env python3
"""Compare three split/selection strategies on Villas 00771-00780.

baseline = v4 (SAM score selection + ambiguity flag, no geometry change).
opt1 = watershed on the inverted centroid heatmap (markers at the two centres).
opt2 = watershed on the distance transform of the combined mask (neck split).
opt3 = context-aware choice among SAM's 3 masks (minimise overlap with neighbours).

No training, no Kaggle, no CVAT writes.
"""
import ast
import glob
import json
import math
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import models
from segment_anything import sam_model_registry, SamPredictor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "centroides", "test_split_261_270")
os.makedirs(OUT, exist_ok=True)
NAMES = {"CenterNetLite", "decode", "validate", "annotate_confidence",
         "_as_ring", "polygon_centroid", "equivalent_radius", "build_targets",
         "canopy_area", "MEDIAN_AREA", "MEDIAN_R"}


def load_ns():
    tree = ast.parse(open(os.path.join(ROOT, "kernels", "centroides_cascade_v4",
                                       "kernel_source.py")).read())
    keep = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in NAMES:
            keep.append(n)
        if isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in NAMES for t in n.targets):
            keep.append(n)
    ns = dict(torch=torch, nn=nn, F=F, np=np, cv2=cv2, models=models,
              random=random, math=math, os=os, Image=Image)
    exec(compile(ast.Module(body=keep, type_ignores=[]), "<k>", "exec"), ns)
    return ns


def seed_mask(mask, centre):
    mk = np.zeros(mask.shape, np.int32)
    er = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8), 1)
    mk[er > 0] = 1
    x, y = int(centre[0]), int(centre[1])
    if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
        mk[y, x] = 1
    return mk


def split_pair(m1, m2, c1, c2, heat=None, mode="opt1"):
    union = m1 | m2
    if mode == "opt1":
        h = cv2.resize(heat, (512, 512), interpolation=cv2.INTER_CUBIC)
        elev = (1.0 - h).astype(np.float32)
    else:
        dist = cv2.distanceTransform(union.astype(np.uint8), cv2.DIST_L2, 5)
        elev = (float(dist.max()) - dist).astype(np.float32)
    mk = seed_mask(m1, c1) + 2 * seed_mask(m2, c2)
    elev_img = cv2.cvtColor(((elev / max(float(elev.max()), 1e-6)) * 255).astype(np.uint8),
                            cv2.COLOR_GRAY2BGR)
    ws = cv2.watershed(elev_img, mk)
    a, b = union & (ws == 1), union & (ws == 2)
    for out, orig, c in ((a, m1, c1), (b, m2, c2)):
        x, y = int(c[0]), int(c[1])
        if 0 <= x < 512 and 0 <= y < 512 and orig[y, x] and not out[y, x]:
            out[y, x] = True
    return a, b


def context_select(cands, order):
    chosen = []
    for k in order:
        best, best_cost = cands[k][0], None
        for m in cands[k]:
            cost = 0.0
            for _, other in chosen:
                inter = np.count_nonzero(m & other)
                union = np.count_nonzero(m | other)
                cost += inter / union if union else 0.0
            if best_cost is None or cost < best_cost:
                best, best_cost = m, cost
        chosen.append((k, best))
    return [m for _, m in sorted(chosen)]


def shape_stats(m):
    area = float(m.sum())
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return area, 0.0, 0.0
    c = max(cnts, key=cv2.contourArea)
    per = cv2.arcLength(c, True)
    ha = cv2.contourArea(cv2.convexHull(c))
    return (area, 4 * math.pi * area / (per * per) if per else 0.0,
            area / ha if ha else 0.0)


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
            heat = torch.sigmoid(ph.squeeze()).cpu().numpy()
            predictor.set_image(img)
            props = ns["decode"](ph, pr, thr=0.5, dist_factor=0.7, fixed_r=58.5)
            cands, sam_scores = [], []
            for cx, cy, r, _ in props:
                box = np.array([max(0, cx - 1.15 * r), max(0, cy - 1.15 * r),
                                min(512, cx + 1.15 * r), min(512, cy + 1.15 * r)])
                ms, sc, _ = predictor.predict(box=box, multimask_output=True)
                cands.append(ms)
                sam_scores.append([float(x) for x in sc])
            masks_score = [c[int(np.argmax(sc))] for c, sc in zip(cands, sam_scores)]
            order = sorted(range(len(props)), key=lambda k: -sam_scores[k][int(np.argmax(sam_scores[k]))])
            variants = {"baseline": masks_score, "opt1": masks_score,
                        "opt2": masks_score, "opt3": context_select(cands, order)}

            frame_res = {}
            kept_by_tag = {}
            ref_centres = None
            for tag, masks in variants.items():
                kept, rejected = ns["validate"](props, masks, img, name)
                flags, ambiguas, npares = ns["annotate_confidence"](
                    [p for p, _ in kept], [m for _, m in kept], img)
                pairs = []
                seen = set()
                for amb in ambiguas:
                    ids = []
                    for k, (p, _) in enumerate(kept):
                        c = [round(p[0], 1), round(p[1], 1)]
                        if c in amb["centros"] and k not in seen:
                            ids.append(k)
                    if len(ids) == 2:
                        seen.update(ids)
                        pairs.append(tuple(ids))
                metrics = []
                for i, j in pairs:
                    p1, p2 = kept[i][0], kept[j][0]
                    m1, m2 = kept[i][1], kept[j][1]
                    if tag in ("opt1", "opt2"):
                        m1, m2 = split_pair(m1, m2, p1[:2], p2[:2], heat, tag)
                    inter = np.count_nonzero(m1 & m2)
                    union = np.count_nonzero(m1 | m2)
                    a1, c1, s1 = shape_stats(m1)
                    a2, c2, s2 = shape_stats(m2)
                    metrics.append({
                        "iou": round(inter / union, 3) if union else 0.0,
                        "area_ratio_i": round(a1 / (math.pi * p1[2] ** 2), 2),
                        "area_ratio_j": round(a2 / (math.pi * p2[2] ** 2), 2),
                        "circ_i": round(c1, 2), "circ_j": round(c2, 2),
                        "solid_i": round(s1, 2), "solid_j": round(s2, 2),
                    })
                    ov = img.copy()
                    ov[m1] = (0.55 * ov[m1] + 0.45 * np.array([70, 220, 110])).astype(np.uint8)
                    ov[m2] = (0.55 * ov[m2] + 0.45 * np.array([230, 120, 60])).astype(np.uint8)
                    Image.fromarray(ov).save(os.path.join(OUT, f"{tag}_{name}"))
                kept_by_tag[tag] = (kept, masks, heat)
                if tag == "baseline" and ambiguas:
                    ref_centres = ambiguas[0]["centros"]
                all_ious = []
                for i in range(len(kept)):
                    for j in range(i + 1, len(kept)):
                        a, b = kept[i][1], kept[j][1]
                        inter = np.count_nonzero(a & b)
                        union = np.count_nonzero(a | b)
                        iou = inter / union if union else 0.0
                        if iou > 0.02:
                            all_ious.append({
                                "iou": round(float(iou), 3),
                                "centros": [[round(kept[i][0][0], 1), round(kept[i][0][1], 1)],
                                            [round(kept[j][0][0], 1), round(kept[j][0][1], 1)]],
                            })
                all_ious.sort(key=lambda x: -x["iou"])
                frame_res[tag] = {"aceptadas": len(kept), "pares": npares,
                                  "ambiguos": len(ambiguas), "metrics": metrics,
                                  "max_iou": all_ious[0]["iou"] if all_ious else 0.0,
                                  "n_gt_005": sum(1 for x in all_ious if x["iou"] > 0.05),
                                  "n_gt_015": sum(1 for x in all_ious if x["iou"] > 0.15),
                                  "top_ious": all_ious[:3]}
            # full overlays: every accepted mask, for every variant
            for tag, (kept, masks, heat) in kept_by_tag.items():
                ov = img.copy()
                for p, m in kept:
                    ov[m] = (0.55 * ov[m] + 0.45 * np.array([70, 220, 110])).astype(np.uint8)
                im = Image.fromarray(ov)
                d = ImageDraw.Draw(im)
                for p, m in kept:
                    cx, cy = int(p[0]), int(p[1])
                    d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], outline=(255, 40, 40), width=2)
                im.save(os.path.join(OUT, f"full_{tag}_{name}"))
            # forced comparison overlays: same two centres for all four variants
            if ref_centres:
                for tag, (kept, masks, heat) in kept_by_tag.items():
                    ids = []
                    for k, (p, _) in enumerate(kept):
                        if [round(p[0], 1), round(p[1], 1)] in ref_centres:
                            ids.append(k)
                    if len(ids) != 2:
                        continue
                    i, j = ids
                    m1, m2 = kept[i][1], kept[j][1]
                    if tag in ("opt1", "opt2"):
                        m1, m2 = split_pair(m1, m2, kept[i][0][:2], kept[j][0][:2], heat, tag)
                    ov = img.copy()
                    ov[m1] = (0.55 * ov[m1] + 0.45 * np.array([70, 220, 110])).astype(np.uint8)
                    ov[m2] = (0.55 * ov[m2] + 0.45 * np.array([230, 120, 60])).astype(np.uint8)
                    Image.fromarray(ov).save(os.path.join(OUT, f"cmp_{tag}_{name}"))
            report[name] = frame_res
            print(name, {t: (r["aceptadas"], r["ambiguos"]) for t, r in frame_res.items()},
                  flush=True)

    json.dump(report, open(os.path.join(OUT, "report.json"), "w"), indent=1)
    for tag in ("baseline", "opt1", "opt2", "opt3"):
        ms = [m for r in report.values() for m in r[tag]["metrics"]]
        if not ms:
            print(tag, "sin pares")
            continue
        ratios = [m["area_ratio_i"] for m in ms] + [m["area_ratio_j"] for m in ms]
        print(tag, "pares", len(ms),
              "IoU med", round(float(np.median([m["iou"] for m in ms])), 3),
              "area_ratio med", round(float(np.median(ratios)), 2),
              "circ med", round(float(np.median([m["circ_i"] for m in ms] +
                                                [m["circ_j"] for m in ms])), 2))


if __name__ == "__main__":
    main()
