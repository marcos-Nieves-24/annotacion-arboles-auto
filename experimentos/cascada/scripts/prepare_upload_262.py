#!/usr/bin/env python3
"""Prepare LOCAL pre-annotation package for CVAT Villas frame 262.

Uses the adopted cascade v4 (v1.2 validator + ambiguity flag), not StarDist.
Writes only under data/para_subir/villas_262/. NO CVAT writes.
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
from PIL import Image, ImageDraw
from segment_anything import sam_model_registry, SamPredictor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "para_subir", "villas_262")
os.makedirs(OUT, exist_ok=True)
JOB_ID = 4307835
FRAME = 262
LABEL_ID = 8155070
LABEL = "crop_tree"
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
    ns = dict(torch=torch, nn=__import__('torch.nn', fromlist=['nn']),
              F=__import__('torch.nn.functional', fromlist=['F']),
              np=np, cv2=cv2, models=__import__('torchvision.models', fromlist=['models']),
              random=random, math=math, os=os, Image=Image)
    exec(compile(ast.Module(body=keep, type_ignores=[]), "<k>", "exec"), ns)
    return ns


def main():
    ns = load_ns()
    meta = json.load(open(os.path.join(ROOT, "data", "raw", "villas", "meta.json")))
    name = meta["frames"][FRAME]["name"]
    img_path = os.path.join(ROOT, "data", "r1", "images", "villas", name)
    img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)

    net = ns["CenterNetLite"]()
    net.load_state_dict(torch.load(os.path.join(ROOT, "data", "centroides", "v1",
                                                "model.pt"), map_location="cpu",
                                   weights_only=True))
    net.eval()
    sam = sam_model_registry["vit_b"](checkpoint=os.path.join(
        ROOT, "data", "centroides", "cascade", "sam_vit_b_01ec64.pth"))
    predictor = SamPredictor(sam)

    with torch.no_grad():
        ph, pr = net(torch.from_numpy(img.astype(np.float32) / 255.).permute(2, 0, 1)[None])
        props = ns["decode"](ph, pr, thr=0.5, dist_factor=0.7, fixed_r=58.5)
        predictor.set_image(img)
        masks = []
        for cx, cy, r, sc in props:
            box = np.array([max(0, cx - 1.15 * r), max(0, cy - 1.15 * r),
                            min(512, cx + 1.15 * r), min(512, cy + 1.15 * r)])
            ms, sc_sam, _ = predictor.predict(box=box, multimask_output=True)
            masks.append(ms[int(np.argmax(sc_sam))])

    kept, rejected = ns["validate"](props, masks, img, name)
    flags, ambiguas, npares = ns["annotate_confidence"](
        [p for p, _ in kept], [m for _, m in kept], img)

    shapes = []
    low_conf = []
    for k, (prop, m) in enumerate(kept):
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        poly = cv2.approxPolyDP(cnt, 0.005 * cv2.arcLength(cnt, True), True).ravel()
        if len(poly) < 6:
            continue
        shapes.append({
            "type": "polygon", "label_id": LABEL_ID, "label": LABEL,
            "frame": FRAME, "group": 0, "source": "auto", "score": float(prop[3]),
            "points": [round(float(v), 2) for v in poly],
            "occluded": False, "outside": False, "z_order": 0,
        })
        if flags[k] == "baja":
            low_conf.append({"center": [round(prop[0], 1), round(prop[1], 1)],
                             "r": round(prop[2], 1)})

    cvat = {"version": 2.0, "tags": [], "shapes": shapes,
            "tracks": [], "intervals": []}
    json.dump(cvat, open(os.path.join(OUT, "annotations.json"), "w"), indent=1)

    im = Image.fromarray(img.copy())
    d = ImageDraw.Draw(im)
    for s in shapes:
        pts = np.asarray(s["points"], dtype=np.int32).reshape(-1, 2)
        d.line([tuple(pts)] + [tuple(pts[0])], fill=(0, 255, 0), width=2)
    for x in low_conf:
        cx, cy = x["center"]
        d.ellipse([cx - 12, cy - 12, cx + 12, cy + 12], outline=(0, 200, 255), width=3)
    im.save(os.path.join(OUT, "overlay.png"))

    manifest = {
        "status": "PREPARED_NOT_UPLOADED",
        "pipeline": "cascade v4 (v1.2 validator + ambiguity flag)",
        "model_sha256_short": "ec2bc282e40bd94a",
        "cvat": {"job_id": JOB_ID, "frame": FRAME, "image": name,
                 "label_id": LABEL_ID, "label": LABEL},
        "propuestas": len(props), "aceptadas": len(kept),
        "rechazadas": len(rejected),
        "motivos_rechazo": {r: sum(1 for x in rejected if x["reason"] == r)
                            for r in sorted({x["reason"] for x in rejected})},
        "pares_que_se_tocan": npares, "pares_ambiguos": len(ambiguas),
        "baja_confianza": low_conf,
        "files": ["annotations.json", "overlay.png", "manifest.json"],
        "upload_auth": "REQUIRES explicit user authorisation; no CVAT write performed",
    }
    json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
