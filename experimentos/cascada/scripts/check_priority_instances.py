#!/usr/bin/env python3
"""Recolour each priority-painted mask with a different colour and audit instances."""
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_split_261_270 import load_ns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "centroides", "priority_261_270_colores")
os.makedirs(OUT, exist_ok=True)
PALETTE = [(255, 60, 60), (60, 160, 255), (255, 200, 40), (130, 255, 90),
           (220, 100, 255), (255, 130, 60), (60, 230, 200), (255, 90, 160),
           (150, 150, 255), (200, 255, 60), (255, 170, 200), (90, 200, 130),
           (180, 120, 255), (255, 220, 120), (100, 255, 180), (240, 120, 120)]


def main():
    ns = load_ns()
    net = ns["CenterNetLite"]()
    net.load_state_dict(torch.load(os.path.join(ROOT, "data", "centroides", "v1",
                                                "model.pt"), map_location="cpu",
                                   weights_only=True))
    net.eval()
    from segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry["vit_b"](checkpoint=os.path.join(
        ROOT, "data", "centroides", "cascade", "sam_vit_b_01ec64.pth"))
    pred = SamPredictor(sam)

    meta = json.load(open(os.path.join(ROOT, "data", "raw", "villas", "meta.json")))
    frames = [meta["frames"][i]["name"] for i in range(261, 271)]
    idx = {}
    for p in glob.glob(os.path.join(ROOT, "data", "r1", "images", "villas", "*.png")):
        idx.setdefault(os.path.basename(p), p)
    audit = {}
    with torch.no_grad():
        for name in frames:
            img = np.asarray(Image.open(idx[name]).convert("RGB"), dtype=np.uint8)
            ph, pr = net(torch.from_numpy(img.astype(np.float32) / 255.).permute(2, 0, 1)[None])
            pred.set_image(img)
            props = ns["decode"](ph, pr, thr=0.5, dist_factor=0.7, fixed_r=58.5)
            masks, scores = [], []
            for cx, cy, r, _ in props:
                box = np.array([max(0, cx - 1.15 * r), max(0, cy - 1.15 * r),
                                min(512, cx + 1.15 * r), min(512, cy + 1.15 * r)])
                ms, sc, _ = pred.predict(box=box, multimask_output=True)
                k = int(np.argmax(sc))
                masks.append(ms[k]); scores.append(float(sc[k]))
            kept, _ = ns["validate"](props, masks, img, name)
            by_centre = {(round(p[0], 1), round(p[1], 1)): scores[i]
                         for i, p in enumerate(props)}
            items = [{"prop": p, "mask": m,
                      "score": by_centre.get((round(p[0], 1), round(p[1], 1)), float(p[3]))}
                     for p, m in kept]
            items.sort(key=lambda x: -x["score"])
            claimed = np.zeros((512, 512), bool)
            painted = []
            for it in items:
                pm = it["mask"] & ~claimed
                claimed |= it["mask"]
                painted.append((it["prop"], pm))

            n_inst = len(painted)
            union = claimed.astype(np.uint8)
            n_comp, labels = cv2.connectedComponents(union, 8)
            n_comp -= 1  # background
            multi = 0
            for lab in range(1, n_comp + 1):
                c = sum(1 for p, m in painted if m[labels == lab].any())
                if c > 1:
                    multi += 1
            centroid_lost = sum(1 for p, m in painted
                                if not (0 <= int(p[1]) < 512 and 0 <= int(p[0]) < 512
                                        and m[int(p[1]), int(p[0])]))
            vis = img.copy()
            for k, (p, m) in enumerate(painted):
                col = PALETTE[k % len(PALETTE)]
                vis[m] = (0.45 * vis[m] + 0.55 * np.array(col)).astype(np.uint8)
                cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cnts, -1, col, 1)
            im = Image.fromarray(vis)
            d = ImageDraw.Draw(im)
            for p, _ in painted:
                cx, cy = int(p[0]), int(p[1])
                d.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(255, 255, 255))
            im.save(os.path.join(OUT, name))
            audit[name] = {"instancias": n_inst, "componentes_conexos_union": n_comp,
                           "componentes_con_varias_instancias": multi,
                           "centroides_fuera_de_su_mascara": centroid_lost}
            print(name, audit[name], flush=True)
    json.dump(audit, open(os.path.join(OUT, "audit.json"), "w"), indent=1)
    files = sorted(f for f in os.listdir(OUT) if f.startswith("Limon_") and f.endswith(".png"))
    tw, th, lab, cols = 400, 400, 22, 2
    sheet = Image.new("RGB", (cols * tw, ((len(files) + 1) // 2) * (th + lab)), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for i, f in enumerate(files):
        im = Image.open(os.path.join(OUT, f)).convert("RGB").resize((tw, th))
        x, y = (i % 2) * tw, (i // 2) * (th + lab)
        sheet.paste(im, (x, y)); d.text((x + 3, y + th + 2), f.replace(".png", ""), fill=(0, 0, 0))
    sheet.save(os.path.join(OUT, "contact.png"))


if __name__ == "__main__":
    main()
