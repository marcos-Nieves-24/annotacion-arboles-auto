#!/usr/bin/env python3
"""Prepare ONLY the polygons that appear at prob=0.30 but not at 0.5929.

Baseline sources:
  273-279 -> data/para_subir/villas_263_300/annotations.json
  280-300 -> data/stardist_resto/cvat_<frame>_stardist.json

Matching: rasterized polygon IoU >= 0.5 within the same frame.
Preparation only: no CVAT writes.
"""
import json, os
from collections import Counter

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P30 = os.path.join(ROOT, "data", "stardist_villas_273_300_sweep", "0.3000")
BASE_PKG = os.path.join(ROOT, "data", "para_subir", "villas_263_300", "annotations.json")
BASE_RESTO = os.path.join(ROOT, "data", "stardist_resto")
FMAP = os.path.join(ROOT, "data", "kaggle-villas-273-300", "frame_map.json")
PKG = os.path.join(ROOT, "data", "para_subir", "villas_273_300_p03_diff")
JOB = 4307835
FRAMES = list(range(273, 301))
os.makedirs(PKG, exist_ok=True)

fmap = json.load(open(FMAP))
base_all = json.load(open(BASE_PKG))["shapes"]
base = {f: [s for s in base_all if s["frame"] == f] for f in range(273, 280)}
for f in range(280, 301):
    base[f] = json.load(open(os.path.join(BASE_RESTO, f"cvat_{f}_stardist.json")))["shapes"]

new_all = []
for f in FRAMES:
    new_all.extend(json.load(open(os.path.join(P30, f"cvat_{f}_stardist.json")))["shapes"])


def mask_of(shape):
    m = np.zeros((512, 512), np.uint8)
    pts = np.asarray(shape["points"], dtype=np.int32).reshape(-1, 2)
    cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


base_masks = {f: [mask_of(s) for s in base[f]] for f in FRAMES}
new_only = []
per_frame = {}
for f in FRAMES:
    cand = [s for s in new_all if s["frame"] == f]
    kept = []
    for s in cand:
        m = mask_of(s)
        best = 0.0
        for bm in base_masks[f]:
            u = np.count_nonzero(m | bm)
            if u:
                best = max(best, np.count_nonzero(m & bm) / u)
        if best < 0.5:
            kept.append(s)
    new_only.extend(kept)
    per_frame[str(f)] = {"image": fmap[str(f)],
                         "n_p30": len(cand),
                         "n_baseline": len(base[f]),
                         "n_new": len(kept)}
    print(f"frame {f}: baseline {len(base[f])} | 0.30 {len(cand)} | nuevos {len(kept)}")

json.dump({"version": 2.0, "tags": [], "shapes": new_only, "tracks": [], "intervals": []},
          open(os.path.join(PKG, "annotations.json"), "w"), indent=1)

manifest = {
    "status": "PREPARED — no CVAT write performed",
    "pipeline": "StarDist2D + NMS 0.02 — DIFERENCIA prob 0.30 vs 0.5929",
    "criterion": "poligonos de 0.30 sin match IoU>=0.5 en el baseline del mismo frame",
    "baseline_sources": ["data/para_subir/villas_263_300/annotations.json (273-279)",
                         "data/stardist_resto/ (280-300)"],
    "cvat": {"job_id": JOB, "label_id": 8155070, "label": "crop_tree",
             "frame_start": 273, "frame_end": 300},
    "n_poligonos_p30": len(new_all), "n_poligonos_baseline": sum(len(base[f]) for f in FRAMES),
    "n_poligonos_nuevos": len(new_only), "n_frames": len(FRAMES),
    "warning": "Umbral 0.30 permisivo: puede incluir arboles que no son del cultivo.",
    "per_frame": per_frame,
    "upload_auth": "REQUIRES explicit user authorisation per batch; no CVAT write performed",
    "files": ["annotations.json", "manifest.json", "overlay_contact.png",
              "backup_annotations_before_upload.json"],
}
json.dump(manifest, open(os.path.join(PKG, "manifest.json"), "w"), indent=1)

# Overlays: solo los nuevos, en verde sobre la imagen
imgs = []
for f in FRAMES:
    name = fmap[str(f)]
    path = os.path.join(ROOT, "data", "r1", "images", "villas", name)
    if not os.path.exists(path):
        continue
    vis = np.asarray(Image.open(path).convert("RGB")).copy()
    fs = [s for s in new_only if s["frame"] == f]
    for s in fs:
        pts = np.asarray(s["points"], dtype=np.int32).reshape(-1, 2)
        cv2.polylines(vis, [pts], True, (60, 255, 90), 2)
    out = os.path.join(PKG, f"overlay_{f}.png")
    Image.fromarray(vis).save(out)
    imgs.append((f, out))

cols, rows = 7, (len(imgs) + 6) // 7
fig, axs = plt.subplots(rows, cols, figsize=(19, 2.9 * rows))
for ax in axs.flat:
    ax.axis("off")
for ax, (f, p) in zip(axs.flat, imgs):
    ax.imshow(Image.open(p))
    ax.set_title(f"frame {f} · {per_frame[str(f)]['n_new']} nuevos", fontsize=8)
    ax.axis("off")
fig.suptitle(f"Diferencia prob 0.30 vs 0.5929 — {len(new_only)} poligonos nuevos (verde)",
             fontsize=13, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.99])
fig.savefig(os.path.join(PKG, "overlay_contact.png"), dpi=105)
plt.close(fig)
print("TOTALES: baseline", sum(len(base[f]) for f in FRAMES))
print("p30", len(new_all), "NUEVOS", len(new_only))
