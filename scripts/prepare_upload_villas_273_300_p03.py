#!/usr/bin/env python3
"""Build the CVAT-ready package for Villas 273-300 at prob_thresh=0.30.

Preparation only: no CVAT writes. Expected more permissive detections.
"""
import glob, json, os
from collections import Counter

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "stardist_villas_273_300_sweep", "0.3000")
PKG = os.path.join(ROOT, "data", "para_subir", "villas_273_300_p03")
FMAP = os.path.join(ROOT, "data", "kaggle-villas-273-300", "frame_map.json")
JOB = 4307835
FRAMES = list(range(273, 301))

os.makedirs(PKG, exist_ok=True)
fmap = json.load(open(FMAP))
shapes = []
for f in FRAMES:
    shapes.extend(json.load(open(os.path.join(SRC, f"cvat_{f}_stardist.json")))["shapes"])
by_frame = Counter(s["frame"] for s in shapes)

json.dump({"version": 2.0, "tags": [], "shapes": shapes, "tracks": [], "intervals": []},
          open(os.path.join(PKG, "annotations.json"), "w"), indent=1)

manifest = {
    "status": "PREPARED — no CVAT write performed",
    "pipeline": "StarDist2D (pesos limon-stardist-v1) + NMS 0.02 + prob_thresh=0.30",
    "model_source": "dataset marcosalvarez234/limon-stardist-v1-weights",
    "images_source": "dataset marcosalvarez234/villas-273-300 (frame_map de meta.json)",
    "cvat": {"job_id": JOB, "label_id": 8155070, "label": "crop_tree",
             "frame_start": 273, "frame_end": 300},
    "n_frames": len(FRAMES), "n_poligonos": len(shapes),
    "prob_thresh": 0.30, "nms": 0.02,
    "warning": "Umbral permisivo: puede incluir arboles que no son del cultivo. "
               "312 poligonos vs 224 con el umbral por defecto 0.5929.",
    "per_frame": {str(f): {"image": fmap[str(f)], "n_poligonos": by_frame[f]}
                  for f in FRAMES},
    "upload_auth": "REQUIRES explicit user authorisation per batch; no CVAT write performed",
    "files": ["annotations.json", "manifest.json", "overlay_contact.png",
              "backup_annotations_before_upload.json"],
}
json.dump(manifest, open(os.path.join(PKG, "manifest.json"), "w"), indent=1)

# Overlays locales: dibujamos los poligonos del JSON sobre las imagenes originales
imgs = []
for f in FRAMES:
    name = fmap[str(f)]
    path = os.path.join(ROOT, "data", "r1", "images", "villas", name)
    if not os.path.exists(path):
        continue
    vis = np.asarray(Image.open(path).convert("RGB")).copy()
    fs = [s for s in shapes if s["frame"] == f]
    for s in fs:
        pts = np.asarray(s["points"], dtype=np.int32).reshape(-1, 2)
        cv2.polylines(vis, [pts], True, (255, 80, 80), 1)
    out = os.path.join(PKG, f"overlay_{f}.png")
    Image.fromarray(vis).save(out)
    imgs.append((f, out))

cols, rows = 7, (len(imgs) + 6) // 7
fig, axs = plt.subplots(rows, cols, figsize=(19, 2.9 * rows))
for ax in axs.flat:
    ax.axis("off")
for ax, (f, p) in zip(axs.flat, imgs):
    ax.imshow(Image.open(p))
    ax.set_title(f"frame {f} · {by_frame[f]} pol.", fontsize=8)
    ax.axis("off")
fig.suptitle("Villas 273-300 — StarDist prob=0.30 (312 polígonos)", fontsize=13, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.99])
fig.savefig(os.path.join(PKG, "overlay_contact.png"), dpi=105)
plt.close(fig)

print(f"frames={len(FRAMES)} poligonos={len(shapes)} overlays={len(imgs)}")
