#!/usr/bin/env python3
"""Build the CVAT-ready package for Magdala frames 240-300. Preparation only."""
import json, os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "stardist_magdala_240_300")
PKG = os.path.join(ROOT, "data", "para_subir", "magdala_240_300")
JOB = 4307818
LABEL_ID = 8155070
FRAMES = list(range(240, 301))

os.makedirs(PKG, exist_ok=True)
rep = json.load(open(os.path.join(SRC, "stardist_resto_report.json")))
shapes = []
for f in FRAMES:
    shapes.extend(json.load(open(os.path.join(SRC, f"cvat_{f}_stardist.json")))["shapes"])
by_frame = Counter(s["frame"] for s in shapes)

json.dump({"version": 2.0, "tags": [], "shapes": shapes, "tracks": [], "intervals": []},
          open(os.path.join(PKG, "annotations.json"), "w"), indent=1)

manifest = {
    "status": "PREPARED — no CVAT write performed",
    "pipeline": "StarDist2D (pesos limon-stardist-v1) + NMS 0.02, kernel marcosalvarez234/limon-stardist-magdala-240-300",
    "model_source": "dataset marcosalvarez234/limon-stardist-v1-weights",
    "images_source": "dataset marcosalvarez234/magdala-240-300 (mapeo real de data/raw/magdala/meta.json)",
    "cvat": {"job_id": JOB, "label_id": LABEL_ID, "label": "crop_tree",
             "frame_start": 240, "frame_end": 300},
    "n_frames": len(FRAMES), "n_poligonos": len(shapes),
    "prob_thresh": 0.5928899049758911, "nms_export": 0.02,
    "per_frame": {str(f): {"image": rep[str(f)]["image"],
                           "n_poligonos": by_frame[f]} for f in FRAMES},
    "upload_auth": "REQUIRES explicit user authorisation per batch; no CVAT write performed",
    "files": ["annotations.json", "manifest.json", "overlay_contact.png",
              "backup_annotations_before_upload.json"],
}
json.dump(manifest, open(os.path.join(PKG, "manifest.json"), "w"), indent=1)

imgs = []
for f in FRAMES:
    p = os.path.join(SRC, f"nms0.02_{rep[str(f)]['image']}")
    if os.path.exists(p):
        imgs.append((f, rep[str(f)]["image"], p))
cols, rows = 8, (len(imgs) + 7) // 8
fig, axs = plt.subplots(rows, cols, figsize=(20, 2.7 * rows))
for ax in axs.flat:
    ax.axis("off")
for ax, (f, name, p) in zip(axs.flat, imgs):
    ax.imshow(Image.open(p))
    ax.set_title(f"{f} · {name[-9:-4]}", fontsize=7)
    ax.axis("off")
fig.suptitle(f"StarDist Magdala NMS 0.02 — frames 240-300 ({len(shapes)} poligonos)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(os.path.join(PKG, "overlay_contact.png"), dpi=110)
plt.close(fig)
print(f"frames={len(FRAMES)} poligonos={len(shapes)} overlays={len(imgs)}")
