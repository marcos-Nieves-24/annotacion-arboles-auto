#!/usr/bin/env python3
"""Build the CVAT-ready package for frames 263-300 from the StarDist run.

Preparation only: no CVAT requests, no writes to the job.
"""
import json, os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "stardist_262_300")
PKG = os.path.join(ROOT, "data", "para_subir", "villas_263_300")
JOB = 4307835
LABEL_ID = 8155070

os.makedirs(PKG, exist_ok=True)

combined = json.load(open(os.path.join(SRC, "cvat_262_300_stardist.json")))
run_manifest = json.load(open(os.path.join(SRC, "manifest_262_300.json")))
shapes = [s for s in combined["shapes"] if 263 <= s["frame"] <= 300]
frames = sorted({s["frame"] for s in shapes})
missing = [f for f in range(263, 301) if f not in set(frames)]
by_frame = Counter(s["frame"] for s in shapes)

ann = {"version": 2.0, "tags": [], "shapes": shapes, "tracks": [], "intervals": []}
json.dump(ann, open(os.path.join(PKG, "annotations.json"), "w"), indent=1)

per_frame = {}
pf = run_manifest.get("per_frame", {})
for f in frames:
    info = pf.get(str(f), {})
    per_frame[str(f)] = {
        "image": info.get("image", ""),
        "n_poligonos": by_frame[f],
    }

manifest = {
    "status": "PREPARED — no CVAT write performed",
    "pipeline": "StarDist2D (pesos limon-stardist-v1) + NMS 0.02, kernel marcosalvarez234/limon-stardist-262-300",
    "model_source": "dataset marcosalvarez234/limon-stardist-v1-weights",
    "images_source": "dataset marcosalvarez234/villas-262-300",
    "cvat": {"job_id": JOB, "label_id": LABEL_ID, "label": "crop_tree",
             "frame_start": 263, "frame_end": 300},
    "n_frames": len(frames),
    "n_poligonos": len(shapes),
    "frames": frames,
    "missing_frames": missing,
    "missing_reason": "imagenes 00790-00804 no disponibles",
    "prob_thresh": run_manifest.get("prob_thresh"),
    "nms_export": 0.02,
    "upload_auth": "REQUIRES explicit user authorisation per batch; no CVAT write performed",
    "files": ["annotations.json", "manifest.json", "overlay_contact.png",
              "backup_annotations_before_upload.json"],
    "per_frame": per_frame,
}
json.dump(manifest, open(os.path.join(PKG, "manifest.json"), "w"), indent=1)

# Contact sheet with the 23 overlays used for the export (NMS 0.02)
imgs = []
for f in frames:
    name = per_frame[str(f)]["image"]
    p = os.path.join(SRC, f"nms0.02_{name}")
    if os.path.exists(p):
        imgs.append((f, name, p))
cols, rows = 6, (len(imgs) + 5) // 6
fig, axs = plt.subplots(rows, cols, figsize=(18, 3.1 * rows))
for ax in axs.flat:
    ax.axis("off")
for ax, (f, name, p) in zip(axs.flat, imgs):
    ax.imshow(Image.open(p))
    ax.set_title(f"frame {f}\n{name}", fontsize=8)
    ax.axis("off")
fig.suptitle(f"StarDist NMS 0.02 — frames 263-300 ({len(shapes)} poligonos)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(os.path.join(PKG, "overlay_contact.png"), dpi=110)
plt.close(fig)

print(f"paquete: {PKG}")
print(f"frames={len(frames)} poligonos={len(shapes)} faltan={missing}")
print(f"overlays en laminas={len(imgs)}")
