#!/usr/bin/env python3
"""Build a CVAT-ready package with ONLY new StarDist polygons for a range.

Usage:
  python3 scripts/make_diff_package.py \
    --src data/stardist_villas_restante \
    --fmap data/kaggle-villas-restante/frame_map.json \
    --job 4307835 --prefix villas_restante --first 0 --last 260

Baseline for comparison = current CVAT annotations (GET, read-only).
A polygon is "new" if no existing shape in the same frame has IoU >= 0.5.
"""
import argparse, json, os, re, urllib.request
from collections import Counter

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--fmap", required=True)
ap.add_argument("--job", type=int, required=True)
ap.add_argument("--prefix", required=True)
ap.add_argument("--first", type=int, required=True)
ap.add_argument("--last", type=int, required=True)
a = ap.parse_args()

PKG = os.path.join(ROOT, "data", "para_subir", a.prefix)
os.makedirs(PKG, exist_ok=True)
fmap = json.load(open(os.path.join(ROOT, a.fmap)))
shapes = []
for f in range(a.first, a.last + 1):
    p = os.path.join(ROOT, a.src, f"cvat_{f}_stardist.json")
    if os.path.exists(p):
        shapes.extend(json.load(open(p))["shapes"])
assert shapes, "sin poligonos de entrada"

token = re.search(r'TOKEN\s*=\s*"([^"]+)"',
                  open(os.path.join(ROOT, "experimentos/yolo/scripts/ronda1.py")).read()).group(1)
req = urllib.request.Request(f'https://app.cvat.ai/api/jobs/{a.job}/annotations',
                             headers={"Authorization": f"Token {token}"})
with urllib.request.urlopen(req, timeout=240) as r:
    current = json.loads(r.read().decode("utf-8"))
json.dump(current, open(os.path.join(PKG, "backup_annotations_before_upload.json"), "w"))

baseline = {}
for s in current.get("shapes", []):
    baseline.setdefault(s.get("frame"), []).append(s)

def mask_of(shape):
    m = np.zeros((512, 512), np.uint8)
    pts = np.asarray(shape["points"], dtype=np.int32).reshape(-1, 2)
    cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)

base_masks = {}
for f, ss in baseline.items():
    if a.first <= f <= a.last:
        base_masks[f] = [mask_of(s) for s in ss]

new_only = []
per_frame = {}
for f in range(a.first, a.last + 1):
    cand = [s for s in shapes if s["frame"] == f]
    kept = []
    for s in cand:
        m = mask_of(s)
        best = 0.0
        for bm in base_masks.get(f, []):
            u = np.count_nonzero(m | bm)
            if u:
                best = max(best, np.count_nonzero(m & bm) / u)
        if best < 0.5:
            kept.append(s)
    new_only.extend(kept)
    per_frame[str(f)] = {"image": fmap.get(str(f), ""),
                         "n_pred": len(cand), "n_new": len(kept)}

json.dump({"version": 2.0, "tags": [], "shapes": new_only,
           "tracks": [], "intervals": []},
          open(os.path.join(PKG, "annotations.json"), "w"), indent=1)
manifest = {
    "status": "PREPARED — no CVAT write performed",
    "pipeline": "StarDist2D prob=0.30 NMS=0.02 — diferencia contra shapes actuales en CVAT",
    "criterion": "IoU < 0.5 contra cualquier shape del mismo frame",
    "job": a.job, "frame_start": a.first, "frame_end": a.last,
    "n_pred": len(shapes), "n_new": len(new_only),
    "warning": "Umbral 0.30 permisivo: puede incluir arboles que no son del cultivo",
    "upload_auth": "REQUIRES explicit user authorisation per batch; no CVAT write performed",
    "per_frame": per_frame,
}
json.dump(manifest, open(os.path.join(PKG, "manifest.json"), "w"), indent=1)
print(f"paquete={PKG} pred={len(shapes)} nuevos={len(new_only)} frames={a.first}-{a.last}")
print("conteo por frame:", {k: v["n_new"] for k, v in list(per_frame.items())[:8]})
