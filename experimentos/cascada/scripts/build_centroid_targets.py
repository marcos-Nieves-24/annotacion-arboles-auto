#!/usr/bin/env python3
"""Build centroid/radius targets from gold COCO polygons (Frente B, E0).

The ``bbox`` field in gold_v2 is zero for every annotation, so centroids are
computed with the shoelace formula on the segmentation polygon.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def _as_ring(poly):
    if not poly:
        return []
    if isinstance(poly[0], (int, float)):
        return poly
    return poly[0]


def polygon_centroid(poly):
    """Area-weighted centroid of a COCO polygon (shoelace)."""
    ring = _as_ring(poly)
    pts = [(float(ring[i]), float(ring[i + 1])) for i in range(0, len(ring) - 1, 2)]
    if not pts:
        return (0.0, 0.0)
    area2 = cx = cy = 0.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        cross = x1 * y2 - x2 * y1
        area2 += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    if abs(area2) < 1e-9:
        return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
    return (cx / (3.0 * area2), cy / (3.0 * area2))


def equivalent_radius(area):
    """Radius of a circle with the same area as the crown mask."""
    return math.sqrt(max(float(area), 0.0) / math.pi)


def _touches_border(ring, size):
    pts = [(float(ring[i]), float(ring[i + 1])) for i in range(0, len(ring) - 1, 2)]
    return any(x <= 0 or x >= size or y <= 0 or y >= size for x, y in pts)


def build_targets(anns, size=512, stride=4, sigma_frac=0.25,
                  sigma_min=2.0, sigma_max=16.0):
    """Return (heatmap, radius, weight, centers) for one 512x512 tile.

    Heatmaps are stride/4-resolution Gaussian blobs combined by element-wise
    maximum; radius is supervised only at the centre pixel; clipped crowns get
    weight 0.5 on the radius loss.
    """
    h = size // stride
    heat = np.zeros((h, h), dtype=np.float32)
    radius = np.zeros((h, h), dtype=np.float32)
    weight = np.zeros((h, h), dtype=np.float32)
    centers = []
    grid_y, grid_x = np.indices((h, h), dtype=np.float32)

    for ann in anns:
        ring = _as_ring(ann.get("segmentation", []))
        if not ring:
            continue
        cx, cy = polygon_centroid(ring)
        area = ann.get("area") or 0.0
        r = equivalent_radius(area)
        sigma_full = min(max(sigma_frac * r, sigma_min), sigma_max)
        sigma_h = sigma_full / stride
        d2 = (grid_x - cx / stride) ** 2 + (grid_y - cy / stride) ** 2
        core = np.exp(-d2 / (2.0 * sigma_h ** 2)) >= 0.5
        w = 0.5 if _touches_border(ring, size) else 1.0
        heat[core] = 1.0
        radius[core] = r
        weight[core] = w
        centers.append((round(cx, 2), round(cy, 2), round(r, 2), w))

    return heat, radius, weight, centers


def visualize(image_path, centers, out_path):
    """Draw crown circles over the original tile for E0 visual inspection."""
    from PIL import Image, ImageDraw

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for cx, cy, r, w in centers:
        color = (255, 60, 60) if w == 0.5 else (60, 220, 120)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=2)
        draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=color)
    img.save(out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coco", default="data/gold_v2/coco.json")
    ap.add_argument("--images", default="data/gold_v1/images")
    ap.add_argument("--out", default="data/centroides")
    ap.add_argument("--limit", type=int, default=0, help="0 = all images")
    ap.add_argument("--viz", action="store_true", help="draw 20 frames")
    args = ap.parse_args()

    coco = json.loads(Path(args.coco).read_text())
    by_image = defaultdict(list)
    for ann in coco["annotations"]:
        by_image[ann["image_id"]].append(ann)

    out = Path(args.out)
    (out / "targets").mkdir(parents=True, exist_ok=True)
    if args.viz:
        (out / "viz").mkdir(parents=True, exist_ok=True)

    images = coco["images"][:args.limit] if args.limit else coco["images"]
    summary = []
    for im in images:
        heat, radius, weight, centers = build_targets(by_image.get(im["id"], []))
        arr = np.asarray(centers, dtype=np.float32).reshape(-1, 4)
        np.savez_compressed(
            out / "targets" / f"{im['id']}.npz",
            heat=heat, radius=radius, weight=weight, centers=arr,
        )
        summary.append({
            "image_id": im["id"],
            "file_name": im["file_name"],
            "n_centers": len(centers),
        })

    (out / "centros_summary.json").write_text(json.dumps(summary, indent=2))

    if args.viz:
        for im in images[:20]:
            src = Path(args.images) / im["file_name"]
            if src.exists():
                _, _, _, centers = build_targets(by_image.get(im["id"], []))
                visualize(src, centers, out / "viz" / im["file_name"])

    total = sum(s["n_centers"] for s in summary)
    print(f"targets={len(summary)} centers={total} out={out}")


if __name__ == "__main__":
    main()
