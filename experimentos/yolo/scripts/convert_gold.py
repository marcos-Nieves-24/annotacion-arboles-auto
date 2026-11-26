"""Fase 0: CVAT gold ranges -> COCO + YOLO-seg (training-grade GT v1.0).

Reads data/raw/{villas,magdala}/{annotations.json, meta.json}, keeps only
gold frames (Villas 0-199, Magdala 0-130), applies GSD metric area filters,
simplifies polygons with Douglas-Peucker, flags border-truncated crowns as
iscrowd (COCO) and skips them in YOLO-seg output.

Outputs under data/gold_v1/:
  coco.json, yolo/images symlinks? no - yolo/labels/*.txt, splits.json,
  quarantine.json, REPORT.md
"""

import json
import math
import os

ROOT = "/home/biohacker24/annotacion-arboles-auto"
GOLD = os.path.join(ROOT, "data", "gold_v1")
RAW = os.path.join(ROOT, "data", "raw")

FARMS = {
    "villas": {"gsd_cm": 3.58, "frames": range(0, 200), "area_min": 882, "area_max": 39224},
    "magdala": {"gsd_cm": 5.25, "frames": range(0, 131), "area_min": 410, "area_max": 18240},
}

W = H = 512
BORDER_TOL = 2.0
BORDER_MIN_VERTICES = 3  # iscrowd only if >=3 vertices on the tile edge;
# exactly 2 usually means a slight clip (enters/exits), still trainable
DP_EPSILON = 1.5  # px, keeps shape while cutting SAM2 vertex noise


def poly_area(pts):
    a = 0.0
    n = len(pts) // 2
    for i in range(n):
        x1, y1 = pts[2 * i], pts[2 * i + 1]
        x2, y2 = pts[2 * ((i + 1) % n)], pts[2 * ((i + 1) % n) + 1]
        a += x1 * y2 - x2 * y1
    return abs(a / 2.0)


def perp_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def douglas_peucker(points, eps):
    """points: list of (x, y). Returns simplified list (open polyline)."""
    if len(points) < 3:
        return points
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        a, b = stack.pop()
        dmax, idx = 0.0, -1
        for i in range(a + 1, b):
            d = perp_dist(points[i][0], points[i][1],
                          points[a][0], points[a][1],
                          points[b][0], points[b][1])
            if d > dmax:
                dmax, idx = d, i
        if dmax > eps:
            keep[idx] = True
            stack.append((a, idx))
            stack.append((idx, b))
    return [p for p, k in zip(points, keep) if k]


def simplify_closed(flat, eps):
    n = len(flat) // 2
    pts = [(flat[2 * i], flat[2 * i + 1]) for i in range(n)]
    if pts[0] != pts[-1]:
        pts = pts + [pts[0]]
    simp = douglas_peucker(pts, eps)
    if simp[0] != simp[-1]:
        simp.append(simp[0])
    # drop duplicated closing point for flat output
    closed = simp[:-1]
    flat_out = [c for p in closed for c in p]
    return flat_out, len(closed)


def touches_border(flat, w=W, h=H, tol=BORDER_TOL):
    n = sum(1 for i in range(0, len(flat), 2)
            if flat[i] <= tol or flat[i + 1] <= tol
            or flat[i] >= w - tol or flat[i + 1] >= h - tol)
    return n >= BORDER_MIN_VERTICES


def main():
    os.makedirs(os.path.join(GOLD, "yolo", "labels"), exist_ok=True)
    images, annotations = [], []
    quarantine = []
    yolo_counts = {"kept": 0, "skipped_iscrowd": 0}
    vtx_before, vtx_after = [], []
    img_id = anno_id = 0

    for farm, cfg in FARMS.items():
        ann = json.load(open(os.path.join(RAW, farm, "annotations.json")))
        meta = json.load(open(os.path.join(RAW, farm, "meta.json")))
        names = [f["name"] for f in meta["frames"]]
        by_frame = {}
        for s in ann.get("shapes", []):
            if s.get("type") != "polygon":
                continue
            by_frame.setdefault(s.get("frame"), []).append(s)

        for f in cfg["frames"]:
            fname = names[f]
            images.append({"id": img_id, "file_name": fname,
                           "width": W, "height": H, "farm": farm, "frame": f})
            label_lines = []
            for s in by_frame.get(f, []):
                pts = s.get("points", [])
                nv0 = len(pts) // 2
                if nv0 < 3:
                    quarantine.append({"farm": farm, "frame": f, "shape_id": s.get("id"),
                                       "reason": "fewer_than_3_points"})
                    continue
                area0 = poly_area(pts)
                simp, nv1 = simplify_closed(pts, DP_EPSILON)
                if nv1 < 3:
                    quarantine.append({"farm": farm, "frame": f, "shape_id": s.get("id"),
                                       "reason": "collapsed_after_simplify"})
                    continue
                area = poly_area(simp)
                vtx_before.append(nv0)
                vtx_after.append(nv1)
                xs = simp[0::2]
                ys = simp[1::2]
                x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
                iscrowd = 1 if touches_border(simp) else 0
                if not (cfg["area_min"] <= area <= cfg["area_max"]):
                    quarantine.append({"farm": farm, "frame": f, "shape_id": s.get("id"),
                                       "reason": "area_out_of_range",
                                       "area_px": round(area, 1)})
                    continue
                annotations.append({"id": anno_id, "image_id": img_id,
                                    "category_id": 1, "segmentation": [simp],
                                    "area": area, "bbox": [x0, y0, x1 - x0, y1 - y0],
                                    "iscrowd": iscrowd})
                anno_id += 1
                if iscrowd:
                    yolo_counts["skipped_iscrowd"] += 1
                else:
                    norm = []
                    for i in range(0, len(simp), 2):
                        norm.append(f"{simp[i] / W:.6f} {simp[i + 1] / H:.6f}")
                    label_lines.append("0 " + " ".join(norm))
                    yolo_counts["kept"] += 1
            with open(os.path.join(GOLD, "yolo", "labels", fname.rsplit(".", 1)[0] + ".txt"), "w") as fh:
                fh.write("\n".join(label_lines))
                if label_lines:
                    fh.write("\n")
            img_id += 1

    coco = {"images": images,
            "annotations": annotations,
            "categories": [{"id": 1, "name": "crop_tree"}]}
    json.dump(coco, open(os.path.join(GOLD, "coco.json"), "w"))
    json.dump(quarantine, open(os.path.join(GOLD, "quarantine.json"), "w"), indent=1)

    villas_imgs = [i["id"] for i in images if i["farm"] == "villas"]
    magdala_imgs = [i["id"] for i in images if i["farm"] == "magdala"]
    splits = {
        "split_A": {"train": villas_imgs, "val": magdala_imgs},
        "split_B": {"train": magdala_imgs, "val": villas_imgs},
        "note": "farm-wise splits, no random shuffle across farms",
    }
    json.dump(splits, open(os.path.join(GOLD, "splits.json"), "w"))

    def stats(a):
        a = sorted(a)
        if not a:
            return {}
        return {"n": len(a), "min": a[0], "median": a[len(a) // 2],
                "max": a[-1], "mean": round(sum(a) / len(a), 1)}

    report = {
        "images": len(images),
        "annotations_kept": len(annotations),
        "iscrowd": sum(1 for a in annotations if a["iscrowd"]),
        "quarantined": len(quarantine),
        "quarantine_reasons": {r: sum(1 for q in quarantine if q["reason"] == r)
                               for r in set(q["reason"] for q in quarantine)} if quarantine else {},
        "vertices_before": stats(vtx_before),
        "vertices_after": stats(vtx_after),
        "yolo": yolo_counts,
    }
    json.dump(report, open(os.path.join(GOLD, "conversion_report.json"), "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
