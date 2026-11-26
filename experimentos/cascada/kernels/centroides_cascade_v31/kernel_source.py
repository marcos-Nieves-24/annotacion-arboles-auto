# %% [markdown]
# # Frente B — Cascada: propuestas por centroides → SAM → validador
#
# Usa los pesos del kernel `limon-centroides-v1` como kernel source.
# Mide E2 (instance F1 ≥ 0.748) y E3 (precisión ≥ 0.802, es decir -2 pp vs baseline)
# en dev y holdout. Rechazos van a lista de revisión, nunca se borran en silencio.

# %%
import subprocess
import sys

for pkg in ["segment-anything", "opencv-python", "pycocotools"]:
    try:
        __import__(pkg.replace("-", "_"))
        print(pkg, "presente")
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
        print(pkg, "instalado")

import glob
import json
import math
import os
import urllib.request

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models
from segment_anything import sam_model_registry, SamPredictor

assert torch.cuda.is_available(), "se requiere GPU T4"
torch.manual_seed(7)
np.random.seed(7)
OUT = "/kaggle/working"

W = glob.glob("/kaggle/input/**/centroides_last.pt", recursive=True)
assert W, "no apareció centroides_last.pt del kernel source"
WEIGHTS = W[0]
print("WEIGHTS:", WEIGHTS)

roots = glob.glob("/kaggle/input/*") + glob.glob("/kaggle/input/datasets/*/*")
ds = [p for p in roots
      if os.path.exists(os.path.join(p, "coco.json"))
      and os.path.isdir(os.path.join(p, "all", "images"))]
assert len(ds) == 1, f"dataset ambiguo: {ds}"
BASE = ds[0]
DATA = os.path.join(BASE, "all")
IMG_DIR = os.path.join(DATA, "images")
print("BASE:", BASE)

report = {"weights": WEIGHTS, "thr": 0.5, "dist_factor": 0.7,
          "radius_source": "estimator A: heatmap blob + NN fallback, clipped to [0.6,1.6]*farm median",
          "blob_factor": 2.17, "blob_thr": 0.5, "blob_window": 10,
          "use_prior": True,
          "use_voronoi": False,
          "baseline_precision": 0.822}

# %%
# --- Modelo y decode (idénticos al kernel v5) ---
def _as_ring(poly):
    if not poly:
        return []
    if isinstance(poly[0], (int, float)):
        return poly
    return poly[0]

def polygon_centroid(poly):
    ring = _as_ring(poly)
    pts = [(float(ring[i]), float(ring[i + 1])) for i in range(0, len(ring) - 1, 2)]
    if not pts:
        return (0.0, 0.0)
    a2 = cx = cy = 0.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        cr = x1 * y2 - x2 * y1
        a2 += cr
        cx += (x1 + x2) * cr
        cy += (y1 + y2) * cr
    if abs(a2) < 1e-9:
        return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
    return (cx / (3.0 * a2), cy / (3.0 * a2))

def equivalent_radius(area):
    return math.sqrt(max(float(area), 0.0) / math.pi)

def build_targets(anns, size=512, stride=4, sigma_frac=0.25,
                  sigma_min=2.0, sigma_max=16.0):
    h = size // stride
    heat = np.zeros((h, h), dtype=np.float32)
    radius = np.zeros((h, h), dtype=np.float32)
    weight = np.zeros((h, h), dtype=np.float32)
    centers = []
    gy, gx = np.indices((h, h), dtype=np.float32)
    for ann in anns:
        ring = _as_ring(ann.get("segmentation", []))
        if not ring:
            continue
        cx, cy = polygon_centroid(ring)
        r = equivalent_radius(ann.get("area") or 0.0)
        sig = min(max(sigma_frac * r, sigma_min), sigma_max) / stride
        d2 = (gx - cx / stride) ** 2 + (gy - cy / stride) ** 2
        core = np.exp(-d2 / (2.0 * sig ** 2)) >= 0.5
        pts = [(float(ring[i]), float(ring[i + 1])) for i in range(0, len(ring) - 1, 2)]
        clipped = any(x <= 0 or x >= size or y <= 0 or y >= size for x, y in pts)
        w = 0.5 if clipped else 1.0
        heat[core] = 1.0
        radius[core] = r
        weight[core] = w
        centers.append((cx, cy, r, w))
    return heat, radius, weight, centers

class CenterNetLite(nn.Module):
    def __init__(self):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.l1, self.l2, self.l3, self.l4 = m.layer1, m.layer2, m.layer3, m.layer4
        self.c1 = nn.Conv2d(64, 64, 1)
        self.c2 = nn.Conv2d(128, 64, 1)
        self.c3 = nn.Conv2d(256, 64, 1)
        self.c4 = nn.Conv2d(512, 64, 1)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.smooth = nn.Conv2d(64, 64, 3, padding=1)
        self.heat = nn.Conv2d(64, 1, 1)
        self.rad = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        p1 = self.l1(self.stem(x))
        p2 = self.l2(p1)
        p3 = self.l3(p2)
        p4 = self.l4(p3)
        f = self.c4(p4)
        f = self.up(f) + self.c3(p3)
        f = self.up(f) + self.c2(p2)
        f = self.up(f) + self.c1(p1)
        f = self.smooth(f)
        return self.heat(f), F.softplus(self.rad(f))

def decode(heat, rad=None, topk=40, thr=0.5, dist_factor=0.7, fixed_r=58.0,
           blob_thr=0.5, blob_factor=2.17, stride=4, win=10, lo=0.6, hi=1.6):
    """Peaks from the heatmap; per-crown radius from the heat blob (estimator A).

    r = blob_factor * sqrt(N_{heat>blob_thr}/pi) * stride, with the nearest-neighbour
    distance as fallback and a clip to [lo, hi] * farm median. The radius head of
    the v1 model collapsed (~0.03) and is not used.
    """
    heat = torch.sigmoid(heat.squeeze())
    hm = heat.cpu().numpy()
    mx = F.max_pool2d(heat[None, None], 3, 1, 1)[0, 0]
    ys, xs = ((heat >= mx) & (heat > thr)).nonzero(as_tuple=True)
    if len(xs) == 0:
        return []
    scores = heat[ys, xs]
    order = scores.argsort(descending=True)[:topk]
    peaks = []
    for k in order:
        y, x = int(ys[k]), int(xs[k])
        y0, y1 = max(0, y - win), min(hm.shape[0], y + win + 1)
        x0, x1 = max(0, x - win), min(hm.shape[1], x + win + 1)
        n = int((hm[y0:y1, x0:x1] >= blob_thr).sum())
        r_blob = blob_factor * math.sqrt(n / math.pi) * stride if n > 0 else 0.0
        peaks.append((x, y, r_blob, float(scores[k])))
    # NMS first, using the blob radius (or the farm median when unavailable).
    # Nearest-neighbour fallback must be computed on SURVIVORS, otherwise the
    # duplicate peaks make every distance tiny and every radius hits the floor.
    rough = []
    for x, y, r_blob, sc in peaks:
        rr = r_blob if r_blob > 0 else fixed_r
        rough.append((float(x * stride), float(y * stride), float(rr), sc))
    survivors = []
    for c in rough:
        ok = True
        for d in survivors:
            md = dist_factor * (c[2] + d[2])
            if (c[0] - d[0]) ** 2 + (c[1] - d[1]) ** 2 < md ** 2:
                ok = False
                break
        if ok:
            survivors.append(c)
    kept = []
    for i, (px, py, r_blob, sc) in enumerate(survivors):
        ds = [math.hypot(px - px2, py - py2)
              for j, (px2, py2, _, _) in enumerate(survivors) if j != i]
        r_nn = min(ds) / 2 if ds else fixed_r
        r = r_blob if r_blob > 0 else r_nn
        if r_blob > 0 and r_blob > 1.5 * r_nn:   # merged blob guard
            r = r_nn
        r = max(lo * fixed_r, min(hi * fixed_r, r))
        kept.append((px, py, float(r), sc))
    return kept

net = CenterNetLite().cuda()
net.load_state_dict(torch.load(WEIGHTS, map_location="cuda", weights_only=True))
net.eval()
print("centroides listo")

# %%
SAM_CKPT = os.path.join(OUT, "sam_vit_b_01ec64.pth")
if not os.path.exists(SAM_CKPT):
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    urllib.request.urlretrieve(url, SAM_CKPT)
print("SAM ckpt:", os.path.getsize(SAM_CKPT), "bytes")
sam = sam_model_registry["vit_b"](checkpoint=SAM_CKPT).to("cuda")
predictor = SamPredictor(sam)

# %%
def jload(fn):
    for d in (DATA, BASE):
        p = os.path.join(d, fn)
        if os.path.exists(p):
            with open(p) as fh:
                return json.load(fh)
    raise FileNotFoundError(fn)

coco = jload("coco.json")
by_id = {im["id"]: im["file_name"] for im in coco["images"]}
name_to_id = {v: k for k, v in by_id.items()}
anns_by_id = {}
for a in coco["annotations"]:
    anns_by_id.setdefault(a["image_id"], []).append(a)

dev = jload("dev.json")
dev_names = {by_id[i] for i in dev.get("A", []) + dev.get("B", []) if i in by_id}
hold_raw = jload("holdout.json")
hold_names = set(hold_raw["holdout"] if isinstance(hold_raw, dict) else hold_raw)
report["n_dev"] = len(dev_names)
report["n_holdout"] = len(hold_names)

def gold_masks(anns):
    out = []
    for a in anns:
        m = np.zeros((512, 512), np.uint8)
        for poly in a.get("segmentation", []):
            pts = np.asarray(poly, dtype=np.int32).reshape(-1, 2)
            if len(pts) >= 3:
                cv2.fillPoly(m, [pts], 1)
        if m.sum() > 0:
            out.append(m.astype(bool))
    return out

def match_masks(preds, golds, thr=0.5):
    pairs = []
    for i, p in enumerate(preds):
        for j, g in enumerate(golds):
            inter = np.count_nonzero(p & g)
            union = np.count_nonzero(p | g)
            if union:
                pairs.append((inter / union, i, j))
    pairs.sort(reverse=True)
    used_p, used_g, tp = set(), set(), 0
    for iou, i, j in pairs:
        if iou < thr or i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        tp += 1
    return tp, len(preds) - tp, len(golds) - tp

MEDIAN_AREA = {"villas": 9798.0, "magdala": 6636.0}
MEDIAN_R = {"villas": 58.5, "magdala": 45.4}
AUDITED_NAMES = {
    "Limon_LasVillas_00509.png", "Limon_LasVillas_00196.png",
    "Limon_LasVillas_00499.png", "Limon_LasVillas_00563.png",
    "Limon_LasVillas_00519.png", "Limon_LasVillas_00408.png",
    "Limon_LasVillas_00626.png", "Limon_Magdala_00866.png",
    "Limon_Magdala_00946.png",
}
OVERLAY_DIR = os.path.join(OUT, "overlays")
os.makedirs(OVERLAY_DIR, exist_ok=True)

def draw_overlay(img, kept, rejected, golds, split, name):
    """Green = accepted SAM mask, yellow = gold contour, red = rejected proposal."""
    ov = img.copy()
    for g in golds:
        cnts, _ = cv2.findContours(g.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cv2.drawContours(ov, cnts, -1, (0, 220, 220), 1)
    for _, m in kept:
        ov[m] = (0.45 * ov[m] + 0.55 * np.array([60, 230, 90])).astype(np.uint8)
    for x in rejected:
        cx, cy = int(x["center"][0]), int(x["center"][1])
        cv2.circle(ov, (cx, cy), max(3, int(x["r"])), (60, 60, 255), 2)
    Image.fromarray(ov).save(os.path.join(OVERLAY_DIR, f"{split}_{name}"))

def canopy_area(img):
    r, g, b = img[..., 0].astype(int), img[..., 1].astype(int), img[..., 2].astype(int)
    veg = (g > r * 1.05) & (g > b * 1.05) & (g > 60)
    return int(veg.sum())

def validate(props, masks, img, name):
    kept, rejected = [], []
    farm = "magdala" if "Magdala" in name else "villas"
    for prop, m in zip(props, masks):
        cx, cy, r, score = prop
        area = float(m.sum())
        reason = None
        if not (0.35 * math.pi * r * r <= area <= 2.0 * math.pi * r * r):
            reason = "area"
        else:
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                reason = "contour"
            else:
                cnt = max(cnts, key=cv2.contourArea)
                per = cv2.arcLength(cnt, True)
                hull_area = cv2.contourArea(cv2.convexHull(cnt))
                circ = 4 * math.pi * area / (per * per) if per > 0 else 0
                solid = area / hull_area if hull_area > 0 else 0
                if circ < 0.30 or solid < 0.65:
                    reason = "forma"
        if reason is None:
            kept.append((prop, m))
        else:
            rejected.append({"center": [round(cx, 1), round(cy, 1)], "r": round(r, 1),
                             "reason": reason})
    # NMS de máscaras por IoU >= 0.5, queda el de mayor score
    order = sorted(range(len(kept)), key=lambda k: -kept[k][0][3])
    survivors = []
    for k in order:
        prop, m = kept[k]
        dup = False
        for _, m2 in survivors:
            inter = np.count_nonzero(m & m2)
            union = np.count_nonzero(m | m2)
            if union and inter / union >= 0.5:
                dup = True
                break
        if dup:
            rejected.append({"center": [round(prop[0], 1), round(prop[1], 1)],
                             "r": round(prop[2], 1), "reason": "nms"})
        else:
            survivors.append((prop, m))
    # conteo esperado como alarma blanda: nunca inventa, solo recorta exceso
    expected = canopy_area(img) / MEDIAN_AREA[farm]
    if len(survivors) > 1.3 * expected:
        limit = max(1, int(1.1 * expected + 1))
        for prop, m in survivors[limit:]:
            rejected.append({"center": [round(prop[0], 1), round(prop[1], 1)],
                             "r": round(prop[2], 1), "reason": "conteo"})
        survivors = survivors[:limit]
    return survivors, rejected

# %%
def select_mask_by_prior(cands, cx, cy, r):
    """Pick among SAM's 3 masks using OUR centre and area prior, not SAM's score."""
    target = math.pi * r * r
    best_i, best = 0, None
    for i, m in enumerate(cands):
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            continue
        cost = (math.hypot(float(xs.mean()) - cx, float(ys.mean()) - cy) / max(r, 1.0)
                + abs(float(m.sum()) - target) / target)
        if best is None or cost < best:
            best, best_i = cost, i
    return cands[best_i]

def resolve_voronoi(masks, props):
    """Reassign disputed pixels (two masks claiming them) to the nearer centre."""
    n = len(masks)
    if n < 2:
        return masks, 0
    yy, xx = np.ogrid[:512, :512]
    d2 = [(xx - cx) ** 2 + (yy - cy) ** 2 for cx, cy, _, _ in props]
    out = [m.copy() for m in masks]
    disputed = 0
    for i in range(n):
        for j in range(i + 1, n):
            ov = out[i] & out[j]
            if not ov.any():
                continue
            disputed += int(ov.sum())
            di, dj = d2[i][ov], d2[j][ov]
            idx = np.argwhere(ov)
            out[i][idx[di > dj]] = False
            out[j][idx[di <= dj]] = False
    return out, disputed

def evaluate(names, split, max_overlay=8):
    tp_sum = fp_sum = fn_sum = 0
    count_errors = []
    disputed_total = []
    n_props = n_kept = 0
    review = {}
    saved = 0
    with torch.no_grad():
        for name in sorted(names):
            img = np.asarray(Image.open(os.path.join(IMG_DIR, name)).convert("RGB"),
                             dtype=np.uint8)
            predictor.set_image(img)
            x = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)[None].cuda()
            ph, pr = net(x)
            farm = "magdala" if "Magdala" in name else "villas"
            props = decode(ph, pr, thr=report["thr"],
                           dist_factor=report["dist_factor"],
                           fixed_r=MEDIAN_R[farm])
            masks = []
            for cx, cy, r, _ in props:
                box = np.array([max(0, cx - 1.15 * r), max(0, cy - 1.15 * r),
                                min(512, cx + 1.15 * r), min(512, cy + 1.15 * r)])
                ms, scores, _ = predictor.predict(box=box, multimask_output=True)
                masks.append(select_mask_by_prior(ms, cx, cy, r) if report["use_prior"] else ms[int(np.argmax(scores))])
            if report["use_voronoi"]:
                masks, n_disputed = resolve_voronoi(masks, props)
            else:
                n_disputed = 0
            disputed_total.append(n_disputed)
            kept, rejected = validate(props, masks, img, name)
            n_props += len(props)
            n_kept += len(kept)
            golds = gold_masks(anns_by_id.get(name_to_id[name], []))
            if saved < max_overlay or name in AUDITED_NAMES:
                draw_overlay(img, kept, rejected, golds, split, name)
                saved += 1
            a, b, c = match_masks([m for _, m in kept], golds)
            tp_sum += a
            fp_sum += b
            fn_sum += c
            count_errors.append(abs(len(kept) - len(golds)))
            if rejected:
                review[name] = rejected
    p = tp_sum / max(tp_sum + fp_sum, 1)
    r = tp_sum / max(tp_sum + fn_sum, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    metrics = {"precision": round(float(p), 4), "recall": round(float(r), 4),
               "F1": round(float(f1), 4), "tp": tp_sum, "fp": fp_sum, "fn": fn_sum,
               "propuestas_por_frame": round(n_props / max(len(names), 1), 2),
               "aceptadas_por_frame": round(n_kept / max(len(names), 1), 2),
               "count_MAE": round(float(np.mean(count_errors)), 2),
               "revision_frames": len(review),
               "pixeles_disputados_por_frame": int(np.mean(disputed_total)) if disputed_total else 0}
    return metrics, review

report["dev"], review_dev = evaluate(dev_names, "dev")
report["holdout"], review_holdout = evaluate(hold_names, "holdout")
with open(os.path.join(OUT, "cascade_review.json"), "w") as fh:
    json.dump({"dev": review_dev, "holdout": review_holdout}, fh, indent=1)
report["gates"] = {
    "E2_dev_F1_ge_0.748": report["dev"]["F1"] >= 0.748,
    "E2_holdout_F1_ge_0.748": report["holdout"]["F1"] >= 0.748,
    "E3_precision_drop_le_2pp": report["holdout"]["precision"] >= report["baseline_precision"] - 0.02,
}
with open(os.path.join(OUT, "centroides_cascade_report.json"), "w") as fh:
    json.dump(report, fh, indent=1)
print(json.dumps(report, indent=1))
