# %% [markdown]
# # Frente B — Generador de propuestas por centroides (v1)
#
# CenterNet-lite: ResNet-18 + FPN stride 4, heatmap focal (α=2, β=4) y radio SmoothL1.
# Entrena SOLO con el split train de gold_v2; dev (38) y los 9 frames auditados se excluyen.
# Holdout 20 intacto. Salida: `/kaggle/working/centroides_best.pt` + `centroides_report.json`.

# %%
import glob
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models

print("cuda:", torch.cuda.is_available())
assert torch.cuda.is_available(), "se requiere GPU T4"
torch.manual_seed(7)
random.seed(7)
np.random.seed(7)

roots = glob.glob("/kaggle/input/*") + glob.glob("/kaggle/input/datasets/*/*")
ds = [p for p in roots
      if os.path.exists(os.path.join(p, "coco.json"))
      and os.path.isdir(os.path.join(p, "all", "images"))]
assert len(ds) == 1, f"dataset no encontrado: {ds}"
BASE = ds[0]
DATA = os.path.join(BASE, "all")
IMG_DIR = os.path.join(DATA, "images")
print("BASE:", BASE)

OUT = "/kaggle/working"
report = {"dataset": BASE, "holdout_excluded": True}
def save_report():
    with open(os.path.join(OUT, "centroides_report.json"), "w") as fh:
        json.dump(report, fh, indent=1)

# %%
# --- Targets (idénticos a scripts/build_centroid_targets.py) ---
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
    area2 = cx = cy = 0.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        cr = x1 * y2 - x2 * y1
        area2 += cr
        cx += (x1 + x2) * cr
        cy += (y1 + y2) * cr
    if abs(area2) < 1e-9:
        return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
    return (cx / (3.0 * area2), cy / (3.0 * area2))

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

hold_raw = jload("holdout.json")
hold_names = set(hold_raw["holdout"] if isinstance(hold_raw, dict) else hold_raw)
dev_raw = jload("dev.json")
dev_ids = set(dev_raw.get("A", []) + dev_raw.get("B", []))
dev_names = {by_id[i] for i in dev_ids if i in by_id}

AUDITED_NAMES = [
    "Limon_LasVillas_00509.png", "Limon_LasVillas_00196.png",
    "Limon_LasVillas_00499.png", "Limon_LasVillas_00563.png",
    "Limon_LasVillas_00519.png", "Limon_LasVillas_00408.png",
    "Limon_LasVillas_00626.png", "Limon_Magdala_00866.png",
    "Limon_Magdala_00946.png",
]
present_names = set(by_id.values())
audited_names = {n for n in AUDITED_NAMES if n in present_names}
audited_missing = sorted(set(AUDITED_NAMES) - audited_names)
report["audited_missing"] = audited_missing
assert len(audited_names) >= 7, f"faltan demasiados frames auditados: {audited_missing}"

all_names = set(by_id.values())
train_names = sorted(all_names - hold_names - dev_names - audited_names)
assert not (set(train_names) & hold_names), "HOLDOUT LEAK"
assert not (set(train_names) & dev_names), "DEV LEAK"
assert not (set(train_names) & audited_names), "AUDIT LEAK"
print("total", len(all_names), "train", len(train_names),
      "dev", len(dev_names), "audited", len(audited_names),
      "holdout", len(hold_names))
report.update({"n_total": len(all_names), "n_train": len(train_names),
               "n_dev": len(dev_names), "n_audited": len(audited_names)})

# %%
class CrownTiles(Dataset):
    def __init__(self, names, augment=False):
        self.names = names
        self.augment = augment

    def __len__(self):
        return len(self.names)

    def _targets(self, name):
        heat, rad, w, centers = build_targets(anns_by_id.get(name_to_id[name], []))
        return (torch.from_numpy(heat), torch.from_numpy(rad),
                torch.from_numpy(w), centers)

    def __getitem__(self, i):
        name = self.names[i]
        with Image.open(os.path.join(IMG_DIR, name)) as im:
            img = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        heat, rad, w, centers = self._targets(name)
        if self.augment:
            if random.random() < 0.5:
                img = torch.flip(img, [2])
                heat = torch.flip(heat, [1])
                rad = torch.flip(rad, [1])
                w = torch.flip(w, [1])
            if random.random() < 0.5:
                img = torch.flip(img, [1])
                heat = torch.flip(heat, [0])
                rad = torch.flip(rad, [0])
                w = torch.flip(w, [0])
            angle = math.radians(random.uniform(-15, 15))
            scale = random.uniform(0.9, 1.1)
            cos_a, sin_a = math.cos(angle) * scale, math.sin(angle) * scale
            theta = torch.tensor([[[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]]])
            def warp(t, mode):
                channels = 1 if t.ndim == 2 else t.shape[0]
                shape = (1, channels) + tuple(t.shape[-2:])
                grid = F.affine_grid(theta, shape, align_corners=False)
                inp = t[None, None] if t.ndim == 2 else t[None]
                return F.grid_sample(inp, grid, mode=mode, padding_mode="zeros",
                                     align_corners=False).squeeze()
            img = warp(img, "bilinear")
            heat = warp(heat, "bilinear")
            rad = warp(rad, "bilinear") * scale
            w = warp(w, "nearest")
            gains = torch.tensor([random.uniform(0.985, 1.015) for _ in range(3)])
            img = (img * gains.view(3, 1, 1)).clamp(0, 1)
        return img, heat, rad, w

# %%
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

def heat_loss(logits, target, pos_weight=100.0):
    return F.binary_cross_entropy_with_logits(
        logits.squeeze(1), target, pos_weight=torch.tensor(pos_weight))

def radius_loss(pred, target, w):
    diff = F.smooth_l1_loss(pred, torch.zeros_like(pred), reduction="none")
    return (diff * w).sum() / max(w.sum().item(), 1.0)

# %%
train_dl = DataLoader(CrownTiles(train_names, augment=True), batch_size=16,
                      shuffle=True, num_workers=2, pin_memory=True)
dev_dl = DataLoader(CrownTiles(sorted(dev_names), augment=False), batch_size=16,
                    shuffle=False, num_workers=2, pin_memory=True)

model = CenterNetLite().cuda()
opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)
best, bad, best_ep = 1e9, 0, -1

for ep in range(80):
    model.train()
    tot = 0.0
    for imgs, heat, rad, w in train_dl:
        imgs = imgs.cuda(non_blocking=True)
        heat, rad, w = heat.cuda(), rad.cuda(), w.cuda()
        ph, pr = model(imgs)
        loss = heat_loss(ph, heat) + 1.0 * radius_loss(pr, rad, w)
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += loss.item() * len(imgs)
    sched.step()
    model.eval()
    vl = 0.0
    with torch.no_grad():
        for imgs, heat, rad, w in dev_dl:
            ph, pr = model(imgs.cuda())
            vl += (heat_loss(ph, heat.cuda()) +
                   radius_loss(pr, rad.cuda(), w.cuda())).item() * len(imgs)
    vl /= max(len(dev_dl.dataset), 1)
    if vl < best - 1e-4:
        best, best_ep, bad = vl, ep, 0
        torch.save(model.state_dict(), os.path.join(OUT, "centroides_best.pt"))
    else:
        bad += 1
    if ep % 5 == 0:
        print(f"ep={ep} train={tot / len(train_dl.dataset):.4f} val={vl:.4f} bad={bad}")

torch.save(model.state_dict(), os.path.join(OUT, "centroides_last.pt"))
report["training"] = {"epochs_run": 80, "best_epoch": best_ep,
                      "best_val_loss": round(best, 5)}
model.load_state_dict(torch.load(os.path.join(OUT, "centroides_last.pt")))
model.eval()
save_report()

# %%
def decode(heat, rad, topk=40, thr=0.15, dist_factor=0.7):
    heat = heat.cpu()
    rad = rad.cpu()
    heat = torch.sigmoid(heat.squeeze())
    rad = rad.squeeze()
    mx = F.max_pool2d(heat[None, None], 3, 1, 1)[0, 0]
    ys, xs = ((heat >= mx) & (heat > thr)).nonzero(as_tuple=True)
    if len(xs) == 0:
        return []
    scores = heat[ys, xs]
    order = scores.argsort(descending=True)[:topk]
    cand = []
    for k in order:
        y, x = int(ys[k]), int(xs[k])
        rr = rad[max(0, y - 1): y + 2, max(0, x - 1): x + 2].max().item()
        cand.append((float(x * 4), float(y * 4), float(rr), float(scores[k])))
    kept = []
    for c in cand:
        ok = True
        for d in kept:
            md = dist_factor * (c[2] + d[2])
            if (c[0] - d[0]) ** 2 + (c[1] - d[1]) ** 2 < md ** 2:
                ok = False
                break
        if ok:
            kept.append(c)
    return kept

def evaluate(names):
    recalls, fps = [], []
    with torch.no_grad():
        for name in sorted(names):
            with Image.open(os.path.join(IMG_DIR, name)) as im:
                arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
            img = torch.from_numpy(arr).permute(2, 0, 1)[None].cuda()
            ph, pr = model(img)
            preds = decode(ph[0], pr[0])
            gold = build_targets(anns_by_id.get(name_to_id[name], []))[3]
            matched = 0
            for gx, gy, gr, _ in gold:
                tol = max(12.0, 0.25 * gr)
                if any((px - gx) ** 2 + (py - gy) ** 2 <= tol ** 2
                       for px, py, _, _ in preds):
                    matched += 1
            recalls.append(matched / max(len(gold), 1))
            fps.append(max(len(preds) - matched, 0))
    return {"proposal_recall": round(float(np.mean(recalls)), 4),
            "fp_per_frame": round(float(np.mean(fps)), 2),
            "frames": len(names)}

report["dev"] = evaluate(dev_names)
report["audited"] = evaluate(audited_names)
report["gates"] = {
    "E1_recall_audited_ge_0.85": report["audited"]["proposal_recall"] >= 0.85,
}
save_report()
print(json.dumps(report, indent=1))
