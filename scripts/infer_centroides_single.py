#!/usr/bin/env python3
"""Run the frozen centroid model on a single tile, CPU-only, foreground.

Replicates CenterNetLite and the cascade decode() exactly:
thr=0.5, topk=40, dist_factor=0.7, fixed per-farm radius (Magdala 45.4).
"""
import argparse, json, os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "experimentos/cascada/data/centroides/v1/model.pt"
RADII = {"villas": 58.5, "magdala": 45.4, "yulima": 35.9}

ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True)
ap.add_argument("--farm", default="magdala", choices=list(RADII))
ap.add_argument("--out", default=None)
args = ap.parse_args()

out = Path(args.out) if args.out else ROOT / "data" / "centroides_single" / Path(args.image).stem
out.mkdir(parents=True, exist_ok=True)


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


def decode(heat, thr=0.5, topk=40, dist_factor=0.7, fixed_r=RADII["magdala"]):
    heat = torch.sigmoid(heat.squeeze())
    mx = F.max_pool2d(heat[None, None], 3, 1, 1)[0, 0]
    ys, xs = ((heat >= mx) & (heat > thr)).nonzero(as_tuple=True)
    if len(xs) == 0:
        return []
    scores = heat[ys, xs]
    order = scores.argsort(descending=True)[:topk]
    cand = [(float(xs[k] * 4), float(ys[k] * 4), float(fixed_r), float(scores[k])) for k in order]
    kept = []
    for c in cand:
        if all((c[0] - d[0]) ** 2 + (c[1] - d[1]) ** 2 >= (dist_factor * (c[2] + d[2])) ** 2
               for d in kept):
            kept.append(c)
    return kept


net = CenterNetLite()
state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
net.load_state_dict(state)
net.eval()

img_path = Path(args.image)
raw = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
x = torch.from_numpy(raw).permute(2, 0, 1)[None]
with torch.no_grad():
    ph, pr = net(x)
peaks = decode(ph[0], fixed_r=RADII[args.farm])

centers = [{"x": round(p[0], 1), "y": round(p[1], 1),
            "r_px": p[2], "score": round(p[3], 4)} for p in peaks]
vis = Image.open(img_path).convert("RGB")
d = ImageDraw.Draw(vis)
for i, p in enumerate(centers):
    x0, y0, r = p["x"], p["y"], p["r_px"]
    d.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], outline=(255, 60, 60), width=2)
    d.ellipse([x0 - 3, y0 - 3, x0 + 3, y0 + 3], fill=(255, 255, 0))
    d.text((x0 + 5, y0 - 8), str(i + 1), fill=(255, 255, 255))
vis.save(out / "centroides_overlay.png")

report = {
    "image": img_path.name,
    "model": "centroides v1 (frozen, sha256 ec2bc282e40bd94a)",
    "farm": args.farm,
    "params": {"thr": 0.5, "topk": 40, "dist_factor": 0.7,
               "fixed_r": RADII[args.farm]},
    "n_centroides": len(centers),
    "centroides": centers,
}
json.dump(report, open(out / "centroides.json", "w"), indent=1)
print(json.dumps({"image": img_path.name, "n": len(centers),
                  "salida": str(out)}, indent=1))
